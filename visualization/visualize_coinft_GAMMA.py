#!/usr/bin/env python3

import json
import os
import queue
import socket
import struct
import threading
import time
from datetime import datetime
from threading import Event, Lock

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons


#########################
#  Configuration        #
#########################

# ---------- CoinFT serial ----------
COM_NAME = "COM7"
BAUD_RATE = 1_000_000
START_BYTE = 2
END_BYTE = 3
SERIAL_TIMEOUT_SECONDS = 0.1

# ---------- ATI Gamma SI-130 / Net F/T UDP-RDT ----------
SENSOR_IP = "192.168.1.1"
SENSOR_PORT = 49152

RDT_HEADER = 0x1234
CMD_STOP = 0x0000
CMD_START_REALTIME = 0x0002
RDT_RECORD_FMT = "!IIIiiiiii"
RDT_RECORD_SIZE = struct.calcsize(RDT_RECORD_FMT)
UDP_RECEIVE_BUFFER_BYTES = 4096
UDP_TIMEOUT_SECONDS = 0.5

# ATI RDT records contain integer counts. Replace these values with the
# counts-per-engineering-unit values for the active ATI calibration.
COUNTS_PER_FORCE = 1_000_000.0   # counts / N
COUNTS_PER_TORQUE = 1_000_000.0  # counts / Nm

# Used only to estimate timestamps when a UDP datagram contains multiple
# records. Set this to the active ATI RDT output rate.
ATI_OUTPUT_RATE_HZ = 100.0

# Validate fixed ATI conversion/timing parameters once at startup instead of
# re-checking or rebuilding them for every received sample.
if COUNTS_PER_FORCE <= 0 or COUNTS_PER_TORQUE <= 0:
    raise ValueError("COUNTS_PER_FORCE and COUNTS_PER_TORQUE must be positive.")
if ATI_OUTPUT_RATE_HZ <= 0:
    raise ValueError("ATI_OUTPUT_RATE_HZ must be positive.")

ATI_COUNT_SCALE = np.repeat(
    np.array([COUNTS_PER_FORCE, COUNTS_PER_TORQUE], dtype=np.float64), 3
)
ATI_SAMPLE_PERIOD = 1.0 / ATI_OUTPUT_RATE_HZ

# ---------- Synchronized tare ----------
# Both sensors are tared inside one common wall-clock window, so the ATI zero,
# the CoinFT input offset, and the CoinFT output bias describe the same
# unloaded instant.
TARE_SECONDS = 15.0
TARE_TIMEOUT_SECONDS = 20.0

# CoinFT is started and allowed to settle before the shared window opens.
COINFT_SETTLE_SECONDS = 2.0
COINFT_MIN_TARE_SAMPLES = 500
ATI_MIN_TARE_SAMPLES = 500

# Distance from the ATI reference sensing plane to the CoinFT sensing plane.
# Verify this value for the current mechanical mounting.
M_ARM = 0.0115  # m

# ---------- Plot and processing ----------
PLOT_DURATION = 10.0
MOVING_AVG_WINDOW = 30
MAX_QUEUE_SIZE = 10_000
ANIMATION_INTERVAL_MS = 50

CHANNEL_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_INDICES = [0, 1, 2]
TORQUE_INDICES = [3, 4, 5]

# Same channel uses the same color; sensor identity is represented by line style.
COLOR_MAP = ["red", "green", "blue"]

# ---------- File paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_MLP.onnx")
NORM_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_norm.json")


#########################
#  Utility functions    #
#########################


def counts_to_engineering_units(ft_counts):
    """Convert [Fx,Fy,Fz,Mx,My,Mz] counts to [N,N,N,Nm,Nm,Nm]."""
    return np.asarray(ft_counts, dtype=np.float64) / ATI_COUNT_SCALE


def put_queue_without_deadlock(target_queue, item):
    """Insert without blocking acquisition; if full, drop the oldest sample."""
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass


def moving_average_2d(arr, window=5):
    """Per-channel moving-average display filter."""
    if arr.shape[0] < window:
        return arr.copy()

    output = np.zeros_like(arr)
    kernel = np.ones(window, dtype=np.float64) / window

    for column in range(arr.shape[1]):
        output[:, column] = np.convolve(arr[:, column], kernel, mode="same")

    return output


#########################
#  Shared thread state  #
#########################


class WorkerStatus:
    """Carries a worker-thread exception back to the main thread."""

    def __init__(self, name):
        self.name = name
        self._lock = Lock()
        self._error = None

    def record(self, exc):
        with self._lock:
            if self._error is None:
                self._error = exc

    def raise_if_failed(self):
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"{self.name} failed: {error}")


class LatestSample:
    """Thread-safe holder for the most recent ATI sample."""

    def __init__(self):
        self._lock = Lock()
        self._value = None

    def set(self, value):
        with self._lock:
            self._value = np.asarray(value, dtype=np.float64).copy()

    def get(self):
        """Return the latest sample, or a six-NaN row when none exists yet."""
        with self._lock:
            if self._value is None:
                return np.full(6, np.nan, dtype=np.float64)
            return self._value.copy()

    def clear(self):
        with self._lock:
            self._value = None


class TareCollector:
    """Accumulates raw ATI samples while the shared tare window is open."""

    def __init__(self):
        self.active = Event()
        self._lock = Lock()
        self._samples = []

    def start(self):
        with self._lock:
            self._samples.clear()
        self.active.set()

    def stop(self):
        self.active.clear()

    def append(self, sample):
        with self._lock:
            self._samples.append(np.asarray(sample, dtype=np.float64).copy())

    def collected(self):
        with self._lock:
            return [sample.copy() for sample in self._samples]


#########################
#  CoinFT sensor        #
#########################


class ForceModel:
    """ONNX F/T model together with its normalization constants."""

    def __init__(self, model_path, norm_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not os.path.isfile(norm_path):
            raise FileNotFoundError(f"Normalization JSON not found: {norm_path}")

        print(f"Loading model from: {model_path}")
        print(f"Loading constants from: {norm_path}")

        self.model_path = model_path
        self.norm_path = norm_path

        self.session = ort.InferenceSession(model_path)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self._declared_input_dim = (
            model_input.shape[-1] if model_input.shape else None
        )

        with open(norm_path, "r", encoding="utf-8") as norm_file:
            norm_data = json.load(norm_file)

        self.mu_x = np.asarray(norm_data["mu_x"], dtype=np.float64)
        self.sd_x = np.asarray(norm_data["sd_x"], dtype=np.float64)
        self.mu_y = np.asarray(norm_data["mu_y"], dtype=np.float64)
        self.sd_y = np.asarray(norm_data["sd_y"], dtype=np.float64)

        if self.mu_x.shape != self.sd_x.shape:
            raise ValueError("mu_x and sd_x have different shapes.")
        if not (self.mu_y.shape == self.sd_y.shape == (6,)):
            raise ValueError("mu_y and sd_y must each contain six values.")
        if np.any(self.sd_x == 0) or np.any(self.sd_y == 0):
            raise ValueError("Normalization standard deviations must be nonzero.")

        self.input_dim = int(self.mu_x.size)
        self.feature_mode = None

    def bind_channels(self, num_channels):
        """Decide whether this model expects raw or raw+quadratic features."""
        if self.input_dim == num_channels:
            self.feature_mode = "linear"
        elif self.input_dim == 2 * num_channels:
            self.feature_mode = "quadratic"
        else:
            raise ValueError(
                "Normalization input dimension is incompatible with CoinFT "
                f"channels: mu_x has {self.input_dim} values, CoinFT has "
                f"{num_channels} channels."
            )

        if (
            isinstance(self._declared_input_dim, int)
            and self._declared_input_dim != self.input_dim
        ):
            raise ValueError(
                "ONNX model input dimension does not match normalization data: "
                f"model={self._declared_input_dim}, "
                f"normalization={self.input_dim}."
            )

        print(
            f"Model input mode: {self.feature_mode} "
            f"({self.input_dim} input features)"
        )

    def _features(self, offsetted):
        if self.feature_mode == "linear":
            return offsetted
        return np.hstack([offsetted, offsetted**2])

    def predict(self, offsetted):
        """Normalization -> ONNX inference -> denormalization."""
        features = self._features(offsetted)

        if features.size != self.input_dim:
            raise RuntimeError(
                "Prepared model feature count is incorrect: "
                f"expected {self.input_dim}, got {features.size}."
            )

        x_norm = (features - self.mu_x) / self.sd_x
        x_input = x_norm.astype(np.float32).reshape(1, -1)

        prediction_normalized = self.session.run(
            None,
            {self.input_name: x_input},
        )[0].flatten()

        if prediction_normalized.size != 6:
            raise RuntimeError(
                "ONNX output must contain six values; "
                f"received {prediction_normalized.size}."
            )

        return prediction_normalized * self.sd_y + self.mu_y


class CoinFTSensor:
    """CoinFT serial transport plus the tare constants that belong to it."""

    def __init__(self, port, baud, model):
        self.port = port
        self.baud = baud
        self.model = model
        self.serial = None
        self.packet_size = 0
        self.num_channels = 0
        self.offset = None
        self.bias = None
        self.framing_errors = 0

    def open(self):
        """Run the CoinFT handshake and match the model to the channel count."""
        self.serial = serial.Serial(
            self.port, self.baud, timeout=SERIAL_TIMEOUT_SECONDS
        )
        try:
            self.serial.write(b"i")
            time.sleep(0.2)
            self.serial.reset_input_buffer()
            self.serial.write(b"q")
            time.sleep(0.01)

            packet_size_raw = self.serial.read(1)
            if len(packet_size_raw) < 1:
                raise RuntimeError("Failed to read packet size from CoinFT.")

            self.packet_size = packet_size_raw[0] - 1
            self.num_channels = (self.packet_size - 1) // 2

            if self.packet_size <= 1 or self.num_channels <= 0:
                raise RuntimeError(
                    f"Invalid CoinFT packet size: {self.packet_size}"
                )

            print(
                "CoinFT ready. "
                f"Channels: {self.num_channels}; packet bytes after start byte: "
                f"{self.packet_size}"
            )

            self.model.bind_channels(self.num_channels)
        except Exception:
            self.close()
            raise

    def start_streaming(self):
        self.flush_input()
        self.serial.write(b"s")

    def stop_streaming(self):
        if self.serial is not None and self.serial.is_open:
            self.serial.write(b"i")

    def flush_input(self):
        self.serial.reset_input_buffer()

    def read_packet(self):
        """
        Read and validate a single CoinFT packet.

        Returns the raw per-channel values as a float64 array, or None when the
        packet was incomplete or badly framed.
        """
        first_byte = self.serial.read(1)
        if not first_byte or first_byte[0] != START_BYTE:
            return None

        data = self.serial.read(self.packet_size)
        if len(data) < self.packet_size:
            return None

        if data[-1] != END_BYTE:
            self.framing_errors += 1
            return None

        sensor_values = []
        for byte_index in range(0, self.packet_size - 1, 2):
            low = data[byte_index]
            high = data[byte_index + 1]
            sensor_values.append(low + 256 * high)

        sensor_data = np.asarray(sensor_values, dtype=np.float64)
        if sensor_data.size != self.num_channels:
            self.framing_errors += 1
            return None

        return sensor_data

    def predict_raw(self, raw):
        """Model output for one raw packet, before the output bias is known."""
        return self.model.predict(raw - self.offset)

    def process(self, raw):
        """Return (offset-removed channels, fully tared F/T) for one packet."""
        offsetted = raw - self.offset
        return offsetted, self.model.predict(offsetted) - self.bias

    def close(self):
        if self.serial is None:
            return
        try:
            if self.serial.is_open:
                self.serial.close()
        finally:
            self.serial = None


#########################
#  ATI UDP/RDT          #
#########################


class AtiRdtTransport:
    """ATI Net F/T UDP/RDT transport that owns its own zero vector."""

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.tare = np.zeros(6, dtype=np.float64)
        self.diagnostics = {
            "sequence_gaps": 0,
            "estimated_missing_records": 0,
            "nonzero_status_records": 0,
            "short_datagrams": 0,
        }
        self._last_rdt_seq = None
        self._status_warning_printed = False
        self._gap_prints = 0

    @staticmethod
    def _request(command):
        """Build an ATI RDT command packet."""
        return struct.pack("!HHI", RDT_HEADER, command, 0)

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(UDP_TIMEOUT_SECONDS)
        self.sock.bind(("", 0))
        print(f"ATI UDP socket bound to local port {self.sock.getsockname()[1]}")

    def drain(self):
        """Discard already-buffered UDP datagrams without blocking."""
        old_timeout = self.sock.gettimeout()
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    self.sock.recvfrom(UDP_RECEIVE_BUFFER_BYTES)
                except BlockingIOError:
                    break
                except OSError:
                    break
        finally:
            self.sock.settimeout(old_timeout)

    def start_streaming(self):
        self.drain()
        self.sock.sendto(self._request(CMD_START_REALTIME), (self.ip, self.port))
        print(f"ATI UDP streaming started from {self.ip}:{self.port}.")

    def stop_streaming(self):
        if self.sock is None:
            return
        try:
            self.sock.sendto(self._request(CMD_STOP), (self.ip, self.port))
        except OSError:
            pass

    def receive_records(self):
        """
        Return (records, receive_time) for one datagram, or None on timeout.

        Each record is a raw engineering-unit F/T array; the tare and the
        reference-plane transform are applied later by tared_sample().
        """
        try:
            data, _ = self.sock.recvfrom(UDP_RECEIVE_BUFFER_BYTES)
        except socket.timeout:
            return None

        receive_time = time.time()
        complete_record_count = len(data) // RDT_RECORD_SIZE
        if complete_record_count == 0:
            self.diagnostics["short_datagrams"] += 1
            return None

        records = []
        for record_index in range(complete_record_count):
            start = record_index * RDT_RECORD_SIZE
            rdt_seq, _ft_seq, status, fx, fy, fz, mx, my, mz = struct.unpack(
                RDT_RECORD_FMT, data[start:start + RDT_RECORD_SIZE]
            )
            self._track_sequence(rdt_seq)
            self._track_status(status)
            records.append(counts_to_engineering_units([fx, fy, fz, mx, my, mz]))

        return records, receive_time

    def _track_sequence(self, rdt_seq):
        if self._last_rdt_seq is not None:
            expected = (self._last_rdt_seq + 1) & 0xFFFFFFFF
            if rdt_seq != expected:
                gap = (int(rdt_seq) - int(self._last_rdt_seq)) & 0xFFFFFFFF
                self.diagnostics["sequence_gaps"] += 1
                self.diagnostics["estimated_missing_records"] += max(0, gap - 1)
                if self._gap_prints < 10:
                    print(
                        "ATI warning: possible dropped/reordered RDT record "
                        f"(expected {expected}, received {rdt_seq})."
                    )
                    self._gap_prints += 1
        self._last_rdt_seq = rdt_seq

    def _track_status(self, status):
        if status == 0:
            return
        self.diagnostics["nonzero_status_records"] += 1
        if not self._status_warning_printed:
            print(
                f"ATI warning: nonzero sensor status word detected: 0x{status:08X}"
            )
            self._status_warning_printed = True

    def tared_sample(self, raw_ft):
        """raw -> tare subtraction -> CoinFT sensing-plane compensation."""
        sample = raw_ft - self.tare
        sample[3] += sample[1] * M_ARM
        sample[4] -= sample[0] * M_ARM
        return sample

    def close(self):
        if self.sock is None:
            return
        try:
            self.stop_streaming()
        finally:
            try:
                self.sock.close()
            finally:
                self.sock = None


#########################
#  Synchronized tare    #
#########################


def perform_synchronized_tare(sensor, ati, tare_collector, ati_status):
    """
    Tare both sensors inside a single shared unloaded window.

    The ATI zero, the CoinFT input offset, and the CoinFT output bias are all
    derived from the same wall-clock window, so no drift separates them. The ATI
    processing order is
        raw -> tare subtraction -> reference-plane compensation.

    The three constants are written onto the sensor/transport objects and the
    describing metadata is returned for the sidecar JSON.
    """
    print("Synchronized tare: keep BOTH the ATI reference and the CoinFT unloaded.")
    print("Do not touch the fixture until the tare reports completion.")

    # Start CoinFT streaming and let it settle, then discard the warm-up bytes.
    sensor.start_streaming()
    time.sleep(COINFT_SETTLE_SECONDS)
    sensor.flush_input()

    coinft_raw = []
    tare_collector.start()
    start = time.perf_counter()
    deadline = start + TARE_TIMEOUT_SECONDS
    timed_out = False

    try:
        while time.perf_counter() - start < TARE_SECONDS:
            ati_status.raise_if_failed()

            if time.perf_counter() > deadline:
                timed_out = True
                break

            packet = sensor.read_packet()
            if packet is not None:
                coinft_raw.append(packet)
    finally:
        tare_collector.stop()

    elapsed = time.perf_counter() - start

    if timed_out:
        raise RuntimeError(
            f"Synchronized tare timed out after {elapsed:.2f} s without "
            f"completing the {TARE_SECONDS:.2f} s window."
        )

    # ---- ATI zero ----
    ati_samples = tare_collector.collected()
    if len(ati_samples) < ATI_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(ati_samples)} ATI samples were received during tare "
            f"(minimum {ATI_MIN_TARE_SAMPLES}). Check SENSOR_IP, the PC Ethernet "
            "configuration, and the sensor streaming settings."
        )

    ati_stack = np.vstack(ati_samples)
    ati.tare = ati_stack.mean(axis=0)

    # ---- CoinFT input offset ----
    if len(coinft_raw) < COINFT_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(coinft_raw)} CoinFT samples were received during tare "
            f"(minimum {COINFT_MIN_TARE_SAMPLES}). Check the serial link and "
            "the configured baud rate."
        )

    coinft_stack = np.vstack(coinft_raw)
    sensor.offset = coinft_stack.mean(axis=0)

    # ---- CoinFT output bias ----
    # The model is nonlinear, so a zero input does not imply a zero output.
    predictions = np.vstack([sensor.predict_raw(row) for row in coinft_stack])
    sensor.bias = predictions.mean(axis=0)

    ati_std = ati_stack.std(axis=0)
    coinft_std = coinft_stack.std(axis=0)
    prediction_std = predictions.std(axis=0)

    print(
        f"Synchronized tare complete in {elapsed:.2f} s "
        f"(ATI {len(ati_samples)} samples, CoinFT {len(coinft_raw)} samples)."
    )
    print(f"  ATI_TARE          : {ati.tare}")
    print(f"  ATI std           : {ati_std}")
    print(f"  CoinFT offset std : {coinft_std}")
    print(f"  ft_bias           : {sensor.bias}")
    print(f"  ft_bias std       : {prediction_std}")
    print(
        "  Compare these std values against a known-good tare; an unusually "
        "large value means the fixture was disturbed during the window."
    )

    return {
        "tare_timestamp": datetime.now().isoformat(timespec="seconds"),
        "tare_seconds": elapsed,
        "ati_tare": ati.tare.tolist(),
        "ati_tare_std": ati_std.tolist(),
        "ati_tare_sample_count": len(ati_samples),
        "coinft_offset": sensor.offset.tolist(),
        "coinft_offset_std": coinft_std.tolist(),
        "coinft_tare_sample_count": len(coinft_raw),
        "ft_bias": sensor.bias.tolist(),
        "ft_bias_std": prediction_std.tolist(),
        "m_arm": M_ARM,
        "counts_per_force": COUNTS_PER_FORCE,
        "counts_per_torque": COUNTS_PER_TORQUE,
        "ati_sensor_ip": SENSOR_IP,
        "ati_output_rate_hz": ATI_OUTPUT_RATE_HZ,
        "model_feature_mode": sensor.model.feature_mode,
        "model_path": sensor.model.model_path,
        "norm_path": sensor.model.norm_path,
    }


#########################
#  Worker functions     #
#########################


def read_ati_udp(
    ati,
    latest_ati,
    ati_queue,
    stop_event,
    experiment_started,
    tare_collector,
    status,
):
    """
    Keep the ATI RDT stream running continuously.

    During the tare window raw engineering-unit samples are accumulated. Samples
    reach the plotting/CSV pipeline only once experiment_started is set.
    """
    last_sample_time = None

    try:
        ati.start_streaming()

        while not stop_event.is_set():
            try:
                received = ati.receive_records()
            except OSError as exc:
                if not stop_event.is_set():
                    print(f"ATI UDP receive error: {exc}")
                break

            if received is None:
                continue

            records, packet_receive_time = received

            # Estimate the first record's acquisition time so records in one
            # datagram do not all receive the same timestamp.
            first_estimated_time = packet_receive_time - (
                (len(records) - 1) * ATI_SAMPLE_PERIOD
            )
            if last_sample_time is not None:
                first_estimated_time = max(
                    first_estimated_time,
                    last_sample_time + ATI_SAMPLE_PERIOD,
                )

            for record_index, raw_ft in enumerate(records):
                sample_time = (
                    first_estimated_time + record_index * ATI_SAMPLE_PERIOD
                )
                last_sample_time = sample_time

                if tare_collector.active.is_set():
                    tare_collector.append(raw_ft)

                if not experiment_started.is_set():
                    continue

                sample_data = ati.tared_sample(raw_ft)
                latest_ati.set(sample_data)
                put_queue_without_deadlock(ati_queue, (sample_time, sample_data))
    except Exception as exc:
        status.record(exc)
        if not stop_event.is_set():
            print(f"ATI UDP thread error: {exc}")
    finally:
        ati.stop_streaming()
        print("ATI UDP streaming stopped.")


def read_coinft(sensor, latest_ati, sensor_queue, records, stop_event, status):
    """
    Read CoinFT serial data, run inference, record, and queue it.

    Both tare constants come from perform_synchronized_tare(), so this loop
    contains only steady-state logic.
    """
    try:
        while not stop_event.is_set():
            raw = sensor.read_packet()
            if raw is None:
                continue

            offsetted, calibrated_ft = sensor.process(raw)
            timestamp = time.time()
            ati_snapshot = latest_ati.get()

            # Raw channels are stored too, so the data can be re-tared offline.
            row = [timestamp]
            row.extend(raw.tolist())
            row.extend(offsetted.tolist())
            row.extend(ati_snapshot.tolist())
            row.extend(calibrated_ft.tolist())
            records.append(row)

            put_queue_without_deadlock(
                sensor_queue,
                (timestamp, calibrated_ft.copy()),
            )
    except Exception as exc:
        status.record(exc)
        if not stop_event.is_set():
            print(f"CoinFT read/inference error: {exc}")


#########################
#  Plotting             #
#########################


class PlotBuffer:
    """Scrolling (timestamp, channel-row) buffer fed by an acquisition queue."""

    def __init__(self, channels=6):
        self.times = np.array([], dtype=np.float64)
        self.data = np.empty((0, channels), dtype=np.float64)

    @property
    def size(self):
        return self.times.size

    def drain(self, source_queue):
        """Move every pending (timestamp, row) pair from a queue into the buffer."""
        times = []
        rows = []

        while True:
            try:
                timestamp, row = source_queue.get_nowait()
            except queue.Empty:
                break

            times.append(timestamp)
            rows.append(row)

        if times:
            self.times = np.concatenate(
                [self.times, np.asarray(times, dtype=np.float64)]
            )
            self.data = np.vstack(
                [self.data, np.asarray(rows, dtype=np.float64)]
            )

    def trim(self, cutoff):
        """Drop the samples that fell out of the scrolling window."""
        if self.times.size > 0:
            mask = self.times >= cutoff
            self.times = self.times[mask]
            self.data = self.data[mask]


class ChannelGroup:
    """One axis holding the ATI/CoinFT line pair of every plotted channel."""

    def __init__(self, axis, indices, ylabel, title):
        self.axis = axis
        self.indices = indices
        self.ati_lines = []
        self.coinft_lines = []

        for line_index, channel_index in enumerate(indices):
            label = CHANNEL_LABELS[channel_index]
            color = COLOR_MAP[line_index]

            ati_line, = axis.plot(
                [],
                [],
                label=f"ATI {label}",
                linestyle="-",
                color=color,
                linewidth=2,
            )
            coinft_line, = axis.plot(
                [],
                [],
                label=f"CoinFT {label}",
                linestyle=":",
                color=color,
                linewidth=2,
            )
            self.ati_lines.append(ati_line)
            self.coinft_lines.append(coinft_line)

        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(loc="upper left")
        axis.grid(True)

    def update(self, relative_time, ati_data, coinft_data, visibility):
        for line_index, channel_index in enumerate(self.indices):
            if visibility[channel_index]:
                self.ati_lines[line_index].set_data(
                    relative_time,
                    ati_data[:, channel_index],
                )
                self.coinft_lines[line_index].set_data(
                    relative_time,
                    coinft_data[:, channel_index],
                )
            else:
                self.ati_lines[line_index].set_data([], [])
                self.coinft_lines[line_index].set_data([], [])

    def autoscale(self):
        self.axis.relim(visible_only=True)
        self.axis.autoscale_view(scalex=False, scaley=True)


class LivePlot:
    """Owns the figure, the scrolling buffers, and the fixed time origin."""

    def __init__(self, ati_queue, sensor_queue, stop_event):
        self.ati_queue = ati_queue
        self.sensor_queue = sensor_queue
        self.stop_event = stop_event

        self.ati_buffer = PlotBuffer()
        self.coinft_buffer = PlotBuffer()
        self.time_origin = None
        self.visibility = [True] * 6

        self.figure, (ax_force, ax_torque) = plt.subplots(
            2,
            1,
            figsize=(14, 10),
            sharex=True,
        )
        plt.subplots_adjust(
            left=0.10,
            bottom=0.10,
            right=0.95,
            top=0.95,
            hspace=0.25,
        )

        self.force_group = ChannelGroup(
            ax_force, FORCE_INDICES, "Force (N)", "Forces"
        )
        self.torque_group = ChannelGroup(
            ax_torque, TORQUE_INDICES, "Torque (Nm)", "Torques"
        )
        self.time_axis = ax_torque
        ax_torque.set_xlabel("Time (s)")

        check_ax = plt.axes([0.01, 0.40, 0.08, 0.20])
        self.check_buttons = CheckButtons(check_ax, CHANNEL_LABELS, self.visibility)
        self.check_buttons.on_clicked(self._toggle_visibility)

        stop_ax = plt.axes([0.85, 0.02, 0.10, 0.05])
        self.stop_button = Button(stop_ax, "Stop")
        self.stop_button.on_clicked(self._on_stop)

        self.figure.canvas.mpl_connect("close_event", self._on_stop)

    def _toggle_visibility(self, label):
        channel_index = CHANNEL_LABELS.index(label)
        self.visibility[channel_index] = not self.visibility[channel_index]

    def _on_stop(self, _event):
        self.stop_event.set()

    def update(self, _frame):
        cutoff = time.time() - PLOT_DURATION

        self.ati_buffer.drain(self.ati_queue)
        self.coinft_buffer.drain(self.sensor_queue)
        self.ati_buffer.trim(cutoff)
        self.coinft_buffer.trim(cutoff)

        if self.coinft_buffer.size == 0:
            return

        # The display filter is applied only to the plotting buffer; the CSV
        # keeps the unfiltered post-tare sensor outputs.
        coinft_plot = moving_average_2d(
            self.coinft_buffer.data,
            window=MOVING_AVG_WINDOW,
        )

        if self.ati_buffer.size > 0:
            # Match the ATI signal to the CoinFT timestamps.
            matched_indices = np.clip(
                np.searchsorted(
                    self.ati_buffer.times,
                    self.coinft_buffer.times,
                    side="right",
                ) - 1,
                0,
                self.ati_buffer.size - 1,
            )
            ati_plot = self.ati_buffer.data[matched_indices, :]
        else:
            ati_plot = np.zeros_like(coinft_plot)

        if self.time_origin is None:
            self.time_origin = self.coinft_buffer.times[0]

        relative_time = self.coinft_buffer.times - self.time_origin

        for group in (self.force_group, self.torque_group):
            group.update(relative_time, ati_plot, coinft_plot, self.visibility)
            group.autoscale()

        # Fixed window until PLOT_DURATION is reached, then continuous scrolling.
        x_right = relative_time[-1]
        if x_right <= PLOT_DURATION:
            self.time_axis.set_xlim(0.0, PLOT_DURATION)
        else:
            self.time_axis.set_xlim(x_right - PLOT_DURATION, x_right)

    def run(self):
        animation = FuncAnimation(
            self.figure,
            self.update,
            interval=ANIMATION_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

        # Keep a live reference for the lifetime of plt.show().
        _ = animation
        plt.show()


#########################
#  Saving and cleanup   #
#########################


def save_recorded_data(records, sensor, tare_metadata):
    """Save the CSV plus a sidecar JSON holding the tare constants."""
    print("Saving data...")

    if not records:
        print("No data recorded; CSV file was not created.")
        return None

    raw_columns = [
        f"CoinFT_raw_{index + 1}" for index in range(sensor.num_channels)
    ]
    offset_columns = [
        f"CoinFT_offset_{index + 1}" for index in range(sensor.num_channels)
    ]
    ati_columns = [f"ATI_{label}" for label in CHANNEL_LABELS]
    coinft_columns = [f"CoinFT_calib_{label}" for label in CHANNEL_LABELS]
    columns = ["Time"] + raw_columns + offset_columns + ati_columns + coinft_columns

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(DATA_DIR, f"CoinFT_data_{timestamp_text}.csv")

    dataframe = pd.DataFrame(records, columns=columns)
    dataframe.to_csv(full_path, index=False)
    print(f"Saved {len(records)} records to: {full_path}")

    if tare_metadata is not None:
        meta_path = os.path.join(
            DATA_DIR, f"CoinFT_data_{timestamp_text}_tare.json"
        )
        try:
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(tare_metadata, meta_file, indent=2)
            print(f"Saved tare constants to: {meta_path}")
        except Exception as exc:
            print(f"WARNING: Could not write tare metadata: {exc}")

    return full_path


def close_hardware(sensor, ati):
    """Stop both streams and close the serial/socket resources."""
    try:
        sensor.stop_streaming()
    except Exception as exc:
        print(f"Warning while stopping CoinFT: {exc}")

    try:
        sensor.close()
    except Exception as exc:
        print(f"Warning while closing serial port: {exc}")

    try:
        ati.close()
    except Exception as exc:
        print(f"Warning while closing ATI socket: {exc}")


#########################
#  Main execution       #
#########################


def main():
    stop_event = Event()
    experiment_started = Event()
    tare_collector = TareCollector()
    ati_status = WorkerStatus("ATI UDP acquisition")
    coinft_status = WorkerStatus("CoinFT acquisition")
    latest_ati = LatestSample()

    ati_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    sensor_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    all_data_records = []

    model = ForceModel(MODEL_PATH, NORM_PATH)
    sensor = CoinFTSensor(COM_NAME, BAUD_RATE, model)
    ati = AtiRdtTransport(SENSOR_IP, SENSOR_PORT)

    tare_metadata = None
    ati_thread = None
    coinft_thread = None

    try:
        sensor.open()
        ati.open()

        # The RDT stream must stay active through the tare and while waiting for
        # the operator, so the ATI thread starts before the prompt.
        ati_thread = threading.Thread(
            target=read_ati_udp,
            args=(
                ati,
                latest_ati,
                ati_queue,
                stop_event,
                experiment_started,
                tare_collector,
                ati_status,
            ),
            daemon=True,
        )
        ati_thread.start()

        print("\n" + "=" * 58)
        print(" >>> CONFIRM BOTH SENSORS ARE UNLOADED")
        print(" >>> PRESS [ENTER] TO START THE SYNCHRONIZED TARE")
        print("=" * 58 + "\n")
        input()

        ati_status.raise_if_failed()
        tare_metadata = perform_synchronized_tare(
            sensor, ati, tare_collector, ati_status
        )
        ati_status.raise_if_failed()

        # CoinFT already streams from the tare; clear the bytes that queued up
        # while the tare predictions were being computed.
        sensor.flush_input()

        # No ATI samples were published before this point, so the plot and CSV
        # still start at the beginning of the experiment.
        latest_ati.clear()
        experiment_started.set()

        print("\n>>> DATA COLLECTION AND PLOTTING STARTED\n")

        coinft_thread = threading.Thread(
            target=read_coinft,
            args=(
                sensor,
                latest_ati,
                sensor_queue,
                all_data_records,
                stop_event,
                coinft_status,
            ),
            daemon=True,
        )
        coinft_thread.start()

        live_plot = LivePlot(ati_queue, sensor_queue, stop_event)
        live_plot.run()

        ati_status.raise_if_failed()
        coinft_status.raise_if_failed()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Stopping acquisition threads...")
        stop_event.set()
        experiment_started.clear()
        tare_collector.stop()

        if coinft_thread is not None:
            coinft_thread.join(timeout=3.0)
        if ati_thread is not None:
            ati_thread.join(timeout=3.0)

        close_hardware(sensor, ati)
        print(f"ATI diagnostics: {ati.diagnostics}")
        print(f"CoinFT framing errors: {sensor.framing_errors}")
        save_recorded_data(all_data_records, sensor, tare_metadata)
        print("Stopped.")


if __name__ == "__main__":
    main()
