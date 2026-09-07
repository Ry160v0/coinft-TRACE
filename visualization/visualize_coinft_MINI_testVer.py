#!/usr/bin/env python3
"""Live CoinFT vs. ATI Mini58 (EtherCAT/ECATBA) visualization.

ATI is read over EtherCAT via pysoem instead of NI-DAQ analog voltage, so no
strain-gauge calibration matrix is needed -- see
intergrated_ati_data_collection_MINI_fixed.py for the same transport used by
the calibration collector. ATI and CoinFT are tared together in one shared
wall-clock window so both zero references describe the same unloaded instant.
"""
import json
import os
import queue
import struct
import threading
import time
from datetime import datetime
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import pysoem
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons

#########################
#  Configuration        #
#########################

# ---------- CoinFT serial ----------
COM_NAME = "COM5"
BAUD_RATE = 1_000_000
START_BYTE = 2
END_BYTE = 3
SERIAL_TIMEOUT_SECONDS = 0.1

# ---------- ATI Mini58 + ECATBA / EtherCAT ----------
ATI_ADAPTER_NAME = None  # set explicitly (see pysoem's find_adapters.py) to skip auto-probing
ATI_VENDOR_ID = 0x00000732
ATI_OUTPUT_RATE_HZ = 1000.0
ATI_PROCESSDATA_TIMEOUT_US = 2000
ATI_STATE_TIMEOUT_US = 50000

# The ECATBA's default TxPDO is six int32 F/T counts (Fx Fy Fz Mx My Mz) at
# the start of the input image -- see ATI's read_vals.c reference. These
# fallback scale factors are replaced by the SDO 0x2040:49/50 values if
# they're readable.
COUNTS_PER_FORCE = 1_000_000.0
COUNTS_PER_TORQUE = 1_000_000.0

# ---------- Synchronized tare ----------
# ATI is zeroed and the CoinFT input offset + model output bias are all
# computed from the same wall-clock window, so all three describe the same
# unloaded instant.
TARE_SECONDS = 15.0
COINFT_SETTLE_SECONDS = 2.0
COINFT_MIN_TARE_SAMPLES = 500
ATI_MIN_TARE_SAMPLES = 500

# Distance from the ATI reference sensing plane to the CoinFT sensing plane.
M_ARM = 0.0115  # m

# ---------- Plot and processing ----------
PLOT_DURATION = 10.0
MOVING_AVG_WINDOW = 20
MAX_QUEUE_SIZE = 10_000
ANIMATION_INTERVAL_MS = 50

# ---------- File paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_MLP.onnx")
NORM_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_norm.json")


#########################
#  Global state         #
#########################

stop_flag = False
ati_lock = Lock()
ati_tare_lock = Lock()

ati_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
sensor_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

all_data_records = []

# Latest ATI value, used for the CSV snapshot recorded with each CoinFT row.
latest_ati = None

ATI_TARE = np.zeros(6, dtype=np.float64)

# Tare constants recorded alongside the CSV so a dataset can be re-tared later.
TARE_META = None

# EtherCAT stays cyclically active from before tare until shutdown. These
# events only control whether raw samples are accumulated for tare or
# published to the plotting/CSV pipeline.
ati_tare_collecting = threading.Event()
experiment_started = threading.Event()
ati_tare_samples = []

ati_thread_error = None
coinft_thread_error = None

ati_diagnostics = {"bad_wkc": 0, "coinft_framing_errors": 0}

# Fixed origin for the scrolling experiment-time axis.
plot_time_origin = None


#########################
#  Utility functions    #
#########################

def put_queue_without_deadlock(target_queue, item):
    """Insert without letting a full plotting queue block acquisition: drop
    the oldest queued sample and insert the newest one instead."""
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
    """Per-channel centered moving-average display filter. Edge samples are
    normalized by the number of contributing samples (not zero-padded), so
    the leading edge of a step response isn't visually pulled toward zero."""
    if window <= 1 or arr.shape[0] < window:
        return arr.copy()

    output = np.zeros_like(arr)
    kernel = np.ones(window, dtype=np.float64)
    normalizer = np.convolve(np.ones(arr.shape[0]), kernel, mode="same")

    for column in range(arr.shape[1]):
        output[:, column] = np.convolve(arr[:, column], kernel, mode="same") / normalizer

    return output


#########################
#  ATI EtherCAT         #
#########################

def find_ati_adapter():
    """Return the Npcap adapter name that sees an ATI EtherCAT slave."""
    if ATI_ADAPTER_NAME:
        return ATI_ADAPTER_NAME

    for adapter in pysoem.find_adapters():
        master = pysoem.Master()
        try:
            master.open(adapter.name)
            found = master.config_init() > 0 and any(
                slave.man == ATI_VENDOR_ID for slave in master.slaves
            )
        except Exception:
            found = False
        master.close()
        if found:
            return adapter.name

    raise RuntimeError("No ATI EtherCAT slave found on any adapter.")


def open_ati_master():
    """Bring the ATI slave to OP state. Returns (master, slave)."""
    master = pysoem.Master()
    master.open(find_ati_adapter())

    if master.config_init() <= 0:
        raise RuntimeError("No EtherCAT slave found.")
    slave = master.slaves[0]

    global COUNTS_PER_FORCE, COUNTS_PER_TORQUE
    try:
        counts_per_force = int.from_bytes(slave.sdo_read(0x2040, 49), "little", signed=True)
        counts_per_torque = int.from_bytes(slave.sdo_read(0x2040, 50), "little", signed=True)
        if counts_per_force <= 0 or counts_per_torque <= 0:
            raise ValueError(
                f"ECATBA returned non-positive scaling: force={counts_per_force}, torque={counts_per_torque}"
            )
        COUNTS_PER_FORCE, COUNTS_PER_TORQUE = counts_per_force, counts_per_torque
        print(f"ATI ECATBA scaling: counts_per_force={COUNTS_PER_FORCE:g}, counts_per_torque={COUNTS_PER_TORQUE:g}")
    except Exception as exc:
        print(f"Could not read ECATBA counts-per-unit SDOs ({exc}); using fallback scaling.")

    master.config_map()
    master.state_check(pysoem.SAFEOP_STATE, ATI_STATE_TIMEOUT_US)

    # Prime one valid exchange before requesting OP -- slaves refuse the OP
    # request until they have received at least one process-data frame.
    master.send_processdata()
    master.receive_processdata(ATI_PROCESSDATA_TIMEOUT_US)
    master.state = pysoem.OP_STATE
    master.write_state()

    for _ in range(40):
        master.send_processdata()
        master.receive_processdata(ATI_PROCESSDATA_TIMEOUT_US)
        if master.state_check(pysoem.OP_STATE, ATI_STATE_TIMEOUT_US) == pysoem.OP_STATE:
            break
    else:
        raise RuntimeError("EtherCAT network did not reach OP state.")

    print(f"ATI EtherCAT is OP. expected WKC={master.expected_wkc}, input bytes={len(slave.input)}")

    # Confirm the assumed fixed PDO layout (6x int32 Fx..Mz) actually parses
    # before the operator starts an experiment on top of it.
    if read_ati_sample(master, slave) is None:
        raise RuntimeError("ATI EtherCAT reached OP but no valid process-data sample was received.")

    return master, slave


def read_ati_sample(master, slave):
    """One EtherCAT cycle. Returns (ft, mono_time, unix_time) or None on a bad WKC."""
    master.send_processdata()
    wkc = master.receive_processdata(ATI_PROCESSDATA_TIMEOUT_US)
    mono = time.perf_counter()
    unix_time = time.time()

    if wkc < master.expected_wkc:
        ati_diagnostics["bad_wkc"] += 1
        return None

    fx, fy, fz, tx, ty, tz = struct.unpack_from("<6i", slave.input)
    ft = np.array([
        fx / COUNTS_PER_FORCE, fy / COUNTS_PER_FORCE, fz / COUNTS_PER_FORCE,
        tx / COUNTS_PER_TORQUE, ty / COUNTS_PER_TORQUE, tz / COUNTS_PER_TORQUE,
    ])
    return ft, mono, unix_time


#########################
#  Resource loading     #
#########################

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
if not os.path.isfile(NORM_PATH):
    raise FileNotFoundError(f"Normalization JSON not found: {NORM_PATH}")

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading constants from: {NORM_PATH}")

ort_session = ort.InferenceSession(MODEL_PATH)
model_input = ort_session.get_inputs()[0]
MODEL_INPUT_NAME = model_input.name

with open(NORM_PATH, "r", encoding="utf-8") as norm_file:
    norm_data = json.load(norm_file)

mu_x = np.asarray(norm_data["mu_x"], dtype=np.float64)
sd_x = np.asarray(norm_data["sd_x"], dtype=np.float64)
mu_y = np.asarray(norm_data["mu_y"], dtype=np.float64)
sd_y = np.asarray(norm_data["sd_y"], dtype=np.float64)

if mu_x.shape != sd_x.shape:
    raise ValueError("mu_x and sd_x have different shapes.")
if not (mu_y.shape == sd_y.shape == (6,)):
    raise ValueError("mu_y and sd_y must each contain six values.")
if np.any(sd_x == 0) or np.any(sd_y == 0):
    raise ValueError("Normalization standard deviations must be nonzero.")


#########################
#  Hardware setup       #
#########################

ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=SERIAL_TIMEOUT_SECONDS)
try:
    ser.write(b"i")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"q")
    time.sleep(0.01)

    packet_size_raw = ser.read(1)
    if len(packet_size_raw) < 1:
        raise RuntimeError("Failed to read packet size from CoinFT.")

    packet_size_exclude_start_byte = packet_size_raw[0] - 1
    num_channels = (packet_size_exclude_start_byte - 1) // 2

    if packet_size_exclude_start_byte <= 1 or num_channels <= 0:
        raise RuntimeError(f"Invalid CoinFT packet size: {packet_size_exclude_start_byte}")

    print(f"CoinFT ready. Channels: {num_channels}; packet bytes after start byte: {packet_size_exclude_start_byte}")
except Exception:
    ser.close()
    raise

# Determine whether the active model expects raw or raw+quadratic features.
normalization_input_dim = int(mu_x.size)
if normalization_input_dim == num_channels:
    MODEL_FEATURE_MODE = "linear"
elif normalization_input_dim == 2 * num_channels:
    MODEL_FEATURE_MODE = "quadratic"
else:
    ser.close()
    raise ValueError(
        "Normalization input dimension is incompatible with CoinFT channels: "
        f"mu_x has {normalization_input_dim} values, CoinFT has {num_channels} channels."
    )

model_shape_last_dim = model_input.shape[-1] if model_input.shape else None
if isinstance(model_shape_last_dim, int) and model_shape_last_dim != normalization_input_dim:
    ser.close()
    raise ValueError(
        "ONNX model input dimension does not match normalization data: "
        f"model={model_shape_last_dim}, normalization={normalization_input_dim}."
    )

print(f"Model input mode: {MODEL_FEATURE_MODE} ({normalization_input_dim} input features)")


#########################
#  Plot-data buffers    #
#########################

t_ati_store = np.array([], dtype=np.float64)
ati_data_store = np.empty((0, 6), dtype=np.float64)

t_sens_store = np.array([], dtype=np.float64)
sens_data_store = np.empty((0, 6), dtype=np.float64)


#########################
#  Acquisition helpers  #
#########################

def read_one_coinft_packet():
    """Read and validate a single CoinFT packet. Returns raw per-channel
    values, or None when the packet was incomplete or badly framed."""
    first_byte = ser.read(1)
    if not first_byte or first_byte[0] != START_BYTE:
        return None

    data = ser.read(packet_size_exclude_start_byte)
    if len(data) < packet_size_exclude_start_byte or data[-1] != END_BYTE:
        ati_diagnostics["coinft_framing_errors"] += 1
        return None

    sensor_values = [
        data[i] + 256 * data[i + 1]
        for i in range(0, packet_size_exclude_start_byte - 1, 2)
    ]
    sensor_data = np.asarray(sensor_values, dtype=np.float64)
    if sensor_data.size != num_channels:
        ati_diagnostics["coinft_framing_errors"] += 1
        return None

    return sensor_data


def prepare_model_features(sensor_data_offsetted):
    """Construct the feature vector expected by the active model/norm file."""
    if MODEL_FEATURE_MODE == "linear":
        return sensor_data_offsetted
    return np.hstack([sensor_data_offsetted, sensor_data_offsetted ** 2])


def predict_ft(sensor_data_offsetted):
    """Normalization -> ONNX -> denormalization."""
    model_features = prepare_model_features(sensor_data_offsetted)
    if model_features.size != normalization_input_dim:
        raise RuntimeError(
            f"Prepared model feature count is incorrect: expected {normalization_input_dim}, "
            f"got {model_features.size}."
        )

    x_norm = (model_features - mu_x) / sd_x
    x_input = x_norm.astype(np.float32).reshape(1, -1)
    prediction_normalized = ort_session.run(None, {MODEL_INPUT_NAME: x_input})[0].flatten()

    if prediction_normalized.size != 6:
        raise RuntimeError(f"ONNX output must contain six values; received {prediction_normalized.size}.")

    return prediction_normalized * sd_y + mu_y


#########################
#  Synchronized tare    #
#########################

def perform_synchronized_tare():
    """Tare ATI, the CoinFT input offset, and the CoinFT model output bias
    inside one shared unloaded window, so all three describe the same instant.
    The model is nonlinear, so the output bias is the mean prediction over the
    window rather than a single noisy sample."""
    global ATI_TARE

    print("Synchronized tare: keep BOTH the ATI reference and the CoinFT unloaded.")

    ser.reset_input_buffer()
    ser.write(b"s")
    time.sleep(COINFT_SETTLE_SECONDS)
    ser.reset_input_buffer()

    with ati_tare_lock:
        ati_tare_samples.clear()
    coinft_raw = []

    ati_tare_collecting.set()
    start = time.perf_counter()
    while time.perf_counter() - start < TARE_SECONDS:
        packet = read_one_coinft_packet()
        if packet is not None:
            coinft_raw.append(packet)
    ati_tare_collecting.clear()

    with ati_tare_lock:
        ati_samples = [ft for _mono, ft in ati_tare_samples]

    if len(ati_samples) < ATI_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(ati_samples)} ATI samples were received during tare "
            f"(minimum {ATI_MIN_TARE_SAMPLES})."
        )
    if len(coinft_raw) < COINFT_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(coinft_raw)} CoinFT samples were received during tare "
            f"(minimum {COINFT_MIN_TARE_SAMPLES})."
        )

    ati_stack = np.vstack(ati_samples)
    ATI_TARE = ati_stack.mean(axis=0)

    coinft_stack = np.vstack(coinft_raw)
    offset_coinft = coinft_stack.mean(axis=0)

    predictions = np.vstack([predict_ft(sample - offset_coinft) for sample in coinft_stack])
    ft_bias = predictions.mean(axis=0)

    print(f"Tare complete: ATI {len(ati_samples)} samples, CoinFT {len(coinft_raw)} samples.")
    print(f"ATI_TARE: {ATI_TARE}  (std {ati_stack.std(axis=0)})")
    print(f"CoinFT offset: {offset_coinft}  (std {coinft_stack.std(axis=0)})")
    print(f"ft_bias: {ft_bias}")

    metadata = {
        "tare_timestamp": datetime.now().isoformat(timespec="seconds"),
        "ati_tare": ATI_TARE.tolist(),
        "ati_tare_sample_count": len(ati_samples),
        "coinft_offset": offset_coinft.tolist(),
        "coinft_tare_sample_count": len(coinft_raw),
        "ft_bias": ft_bias.tolist(),
        "m_arm": M_ARM,
        "counts_per_force": COUNTS_PER_FORCE,
        "counts_per_torque": COUNTS_PER_TORQUE,
        "model_feature_mode": MODEL_FEATURE_MODE,
    }
    return offset_coinft, ft_bias, metadata


#########################
#  Worker functions     #
#########################

def read_ati_ethercat(master, slave):
    """Continuous ATI EtherCAT loop, paced at ATI_OUTPUT_RATE_HZ. Runs from
    just after EtherCAT setup through the end of acquisition. Samples land in
    the tare buffer or the plotting/CSV pipeline depending on which event is
    currently set (or neither, between the two)."""
    global stop_flag, latest_ati, ati_thread_error

    period = 1.0 / ATI_OUTPUT_RATE_HZ
    next_cycle = time.perf_counter()

    try:
        while not stop_flag:
            sample = read_ati_sample(master, slave)

            if sample is not None:
                raw_ft, mono, unix_time = sample

                if ati_tare_collecting.is_set():
                    with ati_tare_lock:
                        ati_tare_samples.append((mono, raw_ft.copy()))

                if experiment_started.is_set():
                    sample_data = raw_ft - ATI_TARE
                    sample_data[3] += sample_data[1] * M_ARM
                    sample_data[4] -= sample_data[0] * M_ARM

                    with ati_lock:
                        latest_ati = sample_data.copy()
                    put_queue_without_deadlock(ati_queue, (unix_time, sample_data))

            next_cycle += period
            now = time.perf_counter()
            if next_cycle > now:
                time.sleep(next_cycle - now)
            else:
                next_cycle = now

    except Exception as exc:
        ati_thread_error = exc
        if not stop_flag:
            print(f"ATI EtherCAT thread error: {exc}")
    finally:
        print("ATI EtherCAT cyclic exchange stopped.")


def read_coinft(offset_coinft, ft_bias):
    """Read CoinFT serial data, run ONNX inference, record, and queue it."""
    global stop_flag, coinft_thread_error

    while not stop_flag:
        try:
            sensor_data = read_one_coinft_packet()
            if sensor_data is None:
                continue

            sensor_data_offsetted = sensor_data - offset_coinft
            calibrated_ft = predict_ft(sensor_data_offsetted) - ft_bias
            timestamp = time.time()

            with ati_lock:
                ati_snapshot = np.full(6, np.nan) if latest_ati is None else latest_ati.copy()

            # Untared raw channels are recorded too, so the dataset can be
            # re-tared offline if a tare turns out to be bad.
            row = [
                timestamp,
                *sensor_data.tolist(),
                *sensor_data_offsetted.tolist(),
                *ati_snapshot.tolist(),
                *calibrated_ft.tolist(),
            ]
            all_data_records.append(row)

            put_queue_without_deadlock(sensor_queue, (timestamp, calibrated_ft.copy()))
        except Exception as exc:
            coinft_thread_error = exc
            if not stop_flag:
                print(f"CoinFT read/inference error: {exc}")
            break


#########################
#  Plotting logic       #
#########################

CHANNEL_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_INDICES = [0, 1, 2]
TORQUE_INDICES = [3, 4, 5]
visibility = [True] * 6


def _drain_plot_queue(source_queue, t_store, data_store):
    """Move every pending (timestamp, row) pair from a queue into the buffers."""
    times, rows = [], []
    while True:
        try:
            timestamp, data = source_queue.get_nowait()
        except queue.Empty:
            break
        times.append(timestamp)
        rows.append(data)

    if times:
        t_store = np.concatenate([t_store, np.asarray(times, dtype=np.float64)])
        data_store = np.vstack([data_store, np.asarray(rows, dtype=np.float64)])
    return t_store, data_store


def _trim_plot_buffer(t_store, data_store, cutoff):
    """Drop the samples that fell out of the scrolling window."""
    if t_store.size > 0:
        mask = t_store >= cutoff
        return t_store[mask], data_store[mask]
    return t_store, data_store


def update_plot(_frame):
    global t_ati_store, ati_data_store
    global t_sens_store, sens_data_store
    global plot_time_origin

    now = time.time()
    cutoff = now - PLOT_DURATION

    t_ati_store, ati_data_store = _drain_plot_queue(ati_queue, t_ati_store, ati_data_store)
    t_sens_store, sens_data_store = _drain_plot_queue(sensor_queue, t_sens_store, sens_data_store)

    t_ati_store, ati_data_store = _trim_plot_buffer(t_ati_store, ati_data_store, cutoff)
    t_sens_store, sens_data_store = _trim_plot_buffer(t_sens_store, sens_data_store, cutoff)

    if t_sens_store.size == 0:
        return

    # The moving-average filter is applied only to the plotting buffers, so
    # the values written to CSV remain the unfiltered post-tare sensor output.
    sens_plot_data = moving_average_2d(sens_data_store, window=MOVING_AVG_WINDOW)

    have_ati = t_ati_store.size > 0
    if have_ati:
        ati_plot_data = moving_average_2d(ati_data_store, window=MOVING_AVG_WINDOW)
        matched_indices = np.searchsorted(t_ati_store, t_sens_store, side="right") - 1
        matched_indices = np.clip(matched_indices, 0, t_ati_store.size - 1)
        ati_plot_matched = ati_plot_data[matched_indices, :]
    else:
        # Leave the ATI curves blank rather than drawing a flat zero line that
        # would look like a real unloaded reading.
        ati_plot_matched = None

    if plot_time_origin is None:
        plot_time_origin = t_sens_store[0]
    relative_time = t_sens_store - plot_time_origin

    for indices, ati_lines, sens_lines in (
        (FORCE_INDICES, lines_ati_force, lines_sens_force),
        (TORQUE_INDICES, lines_ati_torque, lines_sens_torque),
    ):
        for line_index, channel_index in enumerate(indices):
            visible = visibility[channel_index]

            if visible and ati_plot_matched is not None:
                ati_lines[line_index].set_data(relative_time, ati_plot_matched[:, channel_index])
            else:
                ati_lines[line_index].set_data([], [])

            if visible:
                sens_lines[line_index].set_data(relative_time, sens_plot_data[:, channel_index])
            else:
                sens_lines[line_index].set_data([], [])

    ax_force.relim(visible_only=True)
    ax_force.autoscale_view(scalex=False, scaley=True)
    ax_torque.relim(visible_only=True)
    ax_torque.autoscale_view(scalex=False, scaley=True)

    x_right = relative_time[-1]
    if x_right <= PLOT_DURATION:
        ax_torque.set_xlim(0.0, PLOT_DURATION)
    else:
        ax_torque.set_xlim(x_right - PLOT_DURATION, x_right)


#########################
#  GUI setup            #
#########################

fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, hspace=0.25)

color_map = ["red", "green", "blue"]


def _build_channel_lines(axis, indices):
    """Draw the ATI/CoinFT line pair for every channel plotted on one axis."""
    ati_lines, coinft_lines = [], []
    for line_index, channel_index in enumerate(indices):
        label = CHANNEL_LABELS[channel_index]
        color = color_map[line_index]
        ati_line, = axis.plot([], [], label=f"ATI {label}", linestyle="-", color=color, linewidth=2)
        coinft_line, = axis.plot([], [], label=f"CoinFT {label}", linestyle=":", color=color, linewidth=2)
        ati_lines.append(ati_line)
        coinft_lines.append(coinft_line)
    return ati_lines, coinft_lines


lines_ati_force, lines_sens_force = _build_channel_lines(ax_force, FORCE_INDICES)
ax_force.set_ylabel("Force (N)")
ax_force.legend(loc="upper left")
ax_force.set_title("Forces")
ax_force.grid(True)

lines_ati_torque, lines_sens_torque = _build_channel_lines(ax_torque, TORQUE_INDICES)
ax_torque.set_xlabel("Time (s)")
ax_torque.set_ylabel("Torque (Nm)")
ax_torque.legend(loc="upper left")
ax_torque.set_title("Torques")
ax_torque.grid(True)

check_ax = plt.axes([0.01, 0.40, 0.08, 0.20])
check = CheckButtons(check_ax, CHANNEL_LABELS, visibility)


def toggle_visibility(label):
    channel_index = CHANNEL_LABELS.index(label)
    visibility[channel_index] = not visibility[channel_index]


check.on_clicked(toggle_visibility)

stop_ax = plt.axes([0.85, 0.02, 0.10, 0.05])
stop_button = Button(stop_ax, "Stop")


def stop(_event):
    global stop_flag
    stop_flag = True


stop_button.on_clicked(stop)


def on_close(_event):
    global stop_flag
    stop_flag = True


fig.canvas.mpl_connect("close_event", on_close)


#########################
#  Saving and cleanup   #
#########################

def save_recorded_data():
    """Save the recorded data plus a sidecar JSON holding the tare constants,
    so the dataset can be reprocessed if a tare or M_ARM value is later found
    to be wrong."""
    print("Saving data...")

    raw_columns = [f"CoinFT_raw_{index + 1}" for index in range(num_channels)]
    offset_columns = [f"CoinFT_offset_{index + 1}" for index in range(num_channels)]
    ati_columns = [f"ATI_{label}" for label in CHANNEL_LABELS]
    coinft_columns = [f"CoinFT_calib_{label}" for label in CHANNEL_LABELS]
    columns = ["Time"] + raw_columns + offset_columns + ati_columns + coinft_columns

    if not all_data_records:
        print("No data recorded; CSV file was not created.")
        return None

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(DATA_DIR, f"CoinFT_data_{timestamp_text}.csv")

    dataframe = pd.DataFrame(all_data_records, columns=columns)
    dataframe.to_csv(full_path, index=False)
    print(f"Saved {len(all_data_records)} records to: {full_path}")

    if TARE_META is not None:
        meta_path = os.path.join(DATA_DIR, f"CoinFT_data_{timestamp_text}_tare.json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(TARE_META, meta_file, indent=2)
        print(f"Saved tare constants to: {meta_path}")

    return full_path


def close_hardware(master):
    """Stop CoinFT and close the EtherCAT/serial resources."""
    global stop_flag
    stop_flag = True
    experiment_started.clear()
    ati_tare_collecting.clear()

    try:
        if ser.is_open:
            ser.write(b"i")
            ser.close()
    except Exception as exc:
        print(f"Warning while closing CoinFT serial: {exc}")

    if master is not None:
        try:
            master.state = pysoem.INIT_STATE
            master.write_state()
            master.close()
        except Exception as exc:
            print(f"Warning while closing ATI EtherCAT master: {exc}")


#########################
#  Main execution       #
#########################

def main():
    global stop_flag, latest_ati, TARE_META

    ati_thread = None
    coinft_thread = None
    master = None

    try:
        master, slave = open_ati_master()

        ati_thread = threading.Thread(target=read_ati_ethercat, args=(master, slave), daemon=True)
        ati_thread.start()

        print("\n" + "=" * 58)
        print(" >>> CONFIRM BOTH SENSORS ARE UNLOADED")
        print(" >>> PRESS [ENTER] TO START THE SYNCHRONIZED TARE")
        print("=" * 58 + "\n")
        input()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")

        offset_coinft, ft_bias, TARE_META = perform_synchronized_tare()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")

        # CoinFT is already streaming from the tare; clear whatever queued up
        # while the tare predictions were being computed.
        ser.reset_input_buffer()
        with ati_lock:
            latest_ati = None
        experiment_started.set()

        print("\n>>> DATA COLLECTION AND PLOTTING STARTED\n")

        coinft_thread = threading.Thread(target=read_coinft, args=(offset_coinft, ft_bias), daemon=True)
        coinft_thread.start()

        animation = FuncAnimation(
            fig, update_plot, interval=ANIMATION_INTERVAL_MS, blit=False, cache_frame_data=False
        )
        _ = animation  # keep a live reference for the lifetime of plt.show()
        plt.show()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")
        if coinft_thread_error is not None:
            raise RuntimeError(f"CoinFT acquisition failed: {coinft_thread_error}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Stopping acquisition threads...")
        stop_flag = True
        experiment_started.clear()
        ati_tare_collecting.clear()

        if coinft_thread is not None:
            coinft_thread.join(timeout=3.0)
        if ati_thread is not None:
            ati_thread.join(timeout=3.0)

        close_hardware(master)
        print(f"ATI diagnostics: {ati_diagnostics}")
        save_recorded_data()
        print("Stopped.")


if __name__ == "__main__":
    main()
