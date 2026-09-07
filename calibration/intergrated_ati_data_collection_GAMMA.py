#!/usr/bin/env python3
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import serial


#########################
# 1. CONTROL PANEL      #
#########################

# --- CoinFT serial ---
COM_NAME = "COM7"
BAUD_RATE = 1_000_000

# --- ATI UDP RDT ---
SENSOR_IP = "192.168.1.1"
SENSOR_PORT = 49152
ATI_OUTPUT_RATE_HZ = 1000.0
ATI_SOCKET_TIMEOUT_SECONDS = 0.20
ATI_TARE_IN_SECONDS = 2.0
ATI_TARE_SETTLE_SECONDS = 0.25
ATI_LEADING_BUFFER_SECONDS = 1.0
ATI_TRAILING_BUFFER_SECONDS = 1.0

# RDT returns integer counts. These values must match the active ATI
# calibration/configuration and must produce forces in N and moments in N*m.
COUNTS_PER_FORCE = 1_000_000.0
COUNTS_PER_TORQUE = 1_000_000.0
ATI_SCALE_IS_VERIFIED = False

# Keep nonzero-status records, but report and save their status words.
# The exact meaning of each status bit depends on the ATI documentation.
REJECT_NONZERO_STATUS = False

# --- Experiment parameters ---
SAMPLING_DURATION = 40.0
M_ARM = 0.0115
APPLY_LEVER_ARM = True
SENSOR_NAME = "CFT24"

# Use train, val, or test to remain compatible with the original data_processor.
ID = "test"
ATI_MODEL_TAG = "SI13010"

# --- RDT protocol constants ---
RDT_HEADER = 0x1234
CMD_STOP = 0x0000
CMD_START_REALTIME = 0x0002
RDT_RECORD_FMT = "!IIIiiiiii"
RDT_RECORD_SIZE = struct.calcsize(RDT_RECORD_FMT)
UINT32_MOD = 2**32


#########################
# 2. GLOBAL STATE       #
#########################

# Each ATI formal record stores:
# [Fx, Fy, Fz, Mx, My, Mz,
#  monotonic_time, unix_time, rdt_seq, ft_seq, status]
ati_data_list = []

coinft_data_list = []
coinft_time_mono_list = []
coinft_time_unix_list = []

stop_ati = threading.Event()
stop_cft = threading.Event()

ati_thread_error = None
coinft_thread_error = None

ati_diagnostics = {
    "short_datagrams": 0,
    "trailing_bytes": 0,
    "ft_sequence_discontinuities": 0,
    "estimated_missing_ft_samples": 0,
    "duplicate_ft_samples": 0,
    "nonzero_status_records": 0,
}

coinft_diagnostics = {
    "bad_end_byte_packets": 0,
    "short_packets": 0,
}


#########################
# 3. VALIDATION         #
#########################

def validate_configuration():
    """Fail early when a configuration cannot produce the original units."""
    positive_values = {
        "ATI_OUTPUT_RATE_HZ": ATI_OUTPUT_RATE_HZ,
        "ATI_TARE_IN_SECONDS": ATI_TARE_IN_SECONDS,
        "SAMPLING_DURATION": SAMPLING_DURATION,
        "COUNTS_PER_FORCE": COUNTS_PER_FORCE,
        "COUNTS_PER_TORQUE": COUNTS_PER_TORQUE,
    }
    for name, value in positive_values.items():
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value; got {value!r}")

    if ID not in {"train", "val", "test"}:
        print(
            "WARNING: ID is not train/val/test. The original data_processor "
            "may skip the output file."
        )

    if not ATI_SCALE_IS_VERIFIED:
        print(
            "WARNING: ATI_SCALE_IS_VERIFIED is False. Confirm that "
            "COUNTS_PER_FORCE and COUNTS_PER_TORQUE convert RDT counts to "
            "N and N*m before using the data for final calibration."
        )


def force_close_serial(port_name):
    """Retain the helper used by the original script."""
    try:
        test_ser = serial.Serial(port_name)
        test_ser.close()
        return True
    except Exception:
        return False


#########################
# 4. ATI UDP HELPERS    #
#########################

def build_rdt_request(command):
    return struct.pack("!HHI", RDT_HEADER, command, 0)


def send_ati_command(sock, command):
    sock.sendto(build_rdt_request(command), (SENSOR_IP, SENSOR_PORT))


def drain_udp_socket(sock):
    """Remove stale UDP datagrams without blocking."""
    old_timeout = sock.gettimeout()
    try:
        sock.setblocking(False)
        while True:
            try:
                sock.recvfrom(65535)
            except (BlockingIOError, socket.timeout):
                break
            except OSError:
                break
    finally:
        sock.settimeout(old_timeout)


def counts_to_engineering_units(fx, fy, fz, tx, ty, tz):
    """Convert ATI RDT counts to [N, N, N, N*m, N*m, N*m]."""
    return np.array(
        [
            fx / COUNTS_PER_FORCE,
            fy / COUNTS_PER_FORCE,
            fz / COUNTS_PER_FORCE,
            tx / COUNTS_PER_TORQUE,
            ty / COUNTS_PER_TORQUE,
            tz / COUNTS_PER_TORQUE,
        ],
        dtype=np.float64,
    )


def parse_rdt_datagram(data, receive_mono, receive_unix):
    """
    Parse all complete RDT records in one UDP datagram.

    The newest record is anchored to the computer receive timestamp. Earlier
    records in the same datagram are placed ATI_OUTPUT_RATE_HZ seconds apart.
    """
    if len(data) < RDT_RECORD_SIZE:
        ati_diagnostics["short_datagrams"] += 1
        return []

    num_records = len(data) // RDT_RECORD_SIZE
    trailing_bytes = len(data) % RDT_RECORD_SIZE
    if trailing_bytes:
        ati_diagnostics["trailing_bytes"] += trailing_bytes

    records = []
    for record_index in range(num_records):
        start = record_index * RDT_RECORD_SIZE
        chunk = data[start : start + RDT_RECORD_SIZE]
        try:
            rdt_seq, ft_seq, status, fx, fy, fz, tx, ty, tz = struct.unpack(
                RDT_RECORD_FMT, chunk
            )
        except struct.error:
            continue

        records_behind_newest = num_records - 1 - record_index
        time_offset = records_behind_newest / ATI_OUTPUT_RATE_HZ
        sample_mono = receive_mono - time_offset
        sample_unix = receive_unix - time_offset

        ft = counts_to_engineering_units(fx, fy, fz, tx, ty, tz)
        records.append((ft, sample_mono, sample_unix, rdt_seq, ft_seq, status))

    return records


def perform_ati_tare(sock):
    """
    Replicate the original separate 2-second ATI tare before formal collection.
    """
    print("Taring ATI... Keep both sensors completely unloaded.")
    tare_records = []
    nonzero_status_count = 0

    drain_udp_socket(sock)
    send_ati_command(sock, CMD_START_REALTIME)

    try:
        # Discard startup/transient traffic before beginning the timed tare.
        time.sleep(ATI_TARE_SETTLE_SECONDS)
        drain_udp_socket(sock)

        tare_start = time.perf_counter()
        tare_end = tare_start + ATI_TARE_IN_SECONDS

        while time.perf_counter() < tare_end:
            try:
                data, _ = sock.recvfrom(65535)
                receive_mono = time.perf_counter()
                receive_unix = time.time()
            except socket.timeout:
                continue

            for ft, mono, _unix, _rdt_seq, _ft_seq, status in parse_rdt_datagram(
                data, receive_mono, receive_unix
            ):
                if mono < tare_start:
                    continue
                if status != 0:
                    nonzero_status_count += 1
                    if REJECT_NONZERO_STATUS:
                        continue
                tare_records.append(ft)
    finally:
        try:
            send_ati_command(sock, CMD_STOP)
            time.sleep(0.05)
        finally:
            drain_udp_socket(sock)

    if len(tare_records) < 10:
        raise RuntimeError(
            f"ATI tare failed: only {len(tare_records)} usable samples were received."
        )

    ati_tare = np.mean(np.vstack(tare_records), axis=0)
    tare_std = np.std(np.vstack(tare_records), axis=0)

    print(f"ATI tare complete: {len(tare_records)} samples")
    print(f"ATI tare mean [N, N, N, N*m, N*m, N*m]:\n{ati_tare}")
    print(f"ATI tare standard deviation:\n{tare_std}")
    if nonzero_status_count:
        print(
            f"WARNING: {nonzero_status_count} tare records had nonzero ATI status."
        )

    return ati_tare, tare_std, len(tare_records)


#########################
# 5. WORKER THREADS     #
#########################

def read_ati_udp(sock):
    """Formal ATI acquisition thread using host-computer timestamps."""
    global ati_thread_error

    last_ft_seq = None
    warning_prints = 0

    try:
        drain_udp_socket(sock)
        send_ati_command(sock, CMD_START_REALTIME)
        print(f"Started ATI UDP streaming from {SENSOR_IP}:{SENSOR_PORT}")

        while not stop_ati.is_set():
            try:
                data, _ = sock.recvfrom(65535)
                receive_mono = time.perf_counter()
                receive_unix = time.time()
            except socket.timeout:
                continue
            except OSError as exc:
                if stop_ati.is_set():
                    break
                raise exc

            records = parse_rdt_datagram(data, receive_mono, receive_unix)
            for ft, mono, unix_time, rdt_seq, ft_seq, status in records:
                if last_ft_seq is not None:
                    delta = (int(ft_seq) - int(last_ft_seq)) % UINT32_MOD
                    if delta == 0:
                        ati_diagnostics["duplicate_ft_samples"] += 1
                    elif delta != 1:
                        ati_diagnostics["ft_sequence_discontinuities"] += 1
                        ati_diagnostics["estimated_missing_ft_samples"] += max(
                            0, delta - 1
                        )
                        if warning_prints < 10:
                            print(
                                "ATI WARNING: FT sequence discontinuity "
                                f"({last_ft_seq} -> {ft_seq}, delta={delta})."
                            )
                            warning_prints += 1
                last_ft_seq = ft_seq

                if status != 0:
                    ati_diagnostics["nonzero_status_records"] += 1
                    if REJECT_NONZERO_STATUS:
                        continue

                ati_data_list.append(
                    [
                        *ft.tolist(),
                        mono,
                        unix_time,
                        int(rdt_seq),
                        int(ft_seq),
                        int(status),
                    ]
                )

    except Exception as exc:
        ati_thread_error = exc
        print(f"ATI UDP thread error: {exc}")
    finally:
        try:
            send_ati_command(sock, CMD_STOP)
            time.sleep(0.05)
        except OSError:
            pass
        print("ATI UDP streaming stopped.")


def read_coinft(ser, packet_size):
    """CoinFT serial thread retaining the original s/i protocol."""
    global coinft_thread_error

    try:
        ser.write(b"s")

        while not stop_cft.is_set():
            start_byte = ser.read(1)
            if start_byte != b"\x02":
                continue

            data = ser.read(packet_size)
            if len(data) < packet_size:
                coinft_diagnostics["short_packets"] += 1
                continue

            if data[-1] != 3:
                coinft_diagnostics["bad_end_byte_packets"] += 1
                continue

            values = [
                data[i] + 256 * data[i + 1]
                for i in range(0, packet_size - 1, 2)
            ]

            # Timestamp after a complete, valid packet has been received.
            sample_mono = time.perf_counter()
            sample_unix = time.time()

            coinft_data_list.append(values)
            coinft_time_mono_list.append(sample_mono)
            coinft_time_unix_list.append(sample_unix)

    except Exception as exc:
        coinft_thread_error = exc
        if not stop_cft.is_set():
            print(f"CoinFT read error: {exc}")
    finally:
        try:
            ser.write(b"i")
        except serial.SerialException:
            pass


#########################
# 6. ANALYSIS HELPERS   #
#########################

def estimate_rate_from_times(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 3:
        return np.nan
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(1.0 / np.median(dt))


def unwrap_uint32(sequence):
    sequence = np.asarray(sequence, dtype=np.uint64)
    if sequence.size == 0:
        return np.array([], dtype=np.float64)

    unwrapped = np.zeros(sequence.size, dtype=np.float64)
    for i in range(1, sequence.size):
        delta = (int(sequence[i]) - int(sequence[i - 1])) % UINT32_MOD
        # Treat a very large positive modular delta as a negative/restart jump.
        if delta > UINT32_MOD // 2:
            delta -= UINT32_MOD
        unwrapped[i] = unwrapped[i - 1] + delta
    return unwrapped


def estimate_ati_sequence_rate(ft_sequence, timestamps):
    seq = unwrap_uint32(ft_sequence)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if seq.size < 2 or timestamps.size != seq.size:
        return np.nan
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return np.nan
    return float((seq[-1] - seq[0]) / elapsed)


def make_time_based_synchronized_data(
    coinft_tared, coinft_time, ati_ft, ati_time, num_sensors
):
    """
    Preserve the original CoinFT 3-high/3-low window logic, but locate the
    corresponding ATI samples using the common host monotonic time base.
    """
    sync_psoc = np.zeros(len(coinft_tared), dtype=np.uint8)
    for i in range(0, len(sync_psoc), 6):
        sync_psoc[i : i + 3] = 1

    trans_psoc = np.where(np.diff(sync_psoc) != 0)[0]
    num_candidates = max(0, len(trans_psoc) - 2)

    sensor_rows = []
    ati_rows = []
    sync_indices = []
    ati_samples_per_window = []

    for i in range(num_candidates):
        # Same CoinFT sample slices used by the original code.
        cft_start = int(trans_psoc[i + 1] + 1)
        cft_end = int(trans_psoc[i + 2] + 1)

        if cft_start >= len(coinft_time) or cft_end > len(coinft_time):
            continue
        if cft_end <= cft_start:
            continue

        t_start = coinft_time[cft_start]
        if cft_end < len(coinft_time):
            t_end = coinft_time[cft_end]
        else:
            # Usually unreachable, but retain a valid final boundary.
            median_dt = np.median(np.diff(coinft_time))
            t_end = coinft_time[cft_end - 1] + median_dt

        ati_start = int(np.searchsorted(ati_time, t_start, side="left"))
        ati_end = int(np.searchsorted(ati_time, t_end, side="left"))

        if ati_end <= ati_start:
            continue

        sensor_rows.append(np.mean(coinft_tared[cft_start:cft_end, :], axis=0))
        ati_rows.append(np.mean(ati_ft[ati_start:ati_end, :], axis=0))
        sync_indices.append(
            [cft_start, cft_end, ati_start, ati_end, t_start, t_end]
        )
        ati_samples_per_window.append(ati_end - ati_start)

    if sensor_rows:
        sensor_cal_data = np.vstack(sensor_rows).reshape(-1, num_sensors)
        ati_cal_ft = np.vstack(ati_rows).reshape(-1, 6)
        sync_indices = np.asarray(sync_indices, dtype=np.float64)
    else:
        sensor_cal_data = np.empty((0, num_sensors), dtype=np.float64)
        ati_cal_ft = np.empty((0, 6), dtype=np.float64)
        sync_indices = np.empty((0, 6), dtype=np.float64)

    return (
        sensor_cal_data,
        ati_cal_ft,
        sync_indices,
        np.asarray(ati_samples_per_window, dtype=np.int64),
        num_candidates,
    )


#########################
# 7. MAIN               #
#########################

def main():
    validate_configuration()

    # ---- CoinFT serial setup: same handshake as the original code ----
    force_close_serial(COM_NAME)
    try:
        ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=0.1)
        ser.write(b"i")
        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.write(b"q")
        time.sleep(0.01)
        packet_size_raw = ser.read(1)
        if not packet_size_raw:
            raise RuntimeError("PSoC not responding")

        packet_size = ord(packet_size_raw) - 1
        num_sensors = (packet_size - 1) // 2
        print(
            f"PSoC Ready. {num_sensors} Channels. Packet size: {packet_size}"
        )
    except Exception as exc:
        try:
            ser.close()
        except Exception:
            pass
        sys.exit(f"Serial Error: {exc}")

    # ---- ATI socket setup ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(ATI_SOCKET_TIMEOUT_SECONDS)
    try:
        sock.bind(("", 0))
        local_ip, local_port = sock.getsockname()
        print(f"UDP socket bound to {local_ip}:{local_port}")
    except Exception as exc:
        ser.close()
        sock.close()
        sys.exit(f"UDP Socket Error: {exc}")

    try:
        # Separate ATI tare, matching the original program's ordering.
        ati_tare, ati_tare_std, ati_tare_sample_count = perform_ati_tare(sock)

        print("BEGIN DATA COLLECTION (ATI UDP + CoinFT Serial)...")
        t_ati = threading.Thread(target=read_ati_udp, args=(sock,))
        t_cft = threading.Thread(target=read_coinft, args=(ser, packet_size))

        # Replicate original timing: ATI starts first.
        t_ati.start()
        time.sleep(ATI_LEADING_BUFFER_SECONDS)

        t_cft.start()
        time.sleep(SAMPLING_DURATION)

        # Replicate original timing: CoinFT stops first.
        stop_cft.set()
        t_cft.join(timeout=3.0)
        if t_cft.is_alive():
            print("WARNING: CoinFT thread did not stop within the timeout.")

        time.sleep(ATI_TRAILING_BUFFER_SECONDS)

        stop_ati.set()
        t_ati.join(timeout=3.0)
        if t_ati.is_alive():
            print("WARNING: ATI thread did not stop within the timeout.")

    finally:
        stop_cft.set()
        stop_ati.set()

        if ser.is_open:
            try:
                ser.write(b"i")
            except serial.SerialException:
                pass
            ser.close()

        try:
            send_ati_command(sock, CMD_STOP)
        except OSError:
            pass
        sock.close()
        print("Acquisition Stopped. Socket and serial port closed.")

    if ati_thread_error is not None:
        raise RuntimeError(f"ATI acquisition failed: {ati_thread_error}")
    if coinft_thread_error is not None:
        raise RuntimeError(f"CoinFT acquisition failed: {coinft_thread_error}")

    if len(ati_data_list) == 0 or len(coinft_data_list) == 0:
        raise RuntimeError("No data collected from one or both sensors.")

    # ---- Convert recorded lists to arrays ----
    ati_records = np.asarray(ati_data_list, dtype=np.float64)
    ati_raw = ati_records[:, 0:6]
    ati_time_mono = ati_records[:, 6]
    ati_time_unix = ati_records[:, 7]
    ati_rdt_seq = ati_records[:, 8].astype(np.uint32)
    ati_ft_seq = ati_records[:, 9].astype(np.uint32)
    ati_status = ati_records[:, 10].astype(np.uint32)

    # Host timestamps are the synchronization key, so enforce monotonic order
    # before using np.searchsorted. This also protects against occasional UDP
    # packet reordering or overlap introduced by multi-record interpolation.
    ati_time_order = np.argsort(ati_time_mono, kind="stable")
    ati_raw = ati_raw[ati_time_order]
    ati_time_mono = ati_time_mono[ati_time_order]
    ati_time_unix = ati_time_unix[ati_time_order]
    ati_rdt_seq = ati_rdt_seq[ati_time_order]
    ati_ft_seq = ati_ft_seq[ati_time_order]
    ati_status = ati_status[ati_time_order]

    coinft_raw = np.asarray(coinft_data_list, dtype=np.float64)
    coinft_time_mono = np.asarray(coinft_time_mono_list, dtype=np.float64)
    coinft_time_unix = np.asarray(coinft_time_unix_list, dtype=np.float64)

    print(
        f"Collected: ATI={len(ati_raw)} samples, "
        f"CoinFT={len(coinft_raw)} samples"
    )

    if len(coinft_raw) < 50:
        raise RuntimeError(
            "CoinFT produced fewer than 50 samples; original baseline slices "
            "10:50 cannot be reproduced."
        )

    # ---- Original CoinFT baseline ----
    coinft_baseline = np.mean(coinft_raw[10:50, :], axis=0)
    coinft_tared = coinft_raw - coinft_baseline

    # ---- ATI processing ----
    # The new ATI already provides calibrated counts, so conversion to N/N*m
    # happens in counts_to_engineering_units. The separate tare is then removed.
    ati_ft = ati_raw - ati_tare

    # Original lever-arm correction.
    if APPLY_LEVER_ARM:
        ati_ft[:, 3] += ati_ft[:, 1] * M_ARM  # Mx += Fy * r_z
        ati_ft[:, 4] -= ati_ft[:, 0] * M_ARM  # My -= Fx * r_z

    # ---- Host-time synchronization ----
    (
        sensor_cal_data,
        ati_cal_ft,
        sync_window_indices,
        ati_samples_per_window,
        candidate_windows,
    ) = make_time_based_synchronized_data(
        coinft_tared,
        coinft_time_mono,
        ati_ft,
        ati_time_mono,
        num_sensors,
    )

    valid_windows_before_drop = len(sensor_cal_data)
    print(
        f"Time-sync windows: candidates={candidate_windows}, "
        f"valid={valid_windows_before_drop}"
    )

    if ati_samples_per_window.size:
        print(
            "ATI samples per synchronized window: "
            f"min={ati_samples_per_window.min()}, "
            f"median={np.median(ati_samples_per_window):.1f}, "
            f"max={ati_samples_per_window.max()}"
        )

    if valid_windows_before_drop < 2:
        raise RuntimeError(
            "Too few synchronized windows. Check ATI output rate, network "
            "connection, CoinFT acquisition, and timestamp overlap."
        )

    # Original code discards the first processed row due to CoinFT noise.
    sensor_cal_data = sensor_cal_data[1:, :]
    ati_cal_ft = ati_cal_ft[1:, :]
    sync_window_indices = sync_window_indices[1:, :]
    ati_samples_per_window = ati_samples_per_window[1:]

    # ---- Original second-order least-squares calibration ----
    x_features = np.hstack([sensor_cal_data, sensor_cal_data**2])
    a_t, residuals, rank, singular_values = np.linalg.lstsq(
        x_features, ati_cal_ft, rcond=None
    )
    calibrated_ft = x_features @ a_t

    if rank < x_features.shape[1]:
        print(
            f"WARNING: Calibration design matrix rank is {rank}, below "
            f"the full feature count {x_features.shape[1]}."
        )

    # ---- Training-fit validation retained from original ----
    rmse = np.sqrt(np.mean((calibrated_ft - ati_cal_ft) ** 2, axis=0))
    print("\n" + "=" * 50)
    print("LEAST SQUARES TRAINING-FIT RMSE")
    print("=" * 50)
    axis_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units = ["N", "N", "N", "N*m", "N*m", "N*m"]
    for axis, value, unit in zip(axis_labels, rmse, units):
        print(f"{axis}: {value:.6f} {unit}")
    print("-" * 50)
    print(f"Mean of six axis RMSE values: {np.mean(rmse):.6f}")
    print("=" * 50)
    print(
        "This RMSE is evaluated on the same data used to fit A_T. It is a "
        "learnability/training-fit diagnostic, not independent accuracy."
    )

    # ---- Acquisition diagnostics ----
    coinft_rate_measured = estimate_rate_from_times(coinft_time_mono)
    ati_receive_time_rate = estimate_rate_from_times(ati_time_mono)
    ati_sequence_rate = estimate_ati_sequence_rate(ati_ft_seq, ati_time_mono)

    print("\nAcquisition diagnostics")
    print(f"Configured ATI output rate: {ATI_OUTPUT_RATE_HZ:.3f} Hz")
    print(f"ATI timestamp median rate: {ati_receive_time_rate:.3f} Hz")
    print(f"ATI FT-sequence/time rate: {ati_sequence_rate:.3f} Hz")
    print(f"CoinFT timestamp median rate: {coinft_rate_measured:.3f} Hz")
    print(f"ATI diagnostics: {ati_diagnostics}")
    print(f"CoinFT diagnostics: {coinft_diagnostics}")

    # ---- Plotting: same core plots as original ----
    samples_cal = np.arange(len(calibrated_ft))

    plt.figure("Raw History", figsize=(10, 5))
    off_plot = coinft_raw - np.mean(coinft_raw[20:40, :], axis=0)
    for channel in range(num_sensors):
        plt.plot(off_plot[:, channel], label=f"Ch{channel + 1}")
    plt.title("Capacitive Sensor Output (Raw Sample Count)")
    plt.xlabel("Sample Count")
    plt.ylabel("Raw Counts")
    plt.legend(ncol=4, fontsize="small")
    plt.grid(True, alpha=0.3)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    plot_labels = [
        "Fx [N]",
        "Fy [N]",
        "Fz [N]",
        "Mx [N*m]",
        "My [N*m]",
        "Mz [N*m]",
    ]
    for axis_index, axis in enumerate(axes.flatten()):
        axis.plot(
            samples_cal,
            calibrated_ft[:, axis_index],
            label="Calibrated CoinFT",
            linestyle="--",
            color="red",
        )
        axis.plot(
            samples_cal,
            ati_cal_ft[:, axis_index],
            label="ATI Reference",
            color="blue",
            alpha=0.6,
        )
        axis.set_title(plot_labels[axis_index])
        axis.set_xlabel("Sample count")
        axis.legend(fontsize="small")
        axis.grid(True, alpha=0.3)

    fig.suptitle(
        "CoinFT Calibration vs ATI Reference\n"
        "Host-computer timestamp synchronization",
        y=1.01,
    )
    fig.tight_layout()
    plt.show()

    # ---- HDF5 saving ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Keep ID as the final token so the original data_processor still finds
    # _train.h5, _val.h5, or _test.h5.
    h5_filename = (
        f"{SENSOR_NAME}_calibrationData_{timestamp_string}_"
        f"{ATI_MODEL_TAG}_{ID}.h5"
    )
    full_path = os.path.join(data_dir, h5_filename)
    print(f"Saving to: {full_path}")

    common_time_origin = min(ati_time_mono[0], coinft_time_mono[0])
    ati_time_relative = ati_time_mono - common_time_origin
    coinft_time_relative = coinft_time_mono - common_time_origin

    with h5py.File(full_path, "w") as h5_file:
        # Core datasets retained from the original script.
        h5_file.create_dataset("A_T", data=a_t, compression="gzip")
        h5_file.create_dataset(
            "sensor_cal_data", data=sensor_cal_data, compression="gzip"
        )
        h5_file.create_dataset(
            "ati_cal_FT", data=ati_cal_ft, compression="gzip"
        )

        # Additional raw/debug datasets needed because synchronization is now
        # software-time based rather than hardware-line based.
        h5_file.create_dataset("coinft_raw", data=coinft_raw, compression="gzip")
        h5_file.create_dataset("coinft_tared", data=coinft_tared, compression="gzip")
        h5_file.create_dataset(
            "coinft_time", data=coinft_time_relative, compression="gzip"
        )
        h5_file.create_dataset(
            "coinft_unix_time", data=coinft_time_unix, compression="gzip"
        )
        h5_file.create_dataset("coinft_baseline", data=coinft_baseline)

        h5_file.create_dataset("ati_raw", data=ati_raw, compression="gzip")
        h5_file.create_dataset("ati_FT", data=ati_ft, compression="gzip")
        h5_file.create_dataset(
            "ati_time", data=ati_time_relative, compression="gzip"
        )
        h5_file.create_dataset(
            "ati_unix_time", data=ati_time_unix, compression="gzip"
        )
        h5_file.create_dataset("ati_rdt_seq", data=ati_rdt_seq, compression="gzip")
        h5_file.create_dataset("ati_ft_seq", data=ati_ft_seq, compression="gzip")
        h5_file.create_dataset("ati_status", data=ati_status, compression="gzip")
        h5_file.create_dataset("ATI_TARE", data=ati_tare)
        h5_file.create_dataset("ATI_TARE_STD", data=ati_tare_std)

        h5_file.create_dataset(
            "sync_window_indices", data=sync_window_indices, compression="gzip"
        )
        h5_file.create_dataset(
            "ati_samples_per_window",
            data=ati_samples_per_window,
            compression="gzip",
        )
        h5_file.create_dataset("training_fit_rmse", data=rmse)
        h5_file.create_dataset("least_squares_singular_values", data=singular_values)
        h5_file.create_dataset("least_squares_residuals", data=residuals)

        # Original attributes.
        h5_file.attrs["sensor_name"] = SENSOR_NAME
        h5_file.attrs["timestamp"] = timestamp_string
        h5_file.attrs["m_arm"] = M_ARM
        h5_file.attrs["ati_rate"] = ATI_OUTPUT_RATE_HZ

        # New ATI and synchronization metadata.
        h5_file.attrs["id"] = ID
        h5_file.attrs["reference_sensor"] = ATI_MODEL_TAG
        h5_file.attrs["ati_ip"] = SENSOR_IP
        h5_file.attrs["ati_port"] = SENSOR_PORT
        h5_file.attrs["counts_per_force"] = COUNTS_PER_FORCE
        h5_file.attrs["counts_per_torque"] = COUNTS_PER_TORQUE
        h5_file.attrs["ati_scale_is_verified"] = ATI_SCALE_IS_VERIFIED
        h5_file.attrs["apply_lever_arm"] = APPLY_LEVER_ARM
        h5_file.attrs["ati_tare_seconds"] = ATI_TARE_IN_SECONDS
        h5_file.attrs["ati_tare_sample_count"] = ati_tare_sample_count
        h5_file.attrs["ati_timestamp_rate"] = ati_receive_time_rate
        h5_file.attrs["ati_sequence_rate"] = ati_sequence_rate
        h5_file.attrs["coinft_timestamp_rate"] = coinft_rate_measured
        h5_file.attrs["candidate_sync_windows"] = candidate_windows
        h5_file.attrs["valid_sync_windows_before_first_row_drop"] = (
            valid_windows_before_drop
        )
        h5_file.attrs["saved_sync_windows"] = len(sensor_cal_data)
        h5_file.attrs["least_squares_rank"] = rank
        h5_file.attrs["time_sync_method"] = (
            "Host monotonic timestamps; ATI multi-record datagrams interpolated "
            "using ATI_OUTPUT_RATE_HZ; no hardware sync line"
        )
        h5_file.attrs["time_dataset_note"] = (
            "ati_time and coinft_time are seconds relative to a shared host "
            "monotonic origin; *_unix_time contains host wall-clock timestamps"
        )
        h5_file.attrs["ati_nonzero_status_records"] = ati_diagnostics[
            "nonzero_status_records"
        ]
        h5_file.attrs["ati_ft_sequence_discontinuities"] = ati_diagnostics[
            "ft_sequence_discontinuities"
        ]
        h5_file.attrs["ati_estimated_missing_ft_samples"] = ati_diagnostics[
            "estimated_missing_ft_samples"
        ]
        h5_file.attrs["note"] = (
            "ATI reference acquired by UDP RDT. Core CoinFT calibration behavior "
            "matches the original NI-DAQ script except synchronization uses host "
            "computer timestamps."
        )

    print(f"Saved: {h5_filename}")


if __name__ == "__main__":
    main()
