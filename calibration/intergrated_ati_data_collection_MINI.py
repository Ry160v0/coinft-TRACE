#!/usr/bin/env python3
"""ATI Mini58 (EtherCAT/ECATBA) + CoinFT calibration data collection.

ATI is read over EtherCAT via pysoem instead of NI-DAQ analog voltage, so no
strain-gauge calibration matrix is needed: the ECATBA already resolves F/T
counts on-device, and this script only scales counts -> N / N*m.
"""
import os
import struct
import sys
import threading
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pysoem
import serial

#########################
# 1. CONTROL PANEL      #
#########################

# --- CoinFT serial ---
COM_NAME = "COM5"
BAUD_RATE = 1_000_000

# --- ATI Mini58 + ECATBA / EtherCAT ---
ATI_ADAPTER_NAME = None  # set explicitly (see pysoem's find_adapters.py) to skip auto-probing
ATI_VENDOR_ID = 0x00000732
ATI_OUTPUT_RATE_HZ = 1000.0
ATI_PROCESSDATA_TIMEOUT_US = 2000
ATI_STATE_TIMEOUT_US = 50000

# The ECATBA's default TxPDO is six int32 F/T counts (Fx Fy Fz Mx My Mz) at
# the start of the input image -- see ATI's read_vals.c reference. These
# fallback scale factors are replaced by the SDO 0x2040:49/50 sensor and if
# it's readable.
COUNTS_PER_FORCE = 1_000_000.0
COUNTS_PER_TORQUE = 1_000_000.0

# --- Synchronized ATI + CoinFT tare ---
# Mount the gripper before this tare. Both sensors are zeroed over the same
# wall-clock window, so the gripper's static preload becomes the shared zero.
TARE_SECONDS = 2.0
COINFT_SETTLE_SECONDS = 2.0

ATI_LEADING_BUFFER_SECONDS = 1.0
ATI_TRAILING_BUFFER_SECONDS = 1.0

# --- Experiment parameters ---
SAMPLING_DURATION = 40.0
M_ARM = 0.0115  # [m] lever arm between the ATI reference surface and the CoinFT surface
SENSOR_NAME = "CFT24"
ATI_MODEL_TAG = "Mini58_ECATBA"

# train / val / test -- data_processor.py splits on this suffix
ID = "test"  

#########################
# 2. GLOBAL STATE       #
#########################

ati_data_list = []  # [Fx, Fy, Fz, Mx, My, Mz, mono_time]
coinft_data_list = []
coinft_time_list = []

stop_ati = threading.Event()
stop_cft = threading.Event()
ati_tare_collecting = threading.Event()
experiment_started = threading.Event()
ati_tare_lock = threading.Lock()
ati_tare_samples = []

ati_diagnostics = {"bad_wkc": 0}


def force_close_serial(port_name):
    try:
        serial.Serial(port_name).close()
    except Exception:
        pass


#########################
# 3. ATI ETHERCAT       #
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
    """One EtherCAT cycle. Returns (ft, mono_time) or None on a bad WKC."""
    master.send_processdata()
    wkc = master.receive_processdata(ATI_PROCESSDATA_TIMEOUT_US)
    mono = time.perf_counter()

    if wkc < master.expected_wkc:
        ati_diagnostics["bad_wkc"] += 1
        return None

    fx, fy, fz, tx, ty, tz = struct.unpack_from("<6i", slave.input)
    ft = np.array([
        fx / COUNTS_PER_FORCE, fy / COUNTS_PER_FORCE, fz / COUNTS_PER_FORCE,
        tx / COUNTS_PER_TORQUE, ty / COUNTS_PER_TORQUE, tz / COUNTS_PER_TORQUE,
    ])
    return ft, mono


def read_ati_ethercat(master, slave):
    """Continuous ATI EtherCAT loop, paced at ATI_OUTPUT_RATE_HZ.

    Runs from just after EtherCAT setup through the end of acquisition.
    Samples land in the tare buffer or the formal buffer depending on which
    event is currently set (or neither, between the two).
    """
    period = 1.0 / ATI_OUTPUT_RATE_HZ
    next_cycle = time.perf_counter()

    while not stop_ati.is_set():
        sample = read_ati_sample(master, slave)
        if sample is not None:
            ft, mono = sample
            if ati_tare_collecting.is_set():
                with ati_tare_lock:
                    ati_tare_samples.append((mono, ft))
            if experiment_started.is_set():
                ati_data_list.append([*ft, mono])

        next_cycle += period
        now = time.perf_counter()
        if next_cycle > now:
            time.sleep(next_cycle - now)
        else:
            next_cycle = now


#########################
# 4. COINFT SERIAL      #
#########################

def read_one_coinft_packet(ser, packet_size):
    if ser.read(1) != b"\x02":
        return None
    data = ser.read(packet_size)
    if len(data) < packet_size or data[-1] != 3:
        return None
    return [data[i] + 256 * data[i + 1] for i in range(0, packet_size - 1, 2)]


def read_coinft(ser, packet_size, stream_already_started=False):
    if not stream_already_started:
        ser.write(b"s")

    while not stop_cft.is_set():
        packet = read_one_coinft_packet(ser, packet_size)
        if packet is None:
            continue
        coinft_data_list.append(packet)
        coinft_time_list.append(time.perf_counter())

    ser.write(b"i")


#########################
# 5. SYNCHRONIZED TARE  #
#########################

def perform_synchronized_tare(ser, packet_size):
    """Zero ATI and CoinFT over one shared wall-clock window.

    The gripper should already be mounted with no external load applied, so
    its static preload is incorporated into both zero references.
    """
    print("Synchronized tare: keep the gripper mounted, apply no external load.")

    ser.reset_input_buffer()
    ser.write(b"s")
    time.sleep(COINFT_SETTLE_SECONDS)
    ser.reset_input_buffer()

    with ati_tare_lock:
        ati_tare_samples.clear()
    coinft_tare_samples = []

    ati_tare_collecting.set()
    tare_start = time.perf_counter()
    tare_end = tare_start + TARE_SECONDS
    while time.perf_counter() < tare_end:
        packet = read_one_coinft_packet(ser, packet_size)
        if packet is not None:
            coinft_tare_samples.append(packet)
    ati_tare_collecting.clear()

    with ati_tare_lock:
        ati_samples = [ft for mono, ft in ati_tare_samples if mono <= tare_end]

    ati_tare = np.mean(ati_samples, axis=0)
    coinft_baseline = np.mean(coinft_tare_samples, axis=0)

    print(f"Tare complete: ATI {len(ati_samples)} samples, CoinFT {len(coinft_tare_samples)} samples.")
    print(f"ATI tare [N, N, N, N*m, N*m, N*m]: {ati_tare}")
    print(f"CoinFT baseline: {coinft_baseline}")
    return ati_tare, coinft_baseline


#########################
# 6. SYNC & CALIBRATION #
#########################

def make_time_based_synchronized_data(coinft_tared, coinft_time, ati_ft, ati_time, num_sensors):
    """Pair CoinFT/ATI samples using host timestamps.

    The PSoC toggles a virtual sync pulse every 6 samples (3 high, 3 low);
    each pulse window's host-timestamp span is used to slice the matching
    ATI samples.
    """
    sync_psoc = np.zeros(len(coinft_tared), dtype=np.uint8)
    for i in range(0, len(sync_psoc), 6):
        sync_psoc[i:i + 3] = 1
    trans_psoc = np.where(np.diff(sync_psoc) != 0)[0]

    sensor_rows, ati_rows = [], []
    for i in range(len(trans_psoc) - 2):
        cft_start, cft_end = trans_psoc[i + 1] + 1, trans_psoc[i + 2] + 1
        if cft_end >= len(coinft_time):
            continue

        t_start, t_end = coinft_time[cft_start], coinft_time[cft_end]
        ati_start = np.searchsorted(ati_time, t_start)
        ati_end = np.searchsorted(ati_time, t_end)
        if ati_end <= ati_start:
            continue

        sensor_rows.append(coinft_tared[cft_start:cft_end].mean(axis=0))
        ati_rows.append(ati_ft[ati_start:ati_end].mean(axis=0))

    return (
        np.vstack(sensor_rows).reshape(-1, num_sensors),
        np.vstack(ati_rows).reshape(-1, 6),
    )


def process_and_save(ati_tare, coinft_baseline, num_sensors):
    ati_records = np.asarray(ati_data_list, dtype=np.float64)
    ati_raw, ati_time = ati_records[:, :6], ati_records[:, 6]
    order = np.argsort(ati_time, kind="stable")
    ati_raw, ati_time = ati_raw[order], ati_time[order]

    coinft_raw = np.asarray(coinft_data_list, dtype=np.float64)
    coinft_time = np.asarray(coinft_time_list, dtype=np.float64)

    print(f"Collected: ATI={len(ati_raw)} samples, CoinFT={len(coinft_raw)} samples")
    print(f"EtherCAT bad-WKC cycles: {ati_diagnostics['bad_wkc']}")

    coinft_tared = coinft_raw - coinft_baseline
    ati_ft = ati_raw - ati_tare
    ati_ft[:, 3] += ati_ft[:, 1] * M_ARM  # Mx += Fy * r_z
    ati_ft[:, 4] -= ati_ft[:, 0] * M_ARM  # My -= Fx * r_z

    sensor_cal_data, ati_cal_ft = make_time_based_synchronized_data(
        coinft_tared, coinft_time, ati_ft, ati_time, num_sensors
    )
    print(f"Synchronized windows: {len(sensor_cal_data)}")

    # The first window tends to carry CoinFT startup noise.
    sensor_cal_data, ati_cal_ft = sensor_cal_data[1:], ati_cal_ft[1:]

    # Second-order least squares: [raw, raw**2] -> 6-axis F/T.
    x_features = np.hstack([sensor_cal_data, sensor_cal_data ** 2])
    a_t, *_ = np.linalg.lstsq(x_features, ati_cal_ft, rcond=None)
    calibrated_ft = x_features @ a_t

    axis_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units = ["N", "N", "N", "N*m", "N*m", "N*m"]
    rmse = np.sqrt(np.mean((calibrated_ft - ati_cal_ft) ** 2, axis=0))
    print("\nTraining-fit RMSE:")
    for axis, value, unit in zip(axis_labels, rmse, units):
        print(f"  {axis}: {value:.4f} {unit}")

    plot_results(coinft_raw, calibrated_ft, ati_cal_ft, num_sensors, axis_labels, units)
    save_h5(a_t, sensor_cal_data, ati_cal_ft, coinft_raw, ati_raw, ati_tare, coinft_baseline)


#########################
# 7. PLOTTING & SAVING  #
#########################

def plot_results(coinft_raw, calibrated_ft, ati_cal_ft, num_sensors, axis_labels, units):
    plt.figure("Raw History", figsize=(10, 5))
    off_plot = coinft_raw - coinft_raw[20:40].mean(axis=0)
    for ch in range(num_sensors):
        plt.plot(off_plot[:, ch], label=f"Ch{ch + 1}")
    plt.title("Capacitive Sensor Output (Raw Sample Count)")
    plt.xlabel("Sample Count")
    plt.legend(ncol=4, fontsize="small")

    samples = np.arange(len(calibrated_ft))
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(samples, calibrated_ft[:, i], "--", color="red", label="Calibrated CoinFT")
        ax.plot(samples, ati_cal_ft[:, i], color="blue", alpha=0.6, label="ATI Reference")
        ax.set_title(f"{axis_labels[i]} [{units[i]}]")
        ax.set_xlabel("Sample count")
        ax.legend(fontsize="small")

    fig.suptitle("CoinFT Calibration vs ATI Reference\nHost-computer timestamp synchronization", y=1.01)
    fig.tight_layout()
    plt.show()


def save_h5(a_t, sensor_cal_data, ati_cal_ft, coinft_raw, ati_raw, ati_tare, coinft_baseline):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SENSOR_NAME}_calibrationData_{ts}_{ATI_MODEL_TAG}_{ID}.h5"
    path = os.path.join(data_dir, filename)

    with h5py.File(path, "w") as f:
        f.create_dataset("A_T", data=a_t, compression="gzip")
        f.create_dataset("sensor_cal_data", data=sensor_cal_data, compression="gzip")
        f.create_dataset("ati_cal_FT", data=ati_cal_ft, compression="gzip")
        f.create_dataset("coinft_raw", data=coinft_raw, compression="gzip")
        f.create_dataset("ati_raw", data=ati_raw, compression="gzip")
        f.create_dataset("ATI_TARE", data=ati_tare)
        f.create_dataset("coinft_baseline", data=coinft_baseline)

        f.attrs["sensor_name"] = SENSOR_NAME
        f.attrs["timestamp"] = ts
        f.attrs["m_arm"] = M_ARM
        f.attrs["ati_rate"] = ATI_OUTPUT_RATE_HZ
        f.attrs["reference_sensor"] = ATI_MODEL_TAG
        f.attrs["counts_per_force"] = COUNTS_PER_FORCE
        f.attrs["counts_per_torque"] = COUNTS_PER_TORQUE

    print(f"Saved: {filename}")


#########################
# 8. MAIN               #
#########################

def main():
    force_close_serial(COM_NAME)
    ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=0.1)
    ser.write(b"i")
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.write(b"q")
    time.sleep(0.01)
    packet_size_raw = ser.read(1)
    if not packet_size_raw:
        sys.exit(f"PSoC not responding on {COM_NAME}")
    packet_size = ord(packet_size_raw) - 1
    num_sensors = (packet_size - 1) // 2
    print(f"PSoC ready. {num_sensors} channels, packet size {packet_size}.")

    master, slave = open_ati_master()
    t_ati = threading.Thread(target=read_ati_ethercat, args=(master, slave))
    t_ati.start()

    try:
        input("\nMount the gripper, apply no load, then press [Enter] to tare...\n")
        ati_tare, coinft_baseline = perform_synchronized_tare(ser, packet_size)

        ser.reset_input_buffer()
        print("BEGIN DATA COLLECTION...")
        experiment_started.set()
        time.sleep(ATI_LEADING_BUFFER_SECONDS)
        ser.reset_input_buffer()

        t_cft = threading.Thread(target=read_coinft, args=(ser, packet_size, True))
        t_cft.start()
        time.sleep(SAMPLING_DURATION)

        stop_cft.set()
        t_cft.join(timeout=3.0)
        time.sleep(ATI_TRAILING_BUFFER_SECONDS)
    finally:
        experiment_started.clear()
        stop_cft.set()
        stop_ati.set()
        t_ati.join(timeout=3.0)

        ser.write(b"i")
        ser.close()
        master.state = pysoem.INIT_STATE
        master.write_state()
        master.close()
        print("Acquisition stopped. EtherCAT master and serial port closed.")

    process_and_save(ati_tare, coinft_baseline, num_sensors)


if __name__ == "__main__":
    main()
