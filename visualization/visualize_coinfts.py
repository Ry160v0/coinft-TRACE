#!/usr/bin/env python3

import time
import struct
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import serial
import json
import os

# =========================
# User Configuration
# =========================
NUM_COINFTS  = 2   # Currently supports only 1 or 2 CoinFTs. The number of CoinFTs can be easily expanded by modifying the code.

# UART Settings
PORT_NAME    = "COM8"
BAUD_RATE    = 115200
READ_TIMEOUT = 0.1

# Directory Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', 'hardware_configs')

# Sensor / Model Settings
# List ALL potential models here. The script will slice this list based on NUM_COINFTS.
ALL_MODEL_FILES = ['CFT24_C3_MLP.onnx', 'CFT24_C2_MLP.onnx']
ALL_NORM_FILES  = ['CFT24_C3_norm.json', 'CFT24_C2_norm.json']
ALL_LABELS      = ['Left Sensor', 'Right Sensor']

# Apply Selection
MODEL_FILES = ALL_MODEL_FILES[:NUM_COINFTS]
NORM_FILES  = ALL_NORM_FILES[:NUM_COINFTS]
LABELS      = ALL_LABELS[:NUM_COINFTS]

# Data Processing
INITIAL_SAMPLES = 500    # Samples to collect for tare
IGNORED_SAMPLES = 10     # Ignore start transient
WINDOW_SIZE     = 10     # Moving average window size

# Plotting
PLOT_HISTORY    = 5.0    # Seconds of history to show
PLOT_FPS        = 20     # Fixed plot refresh rate, independent of packet rate

# Constants
COINFT_CH        = 12
BYTES_PER_SENSOR = COINFT_CH * 2  # 12 uint16s * 2 bytes
BODY_LEN         = BYTES_PER_SENSOR * NUM_COINFTS

# =========================
# Serial / Model Helpers
# =========================

def load_norms(norm_filenames):
    """Load normalization constants from JSON files."""
    out = []
    for fname in norm_filenames:
        path = os.path.join(CONFIG_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find config file: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        out.append({
            'mu_x': np.array(data['mu_x'], dtype=np.float32),
            'sd_x': np.array(data['sd_x'], dtype=np.float32),
            'mu_y': np.array(data['mu_y'], dtype=np.float32),
            'sd_y': np.array(data['sd_y'], dtype=np.float32),
        })
    return out


def start_stream(ser):
    """Handshake to start streaming."""
    print("Resetting sensors...")
    ser.write(b'i')
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b's')
    time.sleep(0.05)


def read_packet(ser):
    """Reads one frame for N sensors."""
    # 1. Look for Header
    while True:
        b = ser.read(1)
        if not b:
            return None  # Timeout
        if b == b'\x00':
            b2 = ser.read(1)
            if b2 == b'\x00':
                break  # Found header!

    # 2. Read Body
    body = ser.read(BODY_LEN)
    if len(body) != BODY_LEN:
        return None

    # 3. Parse
    vals = struct.unpack('<' + 'H' * (COINFT_CH * NUM_COINFTS), body)

    sensor_data_list = []
    for i in range(NUM_COINFTS):
        start = i * COINFT_CH
        end = (i + 1) * COINFT_CH
        sensor_data_list.append(np.array(vals[start:end], dtype=np.float64))

    return sensor_data_list


def tare(ser):
    """Collects INITIAL_SAMPLES packets and returns per-sensor offsets."""
    print(f"Taring... ({INITIAL_SAMPLES} samples)")
    tare_buffers = [[] for _ in range(NUM_COINFTS)]

    for _ in range(INITIAL_SAMPLES):
        pkt_list = read_packet(ser)
        if pkt_list:
            for i in range(NUM_COINFTS):
                tare_buffers[i].append(pkt_list[i])

    offsets = []
    for i in range(NUM_COINFTS):
        arr = np.array(tare_buffers[i])
        if len(arr) <= IGNORED_SAMPLES:
            raise RuntimeError("Not enough data for tare. Check connection.")
        offsets.append(np.mean(arr[IGNORED_SAMPLES:], axis=0))

    print("Tare complete.")
    return offsets


# =========================
# Shared Plot Buffer
# =========================

class SharedPlotBuffer:
    """Per-sensor ring buffers of (t, force, moment).

    Written by the acquisition thread, snapshotted (under lock) by the main
    thread right before each redraw. Using deque.popleft() instead of
    list.pop(0) keeps history trimming O(1) instead of O(n).
    """

    def __init__(self, num_sensors):
        self.lock = threading.Lock()
        self.t = [deque() for _ in range(num_sensors)]
        self.f = [[deque(), deque(), deque()] for _ in range(num_sensors)]
        self.m = [[deque(), deque(), deque()] for _ in range(num_sensors)]

    def append(self, i, t_now, ft_avg):
        with self.lock:
            self.t[i].append(t_now)
            for j in range(3):
                self.f[i][j].append(ft_avg[j])
            for j in range(3):
                self.m[i][j].append(ft_avg[j + 3])

            while self.t[i] and (self.t[i][-1] - self.t[i][0] > PLOT_HISTORY):
                self.t[i].popleft()
                for j in range(3):
                    self.f[i][j].popleft()
                for j in range(3):
                    self.m[i][j].popleft()

    def snapshot(self, i):
        """Returns plain lists so matplotlib can draw without holding the lock."""
        with self.lock:
            t = list(self.t[i])
            f = [list(channel) for channel in self.f[i]]
            m = [list(channel) for channel in self.m[i]]
        return t, f, m


# =========================
# Acquisition Thread
# =========================

class AcquisitionThread(threading.Thread):
    """Owns the serial port and ONNX sessions. Never touches matplotlib, so a
    slow plot redraw on the main thread cannot delay serial reads here."""

    def __init__(self, ser, sessions, norms, offsets, buffer, start_time):
        super().__init__(daemon=True)
        self.ser = ser
        self.sessions = sessions
        self.norms = norms
        self.offsets = offsets
        self.buffer = buffer
        self.start_time = start_time
        self.ma_queues = [deque(maxlen=WINDOW_SIZE) for _ in range(NUM_COINFTS)]
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            pkt_list = read_packet(self.ser)
            if pkt_list is None:
                continue

            now = time.time() - self.start_time

            for i, raw in enumerate(pkt_list):
                raw_zeroed = raw - self.offsets[i]
                raw_norm = (raw_zeroed.astype(np.float32) - self.norms[i]['mu_x']) / self.norms[i]['sd_x']

                input_name = self.sessions[i].get_inputs()[0].name
                pred_norm = self.sessions[i].run(None, {input_name: raw_norm.reshape(1, 12)})[0].flatten()

                ft_val = pred_norm * self.norms[i]['sd_y'] + self.norms[i]['mu_y']

                self.ma_queues[i].append(ft_val)
                ft_avg = np.mean(self.ma_queues[i], axis=0)

                self.buffer.append(i, now, ft_avg)

    def stop(self):
        self.stop_event.set()


# =========================
# Plotting
# =========================

def setup_plot():
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    lines = [None] * NUM_COINFTS
    for i in range(NUM_COINFTS):
        ax_f = axes[0][i]
        lfx, = ax_f.plot([], [], 'r-', label='Fx')
        lfy, = ax_f.plot([], [], 'g-', label='Fy')
        lfz, = ax_f.plot([], [], 'b-', label='Fz')
        ax_f.set_title(f"{LABELS[i]} - Force")
        ax_f.legend(loc='upper right')
        ax_f.grid(True)

        ax_m = axes[1][i]
        lmx, = ax_m.plot([], [], 'r--', label='Mx')
        lmy, = ax_m.plot([], [], 'g--', label='My')
        lmz, = ax_m.plot([], [], 'b--', label='Mz')
        ax_m.set_title(f"{LABELS[i]} - Moment")
        ax_m.legend(loc='upper right')
        ax_m.grid(True)

        lines[i] = [(lfx, lfy, lfz), (lmx, lmy, lmz)]

    # Hide unused subplots if NUM_COINFTS < 2
    if NUM_COINFTS < 2:
        for r in range(2):
            for c in range(NUM_COINFTS, 2):
                axes[r][c].set_visible(False)

    return fig, axes, lines


def update_plot(axes, lines, buffer):
    for i in range(NUM_COINFTS):
        t, f, m = buffer.snapshot(i)
        if not t:
            continue

        all_f = []
        for j, line in enumerate(lines[i][0]):
            line.set_data(t, f[j])
            all_f.extend(f[j])
        if all_f:
            axes[0][i].set_xlim(t[0], t[-1])
            axes[0][i].set_ylim(min(all_f) - 3.0, max(all_f) + 3.0)

        all_m = []
        for j, line in enumerate(lines[i][1]):
            line.set_data(t, m[j])
            all_m.extend(m[j])
        if all_m:
            axes[1][i].set_xlim(t[0], t[-1])
            axes[1][i].set_ylim(min(all_m) - 0.1, max(all_m) + 0.1)


# =========================
# Main
# =========================

def main():
    print(f"Configured for {NUM_COINFTS} sensors.")
    print(f"Reading configs from: {CONFIG_DIR}")

    model_paths = [os.path.join(CONFIG_DIR, f) for f in MODEL_FILES]
    for p in model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found: {p}")

    sessions = [ort.InferenceSession(p) for p in model_paths]
    norms = load_norms(NORM_FILES)

    print(f"Opening {PORT_NAME}...")
    try:
        ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=READ_TIMEOUT)
    except Exception as e:
        print(f"Error opening serial: {e}")
        return

    start_stream(ser)
    offsets = tare(ser)

    buffer = SharedPlotBuffer(NUM_COINFTS)
    start_time = time.time()

    worker = AcquisitionThread(ser, sessions, norms, offsets, buffer, start_time)
    worker.start()

    fig, axes, lines = setup_plot()

    print("Starting visualization... (Ctrl+C to stop)")
    frame_period = 1.0 / PLOT_FPS

    try:
        while True:
            loop_start = time.time()

            update_plot(axes, lines, buffer)
            plt.pause(0.001)

            sleep_left = frame_period - (time.time() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        worker.stop()
        worker.join(timeout=1.0)
        ser.write(b'i')
        ser.close()
        print("Closed.")


if __name__ == "__main__":
    main()
