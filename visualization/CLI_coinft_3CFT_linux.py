#!/usr/bin/env python3
"""Send filtered 3CFT readings as UTF-8 JSON UDP datagrams.

Each datagram contains time_s (seconds since acquisition started) and a sensors
list with name, Fx, Fy, Fz, Mx, My, and Mz for each configured sensor.
"""

import json
import os
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import serial

# User configuration
NUM_COINFTS = 3  # Supports 1, 2, or 3 sensors.
PORT_NAME = "/dev/ttyACM0"  # On Windows, use the device's COM port, e.g. "COM3".
BAUD_RATE = 115200
READ_TIMEOUT = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', 'hardware_configs')
ALL_MODEL_FILES = ['CFT24_C1_MLP.onnx', 'CFT24_C2_MLP.onnx', 'CFT24_C3_MLP.onnx']
ALL_NORM_FILES = ['CFT24_C1_norm.json', 'CFT24_C2_norm.json', 'CFT24_C3_norm.json']
ALL_LABELS = ['Left Sensor', 'Right Sensor', 'Top Sensor']
MODEL_FILES = ALL_MODEL_FILES[:NUM_COINFTS]
NORM_FILES = ALL_NORM_FILES[:NUM_COINFTS]
LABELS = ALL_LABELS[:NUM_COINFTS]

INITIAL_SAMPLES = 500
IGNORED_SAMPLES = 10
WINDOW_SIZE = 10
OUTPUT_HZ = 20  # Send the latest readings at this rate; acquisition runs independently.
UDP_HOST = "127.0.0.1"  # Receiver IP; use the remote computer's IP for network output.
UDP_PORT = 5005  # The receiving program must bind this UDP port.

COINFT_CH = 12
BYTES_PER_SENSOR = COINFT_CH * 2
BODY_LEN = BYTES_PER_SENSOR * NUM_COINFTS


def load_norms(norm_filenames):
    """Load normalization constants from JSON files."""
    out = []
    for fname in norm_filenames:
        path = os.path.join(CONFIG_DIR, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        out.append({
            key: np.array(data[key], dtype=np.float32)
            for key in ('mu_x', 'sd_x', 'mu_y', 'sd_y')
        })
    return out


def start_stream(ser):
    """Handshake to start streaming."""
    print("Resetting sensors...", flush=True)
    ser.write(b'i')
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b's')
    time.sleep(0.05)


def read_packet(ser, stop_event=None):
    """Read one frame for all sensors, allowing acquisition to be stopped."""
    while stop_event is None or not stop_event.is_set():
        b = ser.read(1)
        if not b:
            return None
        if b == b'\x00' and ser.read(1) == b'\x00':
            break
    else:
        return None

    body = ser.read(BODY_LEN)
    if len(body) != BODY_LEN:
        return None
    vals = struct.unpack('<' + 'H' * (COINFT_CH * NUM_COINFTS), body)
    return [
        np.array(vals[i * COINFT_CH:(i + 1) * COINFT_CH], dtype=np.float64)
        for i in range(NUM_COINFTS)
    ]


def tare(ser):
    """Collect startup packets and calculate per-sensor offsets."""
    print(f"Taring... ({INITIAL_SAMPLES} samples)", flush=True)
    tare_buffers = [[] for _ in range(NUM_COINFTS)]
    for _ in range(INITIAL_SAMPLES):
        pkt_list = read_packet(ser)
        if pkt_list:
            for i in range(NUM_COINFTS):
                tare_buffers[i].append(pkt_list[i])

    offsets = []
    for samples in tare_buffers:
        arr = np.array(samples)
        if len(arr) <= IGNORED_SAMPLES:
            raise RuntimeError("Not enough data for tare. Check connection.")
        offsets.append(np.mean(arr[IGNORED_SAMPLES:], axis=0))
    print("Tare complete.", flush=True)
    return offsets


class LatestReadings:
    """Share the latest complete sensor frame with the UDP output loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def update(self, elapsed, readings):
        with self.lock:
            self.frame = (elapsed, np.array(readings, copy=True))

    def snapshot(self):
        with self.lock:
            if self.frame is None:
                return None
            elapsed, readings = self.frame
            return elapsed, readings.copy()


class AcquisitionThread(threading.Thread):
    """Keep serial acquisition and inference independent of UDP output."""

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
        self.error = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                pkt_list = read_packet(self.ser, self.stop_event)
                if pkt_list is None:
                    continue

                elapsed = time.monotonic() - self.start_time
                readings = []
                for i, raw in enumerate(pkt_list):
                    raw_zeroed = raw - self.offsets[i]
                    raw_norm = (raw_zeroed.astype(np.float32) - self.norms[i]['mu_x']) / self.norms[i]['sd_x']
                    input_name = self.sessions[i].get_inputs()[0].name
                    pred_norm = self.sessions[i].run(
                        None, {input_name: raw_norm.reshape(1, COINFT_CH)}
                    )[0].flatten()
                    ft_val = pred_norm * self.norms[i]['sd_y'] + self.norms[i]['mu_y']
                    self.ma_queues[i].append(ft_val)
                    readings.append(np.mean(self.ma_queues[i], axis=0))

                self.buffer.update(elapsed, readings)
        except Exception as exc:
            self.error = exc
        finally:
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()


def send_readings(sock, destination, elapsed, readings):
    """Send all sensors in one JSON datagram, followed by a newline."""
    sensors = []
    for label, values in zip(LABELS, readings):
        sensor = {'name': label}
        sensor.update(zip(('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'), map(float, values)))
        sensors.append(sensor)
    payload = json.dumps({'time_s': elapsed, 'sensors': sensors}, allow_nan=False, indent = 2)
    sock.sendto((payload + '\n\n').encode('utf-8'), destination)


def main():
    if not 1 <= NUM_COINFTS <= len(ALL_MODEL_FILES):
        raise ValueError("NUM_COINFTS must be 1, 2, or 3.")
    if OUTPUT_HZ <= 0:
        raise ValueError("OUTPUT_HZ must be positive.")
    if not 1 <= UDP_PORT <= 65535:
        raise ValueError("UDP_PORT must be between 1 and 65535.")

    print(f"Configured for {NUM_COINFTS} sensors.", flush=True)
    print(f"Reading configs from: {CONFIG_DIR}", flush=True)
    model_paths = [os.path.join(CONFIG_DIR, f) for f in MODEL_FILES]
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
    sessions = [ort.InferenceSession(path) for path in model_paths]
    norms = load_norms(NORM_FILES)

    print(f"Opening {PORT_NAME}...", flush=True)
    ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=READ_TIMEOUT)
    worker = None
    try:
        start_stream(ser)
        offsets = tare(ser)
        buffer = LatestReadings()
        worker = AcquisitionThread(ser, sessions, norms, offsets, buffer, time.monotonic())
        worker.start()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            destination = (UDP_HOST, UDP_PORT)
            print(f"Sending JSON to udp://{UDP_HOST}:{UDP_PORT} at {OUTPUT_HZ} Hz /n"
                  "(Ctrl+C to stop)", flush=True)
            last_elapsed = None
            while True:
                frame = buffer.snapshot()
                if frame is not None and frame[0] != last_elapsed:
                    send_readings(sock, destination, *frame)
                    last_elapsed = frame[0]
                if worker.stop_event.wait(1.0 / OUTPUT_HZ):
                    if worker.error is not None:
                        raise RuntimeError(f"Acquisition failed: {worker.error}") from worker.error
                    break
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        if worker is not None:
            worker.stop()
            worker.join()
        try:
            ser.write(b'i')
        finally:
            ser.close()
            print("Closed.", flush=True)


if __name__ == '__main__':
    main()
