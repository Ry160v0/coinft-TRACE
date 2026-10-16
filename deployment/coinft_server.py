#!/usr/bin/env python3
"""Acquire CoinFT data and reply to each ZeroMQ request with the latest force/torque."""

import json
import logging
import signal
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial
import zmq


# Configuration: match Teensy Serial2 / Serial3 / Serial4.
DEVICE = "/dev/ttyACM0"
BAUD_RATE = 115200
ENDPOINT = "tcp://0.0.0.0:5560"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "hardware_configs"

SENSORS = [
    ("Left Sensor", "CFT24_C1"),
    ("Right Sensor", "CFT24_C2"),
    ("Top Sensor", "CFT24_C3"),
]

INITIAL_SAMPLES = 500
IGNORED_SAMPLES = 10
WINDOW_SIZE = 10

READ_TIMEOUT = 0.1
STALE_AFTER = 0.2
IDLE_TIMEOUT = 2.0
RETRY_DELAY = 1.0  # Retry before the first received frame only.


class SharedFrames:
    def __init__(self):
        self.condition = threading.Condition()
        self.stopped = threading.Event()

        self.seq = 0
        self.raw = self.result = self.error = None

    def close(self, error=None):
        with self.condition:
            if error is not None:
                self.error = error
                logging.error("CoinFT stopped: %s", error)

            self.stopped.set()
            self.result = None
            self.condition.notify_all()

    def publish_raw(self, raw, timestamp_ns, received_mono):
        with self.condition:
            self.seq += 1
            self.raw = (
                self.seq, timestamp_ns, received_mono, raw
            )
            self.condition.notify_all()

    def next_raw(self, previous_seq):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopped.is_set()
                or (
                    self.raw is not None
                    and self.raw[0] != previous_seq
                )
            )

            return None if self.stopped.is_set() else self.raw

    def publish_result(self, frame, ft):
        with self.condition:
            if not self.stopped.is_set():
                self.result = (
                    frame[2],
                    {
                        "seq": frame[0],
                        "timestamp_ns": frame[1],
                        "ft": ft.tolist(),
                    },
                )

    def snapshot(self):
        with self.condition:
            result = self.result

        if (
            result is None
            or time.monotonic() - result[0] > STALE_AFTER
        ):
            return {
                "seq": None,
                "timestamp_ns": None,
                "ft": None,
            }

        return result[1]


class CoinFTProcessor:
    def __init__(self):
        self.models = []

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        for name, stem in SENSORS:
            data = json.loads(
                (CONFIG_DIR / f"{stem}_norm.json").read_text()
            )

            norm = {}
            for key, size in (
                ("mu_x", 12), ("sd_x", 12),
                ("mu_y", 6), ("sd_y", 6),
            ):
                value = np.asarray(data[key], dtype=np.float32)

                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} for {name}")

                norm[key] = value

            if np.any(norm["sd_x"] <= 0):
                raise ValueError(
                    f"sd_x must be positive for {name}"
                )

            session = ort.InferenceSession(
                str(CONFIG_DIR / f"{stem}_MLP.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            self.models.append(
                (session, session.get_inputs()[0].name, norm)
            )

        self.tare_frames = []
        self.offsets = None
        self.window = deque(maxlen=WINDOW_SIZE)

    def process(self, raw):
        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != (len(SENSORS), 12) or not np.isfinite(raw).all():
            raise ValueError("Invalid complete raw frame")

        if self.offsets is None:
            self.tare_frames.append(raw)

            if len(self.tare_frames) == INITIAL_SAMPLES:
                self.offsets = np.mean(
                    self.tare_frames[IGNORED_SAMPLES:], axis=0
                )
                self.tare_frames.clear()
                logging.info("Tare complete")

            return None

        zeroed = (raw - self.offsets).astype(np.float32)
        readings = []

        for i, (session, input_name, norm) in enumerate(self.models):
            x = (zeroed[i] - norm["mu_x"]) / norm["sd_x"]
            y = np.asarray(
                session.run(None, {input_name: x[None, :]})[0]
            ).flatten()

            if y.shape != (6,):
                raise ValueError(
                    f"Expected six outputs for {SENSORS[i][0]}"
                )

            readings.append(
                y * norm["sd_y"] + norm["mu_y"]
            )

        ft = np.stack(readings)

        if not np.isfinite(ft).all():
            raise ValueError("Non-finite force/torque output")

        self.window.append(ft)
        return np.mean(self.window, axis=0)


def read_frame(port, stop):
    count = len(SENSORS)
    size = 2 + count * 24
    data = bytearray()
    started = time.monotonic()

    while len(data) < size:
        if stop.is_set():
            return None

        data.extend(port.read(size - len(data)))

        if len(data) < size and time.monotonic() - started > IDLE_TIMEOUT:
            raise TimeoutError("No complete CoinFT frame")

    # Timestamp complete USB reception before validation and decoding.
    timestamp_ns = time.time_ns()
    received_mono = time.monotonic()

    if data[:2] != b"\x00\x00":
        raise ValueError("Invalid frame header")

    values = struct.unpack(
        "<" + "H" * (count * 12), data[2:]
    )
    raw = [
        list(values[i:i + 12])
        for i in range(0, len(values), 12)
    ]

    return raw, timestamp_ns, received_mono


def acquisition(shared):
    seen_data = False

    try:
        while not shared.stopped.is_set():
            port = None

            try:
                port = serial.Serial(
                    DEVICE,
                    BAUD_RATE,
                    timeout=READ_TIMEOUT,
                    write_timeout=0.5,
                    exclusive=True,
                )

                port.write(b"i")
                port.flush()

                if shared.stopped.wait(0.2):
                    break

                port.reset_input_buffer()
                port.write(b"s")
                port.flush()

                logging.info(
                    "Streaming from %s; keep sensors unloaded during tare",
                    DEVICE,
                )

                while not shared.stopped.is_set():
                    frame = read_frame(port, shared.stopped)

                    if frame is not None:
                        seen_data = True
                        shared.publish_raw(*frame)

            except (OSError, ValueError) as exc:
                if seen_data:
                    raise

                logging.warning(
                    "%s; retry in %.1fs", exc, RETRY_DELAY
                )

            finally:
                if port is not None:
                    try:
                        port.write(b"i")
                    except OSError:
                        pass
                    port.close()

            shared.stopped.wait(RETRY_DELAY)

    except Exception as exc:
        if not shared.stopped.is_set():
            shared.close(exc)


def processing(shared, processor):
    previous_seq = 0

    try:
        while not shared.stopped.is_set():
            frame = shared.next_raw(previous_seq)

            if frame is None:
                break

            previous_seq = frame[0]
            ft = processor.process(frame[3])

            if ft is not None:
                shared.publish_result(frame, ft)

    except Exception as exc:
        if not shared.stopped.is_set():
            shared.close(exc)


def serve(shared, processor):
    """Each request triggers a latest-frame reply; request contents are ignored."""
    with zmq.Context() as context, context.socket(zmq.REP) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, 500)
        sock.bind(ENDPOINT)

        workers = [
            threading.Thread(
                target=acquisition,
                args=(shared,),
                daemon=True,
            ),
            threading.Thread(
                target=processing,
                args=(shared, processor),
                daemon=True,
            ),
        ]

        for worker in workers:
            worker.start()

        logging.info(
            "CoinFT processed data available on %s", ENDPOINT
        )

        try:
            while not shared.stopped.is_set():
                if sock.poll(100):
                    sock.recv_multipart()
                    sock.send_json(shared.snapshot(), allow_nan=False)

            if shared.error is not None:
                raise RuntimeError(
                    "CoinFT worker failed; "
                    "restart with sensors unloaded"
                ) from shared.error

        finally:
            shared.close()

            for worker in workers:
                worker.join(timeout=3.0)


def main():
    if len(SENSORS) not in (1, 2, 3):
        raise ValueError("Configure one, two, or three sensors")

    if not (0 <= IGNORED_SAMPLES < INITIAL_SAMPLES) or WINDOW_SIZE < 1:
        raise ValueError("Invalid tare or filter settings")

    if min(
        BAUD_RATE, READ_TIMEOUT, STALE_AFTER,
        IDLE_TIMEOUT, RETRY_DELAY,
    ) <= 0:
        raise ValueError("Baud rate and timeouts must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    shared = SharedFrames()

    def shutdown(*_):
        logging.info("Stopping CoinFT")
        shared.close()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    serve(shared, CoinFTProcessor())


if __name__ == "__main__":
    main()
