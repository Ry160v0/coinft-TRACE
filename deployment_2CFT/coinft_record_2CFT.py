#!/usr/bin/env python3
"""Record server-tared CoinFT counts and per-stream tare metadata."""

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import signal
import socket
import threading
import time

import zmq

VERSION = 2
PROCESSING = "server_tared"
CHANNELS = 12
# Psi0's example robot address: replace with the actual G1 wired-interface IP.
ROBOT_IP = "192.168.123.164"
DEFAULT_PORT = 5560
OUTPUT_DIR = "raw_data"  # Relative to this script; each run creates new files.
RECEIVE_HWM = 8192
QUEUE_SIZE = 8192
STALL_AFTER = 2.0
FLUSH_INTERVAL = 1.0  # Buffered writes; flush periodically and on graceful exit.
TIME_COLUMNS = (
    "timestamp_ns", "source_monotonic_ns",
    "received_timestamp_ns", "received_monotonic_ns",
)


class Receiver(threading.Thread):
    """Own the SUB socket in one thread; enqueue bytes before JSON/CSV work."""

    def __init__(self, stop, endpoint):
        super().__init__(name="coinft-receive", daemon=True)
        self.stop = stop
        self.endpoint = endpoint
        self.frames = queue.Queue(maxsize=QUEUE_SIZE)
        self.error = None
        self.stalls = 0

    def run(self):
        last_received = time.monotonic()
        stalled = False
        try:
            with zmq.Context() as context, context.socket(zmq.SUB) as sock:
                sock.setsockopt(zmq.SUBSCRIBE, b"")
                sock.setsockopt(zmq.RCVHWM, RECEIVE_HWM)
                sock.setsockopt(zmq.LINGER, 0)
                # Do not use CONFLATE: the recorder needs every received frame.
                sock.connect(self.endpoint)
                while not self.stop.is_set():
                    if not sock.poll(100):
                        if not stalled and time.monotonic() - last_received >= STALL_AFTER:
                            stalled = True
                            self.stalls += 1
                            logging.warning("No tared messages for %.1fs; server may still be taring. Waiting for %s",
                                            STALL_AFTER, self.endpoint)
                        continue
                    payload = sock.recv()
                    received_timestamp_ns = time.time_ns()
                    received_monotonic_ns = time.monotonic_ns()
                    last_received = received_monotonic_ns / 1e9
                    if stalled:
                        logging.info("Tared messages received")
                        stalled = False
                    try:
                        self.frames.put_nowait((payload, received_timestamp_ns, received_monotonic_ns))
                    except queue.Full as exc:
                        raise RuntimeError("Recording queue overflow; stopping instead of silently dropping counts") from exc
        except Exception as exc:
            self.error = exc
            self.stop.set()


def decode_frame(payload):
    """Validate the wire schema before a frame can affect columns or statistics."""
    frame = json.loads(payload)
    if (not isinstance(frame, dict) or type(frame.get("version")) is not int
            or frame["version"] != VERSION or frame.get("processing") != PROCESSING):
        raise ValueError("Expected CoinFT protocol version 2 with server-tared counts")
    if not isinstance(frame.get("stream_id"), str) or not frame["stream_id"]:
        raise ValueError("Missing stream_id")
    for key, minimum in (("seq", 1), ("timestamp_ns", 1), ("source_monotonic_ns", 0)):
        if type(frame.get(key)) is not int or frame[key] < minimum:
            raise ValueError(f"Invalid {key}")
    sensors = frame.get("sensors")
    if not isinstance(sensors, list) or not 1 <= len(sensors) <= 2:
        raise ValueError("Expected one to two sensors")
    for sensor in sensors:
        if (not isinstance(sensor, list) or len(sensor) != 2
                or any(not isinstance(value, str) or not value for value in sensor)):
            raise ValueError("Each sensor must provide [name, calibration_id]")
    if len({sensor[0] for sensor in sensors}) != len(sensors):
        raise ValueError("Sensor names must be unique")
    tare = frame.get("tare")
    if not isinstance(tare, dict):
        raise ValueError("Missing server tare metadata")
    frames, skip = tare.get("frames"), tare.get("skip")
    if type(frames) is not int or type(skip) is not int or not 0 <= skip < frames:
        raise ValueError("Invalid server tare frame counts")
    for values, minimum in ((frame.get("tared"), -65535), (tare.get("offsets"), 0)):
        if not isinstance(values, list) or len(values) != len(sensors):
            raise ValueError("Channel data does not match sensor count")
        for channels in values:
            if (not isinstance(channels, list) or len(channels) != CHANNELS
                    or any(type(v) not in (int, float) or not minimum <= v <= 65535
                           for v in channels)):
                raise ValueError("Expected 12 numeric channel values per sensor within count range")
    return frame


def save_metadata(file, metadata):
    file.seek(0)
    json.dump(metadata, file, indent=2, allow_nan=False)
    file.truncate()
    file.flush()


def record(stop, port=DEFAULT_PORT):
    endpoint = f"tcp://{ROBOT_IP}:{port}"
    output_dir = Path(OUTPUT_DIR).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "coinft_raw_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    csv_path = output_dir / f"{stem}.csv"
    metadata_path = output_dir / f"{stem}.json"
    metadata = {
        "version": VERSION, "status": "recording", "endpoint": endpoint,
        "csv_file": csv_path.name, "recorder_host": socket.gethostname(),
        "started_timestamp_ns": time.time_ns(),
        "sensors": None, "channels_per_sensor": CHANNELS,
        "counts_format": "Signed floating-point counts: raw uint16 minus server tare offsets, in USB packet order",
        "processing": PROCESSING,
        "clock_sync": "External; not configured or verified by this program",
        "time_fields": {
            "timestamp_ns": "G1 Unix ns at complete USB reception, before decoding",
            "source_monotonic_ns": "G1 monotonic ns at complete USB reception",
            "received_timestamp_ns": "PC Unix ns after ZMQ recv, before JSON decoding",
            "received_monotonic_ns": "PC monotonic ns after ZMQ recv",
        },
        "transport": "ZMQ PUB/SUB JSON; no acknowledgement or replay",
        "receive_hwm": RECEIVE_HWM, "queue_size": QUEUE_SIZE,
        "flush_interval_s": FLUSH_INTERVAL, "rows_written": 0,
        "missing_frames": 0, "streams": {},
    }
    receiver = Receiver(stop, endpoint)
    with csv_path.open("x", newline="", encoding="utf-8") as csv_file, \
            metadata_path.open("x+", encoding="utf-8") as metadata_file:
        writer = csv.writer(csv_file)
        save_metadata(metadata_file, metadata)
        receiver.start()
        last_flush = time.monotonic()
        logging.info("Recording %s to %s", endpoint, csv_path)
        logging.info("Waiting for server tare to complete; recording starts with the first tared frame. Ctrl+C stops recording.")
        try:
            while receiver.is_alive() or not receiver.frames.empty():
                try:
                    payload, received_ns, received_mono_ns = receiver.frames.get(timeout=0.1)
                except queue.Empty:
                    payload = None
                if payload is not None:
                    frame = decode_frame(payload)
                    if metadata["sensors"] is None:
                        metadata["sensors"] = frame["sensors"]
                        columns = ["stream_id", "seq", *TIME_COLUMNS] + [
                            f"{name}_ch{i:02d}" for name, _ in frame["sensors"]
                            for i in range(1, CHANNELS + 1)
                        ]
                        metadata["csv_columns"] = columns
                        writer.writerow(columns)
                    elif frame["sensors"] != metadata["sensors"]:
                        raise ValueError("Sensor layout changed; start a new recording")

                    stream_id, seq = frame["stream_id"], frame["seq"]
                    streams = metadata["streams"]
                    if stream_id not in streams:
                        streams[stream_id] = {
                            "first_seq": seq, "last_seq": seq - 1,
                            "frames_before_subscription": seq - 1,
                            "rows_written": 0, "missing_frames": 0,
                            "tare": frame["tare"],
                        }
                        logging.info("Stream %s: first seq=%d; %d earlier frames unavailable", stream_id, seq, seq - 1)
                        logging.info("Server tare ready: %d frames, first %d skipped; saving corrected counts.",
                                     frame["tare"]["frames"], frame["tare"]["skip"])
                    stats = streams[stream_id]
                    if frame["tare"] != stats["tare"]:
                        raise ValueError("Tare changed within a stream; restart the server for a new tare")
                    if seq <= stats["last_seq"]:
                        raise ValueError(f"Duplicate or backwards seq={seq} in stream {stream_id}")
                    gap = seq - stats["last_seq"] - 1
                    if gap:
                        logging.warning("Stream %s: missing seq %d..%d", stream_id, stats["last_seq"] + 1, seq - 1)
                    writer.writerow([
                        stream_id, seq, frame["timestamp_ns"], frame["source_monotonic_ns"],
                        received_ns, received_mono_ns,
                        *(value for values in frame["tared"] for value in values),
                    ])
                    stats["last_seq"] = seq
                    stats["rows_written"] += 1
                    stats["missing_frames"] += gap
                    metadata["rows_written"] += 1
                    metadata["missing_frames"] += gap

                now = time.monotonic()
                if now - last_flush >= FLUSH_INTERVAL:
                    csv_file.flush()
                    save_metadata(metadata_file, metadata)
                    logging.info("Recorded %d frames; detected %d missing", metadata["rows_written"], metadata["missing_frames"])
                    last_flush = now
            if receiver.error is not None:
                raise RuntimeError("CoinFT receiver stopped") from receiver.error
            metadata["status"] = "stopped"
        except Exception as exc:
            metadata["status"] = "error"
            metadata["error"] = str(exc.__cause__ or exc)
            raise
        finally:
            stop.set()
            receiver.join(timeout=2.0)
            metadata["ended_timestamp_ns"] = time.time_ns()
            metadata["receive_stalls"] = receiver.stalls
            metadata["pending_frames_not_written"] = receiver.frames.qsize()
            csv_file.flush()
            save_metadata(metadata_file, metadata)
            logging.info("Saved %d tared frames to %s", metadata["rows_written"], csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Record server-tared CoinFT ZMQ messages to CSV and JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="G1 server's ZMQ TCP port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if min(RECEIVE_HWM, QUEUE_SIZE, STALL_AFTER, FLUSH_INTERVAL) <= 0:
        raise ValueError("Queue sizes and intervals must be positive")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    record(stop, port=args.port)


if __name__ == "__main__":
    main()
