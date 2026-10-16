#!/usr/bin/env python3
"""Request the latest CoinFT force/torque frame and record it to CSV."""

import csv
import logging
import math
import time
from pathlib import Path

import zmq


# User configuration. Relative output paths are resolved beside this script.
OUTPUT_FILE = "coinft_record.csv"
ENDPOINT = "tcp://127.0.0.1:5560"
REQUEST_INTERVAL = 0.01  # Seconds between requests; at most about 100 requests/s.
REQUEST_TIMEOUT_MS = 3000

# Must match the order of SENSORS in coinft_server.py.
SENSOR_NAMES = ("left", "right", "top")
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")  # N, N, N, Nm, Nm, Nm.
CSV_COLUMNS = ["seq", "timestamp_ns", "received_timestamp_ns"] + [
    f"{sensor}_{axis}" for sensor in SENSOR_NAMES for axis in AXES
]


def connect_socket(context):
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.SNDTIMEO, REQUEST_TIMEOUT_MS)
    sock.setsockopt(zmq.RCVTIMEO, REQUEST_TIMEOUT_MS)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(ENDPOINT)
    return sock


def frame_to_row(frame, received_timestamp_ns):
    """Flatten one valid frame; the server's unavailable reply yields no row."""
    if not isinstance(frame, dict):
        raise ValueError("Expected a JSON object from the server")

    seq = frame.get("seq")
    timestamp_ns = frame.get("timestamp_ns")
    ft = frame.get("ft")

    if seq is None and timestamp_ns is None and ft is None:
        return None

    if type(seq) is not int or seq < 1:
        raise ValueError("Invalid server sequence number")
    if type(timestamp_ns) is not int or timestamp_ns <= 0:
        raise ValueError("Invalid server timestamp")
    if not isinstance(ft, list) or len(ft) != len(SENSOR_NAMES):
        raise ValueError("Server sensor count does not match SENSOR_NAMES")

    values = []
    for sensor_ft in ft:
        if not isinstance(sensor_ft, list) or len(sensor_ft) != len(AXES):
            raise ValueError("Expected six force/torque values per sensor")
        for value in sensor_ft:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Force/torque values must be finite numbers")
            values.append(value)

    return [seq, timestamp_ns, received_timestamp_ns, *values]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not math.isfinite(REQUEST_INTERVAL) or REQUEST_INTERVAL <= 0:
        raise ValueError("REQUEST_INTERVAL must be positive and finite")
    if REQUEST_TIMEOUT_MS <= 0:
        raise ValueError("REQUEST_TIMEOUT_MS must be positive")

    output_path = Path(OUTPUT_FILE).expanduser()
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation protects previous recordings from being overwritten.
    with output_path.open("x", newline="", encoding="utf-8") as csv_file, \
            zmq.Context() as context:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        csv_file.flush()

        sock = connect_socket(context)
        last_frame_key = None
        rows_written = 0
        last_report = time.monotonic()
        logging.info("Requesting data from %s", ENDPOINT)
        logging.info("Recording to %s; press Ctrl+C to stop", output_path)
        logging.info("Waiting for valid data (including server tare)")

        try:
            while True:
                requested_at = time.monotonic()
                try:
                    sock.send(b"get")
                    frame = sock.recv_json()
                    received_timestamp_ns = time.time_ns()
                except zmq.Again:
                    logging.warning("Server request timed out; reconnecting")
                    # A REQ socket cannot send again after a lost reply.
                    sock.close()
                    sock = connect_socket(context)
                    time.sleep(REQUEST_INTERVAL)
                    continue

                row = frame_to_row(frame, received_timestamp_ns)
                if row is not None:
                    # Including the timestamp allows seq to reset after restart.
                    frame_key = (row[0], row[1])
                    if frame_key != last_frame_key:
                        writer.writerow(row)
                        csv_file.flush()
                        last_frame_key = frame_key
                        rows_written += 1
                        now = time.monotonic()
                        if rows_written == 1 or now - last_report >= 1.0:
                            logging.info("Recorded %d frames", rows_written)
                            last_report = now

                remaining = REQUEST_INTERVAL - (time.monotonic() - requested_at)
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            logging.info("Recording stopped")
        finally:
            sock.close()
            logging.info("Saved %d frames to %s", rows_written, output_path)


if __name__ == "__main__":
    main()
