#!/usr/bin/env python3
"""Publish unprocessed CoinFT USB frames over ZMQ PUB (single-frame JSON).

Requires pyserial and pyzmq. Start recorders before starting acquisition.
PUB/SUB has no delivery acknowledgement or replay; sequence gaps reveal loss.
"""

import argparse
import logging
import queue
import signal
import struct
import threading
import time
import uuid

import serial
import zmq


# Match the Teensy firmware's sensor count and Serial2 / Serial3 / Serial4 order.
DEVICE = "/dev/ttyACM0"
BAUD_RATE = 115200
DEFAULT_PORT = 5560
SENSORS = (("left", "CFT24_C1"), ("right", "CFT24_C2"), ("top", "CFT24_C3"))
CHANNELS = 12
READ_TIMEOUT = 0.1
IDLE_TIMEOUT = 2.0
RETRY_DELAY = 1.0  # Retry USB startup only; errors after the first frame are fatal.
QUEUE_SIZE = 4096
SEND_HWM = 4096  # Per-subscriber ZMQ queue; overflow may drop messages silently.


def read_frame(port, stop, decoder):
    """Timestamp complete USB reception, then validate and decode uint16 values."""
    size = 2 + decoder.size
    data = bytearray()
    started = time.monotonic()
    while len(data) < size:
        if stop.is_set():
            return None
        data.extend(port.read(size - len(data)))
        if len(data) < size and time.monotonic() - started > IDLE_TIMEOUT:
            raise TimeoutError("No complete CoinFT USB frame")

    timestamp_ns = time.time_ns()
    source_monotonic_ns = time.monotonic_ns()
    if data[:2] != b"\x00\x00":
        raise ValueError("Invalid CoinFT frame header")
    values = decoder.unpack_from(data, 2)
    return {
        "timestamp_ns": timestamp_ns,
        "source_monotonic_ns": source_monotonic_ns,
        "raw": [values[i:i + CHANNELS] for i in range(0, len(values), CHANNELS)],
    }


class Acquisition(threading.Thread):
    def __init__(self, stop, device=DEVICE):
        super().__init__(name="coinft-usb", daemon=True)
        self.stop = stop
        self.device = device
        self.frames = queue.Queue(maxsize=QUEUE_SIZE)
        self.error = None
        self.stream_id = uuid.uuid4().hex
        self.seq = 0

    def run(self):
        decoder = struct.Struct(f"<{CHANNELS * len(SENSORS)}H")
        try:
            while not self.stop.is_set():
                port = None
                try:
                    port = serial.Serial(
                        self.device, BAUD_RATE, timeout=READ_TIMEOUT,
                        write_timeout=0.5, exclusive=True,
                    )
                    port.write(b"i")
                    port.flush()
                    if self.stop.wait(0.2):
                        break
                    port.reset_input_buffer()
                    port.write(b"s")
                    port.flush()
                    logging.info("Reading raw frames from %s; stream_id=%s", self.device, self.stream_id)

                    while not self.stop.is_set():
                        frame = read_frame(port, self.stop, decoder)
                        if frame is None:
                            break
                        self.seq += 1
                        frame.update(
                            version=1, stream_id=self.stream_id,
                            seq=self.seq, sensors=SENSORS,
                        )
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full as exc:
                            raise RuntimeError(
                                f"USB queue overflow at seq={self.seq}; stopping acquisition"
                            ) from exc
                except (OSError, ValueError) as exc:
                    if self.seq:
                        raise
                    if not self.stop.is_set():
                        logging.warning("USB startup: %s; retry in %.1fs", exc, RETRY_DELAY)
                finally:
                    if port is not None:
                        try:
                            port.write(b"i")
                        except OSError:
                            pass
                        port.close()
                self.stop.wait(RETRY_DELAY)
        except Exception as exc:
            self.error = exc
            self.stop.set()


def serve(stop, port=DEFAULT_PORT, device=DEVICE):
    endpoint = f"tcp://0.0.0.0:{port}"
    acquisition = Acquisition(stop, device)
    sent = 0
    with zmq.Context() as context, context.socket(zmq.PUB) as sock:
        sock.setsockopt(zmq.SNDHWM, SEND_HWM)
        sock.setsockopt(zmq.LINGER, 1000)
        sock.bind(endpoint)
        acquisition.start()
        logging.info("Publishing raw JSON on %s", endpoint)
        try:
            # Drain acquired frames on graceful shutdown or a USB failure.
            while acquisition.is_alive() or not acquisition.frames.empty():
                try:
                    frame = acquisition.frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                sock.send_json(frame, allow_nan=False, separators=(",", ":"))
                sent += 1
            if acquisition.error is not None:
                raise RuntimeError("CoinFT acquisition stopped") from acquisition.error
        finally:
            stop.set()
            acquisition.join(timeout=3.0)
            logging.info("Published %d frames (delivery is not acknowledged)", sent)


def main():
    parser = argparse.ArgumentParser(
        description="Publish raw CoinFT USB frames over ZMQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ TCP port to listen on")
    parser.add_argument("--usb", default=DEVICE, help="USB serial device path")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 1 <= len(SENSORS) <= 3 or len({name for name, _ in SENSORS}) != len(SENSORS):
        raise ValueError("Configure one to three sensors with unique names")
    if min(BAUD_RATE, READ_TIMEOUT, IDLE_TIMEOUT, RETRY_DELAY, QUEUE_SIZE, SEND_HWM) <= 0:
        raise ValueError("Rates, timeouts and queue sizes must be positive")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    serve(stop, port=args.port, device=args.usb)


if __name__ == "__main__":
    main()
