#!/usr/bin/env python3
"""Low-latency, two-sensor CoinFT counts-to-force ZMQ processor (run on G1).

Input:  tcp://127.0.0.1:5560, JSON v2 from coinft_server_2CFT.py.
        Source performs startup tare; this processor NEVER tares again.
Output: tcp://0.0.0.0:5561, one single-part JSON message per processed frame:
        {"ft_filtered": [[Fx,Fy,Fz,Mx,My,Mz], [Fx,Fy,Fz,Mx,My,Mz]],
         "timestamp_ns": <G1 USB-frame-reception Unix nanoseconds>}

The wire message intentionally contains ONLY the two fields requested for the
SONIC episode. On the PC, convert them to shapes (2,6) and (1,), respectively.

Samples are processed as they arrive: no 30-Hz throttle or GUI. The SUB uses
ZMQ_CONFLATE to discard *outdated* waiting messages if inference falls behind.
This is for live feedback, NOT lossless acquisition; use coinft_record_2CFT.py
on port 5560 to retain all received raw/tared count frames.

Example (from the deployment directory on G1):
    python3 coinft_processor.py --model-dir ./hardware_configs

Dependencies: numpy, pyzmq, onnxruntime (CPUExecutionProvider).
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import signal
import time

import numpy as np
import zmq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "hardware_configs"
DEFAULT_INPUT_ENDPOINT = "tcp://127.0.0.1:5560"
DEFAULT_OUTPUT_ENDPOINT = "tcp://0.0.0.0:5561"

SOURCE_VERSION = 2
SOURCE_PROCESSING = "server_tared"
CHANNELS = 12
AXES = 6

# Match the EXACT sensor order and calibration IDs in coinft_server_2CFT.py.
SENSORS = (("CFT1", "CFT24_C1"), ("CFT2", "CFT24_C3"))

DEFAULT_WINDOW = 10  # Matches visualize_coinfts_3CFT.py (causal moving average).
OUTPUT_HWM = 5       # Small PUB queue; slow clients may miss older force messages.
POLL_TIMEOUT_MS = 100
MAX_MESSAGE_BYTES = 32768
STATS_SAMPLES = 2048

logger = logging.getLogger("coinft_processor")


class ForceModel:
    """Cached one-sensor ONNX session and normalization, as in the visualizer."""

    def __init__(self, calibration_id: str, model_dir: Path, ort_module, intra_threads: int):
        model_path = model_dir / f"{calibration_id}_MLP.onnx"
        norm_path = model_dir / f"{calibration_id}_norm.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")
        if not norm_path.is_file():
            raise FileNotFoundError(f"Missing normalization JSON: {norm_path}")

        with norm_path.open("r", encoding="utf-8") as f:
            norm = json.load(f)
        for name, length in (("mu_x", CHANNELS), ("sd_x", CHANNELS),
                             ("mu_y", AXES), ("sd_y", AXES)):
            arr = np.asarray(norm[name], dtype=np.float32)
            if arr.shape != (length,) or not np.isfinite(arr).all():
                raise ValueError(f"{norm_path}: {name} must contain {length} finite values")
            if name.startswith("sd_") and np.any(arr <= 0):
                raise ValueError(f"{norm_path}: {name} must be positive")
            setattr(self, name, arr)

        opts = ort_module.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort_module.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"Expected exactly one input and output in {model_path}")
        if inputs[0].type != "tensor(float)":
            raise ValueError(f"Expected float32 ONNX input in {model_path}")
        if len(inputs[0].shape) != 2 or inputs[0].shape[1] not in (CHANNELS, None, "12"):
            raise ValueError(f"Expected ONNX input [batch, 12] in {model_path}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.calibration_id = calibration_id

    def predict(self, tared_counts: np.ndarray) -> np.ndarray:
        """Tared counts -> normalize -> ONNX -> denormalize; no second tare."""
        x = (tared_counts.astype(np.float32) - self.mu_x) / self.sd_x
        pred = self.session.run([self.output_name], {self.input_name: x.reshape(1, CHANNELS)})[0]
        pred = np.asarray(pred, dtype=np.float32)
        if pred.size != AXES or not np.isfinite(pred).all():
            raise ValueError(f"Invalid 6-axis ONNX output for {self.calibration_id}")
        force = pred.reshape(AXES) * self.sd_y + self.mu_y
        if not np.isfinite(force).all():
            raise ValueError(f"Non-finite denormalized force for {self.calibration_id}")
        return force

    def warm_up(self, iterations: int) -> None:
        # Avoid a cold first inference after the subscriber starts accepting data.
        for _ in range(iterations):
            self.predict(np.zeros(CHANNELS, dtype=np.float32))


class MovingAverage:
    """One rolling force-vector window per sensor; O(1) accumulation."""

    def __init__(self, window: int):
        self.window = window
        self.values: deque[np.ndarray] = deque(maxlen=window)
        self.sum = np.zeros(AXES, dtype=np.float64)

    def reset(self) -> None:
        self.values.clear()
        self.sum.fill(0)

    def update(self, force: np.ndarray) -> np.ndarray:
        if len(self.values) == self.window:
            self.sum -= self.values[0]
        self.values.append(force.copy())
        self.sum += force
        return (self.sum / len(self.values)).astype(np.float32)


class SampleState:
    """Keep sequence/clock continuity and reset filters after source discontinuities."""

    def __init__(self, window: int, reset_gap_ms: float):
        self.filters = [MovingAverage(window) for _ in SENSORS]
        self.reset_gap_ns = int(reset_gap_ms * 1e6)
        self.stream_id: str | None = None
        self.seq: int | None = None
        self.source_monotonic_ns: int | None = None
        self.skipped_frames = 0
        self.filter_resets = 0

    def accept(self, frame: dict) -> bool:
        stream_id = frame["stream_id"]
        seq = frame["seq"]
        src_ns = frame["source_monotonic_ns"]

        if self.stream_id != stream_id:
            if self.stream_id is not None:
                logger.warning("Source stream_id changed: resetting force filters")
            self._reset_filters()
            self.stream_id = stream_id
            self.seq = None
            self.source_monotonic_ns = None

        if self.seq is not None:
            if seq <= self.seq:
                logger.warning("Ignoring duplicate/out-of-order frame seq=%s (last=%s)", seq, self.seq)
                return False
            self.skipped_frames += max(0, seq - self.seq - 1)
            # With CONFLATE, a small seq gap is expected under load. A real
            # pause should not contaminate new measurements with old force.
            if (self.source_monotonic_ns is not None
                    and (src_ns < self.source_monotonic_ns
                         or src_ns - self.source_monotonic_ns > self.reset_gap_ns)):
                self._reset_filters()

        self.seq = seq
        self.source_monotonic_ns = src_ns
        return True

    def _reset_filters(self) -> None:
        for filt in self.filters:
            filt.reset()
        self.filter_resets += 1


def decode_source_message(payload: bytes) -> dict:
    """Validate v2 server_tared packet and sensor->model assignment."""
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("CoinFT source packet exceeds maximum expected size")
    frame = json.loads(payload)
    if not isinstance(frame, dict):
        raise ValueError("Expected a JSON object")
    if type(frame.get("version")) is not int or frame["version"] != SOURCE_VERSION:
        raise ValueError("Expected CoinFT server protocol version 2")
    if frame.get("processing") != SOURCE_PROCESSING:
        raise ValueError("Expected server_tared counts; do not perform tare twice")
    if not isinstance(frame.get("stream_id"), str) or not frame["stream_id"]:
        raise ValueError("Missing stream_id")
    for field, minimum in (("seq", 1), ("timestamp_ns", 1), ("source_monotonic_ns", 0)):
        value = frame.get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {field}: {value!r}")

    # A different order or calibration ID could silently produce wrong forces.
    names = frame.get("sensors")
    if not isinstance(names, list) or names != [list(s) for s in SENSORS]:
        raise RuntimeError(
            f"Sensor layout or calibration ID mismatch: received {names!r}, "
            f"expected {SENSORS!r}. Fix the mapping before processing."
        )
    counts = np.asarray(frame.get("tared"), dtype=np.float64)
    if counts.shape != (len(SENSORS), CHANNELS) or not np.isfinite(counts).all():
        raise ValueError(f"Expected {len(SENSORS)}x{CHANNELS} finite tared counts")
    if np.any(np.abs(counts) > 65535):
        raise ValueError("Tared counts exceed uint16 physical count-difference range")
    frame["tared"] = counts
    return frame


class Stats:
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.started_at = time.monotonic()
        self.last_log = self.started_at
        self.processed = 0
        self.published = 0
        self.dropped_publish = 0
        self.invalid = 0
        self.infer_ms: deque[float] = deque(maxlen=STATS_SAMPLES)
        self.age_ms: deque[float] = deque(maxlen=STATS_SAMPLES)

    def maybe_log(self, state: SampleState) -> None:
        now = time.monotonic()
        duration = now - self.last_log
        if duration < self.interval_sec:
            return
        def summarize(samples):
            if not samples:
                return "n/a"
            return f"p50={np.percentile(samples, 50):.2f},p95={np.percentile(samples, 95):.2f}"

        logger.info(
            "Input=%d (%.1f Hz since start) PUB=%d dropped_pub=%d invalid=%d "
            "seq_skips=%d filter_resets=%d | inference_ms(%s) source_age_ms(%s)",
            self.processed,
            self.processed / max(1e-9, now - self.started_at),
            self.published, self.dropped_publish, self.invalid,
            state.skipped_frames, state.filter_resets,
            summarize(self.infer_ms), summarize(self.age_ms),
        )
        self.last_log = now


def run(args, ort_module=None, stop=None) -> Stats:
    """Run single-threaded event-driven conversion. Exposed for integration testing."""
    if ort_module is None:
        try:
            import onnxruntime as ort_module
        except ImportError as exc:
            raise RuntimeError("Install ONNX Runtime on G1: pip install onnxruntime") from exc
    model_dir = Path(args.model_dir).expanduser().resolve()
    models = [ForceModel(cid, model_dir, ort_module, args.onnx_threads)
              for _, cid in SENSORS]
    for model in models:
        model.warm_up(args.warmup)
        logger.info("Loaded and warmed up model %s", model.calibration_id)

    state = SampleState(args.window, args.reset_gap_ms)
    stats = Stats(args.stats_interval)
    logger.info("Output: JSON keys [ft_filtered, timestamp_ns], force shape=(2,6)")

    with zmq.Context() as context:
        with context.socket(zmq.SUB) as subscriber, context.socket(zmq.PUB) as publisher:
            subscriber.setsockopt(zmq.CONFLATE, 1)
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber.setsockopt(zmq.LINGER, 0)
            subscriber.connect(args.input)

            publisher.setsockopt(zmq.SNDHWM, OUTPUT_HWM)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.bind(args.output)
            logger.info("CoinFT subscribe: %s | force publish: %s", args.input, args.output)
            logger.info("Filtering: causal moving average, window=%d; no rate limit", args.window)
            last_invalid_log = 0.0

            while stop is None or not stop.is_set():
                if subscriber.poll(POLL_TIMEOUT_MS) == 0:
                    stats.maybe_log(state)
                    continue
                payload = subscriber.recv()
                try:
                    frame = decode_source_message(payload)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    stats.invalid += 1
                    now = time.monotonic()
                    if now - last_invalid_log >= 1.0:
                        logger.warning("Discarding invalid count frame: %s", exc)
                        last_invalid_log = now
                    continue
                if not state.accept(frame):
                    continue

                infer_start = time.monotonic_ns()
                forces = [model.predict(counts) for model, counts in zip(models, frame["tared"])]
                infer_end = time.monotonic_ns()
                filtered = [fil.update(force).tolist() for fil, force in zip(state.filters, forces)]

                # ONLY the two requested client fields. Source timestamp is
                # the G1 Unix ns at completion of original USB frame reception.
                message = {
                    "ft_filtered": filtered,
                    "timestamp_ns": frame["timestamp_ns"],
                }
                wire = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
                try:
                    publisher.send(wire, flags=zmq.DONTWAIT)
                    stats.published += 1
                except zmq.Again:
                    stats.dropped_publish += 1

                stats.processed += 1
                stats.infer_ms.append((infer_end - infer_start) / 1e6)
                stats.age_ms.append((time.monotonic_ns() - frame["source_monotonic_ns"]) / 1e6)
                stats.maybe_log(state)
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT_ENDPOINT, help="Input server-tared count ZMQ endpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ENDPOINT, help="Force ZMQ PUB bind endpoint")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing *_MLP.onnx and *_norm.json")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Moving-average window in processed samples (1 disables filtering)")
    parser.add_argument("--reset-gap-ms", type=float, default=200.0, help="Reset MA after this long without a fresh source sample")
    parser.add_argument("--onnx-threads", type=int, default=1, help="Number of ONNX CPU intra-op threads per session")
    parser.add_argument("--warmup", type=int, default=2, help="Number of startup model warmup inferences")
    parser.add_argument("--stats-interval", type=float, default=5.0, help="Interval between aggregate performance logs, seconds")
    args = parser.parse_args(argv)
    if args.window < 1 or args.onnx_threads < 1 or args.warmup < 0 or args.reset_gap_ms <= 0 or args.stats_interval <= 0:
        parser.error("Require window>=1, onnx-threads>=1, warmup>=0, reset-gap-ms>0, stats-interval>0")
    if args.input == args.output:
        parser.error("Input and output ZMQ endpoints must differ")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    # signal.signal is main-thread-only; set a handler for a clean shutdown.
    import threading
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        run(args, stop=stop)
    except (OSError, ValueError, RuntimeError, zmq.ZMQError) as exc:
        logger.error("Processor stopped: %s", exc)
        raise SystemExit(1) from exc
    logger.info("Processor stopped cleanly")


if __name__ == "__main__":
    main()
