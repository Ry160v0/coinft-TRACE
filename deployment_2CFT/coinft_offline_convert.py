#!/usr/bin/env python3
"""Convert coinft_record_2CFT.py server-tared counts into force CSV + manifest.

Input: <current working directory>/raw_data/<filename> with matching JSON.
Layout: deployment_2CFT/{coinft_offline_convert.py,hardware_configs/,force_data/}
Requirements: numpy, onnxruntime; no ZMQ or hardware connection required.
"""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time

import numpy as np

VERSION = 2
PROCESSING = "server_tared"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "hardware_configs"
OUTPUT_DIR = SCRIPT_DIR / "force_data"
ID_COLUMNS = (
    "stream_id", "seq", "timestamp_ns", "source_monotonic_ns",
    "received_timestamp_ns", "received_monotonic_ns",
)
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
RAW_CHANNELS = 12
AVERAGE_WINDOW = 30
BATCH_SIZE = 512


def load_manifest(path):
    """Honor the recorder's sensor order and calibration_id, not guessed filenames."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION or manifest.get("processing") != PROCESSING):
        raise ValueError("Expected version 2 server-tared recording. Legacy raw recordings "
                         "must be processed with the legacy deployment/coinft_offline_convert.py")
    sensors = manifest.get("sensors")
    if (manifest.get("channels_per_sensor") != RAW_CHANNELS
            or not isinstance(sensors, list) or not 1 <= len(sensors) <= 2):
        raise ValueError(f"Invalid recorder manifest: {path}")
    for item in sensors:
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(v, str) and v for v in item)):
            raise ValueError("Each sensor must be [name, calibration_id]")
    if len({name for name, _ in sensors}) != len(sensors):
        raise ValueError("Duplicate sensor name in manifest")
    columns = list(ID_COLUMNS) + [
        f"{name}_ch{i:02d}" for name, _ in sensors for i in range(1, RAW_CHANNELS + 1)
    ]
    if manifest.get("csv_columns", columns) != columns:
        raise ValueError("Manifest csv_columns disagree with sensor layout")
    streams = manifest.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("Manifest has no stream tare metadata")
    for state in streams.values():
        if not isinstance(state, dict):
            raise ValueError("Invalid stream metadata")
        tare = state.get("tare")
        if not isinstance(tare, dict):
            raise ValueError("Missing server tare metadata")
        frames, skip = tare.get("frames"), tare.get("skip")
        if type(frames) is not int or type(skip) is not int or not 0 <= skip < frames:
            raise ValueError("Invalid server tare frame counts")
        offsets = tare.get("offsets")
        if not isinstance(offsets, list) or len(offsets) != len(sensors):
            raise ValueError("Tare offsets do not match sensor count")
        for channels in offsets:
            if (not isinstance(channels, list) or len(channels) != RAW_CHANNELS
                    or any(type(v) not in (int, float) or not 0 <= v <= 65535
                           for v in channels)):
                raise ValueError("Expected 12 numeric tare offsets per sensor within uint16 range")
    return sensors, columns, manifest


def count_rows(path, columns, count):
    """Stream valid records; preserve identification/timestamp strings verbatim."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        if next(reader, None) != columns:
            raise ValueError("CSV header does not match the recorder manifest")
        for line, row in enumerate(reader, 2):
            if len(row) != len(columns) or not row[0]:
                raise ValueError(f"CSV line {line}: missing or incorrect fields")
            try:
                seq = int(row[1])
                times = [int(v) for v in row[2:6]]
                counts = np.fromiter((float(v) for v in row[6:]), dtype=np.float64,
                                     count=count * RAW_CHANNELS).reshape(count, RAW_CHANNELS)
            except ValueError as exc:
                raise ValueError(f"CSV line {line}: invalid integer identity or numeric count") from exc
            if (seq < 1 or times[0] <= 0 or times[1] < 0 or times[2] <= 0
                    or times[3] < 0 or not np.isfinite(counts).all()
                    or np.any(np.abs(counts) > 65535)):
                raise ValueError(f"CSV line {line}: invalid seq, timestamps, or tared counts")
            yield row[:6], row[0], seq, counts


def inspect_streams(path, columns, count, source_streams):
    """Validate sequence order and carry forward the server's tare metadata."""
    streams = {}
    for _, sid, seq, _ in count_rows(path, columns, count):
        if sid not in source_streams:
            raise ValueError(f"Stream {sid}: missing server tare metadata")
        state = streams.setdefault(sid, {
            "rows_written": 0, "first_seq": seq, "last_seq": seq - 1,
            "missing_frames": 0, "tare": source_streams[sid]["tare"],
        })
        if seq <= state["last_seq"]:
            raise ValueError(f"Stream {sid}: repeated or backwards seq {seq}")
        if state["rows_written"]:
            state["missing_frames"] += seq - state["last_seq"] - 1
        state["last_seq"] = seq
        state["rows_written"] += 1
    if not streams:
        raise ValueError("Tared CSV has no data rows")
    return streams


class ForceModel:
    """One cached ONNX session per calibrated CoinFT; raw tare is applied upstream."""

    def __init__(self, calibration_id, folder):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", calibration_id):
            raise ValueError(f"Unsafe calibration_id: {calibration_id!r}")
        self.model = folder / f"{calibration_id}_MLP.onnx"
        self.norm = folder / f"{calibration_id}_norm.json"
        for path in (self.model, self.norm):
            if not path.is_file():
                raise FileNotFoundError(f"Missing calibration file: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install numpy onnxruntime") from exc
        self.session = ort.InferenceSession(str(self.model), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError(f"Expected one float32 input and one output: {self.model}")
        shape = inputs[0].shape
        if len(shape) != 2 or shape[1] not in (RAW_CHANNELS, None, "12"):
            raise ValueError(f"Expected ONNX input [batch, 12], got {shape}")
        self.single_sample = shape[0] == 1
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        params = json.loads(self.norm.read_text(encoding="utf-8"))
        for key, size in (("mu_x", 12), ("sd_x", 12), ("mu_y", 6), ("sd_y", 6)):
            values = np.asarray(params.get(key), dtype=np.float64)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"{self.norm}: {key} must have {size} finite numbers")
            if key.startswith("sd_") and np.any(values <= 0):
                raise ValueError(f"{self.norm}: {key} must be strictly positive")
            setattr(self, key, values)

    def predict(self, tared_counts):
        x = ((tared_counts - self.mu_x) / self.sd_x).astype(np.float32)
        if self.single_sample:
            y = np.stack([
                self.session.run([self.output_name], {self.input_name: a[None, :]})[0]
                .reshape(-1) for a in x
            ])
        else:
            y = self.session.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y, dtype=np.float64)
        if y.size != len(x) * len(AXES) or not np.isfinite(y).all():
            raise ValueError(f"Invalid model output from {self.model}")
        return y.reshape(len(x), len(AXES)) * self.sd_y + self.mu_y


class MovingAverage:
    def __init__(self, window):
        self.values = deque(maxlen=window)
        self.sum = np.zeros(len(AXES))
        self.window = window

    def update(self, vector):
        if len(self.values) == self.window:
            self.sum -= self.values[0]
        self.values.append(vector)
        self.sum += vector
        return self.sum / len(self.values)


def write_chunk(rows, sensors, models, position, filters, last_seq, window, writer):
    """Batch inference for each sensor, then write rows in exact source order."""
    predictions = np.empty((len(rows), len(sensors), len(AXES)))
    for index, (name, _) in enumerate(sensors):
        values = np.stack([row[3][index] for row in rows])
        predictions[:, index] = models[name].predict(values)

    for i, (identity, sid, seq, _) in enumerate(rows):
        position[sid] = position.get(sid, 0) + 1
        previous = last_seq.get(sid)
        if previous is not None and seq > previous + 1:
            for name, _ in sensors:
                filters.pop((sid, name), None)
        last_seq[sid] = seq
        force = []
        for index, (name, _) in enumerate(sensors):
            key = (sid, name)
            if key not in filters:
                filters[key] = MovingAverage(window)
            force.extend(f"{value:.10g}" for value in filters[key].update(predictions[i, index]))
        writer.writerow([*identity, "ok", *force])


def convert(input_csv, model_dir=MODEL_DIR, output=None, window=AVERAGE_WINDOW):
    if type(window) is not int or window < 1:
        raise ValueError("Require average window >= 1")
    input_csv = Path(input_csv).expanduser().resolve()
    manifest_path = input_csv.with_suffix(".json")
    sensors, columns, source = load_manifest(manifest_path)
    streams = inspect_streams(input_csv, columns, len(sensors), source["streams"])

    cache = {}
    models = {}
    for name, calibration_id in sensors:
        if calibration_id not in cache:
            cache[calibration_id] = ForceModel(calibration_id, Path(model_dir))
        models[name] = cache[calibration_id]

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = OUTPUT_DIR / f"coinft_force_{stamp}.csv"
    output = Path(output).expanduser().resolve()
    out_manifest = output.with_suffix(".json")
    if output == input_csv or out_manifest == manifest_path:
        raise ValueError("Output must not overwrite source recordings")
    if output.exists() or out_manifest.exists():
        raise FileExistsError("Output CSV/JSON already exists (recorder files are never overwritten)")
    output.parent.mkdir(parents=True, exist_ok=True)
    out_columns = [*ID_COLUMNS, "processing_status", *[
        f"{name}_{axis}" for name, _ in sensors for axis in AXES
    ]]

    position, filters, last_seq = {}, {}, {}
    created = False
    created_json = False
    try:
        with output.open("x", newline="", encoding="utf-8") as target:
            created = True
            writer = csv.writer(target)
            writer.writerow(out_columns)
            batch = []
            for row in count_rows(input_csv, columns, len(sensors)):
                batch.append(row)
                if len(batch) >= BATCH_SIZE:
                    write_chunk(batch, sensors, models, position,
                                filters, last_seq, window, writer)
                    batch.clear()
            if batch:
                write_chunk(batch, sensors, models, position,
                            filters, last_seq, window, writer)
            if any(position.get(sid, 0) != s["rows_written"] for sid, s in streams.items()):
                raise RuntimeError("Input CSV changed between passes")

        result = {
            "version": VERSION, "status": "converted", "csv_file": output.name,
            "source_csv": str(input_csv), "source_metadata": str(manifest_path),
            "sensors": sensors, "force_axes_per_sensor": 6,
            "csv_columns": out_columns, "rows_written": sum(s["rows_written"] for s in streams.values()),
            "rows_with_force": sum(s["rows_written"] for s in streams.values()),
            "missing_frames": sum(s["missing_frames"] for s in streams.values()),
            "processing": {
                "pipeline": "server-tared counts -> normalize -> ONNX -> denormalize -> causal moving average",
                "tare_applied_by": "server",
                "moving_average_window": window,
                "on_missing_seq": "reset averaging; never interpolate",
                "force_units": "training labels; check model-specific moment units",
            },
            "model_files": {
                name: {"calibration_id": cid,
                       "model": str(models[name].model),
                       "normalization": str(models[name].norm)}
                for name, cid in sensors
            },
            "time_fields": source.get("time_fields", {}),
            "streams": streams,
            "converted_timestamp_ns": time.time_ns(),
        }
        with out_manifest.open("x", encoding="utf-8") as f:
            created_json = True
            json.dump(result, f, indent=2, allow_nan=False)
        return output, result
    except Exception:
        # Partial derived output must not be mistaken for a complete conversion.
        if created:
            output.unlink(missing_ok=True)
        if created_json:
            out_manifest.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV filename in ./raw_data (matching JSON is auto-loaded)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, help="Override timestamped force_data/coinft_force_*.csv")
    parser.add_argument("--average-window", type=int, default=AVERAGE_WINDOW)
    args = parser.parse_args()
    if (not args.input or args.input in (".", "..")
            or any(char in args.input for char in "/\\:")):
        parser.error("input must be a filename only, without a directory")
    input_csv = Path.cwd() / "raw_data" / args.input
    try:
        output, report = convert(input_csv, args.model_dir, args.output,
                                 args.average_window)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("Saved %d frames (%d with force, %d missing) to %s",
                 report["rows_written"], report["rows_with_force"],
                 report["missing_frames"], output)


if __name__ == "__main__":
    main()
