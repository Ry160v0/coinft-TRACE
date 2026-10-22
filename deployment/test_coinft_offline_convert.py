"""Tests processing and recorder CSV compatibility with a mocked ONNX Runtime.

The fake model returns the first six normalized input channels. It tests the
conversion path, not real sensor calibration accuracy.
"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from coinft_offline_convert import convert, load_manifest, measure_baselines, MODEL_DIR

FIXTURE = Path(__file__).with_name("coinft_raw_test.csv")


class FakeTensor:
    name = "input"
    type = "tensor(float)"
    shape = [None, 12]


class FakeOutput:
    name = "output"


class FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        return [FakeTensor()]

    def get_outputs(self):
        return [FakeOutput()]

    def run(self, names, inputs):
        return [inputs["input"][:, :6]]


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.models = self.directory / "hardware_configs"
        self.models.mkdir()
        norm = {"mu_x": [0] * 12, "sd_x": [1] * 12,
                "mu_y": [0] * 6, "sd_y": [1] * 6}
        for i in (1, 2, 3):
            (self.models / f"CFT24_C{i}_MLP.onnx").write_bytes(b"FAKE TEST MODEL")
            (self.models / f"CFT24_C{i}_norm.json").write_text(json.dumps(norm))
        self.patched = patch.dict(sys.modules, {"onnxruntime": types.SimpleNamespace(InferenceSession=FakeSession)})
        self.patched.start()
        self.addCleanup(self.patched.stop)

    def test_processing_and_identifiers(self):
        target = self.directory / "coinft_force_test.csv"
        path, report = convert(FIXTURE, self.models, target, window=5)
        self.assertEqual(path, target)
        self.assertEqual(report["rows_written"], 1450)
        self.assertEqual(report["rows_with_force"], 450)
        self.assertEqual(report["missing_frames"], 2)
        self.assertEqual(set(report["streams"]), {"demo-stream-A", "demo-stream-B"})
        with FIXTURE.open(newline="") as raw_file, path.open(newline="") as force_file:
            raw = list(csv.reader(raw_file))
            out = list(csv.reader(force_file))
        self.assertEqual(len(raw), len(out))
        self.assertEqual(len(out[0]), 6 + 1 + 18)
        for x, y in zip(raw[1:], out[1:]):
            self.assertEqual(x[:6], y[:6])
        self.assertEqual(out[500][6], "tare")
        self.assertEqual(out[501][6], "ok")
        self.assertEqual(out[800][6], "ok")
        self.assertEqual(out[801][6], "tare")
        self.assertEqual(out[1301][6], "ok")
        self.assertTrue(all(v == "" for v in out[500][7:]))
        offset = report["streams"]["demo-stream-A"]["tare_raw_offsets"]["left"][0]
        first_force = float(out[501][7])
        first_raw = int(raw[501][6])
        self.assertAlmostEqual(first_force, first_raw - offset, places=5)
        force_json = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(force_json["csv_columns"], out[0])
        with self.assertRaises(FileExistsError):
            convert(FIXTURE, self.models, target)

    def test_seq_gap_and_stream_baselines(self):
        sensors, columns, _ = load_manifest(FIXTURE.with_suffix(".json"))
        stream_info = measure_baselines(FIXTURE, columns, len(sensors), 500, 10)
        self.assertEqual(stream_info["demo-stream-A"]["missing_frames"], 2)
        self.assertEqual(stream_info["demo-stream-B"]["missing_frames"], 0)
        self.assertEqual(stream_info["demo-stream-A"]["rows_written"], 800)

    def test_missing_calibration_fails_before_writing(self):
        empty = self.directory / "empty"
        empty.mkdir()
        output = self.directory / "missing.csv"
        with self.assertRaises(FileNotFoundError):
            convert(FIXTURE, empty, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
