#!/usr/bin/env python3

import json
import os
import queue
import struct
import threading
import time
from datetime import datetime
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons

try:
    import pysoem
except Exception as exc:
    pysoem = None
    PYSOEM_IMPORT_ERROR = exc
else:
    PYSOEM_IMPORT_ERROR = None


#########################
#  Configuration        #
#########################

# ---------- CoinFT serial ----------
COM_NAME = "COM5"
BAUD_RATE = 1_000_000
START_BYTE = 2
END_BYTE = 3
SERIAL_TIMEOUT_SECONDS = 0.1

# ---------- ATI Mini58 + ECATBA / EtherCAT ----------
# Windows adapter names used by SOEM/PySOEM look like:
#   \\Device\\NPF_{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
# Leave None to probe available adapters for an ATI EtherCAT slave.
ATI_ADAPTER_NAME = None
ATI_ADAPTER_HINT = None
ATI_SLAVE_INDEX = 0  # PySOEM slave list is zero-based.

ATI_VENDOR_ID = 0x00000732
ATI_EXPECTED_PRODUCT_CODE = 0x26483052

# Preserve the original visualization ATI publication/display target.
ATI_OUTPUT_RATE_HZ = 1000.0
ATI_ETHERCAT_CYCLE_RATE_HZ = 1000.0
ATI_PROCESSDATA_TIMEOUT_US = 2000
ATI_STATE_TIMEOUT_US = 50000

# Fallback scaling only. Startup attempts to replace these from ECATBA SDO
# 0x2040:31 (counts/force unit) and 0x2040:32 (counts/torque unit).
COUNTS_PER_FORCE = 1_000_000.0   # fallback counts / N
COUNTS_PER_TORQUE = 1_000_000.0  # fallback counts / Nm
ATI_READ_SCALING_FROM_SDO = True
ATI_SCALE_IS_VERIFIED = False

if COUNTS_PER_FORCE <= 0 or COUNTS_PER_TORQUE <= 0:
    raise ValueError("COUNTS_PER_FORCE and COUNTS_PER_TORQUE must be positive.")
if ATI_OUTPUT_RATE_HZ <= 0:
    raise ValueError("ATI_OUTPUT_RATE_HZ must be positive.")
if ATI_ETHERCAT_CYCLE_RATE_HZ <= 0:
    raise ValueError("ATI_ETHERCAT_CYCLE_RATE_HZ must be positive.")
if ATI_ETHERCAT_CYCLE_RATE_HZ < ATI_OUTPUT_RATE_HZ:
    raise ValueError(
        "ATI_ETHERCAT_CYCLE_RATE_HZ must be >= ATI_OUTPUT_RATE_HZ."
    )
ATI_PUBLISH_DIVISOR = int(round(ATI_ETHERCAT_CYCLE_RATE_HZ / ATI_OUTPUT_RATE_HZ))
if not np.isclose(
    ATI_ETHERCAT_CYCLE_RATE_HZ / ATI_PUBLISH_DIVISOR,
    ATI_OUTPUT_RATE_HZ,
    rtol=0.0,
    atol=1e-9,
):
    raise ValueError(
        "ATI_ETHERCAT_CYCLE_RATE_HZ must be an integer multiple of "
        "ATI_OUTPUT_RATE_HZ for deterministic decimation."
    )

def _refresh_ati_count_scale():
    """Rebuild the counts->engineering-units divisor from active scaling."""
    global ATI_COUNT_SCALE
    ATI_COUNT_SCALE = np.repeat(
        np.array([COUNTS_PER_FORCE, COUNTS_PER_TORQUE], dtype=np.float64), 3
    )


_refresh_ati_count_scale()

# ---------- Synchronized tare ----------
# CHANGED: both sensors are now tared inside one common wall-clock window, so
# the ATI zero, the CoinFT input offset, and the CoinFT output bias all describe
# the same unloaded instant. This removes the drift gap that existed when the
# ATI was tared before the operator prompt and the CoinFT after it.
TARE_SECONDS = 15.0
TARE_TIMEOUT_SECONDS = 20.0

# CoinFT is started and allowed to settle before the shared window opens. This
# replaces the previous "collect 3000 samples, discard the first 100" scheme.
COINFT_SETTLE_SECONDS = 2.0
COINFT_MIN_TARE_SAMPLES = 500
ATI_MIN_TARE_SAMPLES = 500

# Distance from the ATI reference sensing plane to the CoinFT sensing plane.
# Verify this value for the Mini58 mechanical mounting before final experiments.
M_ARM = 0.0115  # m

# ---------- Plot and processing ----------
PLOT_DURATION = 10.0

# Filtering and smoothing for the CoinFT and ATI signals.
MOVING_AVG_WINDOW = 20

MAX_QUEUE_SIZE = 10_000
ANIMATION_INTERVAL_MS = 50

# ---------- File paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_MLP.onnx")
NORM_PATH = os.path.join(PROJECT_ROOT, "hardware_configs", "CFT24_norm.json")


#########################
#  Global state         #
#########################

stop_flag = False
ati_lock = Lock()
ati_tare_lock = Lock()

ati_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
sensor_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

all_data_records = []

# Latest ATI value is used for the CSV snapshot recorded with each CoinFT row.
latest_ati = None

ATI_TARE = np.zeros(6, dtype=np.float64)

# Tare constants recorded alongside the CSV so a dataset can be re-tared later.
TARE_META = None

# EtherCAT stays cyclically active from before tare until shutdown. These events
# only control whether raw samples are accumulated for tare or published to the
# original visualization pipeline.
ati_tare_collecting = threading.Event()
experiment_started = threading.Event()
ati_tare_samples = []
ati_thread_error = None

ati_diagnostics = {
    "bad_wkc": 0,
    "short_pdo": 0,
    "sample_counter_discontinuities": 0,
    "estimated_missing_samples": 0,
    "duplicate_samples": 0,
    "nonzero_status_records": 0,
    "coinft_framing_errors": 0,
}

# Fixed origin for the scrolling experiment-time axis.
plot_time_origin = None

# Created during main().
ati_ecat = None


#########################
#  Utility functions    #
#########################


def _require_pysoem():
    if pysoem is not None:
        return

    detail = repr(PYSOEM_IMPORT_ERROR)
    raise RuntimeError(
        "PySOEM could not be imported. The package may already be installed; "
        "this can also happen when its Windows packet-capture dependency cannot "
        "be loaded. Install 64-bit Npcap with 'WinPcap API-compatible Mode' "
        "enabled and verify that the same Python interpreter can run "
        "`python -c \"import pysoem\"`. Original import error: " + detail
    ) from PYSOEM_IMPORT_ERROR


def counts_to_engineering_units(ft_counts):
    """Convert [Fx,Fy,Fz,Mx,My,Mz] counts to [N,N,N,Nm,Nm,Nm]."""
    return np.asarray(ft_counts, dtype=np.float64) / ATI_COUNT_SCALE


def put_queue_without_deadlock(target_queue, item):
    """
    Insert data without allowing a full plotting queue to block acquisition.
    If full, discard the oldest queued sample and insert the newest sample.
    """
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass


def moving_average_2d(arr, window=5):
    """
    Per-channel centered moving-average display filter.

    CHANGED: the edge samples are normalized by the number of contributing
    samples instead of being zero-padded. The previous mode="same" convolution
    pulled the newest points of the live plot toward zero, which visually
    attenuated the leading edge of every step response.
    """
    if window <= 1 or arr.shape[0] < window:
        return arr.copy()

    output = np.zeros_like(arr)
    kernel = np.ones(window, dtype=np.float64)
    normalizer = np.convolve(
        np.ones(arr.shape[0], dtype=np.float64),
        kernel,
        mode="same",
    )

    for column in range(arr.shape[1]):
        output[:, column] = (
            np.convolve(arr[:, column], kernel, mode="same") / normalizer
        )

    return output


def _sdo_int(slave, index, subindex, *, signed, byte_count=None):
    """Read an integer CoE/SDO entry encoded in EtherCAT little-endian order."""
    data = slave.sdo_read(index, subindex)
    if byte_count is not None:
        data = data[:byte_count]
    if not data:
        raise RuntimeError(f"Empty SDO response for 0x{index:04X}:{subindex:02X}")
    return int.from_bytes(data, byteorder="little", signed=signed)


def _format_adapter_list(adapters):
    if not adapters:
        return "  (no adapters reported by PySOEM)"
    return "\n".join(
        f"  [{i}] {adapter.desc} -> {adapter.name}"
        for i, adapter in enumerate(adapters)
    )


def _sleep_to_cycle(target_time):
    """Pace the host EtherCAT loop without changing any signal processing."""
    remaining = target_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


#########################
#  ATI EtherCAT         #
#########################


class AtiEtherCATTransport:
    """Windows-compatible ATI Mini58 + ECATBA transport using PySOEM."""

    def __init__(self):
        self.master = None
        self.slave = None
        self.adapter_name = None
        self.adapter_desc = ""
        self.slave_name = ""
        self.vendor_id = 0
        self.product_code = 0
        self.revision = 0
        self.serial_number = 0
        self.force_unit_code = None
        self.torque_unit_code = None
        self.expected_wkc = 0
        self.input_layout = {}
        self.status_available = False
        self.sample_counter_available = False
        self.cycle_counter = 0
        self._opened = False
        self._zero_output = b""

    @staticmethod
    def available_adapters():
        _require_pysoem()
        return list(pysoem.find_adapters())

    def _probe_adapter_for_ati(self, adapter):
        probe = pysoem.Master()
        try:
            probe.open(adapter.name)
            count = probe.config_init()
            if count <= 0:
                return False
            return any(
                slave.man == ATI_VENDOR_ID or "ATI" in slave.name.upper()
                for slave in probe.slaves
            )
        except Exception:
            return False
        finally:
            try:
                probe.close()
            except Exception:
                pass

    def _select_adapter(self):
        adapters = self.available_adapters()
        if not adapters:
            raise RuntimeError(
                "PySOEM did not report any network adapters. Verify Npcap is "
                "installed in WinPcap API-compatible mode."
            )

        if ATI_ADAPTER_NAME:
            for adapter in adapters:
                if adapter.name == ATI_ADAPTER_NAME:
                    self.adapter_desc = adapter.desc
                    return adapter.name
            raise RuntimeError(
                f"ATI_ADAPTER_NAME was not found: {ATI_ADAPTER_NAME}\n"
                f"Available adapters:\n{_format_adapter_list(adapters)}"
            )

        candidates = adapters
        if ATI_ADAPTER_HINT:
            hint = ATI_ADAPTER_HINT.lower()
            hinted = [
                adapter
                for adapter in adapters
                if hint in adapter.name.lower() or hint in adapter.desc.lower()
            ]
            if hinted:
                candidates = hinted

        matches = [
            adapter for adapter in candidates if self._probe_adapter_for_ati(adapter)
        ]

        if len(matches) == 1:
            self.adapter_desc = matches[0].desc
            return matches[0].name

        if len(matches) > 1:
            raise RuntimeError(
                "More than one adapter sees an ATI EtherCAT slave. Set "
                "ATI_ADAPTER_NAME explicitly.\n"
                f"Matches:\n{_format_adapter_list(matches)}"
            )

        raise RuntimeError(
            "No ATI EtherCAT slave was discovered while probing Windows network "
            "adapters. Check Mini58 -> ECATBA -> Ethernet wiring, ECATBA power, "
            "and Npcap. You can also set ATI_ADAPTER_NAME explicitly.\n"
            f"Available adapters:\n{_format_adapter_list(adapters)}"
        )

    def _read_calibration_sdo(self):
        global COUNTS_PER_FORCE, COUNTS_PER_TORQUE, ATI_SCALE_IS_VERIFIED

        try:
            self.force_unit_code = _sdo_int(
                self.slave, 0x2040, 41, signed=False, byte_count=1
            )
            self.torque_unit_code = _sdo_int(
                self.slave, 0x2040, 42, signed=False, byte_count=1
            )
        except Exception as exc:
            print(f"WARNING: Could not read ATI unit codes from SDO: {exc}")

        if not ATI_READ_SCALING_FROM_SDO:
            _refresh_ati_count_scale()
            return

        try:
            counts_per_force = _sdo_int(
                self.slave, 0x2040, 49, signed=True, byte_count=4
            )
            counts_per_torque = _sdo_int(
                self.slave, 0x2040, 50, signed=True, byte_count=4
            )
            if counts_per_force <= 0 or counts_per_torque <= 0:
                raise ValueError(
                    "ECATBA returned non-positive counts-per-unit values: "
                    f"force={counts_per_force}, torque={counts_per_torque}"
                )

            COUNTS_PER_FORCE = float(counts_per_force)
            COUNTS_PER_TORQUE = float(counts_per_torque)
            ATI_SCALE_IS_VERIFIED = True
            _refresh_ati_count_scale()

            print(
                "ATI ECATBA SDO scaling loaded: "
                f"counts_per_force={COUNTS_PER_FORCE:g}, "
                f"counts_per_torque={COUNTS_PER_TORQUE:g}"
            )
            if self.force_unit_code is not None:
                print(
                    "ATI active unit codes: "
                    f"force={self.force_unit_code}, torque={self.torque_unit_code}. "
                    "Confirm the active ECATBA calibration is configured for N and N*m."
                )
        except Exception as exc:
            ATI_SCALE_IS_VERIFIED = False
            _refresh_ati_count_scale()
            print(
                "WARNING: Could not read ECATBA counts-per-unit SDO values. "
                "Using configured fallback scaling instead. "
                f"Reason: {exc}"
            )

    def _discover_input_pdo_layout(self):
        """
        Read active TxPDO assignment (0x1C13) and mapping entries through SDO.

        Mapping result:
            (object_index, subindex) -> (bit_offset, bit_length)

        If unavailable, the first six little-endian int32 values are used for
        Fx/Fy/Fz/Mx/My/Mz, matching the reference SOEM implementation.
        """
        layout = {}
        try:
            assignment_count = _sdo_int(
                self.slave, 0x1C13, 0, signed=False, byte_count=1
            )
            bit_offset = 0

            for assignment_subindex in range(1, assignment_count + 1):
                pdo_index = _sdo_int(
                    self.slave,
                    0x1C13,
                    assignment_subindex,
                    signed=False,
                    byte_count=2,
                )
                entry_count = _sdo_int(
                    self.slave, pdo_index, 0, signed=False, byte_count=1
                )

                for entry_subindex in range(1, entry_count + 1):
                    descriptor = _sdo_int(
                        self.slave,
                        pdo_index,
                        entry_subindex,
                        signed=False,
                        byte_count=4,
                    )
                    object_index = (descriptor >> 16) & 0xFFFF
                    object_subindex = (descriptor >> 8) & 0xFF
                    bit_length = descriptor & 0xFF

                    if object_index != 0:
                        layout[(object_index, object_subindex)] = (
                            bit_offset,
                            bit_length,
                        )
                    bit_offset += bit_length

            required_ft = [(0x6000, sub) for sub in range(1, 7)]
            if not all(key in layout for key in required_ft):
                print(
                    "WARNING: Active TxPDO mapping does not expose all six "
                    "0x6000 F/T entries; using the first-24-byte fallback layout."
                )
                return {}

            print("ATI TxPDO mapping discovered from SDO 0x1C13.")
            return layout
        except Exception as exc:
            print(
                "WARNING: Could not read active TxPDO mapping via SDO. "
                "Using the first-24-byte F/T layout. "
                f"Reason: {exc}"
            )
            return {}

    def _mapped_32(self, pdo_bytes, object_index, subindex=None, *, signed):
        if not self.input_layout:
            return None

        if subindex is None:
            candidates = [
                (key, value)
                for key, value in self.input_layout.items()
                if key[0] == object_index
            ]
            if not candidates:
                return None
            (_object_index, _subindex), (bit_offset, bit_length) = candidates[0]
        else:
            mapping = self.input_layout.get((object_index, subindex))
            if mapping is None:
                return None
            bit_offset, bit_length = mapping

        if bit_length != 32 or bit_offset % 8 != 0:
            return None

        byte_offset = bit_offset // 8
        end = byte_offset + 4
        if end > len(pdo_bytes):
            return None

        return int.from_bytes(
            pdo_bytes[byte_offset:end],
            byteorder="little",
            signed=signed,
        )

    def _parse_input_pdo(self, pdo_bytes):
        if len(pdo_bytes) < 24:
            ati_diagnostics["short_pdo"] += 1
            raise RuntimeError(
                f"ATI input PDO is only {len(pdo_bytes)} bytes; at least 24 are required."
            )

        if self.input_layout:
            ft_counts = [
                self._mapped_32(pdo_bytes, 0x6000, sub, signed=True)
                for sub in range(1, 7)
            ]
            if any(value is None for value in ft_counts):
                ft_counts = list(struct.unpack_from("<6i", pdo_bytes, 0))
        else:
            ft_counts = list(struct.unpack_from("<6i", pdo_bytes, 0))

        status = self._mapped_32(pdo_bytes, 0x6010, subindex=None, signed=False)
        sample_counter = self._mapped_32(
            pdo_bytes, 0x6020, subindex=None, signed=False
        )

        self.status_available = status is not None
        self.sample_counter_available = sample_counter is not None

        if status is None:
            status = 0
        if sample_counter is None:
            sample_counter = self.cycle_counter & 0xFFFFFFFF

        return ft_counts, int(status), int(sample_counter)

    def open(self):
        _require_pysoem()
        self.adapter_name = self._select_adapter()
        print(
            "Opening EtherCAT adapter: "
            f"{self.adapter_desc or '(no description)'} -> {self.adapter_name}"
        )

        self.master = pysoem.Master()
        self.master.open(self.adapter_name)
        self._opened = True

        slave_count = self.master.config_init()
        if slave_count <= 0:
            raise RuntimeError("No EtherCAT slave found on the selected adapter.")

        print(f"EtherCAT slaves found: {slave_count}")
        for index, found_slave in enumerate(self.master.slaves):
            print(
                f"  [{index}] {found_slave.name} "
                f"vendor=0x{found_slave.man:08X} "
                f"product=0x{found_slave.id:08X}"
            )

        if ATI_SLAVE_INDEX < 0 or ATI_SLAVE_INDEX >= slave_count:
            raise RuntimeError(
                f"ATI_SLAVE_INDEX={ATI_SLAVE_INDEX} is outside the discovered "
                f"slave range 0..{slave_count - 1}."
            )

        self.slave = self.master.slaves[ATI_SLAVE_INDEX]
        self.slave_name = self.slave.name
        self.vendor_id = int(self.slave.man)
        self.product_code = int(self.slave.id)
        self.revision = int(self.slave.rev)

        if self.vendor_id != ATI_VENDOR_ID:
            print(
                "WARNING: Selected EtherCAT slave vendor ID is "
                f"0x{self.vendor_id:08X}, expected ATI 0x{ATI_VENDOR_ID:08X}."
            )
        if self.product_code != ATI_EXPECTED_PRODUCT_CODE:
            print(
                "WARNING: ATI EtherCAT product code differs from the reference "
                f"ESI: got 0x{self.product_code:08X}, "
                f"reference 0x{ATI_EXPECTED_PRODUCT_CODE:08X}. Continuing because "
                "compatible ECATBA revisions may use a different product code."
            )

        try:
            self.serial_number = _sdo_int(
                self.slave, 0x1018, 4, signed=False, byte_count=4
            )
        except Exception as exc:
            print(f"WARNING: Could not read ATI EtherCAT serial number: {exc}")

        self._read_calibration_sdo()

        self.master.config_map()
        self.input_layout = self._discover_input_pdo_layout()
        self.master.config_dc()

        safeop_state = self.master.state_check(
            pysoem.SAFEOP_STATE, ATI_STATE_TIMEOUT_US
        )
        if safeop_state != pysoem.SAFEOP_STATE:
            raise RuntimeError(
                f"EtherCAT network failed to reach SAFE-OP; state=0x{safeop_state:02X}."
            )

        self._zero_output = bytes(len(self.slave.output))
        if self._zero_output:
            self.slave.output = self._zero_output

        self.master.send_processdata(release_gil=True)
        self.master.receive_processdata(
            ATI_PROCESSDATA_TIMEOUT_US, release_gil=True
        )

        self.master.state = pysoem.OP_STATE
        self.master.write_state()

        op_reached = False
        for _ in range(40):
            if self._zero_output:
                self.slave.output = self._zero_output
            self.master.send_processdata(release_gil=True)
            self.master.receive_processdata(
                ATI_PROCESSDATA_TIMEOUT_US, release_gil=True
            )
            if (
                self.master.state_check(pysoem.OP_STATE, ATI_STATE_TIMEOUT_US)
                == pysoem.OP_STATE
            ):
                op_reached = True
                break

        if not op_reached:
            self.master.read_state()
            state_details = []
            for index, found_slave in enumerate(self.master.slaves):
                if found_slave.state != pysoem.OP_STATE:
                    try:
                        status_text = pysoem.al_status_code_to_string(
                            found_slave.al_status
                        )
                    except Exception:
                        status_text = "unknown"
                    state_details.append(
                        f"slave {index} {found_slave.name}: "
                        f"state=0x{found_slave.state:02X}, "
                        f"AL=0x{found_slave.al_status:04X} ({status_text})"
                    )
            raise RuntimeError(
                "EtherCAT network failed to reach OP state. "
                + "; ".join(state_details)
            )

        self.expected_wkc = int(self.master.expected_wkc)
        print(
            f"ATI EtherCAT is OP. expected WKC={self.expected_wkc}, "
            f"input PDO bytes={len(self.slave.input)}, "
            f"output PDO bytes={len(self.slave.output)}"
        )

        sample = self.read_sample()
        if sample is None:
            raise RuntimeError(
                "ATI EtherCAT reached OP but no valid process-data sample was received."
            )

        print(
            "ATI PDO diagnostics: "
            f"status_mapped={self.status_available}, "
            f"sample_counter_mapped={self.sample_counter_available}"
        )

    def read_sample(self):
        if not self._opened or self.master is None or self.slave is None:
            raise RuntimeError("ATI EtherCAT transport is not open.")

        self.cycle_counter = (self.cycle_counter + 1) & 0xFFFFFFFF

        if self._zero_output:
            self.slave.output = self._zero_output

        self.master.send_processdata(release_gil=True)
        wkc = self.master.receive_processdata(
            ATI_PROCESSDATA_TIMEOUT_US, release_gil=True
        )

        receive_mono = time.perf_counter()
        receive_unix = time.time()

        if wkc < self.expected_wkc:
            ati_diagnostics["bad_wkc"] += 1
            return None

        ft_counts, status, sample_counter = self._parse_input_pdo(self.slave.input)
        ft = counts_to_engineering_units(ft_counts)

        return ft, receive_mono, receive_unix, sample_counter, status

    def close(self):
        if self.master is None:
            return
        try:
            if self._opened:
                try:
                    self.master.state = pysoem.INIT_STATE
                    self.master.write_state()
                except Exception:
                    pass
                self.master.close()
        finally:
            self._opened = False
            self.master = None
            self.slave = None


#########################
#  Resource loading     #
#########################

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
if not os.path.isfile(NORM_PATH):
    raise FileNotFoundError(f"Normalization JSON not found: {NORM_PATH}")

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading constants from: {NORM_PATH}")

ort_session = ort.InferenceSession(MODEL_PATH)
model_input = ort_session.get_inputs()[0]
MODEL_INPUT_NAME = model_input.name

with open(NORM_PATH, "r", encoding="utf-8") as norm_file:
    norm_data = json.load(norm_file)

mu_x = np.asarray(norm_data["mu_x"], dtype=np.float64)
sd_x = np.asarray(norm_data["sd_x"], dtype=np.float64)
mu_y = np.asarray(norm_data["mu_y"], dtype=np.float64)
sd_y = np.asarray(norm_data["sd_y"], dtype=np.float64)

if not (mu_x.shape == sd_x.shape):
    raise ValueError("mu_x and sd_x have different shapes.")
if not (mu_y.shape == sd_y.shape == (6,)):
    raise ValueError("mu_y and sd_y must each contain six values.")
if np.any(sd_x == 0) or np.any(sd_y == 0):
    raise ValueError("Normalization standard deviations must be nonzero.")


#########################
#  Hardware setup       #
#########################

# CoinFT serial handshake: unchanged from the original visualization.
ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=SERIAL_TIMEOUT_SECONDS)
try:
    ser.write(b"i")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"q")
    time.sleep(0.01)

    packet_size_raw = ser.read(1)
    if len(packet_size_raw) < 1:
        raise RuntimeError("Failed to read packet size from CoinFT.")

    packet_size_exclude_start_byte = packet_size_raw[0] - 1
    num_channels = (packet_size_exclude_start_byte - 1) // 2

    if packet_size_exclude_start_byte <= 1 or num_channels <= 0:
        raise RuntimeError(
            f"Invalid CoinFT packet size: {packet_size_exclude_start_byte}"
        )

    print(
        "CoinFT ready. "
        f"Channels: {num_channels}; packet bytes after start byte: "
        f"{packet_size_exclude_start_byte}"
    )
except Exception:
    ser.close()
    raise

# Determine whether the current model expects raw or raw+quadratic features.
normalization_input_dim = int(mu_x.size)
if normalization_input_dim == num_channels:
    MODEL_FEATURE_MODE = "linear"
elif normalization_input_dim == 2 * num_channels:
    MODEL_FEATURE_MODE = "quadratic"
else:
    ser.close()
    raise ValueError(
        "Normalization input dimension is incompatible with CoinFT channels: "
        f"mu_x has {normalization_input_dim} values, CoinFT has {num_channels} channels."
    )

model_shape_last_dim = model_input.shape[-1] if model_input.shape else None
if isinstance(model_shape_last_dim, int) and model_shape_last_dim != normalization_input_dim:
    ser.close()
    raise ValueError(
        "ONNX model input dimension does not match normalization data: "
        f"model={model_shape_last_dim}, normalization={normalization_input_dim}."
    )

print(
    f"Model input mode: {MODEL_FEATURE_MODE} "
    f"({normalization_input_dim} input features)"
)


#########################
#  Plot-data buffers    #
#########################

t_ati_store = np.array([], dtype=np.float64)
ati_data_store = np.empty((0, 6), dtype=np.float64)

t_sens_store = np.array([], dtype=np.float64)
sens_data_store = np.empty((0, 6), dtype=np.float64)


#########################
#  Acquisition helpers  #
#########################


def read_one_coinft_packet():
    """
    Read and validate a single CoinFT packet.

    Returns the raw per-channel values as a float64 array, or None when the
    packet was incomplete or badly framed. Extracted from the original
    read_coinft loop so the tare routine and the acquisition thread share one
    parser instead of duplicating the framing logic.
    """
    first_byte = ser.read(1)
    if not first_byte or first_byte[0] != START_BYTE:
        return None

    data = ser.read(packet_size_exclude_start_byte)
    if len(data) < packet_size_exclude_start_byte:
        return None

    if data[-1] != END_BYTE:
        ati_diagnostics["coinft_framing_errors"] += 1
        return None

    sensor_values = []
    for byte_index in range(0, packet_size_exclude_start_byte - 1, 2):
        low = data[byte_index]
        high = data[byte_index + 1]
        sensor_values.append(low + 256 * high)

    sensor_data = np.asarray(sensor_values, dtype=np.float64)
    if sensor_data.size != num_channels:
        ati_diagnostics["coinft_framing_errors"] += 1
        return None

    return sensor_data


def prepare_model_features(sensor_data_offsetted):
    """Construct the feature vector expected by the active model/norm file."""
    if MODEL_FEATURE_MODE == "linear":
        return sensor_data_offsetted

    return np.hstack(
        [
            sensor_data_offsetted,
            sensor_data_offsetted**2,
        ]
    )


def predict_ft(sensor_data_offsetted):
    """Original normalization -> ONNX -> denormalization path."""
    model_features = prepare_model_features(sensor_data_offsetted)

    if model_features.size != normalization_input_dim:
        raise RuntimeError(
            "Prepared model feature count is incorrect: "
            f"expected {normalization_input_dim}, got {model_features.size}."
        )

    x_norm = (model_features - mu_x) / sd_x
    x_input = x_norm.astype(np.float32).reshape(1, -1)

    prediction_normalized = ort_session.run(
        None,
        {MODEL_INPUT_NAME: x_input},
    )[0].flatten()

    if prediction_normalized.size != 6:
        raise RuntimeError(
            "ONNX output must contain six values; "
            f"received {prediction_normalized.size}."
        )

    return prediction_normalized * sd_y + mu_y


#########################
#  Synchronized tare    #
#########################


def perform_synchronized_tare():
    """
    Tare both sensors inside a single shared unloaded window.

    Replaces the previous three independent tares:
      * ATI zero was a 6 s mean taken before the operator prompt.
      * CoinFT input offset was a mean of samples 101..3000 taken after it.
      * CoinFT output bias was a single prediction.

    All three constants are now derived from the same wall-clock window, and the
    output bias is the mean of every prediction in that window rather than one
    noisy sample. The ATI processing order is unchanged:
        raw -> tare subtraction -> reference-plane compensation.
    """
    global ATI_TARE

    print("Synchronized tare: keep BOTH the ATI reference and the CoinFT unloaded.")
    print("Do not touch the fixture until the tare reports completion.")

    # Start CoinFT streaming and let it settle, then discard the warm-up bytes.
    ser.reset_input_buffer()
    ser.write(b"s")
    time.sleep(COINFT_SETTLE_SECONDS)
    ser.reset_input_buffer()

    with ati_tare_lock:
        ati_tare_samples.clear()

    coinft_raw = []
    ati_tare_collecting.set()
    start = time.perf_counter()
    deadline = start + TARE_TIMEOUT_SECONDS
    timed_out = False

    try:
        while time.perf_counter() - start < TARE_SECONDS:
            if ati_thread_error is not None:
                raise RuntimeError(
                    f"ATI EtherCAT thread failed during tare: {ati_thread_error}"
                )

            if time.perf_counter() > deadline:
                timed_out = True
                break

            packet = read_one_coinft_packet()
            if packet is not None:
                coinft_raw.append(packet)
    finally:
        ati_tare_collecting.clear()

    elapsed = time.perf_counter() - start

    if timed_out:
        raise RuntimeError(
            f"Synchronized tare timed out after {elapsed:.2f} s without "
            f"completing the {TARE_SECONDS:.2f} s window."
        )

    # ---- ATI zero ----
    with ati_tare_lock:
        ati_samples = [sample.copy() for _timestamp, sample in ati_tare_samples]

    if len(ati_samples) < ATI_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(ati_samples)} ATI samples were received during tare "
            f"(minimum {ATI_MIN_TARE_SAMPLES}). Check the Windows adapter, "
            "Npcap, ECATBA power/wiring, and EtherCAT OP state."
        )

    ati_stack = np.vstack(ati_samples)
    ATI_TARE = ati_stack.mean(axis=0)

    # ---- CoinFT input offset ----
    if len(coinft_raw) < COINFT_MIN_TARE_SAMPLES:
        raise RuntimeError(
            f"Only {len(coinft_raw)} CoinFT samples were received during tare "
            f"(minimum {COINFT_MIN_TARE_SAMPLES}). Check the serial link and "
            "the configured baud rate."
        )

    coinft_stack = np.vstack(coinft_raw)
    offset_coinft = coinft_stack.mean(axis=0)

    # ---- CoinFT output bias ----
    # The model is nonlinear, so a zero input does not imply a zero output.
    # Averaging the predictions over the same window gives the true unloaded
    # output zero and suppresses the DC noise that a single sample carried.
    predictions = np.vstack(
        [predict_ft(sample - offset_coinft) for sample in coinft_stack]
    )
    ft_bias = predictions.mean(axis=0)

    ati_std = ati_stack.std(axis=0)
    coinft_std = coinft_stack.std(axis=0)
    prediction_std = predictions.std(axis=0)

    print(
        f"Synchronized tare complete in {elapsed:.2f} s "
        f"(ATI {len(ati_samples)} samples, CoinFT {len(coinft_raw)} samples)."
    )
    print(f"  ATI_TARE          : {ATI_TARE}")
    print(f"  ATI std           : {ati_std}")
    print(f"  CoinFT offset std : {coinft_std}")
    print(f"  ft_bias           : {ft_bias}")
    print(f"  ft_bias std       : {prediction_std}")
    print(
        "  Compare these std values against a known-good tare; an unusually "
        "large value means the fixture was disturbed during the window."
    )

    metadata = {
        "tare_timestamp": datetime.now().isoformat(timespec="seconds"),
        "tare_seconds": elapsed,
        "ati_tare": ATI_TARE.tolist(),
        "ati_tare_std": ati_std.tolist(),
        "ati_tare_sample_count": len(ati_samples),
        "coinft_offset": offset_coinft.tolist(),
        "coinft_offset_std": coinft_std.tolist(),
        "coinft_tare_sample_count": len(coinft_raw),
        "ft_bias": ft_bias.tolist(),
        "ft_bias_std": prediction_std.tolist(),
        "m_arm": M_ARM,
        "counts_per_force": COUNTS_PER_FORCE,
        "counts_per_torque": COUNTS_PER_TORQUE,
        "ati_scale_is_verified": ATI_SCALE_IS_VERIFIED,
        "model_feature_mode": MODEL_FEATURE_MODE,
        "model_path": MODEL_PATH,
        "norm_path": NORM_PATH,
    }

    return offset_coinft, ft_bias, metadata


#########################
#  Worker functions     #
#########################


def read_ati_ethercat():
    """
    Keep cyclic EtherCAT PDO exchange running continuously.

    During tare, raw engineering-unit samples are accumulated. Before the
    experiment starts, PDO exchange continues but samples are not sent to the
    plotting/CSV pipeline. Once experiment_started is set, the original software
    tare, lever-arm compensation, latest_ati, and ati_queue behavior applies.
    """
    global stop_flag, latest_ati, ati_thread_error

    period = 1.0 / ATI_ETHERCAT_CYCLE_RATE_HZ
    next_cycle = time.perf_counter()
    valid_cycle_count = 0
    last_sample_counter = None
    status_warning_printed = False
    discontinuity_prints = 0

    try:
        print(
            "ATI EtherCAT cyclic exchange started: "
            f"PDO target={ATI_ETHERCAT_CYCLE_RATE_HZ:.1f} Hz, "
            f"visualization publication={ATI_OUTPUT_RATE_HZ:.1f} Hz."
        )

        while not stop_flag:
            sample = ati_ecat.read_sample()

            if sample is not None:
                raw_ft, sample_mono, sample_time, sample_counter, status = sample

                if last_sample_counter is not None:
                    delta = (int(sample_counter) - int(last_sample_counter)) & 0xFFFFFFFF
                    if delta == 0:
                        ati_diagnostics["duplicate_samples"] += 1
                    elif delta != 1:
                        ati_diagnostics["sample_counter_discontinuities"] += 1
                        ati_diagnostics["estimated_missing_samples"] += max(0, delta - 1)
                        if discontinuity_prints < 10:
                            print(
                                "ATI warning: sample-counter discontinuity "
                                f"({last_sample_counter} -> {sample_counter}, delta={delta})."
                            )
                            discontinuity_prints += 1
                last_sample_counter = sample_counter

                if status != 0:
                    ati_diagnostics["nonzero_status_records"] += 1
                    if not status_warning_printed:
                        print(
                            "ATI warning: nonzero EtherCAT sensor status word "
                            f"detected: 0x{status:08X}"
                        )
                        status_warning_printed = True

                valid_cycle_count += 1
                publish_now = (valid_cycle_count % ATI_PUBLISH_DIVISOR) == 0

                # CHANGED: tare accumulation is no longer decimated. Averaging
                # every valid cycle lowers the noise on ATI_TARE without
                # affecting the publication rate used during the experiment.
                if ati_tare_collecting.is_set():
                    with ati_tare_lock:
                        ati_tare_samples.append((sample_mono, raw_ft.copy()))

                if publish_now and experiment_started.is_set():
                    # Original ATI software tare.
                    sample_data = raw_ft - ATI_TARE

                    # Original transform to the CoinFT sensing plane.
                    sample_data[3] += sample_data[1] * M_ARM
                    sample_data[4] -= sample_data[0] * M_ARM

                    with ati_lock:
                        latest_ati = sample_data.copy()

                    put_queue_without_deadlock(
                        ati_queue,
                        (sample_time, sample_data.copy()),
                    )

            next_cycle += period
            now = time.perf_counter()
            if next_cycle < now - period:
                # Do not try to execute a burst of overdue cycles after a long
                # Windows scheduling delay; resume from the current host time.
                next_cycle = now
            _sleep_to_cycle(next_cycle)

    except Exception as exc:
        ati_thread_error = exc
        if not stop_flag:
            print(f"ATI EtherCAT thread error: {exc}")
    finally:
        print("ATI EtherCAT cyclic exchange stopped.")


def read_coinft(offset_coinft, ft_bias):
    """
    Read CoinFT serial data, run ONNX inference, record, and queue it.

    Both tare constants are now computed by perform_synchronized_tare() and
    passed in, so this loop contains only steady-state logic.
    """
    global stop_flag

    while not stop_flag:
        try:
            sensor_data = read_one_coinft_packet()
            if sensor_data is None:
                continue

            sensor_data_offsetted = sensor_data - offset_coinft
            calibrated_ft = predict_ft(sensor_data_offsetted) - ft_bias

            timestamp = time.time()

            with ati_lock:
                if latest_ati is None:
                    ati_snapshot = np.full(6, np.nan, dtype=np.float64)
                else:
                    ati_snapshot = latest_ati.copy()

            # CHANGED: the untared raw channels are recorded as well, so the
            # dataset can be re-tared offline if a tare turns out to be bad.
            row = [timestamp]
            row.extend(sensor_data.tolist())
            row.extend(sensor_data_offsetted.tolist())
            row.extend(ati_snapshot.tolist())
            row.extend(calibrated_ft.tolist())
            all_data_records.append(row)

            # The plotting queue only needs timestamp + calibrated CoinFT F/T.
            put_queue_without_deadlock(
                sensor_queue,
                (timestamp, calibrated_ft.copy()),
            )
        except Exception as exc:
            if not stop_flag:
                print(f"CoinFT read/inference error: {exc}")
            break


#########################
#  Plotting logic       #
#########################

CHANNEL_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_INDICES = [0, 1, 2]
TORQUE_INDICES = [3, 4, 5]
visibility = [True] * 6


def _drain_plot_queue(source_queue, t_store, data_store):
    """Move every pending (timestamp, row) pair from a queue into the buffers."""
    times = []
    rows = []

    while True:
        try:
            timestamp, data = source_queue.get_nowait()
        except queue.Empty:
            break

        times.append(timestamp)
        rows.append(data)

    if times:
        t_store = np.concatenate([t_store, np.asarray(times, dtype=np.float64)])
        data_store = np.vstack([data_store, np.asarray(rows, dtype=np.float64)])

    return t_store, data_store


def _trim_plot_buffer(t_store, data_store, cutoff):
    """Drop the samples that fell out of the scrolling window."""
    if t_store.size > 0:
        mask = t_store >= cutoff
        return t_store[mask], data_store[mask]

    return t_store, data_store


def update_plot(_frame):
    global t_ati_store, ati_data_store
    global t_sens_store, sens_data_store
    global plot_time_origin

    now = time.time()
    cutoff = now - PLOT_DURATION

    # Consume both acquisition queues.
    t_ati_store, ati_data_store = _drain_plot_queue(
        ati_queue, t_ati_store, ati_data_store
    )
    t_sens_store, sens_data_store = _drain_plot_queue(
        sensor_queue, t_sens_store, sens_data_store
    )

    # Keep only most recent PLOT_DURATION seconds
    t_ati_store, ati_data_store = _trim_plot_buffer(
        t_ati_store, ati_data_store, cutoff
    )
    t_sens_store, sens_data_store = _trim_plot_buffer(
        t_sens_store, sens_data_store, cutoff
    )

    if t_sens_store.size == 0:
        return

    # Apply the same moving-average display filter to both sensors.
    # The filter is intentionally applied only to the plotting buffers, so the
    # values written to CSV remain the unfiltered post-tare sensor outputs.
    sens_plot_data = moving_average_2d(
        sens_data_store,
        window=MOVING_AVG_WINDOW,
    )

    have_ati = t_ati_store.size > 0

    if have_ati:
        ati_plot_data = moving_average_2d(
            ati_data_store,
            window=MOVING_AVG_WINDOW,
        )

        # Match the filtered ATI signal to the CoinFT timestamps.
        matched_indices = np.searchsorted(
            t_ati_store,
            t_sens_store,
            side="right",
        ) - 1

        matched_indices = np.clip(
            matched_indices,
            0,
            t_ati_store.size - 1,
        )

        ati_plot_matched = ati_plot_data[matched_indices, :]
    else:
        # CHANGED: previously this drew a flat zero line, which looked like a
        # real unloaded ATI reading. Now the ATI curves are simply blank until
        # actual reference data arrives.
        ati_plot_matched = None

    # =====================================================
    # FIXED EXPERIMENT TIME AXIS
    # =====================================================

    if plot_time_origin is None:
        plot_time_origin = t_sens_store[0]

    relative_time = t_sens_store - plot_time_origin

    # =====================================================
    # Update curves
    # =====================================================

    for indices, ati_lines, sens_lines in (
        (FORCE_INDICES, lines_ati_force, lines_sens_force),
        (TORQUE_INDICES, lines_ati_torque, lines_sens_torque),
    ):
        for line_index, channel_index in enumerate(indices):
            visible = visibility[channel_index]

            if visible and ati_plot_matched is not None:
                ati_lines[line_index].set_data(
                    relative_time,
                    ati_plot_matched[:, channel_index],
                )
            else:
                ati_lines[line_index].set_data([], [])

            if visible:
                sens_lines[line_index].set_data(
                    relative_time,
                    sens_plot_data[:, channel_index],
                )
            else:
                sens_lines[line_index].set_data([], [])

    # =====================================================
    # Y-axis autoscaling
    # =====================================================

    ax_force.relim(visible_only=True)
    ax_force.autoscale_view(scalex=False, scaley=True)

    ax_torque.relim(visible_only=True)
    ax_torque.autoscale_view(scalex=False, scaley=True)

    # =====================================================
    # Scrolling X-axis
    # =====================================================

    x_right = relative_time[-1]

    if x_right <= PLOT_DURATION:
        # First 10 seconds
        ax_torque.set_xlim(0.0, PLOT_DURATION)

    else:
        # Scroll continuously after 10 seconds
        ax_torque.set_xlim(
            x_right - PLOT_DURATION,
            x_right,
        )


#########################
#  GUI setup            #
#########################

fig, (ax_force, ax_torque) = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True,
)
plt.subplots_adjust(
    left=0.10,
    bottom=0.10,
    right=0.95,
    top=0.95,
    hspace=0.25,
)

# Same channel uses the same color; sensor identity is represented by line style.
color_map = ["red", "green", "blue"]


def _build_channel_lines(axis, indices):
    """Draw the ATI/CoinFT line pair for every channel plotted on one axis."""
    ati_lines = []
    coinft_lines = []

    for line_index, channel_index in enumerate(indices):
        label = CHANNEL_LABELS[channel_index]
        color = color_map[line_index]

        ati_line, = axis.plot(
            [],
            [],
            label=f"ATI {label}",
            linestyle="-",
            color=color,
            linewidth=2,
        )
        coinft_line, = axis.plot(
            [],
            [],
            label=f"CoinFT {label}",
            linestyle=":",
            color=color,
            linewidth=2,
        )
        ati_lines.append(ati_line)
        coinft_lines.append(coinft_line)

    return ati_lines, coinft_lines


lines_ati_force, lines_sens_force = _build_channel_lines(ax_force, FORCE_INDICES)

ax_force.set_ylabel("Force (N)")
ax_force.legend(loc="upper left")
ax_force.set_title("Forces")
ax_force.grid(True)

lines_ati_torque, lines_sens_torque = _build_channel_lines(ax_torque, TORQUE_INDICES)

ax_torque.set_xlabel("Time (s)")
ax_torque.set_ylabel("Torque (Nm)")
ax_torque.legend(loc="upper left")
ax_torque.set_title("Torques")
ax_torque.grid(True)

check_ax = plt.axes([0.01, 0.40, 0.08, 0.20])
check = CheckButtons(check_ax, CHANNEL_LABELS, visibility)


def toggle_visibility(label):
    channel_index = CHANNEL_LABELS.index(label)
    visibility[channel_index] = not visibility[channel_index]


check.on_clicked(toggle_visibility)

stop_ax = plt.axes([0.85, 0.02, 0.10, 0.05])
stop_button = Button(stop_ax, "Stop")


def stop(_event):
    global stop_flag
    stop_flag = True


stop_button.on_clicked(stop)


def on_close(_event):
    global stop_flag
    stop_flag = True


fig.canvas.mpl_connect("close_event", on_close)


#########################
#  Saving and cleanup   #
#########################


def save_recorded_data():
    """
    Save the recorded data groups plus a sidecar JSON holding the tare
    constants, so the dataset can be reprocessed if a tare or M_ARM value
    later turns out to be wrong.
    """
    print("Saving data...")

    raw_columns = [f"CoinFT_raw_{index + 1}" for index in range(num_channels)]
    offset_columns = [f"CoinFT_offset_{index + 1}" for index in range(num_channels)]
    ati_columns = [f"ATI_{label}" for label in CHANNEL_LABELS]
    coinft_columns = [f"CoinFT_calib_{label}" for label in CHANNEL_LABELS]
    columns = ["Time"] + raw_columns + offset_columns + ati_columns + coinft_columns

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CoinFT_data_{timestamp_text}.csv"
    full_path = os.path.join(DATA_DIR, filename)

    if not all_data_records:
        print("No data recorded; CSV file was not created.")
        return None

    dataframe = pd.DataFrame(all_data_records, columns=columns)
    dataframe.to_csv(full_path, index=False)
    print(f"Saved {len(all_data_records)} records to: {full_path}")

    if TARE_META is not None:
        meta_path = os.path.join(DATA_DIR, f"CoinFT_data_{timestamp_text}_tare.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(TARE_META, meta_file, indent=2)
            print(f"Saved tare constants to: {meta_path}")
        except Exception as exc:
            print(f"WARNING: Could not write tare metadata: {exc}")

    return full_path


def close_hardware():
    """Stop CoinFT and close the EtherCAT/serial resources."""
    global stop_flag
    stop_flag = True
    experiment_started.clear()
    ati_tare_collecting.clear()

    try:
        if ser.is_open:
            ser.write(b"i")
    except Exception as exc:
        print(f"Warning while stopping CoinFT: {exc}")

    try:
        if ser.is_open:
            ser.close()
    except Exception as exc:
        print(f"Warning while closing serial port: {exc}")

    try:
        if ati_ecat is not None:
            ati_ecat.close()
    except Exception as exc:
        print(f"Warning while closing ATI EtherCAT master: {exc}")


#########################
#  Main execution       #
#########################


def main():
    global stop_flag, ati_ecat, latest_ati, TARE_META

    ati_thread = None
    coinft_thread = None

    try:
        _require_pysoem()

        # Initialize Mini58/ECATBA and enter EtherCAT OP state.
        ati_ecat = AtiEtherCATTransport()
        ati_ecat.open()

        # EtherCAT cyclic process-data exchange must remain active through tare
        # and while waiting for the operator.
        ati_thread = threading.Thread(target=read_ati_ethercat, daemon=True)
        ati_thread.start()

        # CHANGED: the operator prompt now comes BEFORE the tare. Nothing
        # separates the two sensors' zero points in time, so however long the
        # operator waits here, both constants still describe the same instant.
        print("\n" + "=" * 58)
        print(" >>> CONFIRM BOTH SENSORS ARE UNLOADED")
        print(" >>> PRESS [ENTER] TO START THE SYNCHRONIZED TARE")
        print("=" * 58 + "\n")
        input()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")

        offset_coinft, ft_bias, TARE_META = perform_synchronized_tare()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")

        # CoinFT is already streaming from the tare, so no second 's' command is
        # sent. Clearing the buffer drops the bytes that queued up while the
        # tare predictions were being computed.
        ser.reset_input_buffer()

        # No ATI samples were published before this point, so the original plot
        # and CSV start semantics are preserved while EtherCAT remained cyclic.
        with ati_lock:
            latest_ati = None
        experiment_started.set()

        print("\n>>> DATA COLLECTION AND PLOTTING STARTED\n")

        coinft_thread = threading.Thread(
            target=read_coinft,
            args=(offset_coinft, ft_bias),
            daemon=True,
        )
        coinft_thread.start()

        animation = FuncAnimation(
            fig,
            update_plot,
            interval=ANIMATION_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

        # Keep a live reference for the lifetime of plt.show().
        _ = animation
        plt.show()

        if ati_thread_error is not None:
            raise RuntimeError(f"ATI EtherCAT acquisition failed: {ati_thread_error}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Stopping acquisition threads...")
        stop_flag = True
        experiment_started.clear()
        ati_tare_collecting.clear()

        if coinft_thread is not None:
            coinft_thread.join(timeout=3.0)
        if ati_thread is not None:
            ati_thread.join(timeout=3.0)

        close_hardware()
        print(f"ATI diagnostics: {ati_diagnostics}")
        save_recorded_data()
        print("Stopped.")


if __name__ == "__main__":
    main()