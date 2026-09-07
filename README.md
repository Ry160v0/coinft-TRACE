# CoinFT-TRACE

Calibration, training, and visualization pipeline for **CoinFT** — a coin-sized capacitive 6-axis force/torque (F/T) sensor. This repo covers the PC-side software stack: reading raw sensor packets over serial, collecting synchronized ground-truth data against an ATI reference F/T sensor, training a neural-network calibration model, and visualizing live/recorded force-torque data.

## How it works

1. **Sensor firmware** streams 12 raw capacitive channel readings over USB serial.
2. **Data collection scripts** synchronize those raw readings against an ATI reference F/T sensor (via NI-DAQ analog I/O, UDP/RDT over Ethernet, or EtherCAT depending on the ATI model) and save labeled sessions to `.h5` files.
3. **`data_processor.py`** aggregates and splits the collected sessions into `train.h5` / `val.h5` / `test.h5`, and computes normalization statistics.
4. **`coinft_MLP_train.py`** trains a small MLP that maps the 12 raw channels to calibrated `[Fx, Fy, Fz, Mx, My, Mz]`, exporting both a PyTorch checkpoint and an ONNX model.
5. **Visualization scripts** run the exported ONNX model live against the streaming sensor (optionally alongside the ATI reference) for real-time force/torque display.

## Repository structure

```
calibration/     Data collection, dataset processing, model training/tuning
visualization/   Live plotting tools (CoinFT alone, or CoinFT vs. ATI reference)
hardware_configs/  Trained model artifacts + tuning parameters (checked in)
hardware_files/    PCB fabrication files, firmware image, Teensy bridge firmware
data/            Raw/processed calibration sessions (gitignored — see below)
```

## Firmware / serial protocol

The CoinFT board exposes a simple single-byte command protocol over USB serial at **1,000,000 baud**:

| Command | Effect |
|---|---|
| `s` | Start streaming raw sensor packets |
| `i` | Stop streaming / go idle |
| `q` | Query packet size (reply: 1 byte, `packet_size + 1`) |
| `p` | Set sample period (2-byte little-endian tick count) |
| `t` | Tune one channel's front-end (`sensor_idx, resolution, ref, compensation`) |

Each streamed packet is framed as `0x02` (start) + 12 × `uint16` little-endian raw channel counts (24 bytes) + `0x03` (end).

`hardware_files/Teensy/teensy_coinft_serial_interface.cpp` is a separate Teensy 4.0 bridge (PlatformIO, `board = teensy40`) used when running **one or two** CoinFT boards through a single USB connection: it relays `i`/`s`/`t` commands out to each board's UART, reassembles their framed packets, and forwards them upstream as a single `0x00 0x00`-prefixed multi-sensor packet at 115200 baud (consumed by `visualization/visualize_coinfts.py`).

## Calibration scripts (`calibration/`)

- **`read_ati.py`** — standalone diagnostic reader for an ATI Net F/T box over UDP/RDT (`192.168.1.1:49152`); prints live Fx..Mz to the console.
- **`intergrated_ati_data_collection_GAMMA.py`** — synchronized CoinFT + ATI Gamma/SI-130 data collection over UDP/RDT (Ethernet). Tares both sensors, records a session, and saves it as an `.h5` file under `data/`.
- **`intergrated_ati_data_collection_MINI.py`** — same collection pipeline, but reads the ATI **Mini58** reference sensor over **EtherCAT** (via `pysoem`). Both sensors are tared over the same wall-clock window so a mounted gripper's preload becomes the shared zero.
- **`data_processor.py`** — scans `data/*_calibrationData_*.h5`, groups sessions by reference-sensor model (to avoid mixing calibration campaigns), splits into train/val/test, computes per-channel normalization from the training split, and writes `data/train.h5`, `data/val.h5`, `data/test.h5`, `hardware_configs/CFT24_norm.json`, and a dataset manifest.
- **`coinft_MLP_train.py`** — trains the calibration MLP on the processed dataset and exports `hardware_configs/CFT24_MLP.pth` + `CFT24_MLP.onnx`, along with a results plot.
- **`coinft_tuner.py`** — Tkinter GUI for live-tuning each capacitive channel's front-end (resolution/reference/compensation) over serial, saving parameters to `hardware_configs/Tuning_saved.mat` / `.txt`.
- **`h5py_visualizer.py`** — quick script to load one recorded `.h5` session and plot the raw 12-channel sensor data alongside the synced ATI reference channels.

All collection/training scripts are configured by editing the constants at the top of the file (COM port, ATI adapter, session ID, etc.) and run directly, e.g.:

```bash
python calibration/intergrated_ati_data_collection_MINI.py
python calibration/data_processor.py
python calibration/coinft_MLP_train.py
```

## Visualization (`visualization/`)

- **`visualize_coinfts.py`** — real-time display for one or two CoinFT boards via the Teensy bridge (no reference sensor), running ONNX inference live.
- **`visualize_coinft_GAMMA.py`** — live CoinFT vs. ATI Gamma/SI-130 (UDP/RDT) comparison.
- **`visualize_coinft_MINI.py`** / **`visualize_coinft_MINI_testVer.py`** — live CoinFT vs. ATI Mini58 (EtherCAT) comparison; the `testVer` variant is a lighter-weight/experimental build.
- **`visualize_coinft_with_reference_sensor.py`** — live CoinFT vs. ATI comparison using NI-DAQ analog input for the reference sensor.

## Hardware config artifacts (`hardware_configs/`)

- `CFT24_MLP.pth`, `CFT24_MLP.onnx` (+ `.onnx.data`) — the trained calibration model, produced by `coinft_MLP_train.py` and loaded via `onnxruntime` at inference time by every visualization script.
- `CFT24_norm.json` — per-channel input/output normalization constants, produced by `data_processor.py`.
- `Tuning_saved.mat` / `Tuning_saved.txt` — per-channel front-end tuning parameters produced by `coinft_tuner.py`.

## Hardware files (`hardware_files/`)

Board fabrication and firmware artifacts for CoinFT: BOM (`CFT_V2_BOM_PCBA.xlsx`), component placement list (`CFT_V2_CPL_revised.xlsx`), Gerbers (`CFT_V2_Top_Gerber.zip`, `CFT_V2_Middle_Gerbers.zip`), a precompiled firmware image (`CoinFT_V2_firmware.hex`), mechanical/shielding variants, and the `Teensy/` multi-sensor bridge firmware described above.

## Dependencies

The scripts rely on: `numpy`, `scipy`, `matplotlib`, `h5py`, `pandas`, `pyserial`, `pytorch`, `onnx`, `onnxruntime`, `pysoem` (EtherCAT), and `nidaqmx` (for the NI-DAQ-based collection/visualization scripts).

## Data

`data/` is excluded from version control — it holds raw and processed calibration sessions (`.h5`, `.csv`) that are large and machine/session-specific. Run the collection scripts above to regenerate it locally, or point `data_processor.py` at your own sessions.
