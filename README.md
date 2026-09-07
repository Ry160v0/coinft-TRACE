# CoinFT-TRACE

Calibration, training, and visualization pipeline for **CoinFT** — a coin-sized capacitive 6-axis force/torque (F/T) sensor. This repo covers the PC-side software stack: reading raw sensor packets over serial, collecting synchronized ground-truth data against an ATI reference F/T sensor, training a neural-network calibration model, and visualizing live/recorded force-torque data.

## How it works

1. **Sensor firmware** streams 12 raw capacitive channel readings over USB serial.
2. **Data collection scripts** synchronize those raw readings against an ATI reference F/T sensor (over UDP/RDT for the Gamma/SI-130, or EtherCAT for the Mini58) and save labeled sessions to `.h5` files.
3. **`data_processor.py`** aggregates and splits the collected sessions into `train.h5` / `val.h5` / `test.h5`, and computes normalization statistics.
4. **`coinft_MLP_train.py`** trains a small MLP that maps the 12 raw channels to calibrated `[Fx, Fy, Fz, Mx, My, Mz]`, exporting both a PyTorch checkpoint and an ONNX model.
5. **Visualization scripts** run the exported ONNX model live against the streaming sensor (optionally alongside the ATI reference) so you can see and validate the calibration in real time.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `h5py`, `pandas`, `pyserial`, `pytorch`, `onnx`, `onnxruntime`, `pysoem` (EtherCAT, needed for the Mini58 path).

## The calibration workflow

The full pipeline is four steps, run in order. Each script is configured by editing the constants in its "CONTROL PANEL" section at the top of the file (COM port, ATI adapter, session ID, duration, etc.) before running it.

### Step 1 — Collect data (`calibration/`)

Connect the CoinFT over USB and mount it against your ATI reference sensor, then run one of:

```bash
python calibration/intergrated_ati_data_collection_GAMMA.py   # ATI Gamma / SI-130, over UDP/RDT (Ethernet)
python calibration/intergrated_ati_data_collection_MINI.py    # ATI Mini58, over EtherCAT
```

Both scripts tare the CoinFT and the ATI sensor together at the start (so any mounting preload becomes the shared zero), then record a synchronized session for `SAMPLING_DURATION` seconds (40s by default) and save it to `data/` as `CFT24_calibrationData_<timestamp>_<ATI_model>_<ID>.h5`. Set `ID` to `train`, `val`, or `test` before each run — `data_processor.py` uses that suffix to build the dataset splits.

**Repeat this for many short sessions**, each one exciting a different combination of axes. A single session with only straight-down presses will calibrate `Fz` well but leave the model blind to shear and torque. Use the recipes below as a guide for what to physically do with the sensor during each 40-second recording — mix and repeat them across your `train`/`val`/`test` sessions:

| Recipe | What to do in ~40s | Axes excited |
|---|---|---|
| A — Pure normal | Slow press from zero to full scale and back to zero, 3-4 cycles | Fz |
| B — Normal, varying rate | Fast press/release, plus hold-and-watch-creep | Fz (hysteresis / creep) |
| C — Pure shear X | Push back and forth along X | Fx (+ a little My) |
| D — Pure shear Y | Push back and forth along Y | Fy (+ a little Mx) |
| E — Off-center press | Press different points around the sensor's edge, working around the rim | Fz + Mx + My |
| F — Twist | Press down with a finger and rotate | Mz |
| G — Mixed scrub | Random combination of press + push + twist | All axes |
| H — Pull-up | Pull upward (only if your application actually needs tension) | +Fz (not covered by A/B) |

The more of these you cover — and the more you vary force magnitude and speed — the better the trained model generalizes.

### Step 2 — Process the dataset

```bash
python calibration/data_processor.py
```

Scans `data/*_calibrationData_*.h5`, groups sessions by reference-sensor model (so a Gamma campaign and a Mini58 campaign aren't mixed together), splits into train/val/test, and writes `data/train.h5`, `data/val.h5`, `data/test.h5` plus `hardware_configs/CFT24_norm.json` (per-channel normalization stats used later for inference).

### Step 3 — Train the model

```bash
python calibration/coinft_MLP_train.py
```

Trains the calibration MLP on the processed dataset and saves `hardware_configs/CFT24_MLP.pth` (PyTorch) and `hardware_configs/CFT24_MLP.onnx` (used by every visualization script), along with a results plot comparing predicted vs. reference force/torque on the test split.

### Step 4 — Visualize and validate

```bash
python visualization/visualize_coinft_GAMMA.py     # live CoinFT vs. ATI Gamma/SI-130
python visualization/visualize_coinft_MINI.py       # live CoinFT vs. ATI Mini58 (EtherCAT)
python visualization/visualize_coinfts.py           # CoinFT only, no reference sensor (via Teensy bridge)
```

These run the exported ONNX model live and plot calibrated Fx..Mz in real time, next to the ATI reference where available — use them to sanity-check accuracy before trusting the calibration, or to re-tune with `calibration/coinft_tuner.py` if a channel looks off.

## Notes

- `data/` is excluded from version control (`.gitignore`) — it holds raw and processed sessions that are large and machine/session-specific. Regenerate it locally by running Step 1.
- `hardware_configs/` and `hardware_files/` (trained model, tuning parameters, PCB/firmware files) are checked in, since they're small and needed to reproduce or deploy a calibration without re-collecting data.
- The CoinFT board streams over USB serial at 1,000,000 baud (`s` = start streaming, `i` = stop, `q` = query packet size, `t` = tune a channel). `hardware_files/Teensy/` contains bridge firmware for running one or two CoinFT boards through a single USB connection.
