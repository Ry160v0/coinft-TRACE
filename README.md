# CoinFT-TRACE

Calibration, training, and visualization pipeline for **CoinFT** — a coin-sized capacitive 6-axis force/torque sensor.

This repo covers the PC-side software stack: reading raw sensor packets over serial, collecting synchronized ground-truth data against an ATI reference sensor, training a neural-network calibration model, and visualizing the result in real time.

---

## Overview

The pipeline has five stages:

1. **Firmware** — the CoinFT board streams 12 raw capacitive channels over USB serial.
2. **Data collection** — a script reads those raw channels and a reference ATI sensor at the same time, and saves each labeled session to an `.h5` file.
3. **Processing** — `data_processor.py` combines sessions into `train` / `val` / `test` sets and computes normalization stats.
4. **Training** — `coinft_MLP_train.py` fits an MLP that maps the 12 raw channels to calibrated `Fx, Fy, Fz, Mx, My, Mz`.
5. **Visualization** — a viewer runs the trained model live, so you can watch calibrated readings next to the ATI reference.

---

## Requirements

- Python 3, with `numpy`, `scipy`, `matplotlib`, `h5py`, `pandas`, `pyserial`, `pytorch`, `onnx`, `onnxruntime`
- `pysoem` — only needed for the EtherCAT (ATI Mini58) path

---

## The calibration workflow

Four steps, run in order. Each script is configured by editing the constants in its **CONTROL PANEL** section near the top of the file (COM port, ATI adapter, session ID, duration, ...) before you run it.

### Step 1 · Collect data

Connect the CoinFT over USB, mount it against your ATI reference sensor, and run whichever one matches your reference hardware:

```bash
python calibration/intergrated_ati_data_collection_GAMMA.py   # ATI Gamma / SI-130, over UDP/RDT (Ethernet)
python calibration/intergrated_ati_data_collection_MINI.py    # ATI Mini58, over EtherCAT
```

What happens during a run:

- Both sensors are **tared together** at the start, so any mounting preload becomes the shared zero.
- The session records for `SAMPLING_DURATION` seconds (**40s by default**).
- The result is saved to `data/` as `CFT24_calibrationData_<timestamp>_<ATI_model>_<ID>.h5`.

Set `ID` to `train`, `val`, or `test` before each run — `data_processor.py` reads that suffix to build the dataset splits.

#### Cover every axis, not just Fz

Plan on running this many times, not once. A single session of straight-down presses calibrates `Fz` nicely but leaves the model blind to shear and torque. Use the recipes below as a script for what to physically do to the sensor during each 40-second recording, and mix them across your `train` / `val` / `test` sessions:

| Recipe | What to do in ~40s | Axes excited |
|---|---|---|
| A — Pure normal | Slow press from zero to full scale and back, 3–4 cycles | Fz |
| B — Normal, varying rate | Fast press/release, plus a hold to watch creep | Fz (hysteresis / creep) |
| C — Pure shear X | Push back and forth along X | Fx (+ a little My) |
| D — Pure shear Y | Push back and forth along Y | Fy (+ a little Mx) |
| E — Off-center press | Press different points around the sensor's edge | Fz + Mx + My |
| F — Twist | Press down and rotate in place | Mz |
| G — Mixed scrub | Random combination of press + push + twist | All axes |
| H — Pull-up | Pull upward, if your application needs tension | +Fz (A/B don't cover this) |

The more of these you cover — and the more you vary force magnitude and speed — the better the trained model generalizes.

### Step 2 · Process the dataset

```bash
python calibration/data_processor.py
```

This scans `data/*_calibrationData_*.h5`, groups sessions by reference-sensor model so a Gamma campaign and a Mini58 campaign never get mixed, and splits everything into train/val/test. It writes:

- `data/train.h5`, `data/val.h5`, `data/test.h5`
- `hardware_configs/CFT24_norm.json` — per-channel normalization stats, used later at inference time

### Step 3 · Train the model

```bash
python calibration/coinft_MLP_train.py
```

Trains the calibration MLP on the processed dataset and saves:

- `hardware_configs/CFT24_MLP.pth` — PyTorch checkpoint
- `hardware_configs/CFT24_MLP.onnx` — ONNX export, used by every visualization script
- a results plot comparing predicted vs. reference force/torque on the test split

### Step 4 · Visualize and validate

```bash
python visualization/visualize_coinft_GAMMA.py     # live CoinFT vs. ATI Gamma/SI-130
python visualization/visualize_coinft_MINI.py       # live CoinFT vs. ATI Mini58 (EtherCAT)
python visualization/visualize_coinfts.py           # CoinFT only, no reference sensor (via Teensy bridge)
```

Each of these runs the exported ONNX model live and plots calibrated `Fx..Mz` in real time, next to the ATI reference where one is connected. Use them to sanity-check the calibration — and if a channel looks off, re-tune it with `calibration/coinft_tuner.py`.

---

## Good to know

- **`data/` is gitignored.** It holds raw and processed sessions, which are large and specific to your machine/setup — regenerate it locally with Step 1.
- **`hardware_configs/` and `hardware_files/` are checked in.** The trained model, tuning parameters, and PCB/firmware files are small and let you reproduce or deploy a calibration without re-collecting data.
- **Serial protocol:** the CoinFT streams over USB at 1,000,000 baud. Single-byte commands: `s` start streaming, `i` stop, `q` query packet size, `t` tune a channel. `hardware_files/Teensy/` holds bridge firmware for running one or two CoinFT boards through a single USB connection.
