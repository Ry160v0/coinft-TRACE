#!/usr/bin/env python3
"""
data_processor_ver2.py

Merges the two earlier processors:

  * data_processor_origin.py reads EVERY calibration file and lets the
    train/val/test split follow the ID suffix written at collection time.
  * data_processor.py adds the [raw, raw**2] quadratic features but only ever
    used the single newest file.

This version keeps both: all matching files are loaded, the split follows the
filename suffix when it is available, and the quadratic features are preserved.
An ATI_MODEL_TAG filter selects one reference-sensor campaign without having to
move files around.

Expected filename layout (written by the intergrated_ati_data_collection_*
scripts):

    {SENSOR_NAME}_calibrationData_{YYYYmmdd}_{HHMMSS}_{ATI_MODEL_TAG}_{ID}.h5

ATI_MODEL_TAG may itself contain underscores (e.g. Mini58_ECATBA); the last
token before ".h5" is always the ID.
"""

import glob
import json
import os
import re

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# ====================== CONFIGURATION ======================
SENSOR_NAME = 'CFT24'

# Reference sensor campaign to use. None uses every tag found.
# The tags present in DATA_DIR are printed at startup, e.g.:
#   'SI13010'        ATI Gamma SI-130-10 over UDP/RDT
#   'Mini58_ECATBA'  ATI Mini58 over EtherCAT
ATI_MODEL_TAG = 'Mini58_ECATBA'

# How the train/val/test split is decided:
#   'auto'     use the _train/_val/_test suffixes when a train bucket exists,
#              otherwise fall back to assigning whole sessions
#   'filename' original behavior: the ID suffix alone decides the split
#   'session'  whole files go to train/val/test (no intra-session leakage)
#   'row'      pool every row and shuffle (leaks correlated neighbors; last resort)
SPLIT_MODE = 'auto'

# [raw, raw**2] features. False reproduces the 12-dimensional original.
USE_QUADRATIC_FEATURES = True

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
SAVE_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
SAVE_JSON_DIR = os.path.join(SCRIPT_DIR, '..', 'hardware_configs')
JSON_FILENAME = f'{SENSOR_NAME}_norm.json'
MANIFEST_FILENAME = f'{SENSOR_NAME}_dataset_manifest.json'
# ============================================================

SPLIT_NAMES = ('train', 'val', 'test')

FILENAME_PATTERN = re.compile(
    r'^(?P<sensor>.+?)_calibrationData_'
    r'(?P<date>\d{8})_(?P<time>\d{6})_'
    r'(?P<rest>.+)\.h5$'
)


def parse_filename(filename):
    """
    Split a calibration filename into sensor / timestamp / tag / id.

    Returns None when the name does not follow the collection-script layout.
    """
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None

    tokens = match.group('rest').split('_')
    session_id = tokens[-1]
    tag = '_'.join(tokens[:-1]) if len(tokens) > 1 else None

    return {
        'sensor': match.group('sensor'),
        'timestamp': f"{match.group('date')}_{match.group('time')}",
        'tag': tag,
        'id': session_id,
        'split': session_id if session_id in SPLIT_NAMES else None,
    }


def discover_sessions():
    """Load every calibration file that passes the ATI_MODEL_TAG filter."""
    search_path = os.path.join(DATA_DIR, f'*{SENSOR_NAME}_calibrationData_*.h5')
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f'ERROR: no calibration file matched {search_path}')
        return []

    tag_counts = {}
    sessions = []
    skipped = []

    for file_path in file_list:
        filename = os.path.basename(file_path)
        info = parse_filename(filename)

        if info is None:
            skipped.append((filename, 'filename layout not recognized'))
            continue

        tag_counts[info['tag']] = tag_counts.get(info['tag'], 0) + 1

        if ATI_MODEL_TAG is not None and info['tag'] != ATI_MODEL_TAG:
            continue

        with h5py.File(file_path, 'r') as h5_file:
            if 'sensor_cal_data' not in h5_file or 'ati_cal_FT' not in h5_file:
                skipped.append((filename, 'missing sensor_cal_data / ati_cal_FT'))
                continue

            x_raw = h5_file['sensor_cal_data'][:]
            y_raw = h5_file['ati_cal_FT'][:]

        if x_raw.shape[0] != y_raw.shape[0]:
            skipped.append((filename, f'row mismatch {x_raw.shape} vs {y_raw.shape}'))
            continue

        info['name'] = filename
        info['X'] = np.asarray(x_raw, dtype=np.float64)
        info['Y'] = np.asarray(y_raw, dtype=np.float64)
        sessions.append(info)

    print(f'Found {len(file_list)} calibration files in {DATA_DIR}')
    print('Tags present:')
    for tag, count in sorted(tag_counts.items(), key=lambda item: str(item[0])):
        marker = '  <- selected' if (
            ATI_MODEL_TAG is None or tag == ATI_MODEL_TAG
        ) else ''
        print(f"    {str(tag):<20} {count} file(s){marker}")

    for filename, reason in skipped:
        print(f'  Skipping {filename}: {reason}')

    if ATI_MODEL_TAG is None and len(tag_counts) > 1:
        print(
            'WARNING: ATI_MODEL_TAG is None and several reference sensors are '
            'present. Merging campaigns from different reference sensors is '
            'usually not what you want.'
        )

    return sessions


def check_channel_consistency(sessions):
    """Reject a mixed set of files before they are stacked."""
    x_widths = {session['X'].shape[1] for session in sessions}
    y_widths = {session['Y'].shape[1] for session in sessions}

    if len(x_widths) != 1 or len(y_widths) != 1:
        raise ValueError(
            'Selected files do not share one channel layout: '
            f'X widths={sorted(x_widths)}, Y widths={sorted(y_widths)}. '
            'Set ATI_MODEL_TAG to a single campaign.'
        )

    return x_widths.pop(), y_widths.pop()


def split_by_filename(sessions):
    """Original behavior: the _train/_val/_test suffix decides the split."""
    buckets = {name: [] for name in SPLIT_NAMES}

    for session in sessions:
        if session['split'] is None:
            print(
                f"  Skipping {session['name']}: unknown ID suffix "
                f"'{session['id']}'"
            )
            continue
        buckets[session['split']].append(session)

    return buckets


def split_by_session(sessions):
    """Assign whole files, so no calibration run spans two splits."""
    if len(sessions) < 3:
        return None

    order = np.random.default_rng(RANDOM_SEED).permutation(len(sessions))
    shuffled = [sessions[index] for index in order]
    count = len(shuffled)

    train_count = max(1, min(int(round(count * TRAIN_RATIO)), count - 2))
    val_count = max(1, min(int(round(count * VAL_RATIO)), count - train_count - 1))

    return {
        'train': shuffled[:train_count],
        'val': shuffled[train_count:train_count + val_count],
        'test': shuffled[train_count + val_count:],
    }


def split_by_row(sessions):
    """Pool every row and shuffle. Kept only as an explicit last resort."""
    x_all = np.vstack([session['X'] for session in sessions])
    y_all = np.vstack([session['Y'] for session in sessions])

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, shuffle=True
    )
    val_share = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, train_size=val_share, random_state=RANDOM_SEED, shuffle=True
    )

    def as_session(name, x_data, y_data):
        return [{
            'name': f'<pooled {name}>',
            'tag': ATI_MODEL_TAG,
            'id': name,
            'timestamp': '',
            'X': x_data,
            'Y': y_data,
        }]

    return {
        'train': as_session('train', x_train, y_train),
        'val': as_session('val', x_val, y_val),
        'test': as_session('test', x_test, y_test),
    }


def choose_split(sessions):
    """Apply SPLIT_MODE and return (buckets, mode_actually_used)."""
    if SPLIT_MODE not in ('auto', 'filename', 'session', 'row'):
        raise ValueError(f'Unknown SPLIT_MODE: {SPLIT_MODE}')

    if SPLIT_MODE in ('auto', 'filename'):
        buckets = split_by_filename(sessions)
        if buckets['train']:
            return buckets, 'filename'
        if SPLIT_MODE == 'filename':
            raise RuntimeError(
                'No _train.h5 file found. Set the ID variable in the collection '
                'script per session, or use SPLIT_MODE = "session".'
            )
        print(
            'No _train.h5 file found; falling back to a session-level split '
            'so no calibration run spans two splits.'
        )

    if SPLIT_MODE in ('auto', 'session'):
        buckets = split_by_session(sessions)
        if buckets is not None:
            return buckets, 'session'
        if SPLIT_MODE == 'session':
            raise RuntimeError(
                f'Only {len(sessions)} session(s) available; a session-level '
                'split needs at least 3.'
            )
        print(
            f'WARNING: only {len(sessions)} session(s) available; falling back '
            'to a row-level shuffle. Neighboring samples of one calibration run '
            'will land in different splits, so the test score will be optimistic.'
        )

    return split_by_row(sessions), 'row'


def stack_bucket(bucket):
    """Concatenate one split, or return (None, None) when it is empty."""
    if not bucket:
        return None, None
    return (
        np.vstack([session['X'] for session in bucket]),
        np.vstack([session['Y'] for session in bucket]),
    )


def add_quadratic_features(x_data):
    """[raw, raw**2] feature expansion."""
    return np.hstack([x_data, x_data ** 2])


def save_h5(filename, x_data, y_data, stats):
    full_path = os.path.join(SAVE_DATA_DIR, filename)
    with h5py.File(full_path, 'w') as h5_file:
        h5_file.create_dataset('data', data=x_data, compression='gzip')
        h5_file.create_dataset('label', data=y_data, compression='gzip')
        # Retained from data_processor_origin.py for offline convenience.
        h5_file.create_dataset('mean_X', data=stats['mean_X'])
        h5_file.create_dataset('std_X', data=stats['std_X'])
        h5_file.create_dataset('mean_Y', data=stats['mean_Y'])
        h5_file.create_dataset('std_Y', data=stats['std_Y'])
    print(f'  Saved: {filename}  X={x_data.shape}  Y={y_data.shape}')


def main():
    print('=' * 68)
    print('CoinFT Data Processor v2 (all files + session split + quadratic)')
    print('=' * 68)

    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_JSON_DIR, exist_ok=True)

    sessions = discover_sessions()
    if not sessions:
        print(
            'ERROR: no usable calibration file after filtering. '
            f'ATI_MODEL_TAG = {ATI_MODEL_TAG!r}'
        )
        return

    x_width, y_width = check_channel_consistency(sessions)
    total_rows = sum(session['X'].shape[0] for session in sessions)
    print(
        f'\nSelected {len(sessions)} session(s), {total_rows} rows, '
        f'{x_width} raw channels -> {y_width} F/T axes'
    )

    buckets, used_mode = choose_split(sessions)
    print(f"\nSplit mode used: '{used_mode}'")

    for name in SPLIT_NAMES:
        print(f'  {name}:')
        for session in buckets[name]:
            print(f"    {session['name']}  ({session['X'].shape[0]} rows)")
        if not buckets[name]:
            print('    (empty)')

    x_train, y_train = stack_bucket(buckets['train'])
    x_val, y_val = stack_bucket(buckets['val'])
    x_test, y_test = stack_bucket(buckets['test'])

    missing = [
        name for name, data in
        (('train', x_train), ('val', x_val), ('test', x_test))
        if data is None
    ]
    if missing:
        print(
            f"\nERROR: split(s) {missing} are empty. coinft_MLP_train.py needs "
            'train.h5, val.h5 and test.h5. Add more sessions or change SPLIT_MODE.'
        )
        return

    if USE_QUADRATIC_FEATURES:
        x_train = add_quadratic_features(x_train)
        x_val = add_quadratic_features(x_val)
        x_test = add_quadratic_features(x_test)
        print(f'\nQuadratic features on: {x_width} -> {x_train.shape[1]} dimensions')
    else:
        print(f'\nQuadratic features off: {x_train.shape[1]} dimensions')

    # Normalization is computed from the training split only.
    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0)
    std_x[std_x == 0] = 1.0
    mean_y = np.mean(y_train, axis=0)
    std_y = np.std(y_train, axis=0)
    std_y[std_y == 0] = 1.0

    json_path = os.path.join(SAVE_JSON_DIR, JSON_FILENAME)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(
            {
                'mu_x': mean_x.tolist(),
                'sd_x': std_x.tolist(),
                'mu_y': mean_y.tolist(),
                'sd_y': std_y.tolist(),
            },
            json_file,
            indent=2,
        )
    print(f'\nSaved {mean_x.size}-dimensional norm constants to: {json_path}')

    stats = {
        'mean_X': mean_x,
        'std_X': std_x,
        'mean_Y': mean_y,
        'std_Y': std_y,
    }

    x_train = (x_train - mean_x) / std_x
    y_train = (y_train - mean_y) / std_y
    x_val = (x_val - mean_x) / std_x
    y_val = (y_val - mean_y) / std_y
    x_test = (x_test - mean_x) / std_x
    y_test = (y_test - mean_y) / std_y

    save_h5('train.h5', x_train, y_train, stats)
    save_h5('val.h5', x_val, y_val, stats)
    save_h5('test.h5', x_test, y_test, stats)

    # Which sessions produced this norm.json / model.
    manifest_path = os.path.join(SAVE_DATA_DIR, MANIFEST_FILENAME)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(
            {
                'sensor_name': SENSOR_NAME,
                'ati_model_tag': ATI_MODEL_TAG,
                'split_mode_requested': SPLIT_MODE,
                'split_mode_used': used_mode,
                'use_quadratic_features': USE_QUADRATIC_FEATURES,
                'input_dim': int(x_train.shape[1]),
                'random_seed': RANDOM_SEED,
                'splits': {
                    name: [session['name'] for session in buckets[name]]
                    for name in SPLIT_NAMES
                },
                'rows': {
                    'train': int(x_train.shape[0]),
                    'val': int(x_val.shape[0]),
                    'test': int(x_test.shape[0]),
                },
            },
            manifest_file,
            indent=2,
        )
    print(f'Saved dataset manifest to: {manifest_path}')

    print('\nDone.')
    print('=' * 68)


if __name__ == '__main__':
    main()
