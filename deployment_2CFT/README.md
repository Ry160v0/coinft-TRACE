## Dependencies

On G1 (server & data processor):

```bash
python3 -m pip install pyserial pyzmq numpy onnxruntime
```

On PC:
```bash
python -m pip install pyzmq numpy onnxruntime
```


## Launch Recorder


Start the recorder on PC first:

```bash
python deployment_2CFT/coinft_record_2CFT.py --port 5560
```

Then start the server on G1:

```bash
python3 deployment_2CFT/coinft_server_2CFT.py --port 5560 --usb /dev/ttyACM0
```

Optional: Adjust the tare frames
```bash
python3 deployment_2CFT/coinft_server_2CFT.py --tare-frames 500 --tare-skip 10
```


After finish recording, use Ctrl+C to pause, and the csv file is saved in `raw_data`.  

Launch converter to convert count data to force data: 

```bash
python deployment_2CFT/coinft_offline_convert.py <file_name>
```

## Launch Data Collection with run_data_exporter.py

Launch the server:

```bash
python3 deployment_2CFT/coinft_server_2CFT.py --port 5560 --usb /dev/ttyACM0
```

Then launch data exporter: 
```bash
python gear_sonic/scripts/run_data_exporter.py \
    --camera-host 192.168.123.164 \
    --task-prompt "Pick bottle and turn and pour into cup." \
    --data-collection-frequency 30 \
    --record-coinft
```

## Timestamp fields

All timestamps are stored as integer nanoseconds. System timestamps use the Unix epoch.

| Field | Host | Event and purpose |
| --- | --- | --- |
| `timestamp_ns` | G1 | System time recorded when a complete USB frame has been received, before decoding. Used for cross-device time correlation. |
| `source_monotonic_ns` | G1 | Monotonic time recorded immediately after `timestamp_ns`. Used to measure frame intervals on G1. |
| `received_timestamp_ns` | PC | System time recorded immediately after a complete ZMQ message is received, before JSON decoding. Used for the PC reception timeline. |
| `received_monotonic_ns` | PC | Monotonic time recorded immediately after `received_timestamp_ns`. Used to measure reception intervals on the PC. |

