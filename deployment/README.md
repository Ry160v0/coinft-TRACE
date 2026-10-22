## Dependencies

On G1 (server):

```bash
python3 -m pip install pyserial pyzmq
```

On PC (recorder/client):

```bash
python -m pip install pyzmq
```

## Launch

Run these commands from the project root on each host. Set `ROBOT_IP` in
`coinft_record_testVer.py` to G1's wired-interface IP (default: `192.168.123.164`).

Start the recorder on PC first:

```bash
python deployment/coinft_record_testVer.py --port 5560
```

Then start the server on G1:

```bash
python3 deployment/coinft_server_testVer.py --port 5560 --usb /dev/ttyACM0
```



## Timestamp fields

All timestamps are stored as integer nanoseconds. System timestamps use the Unix epoch.

| Field | Host | Event and purpose |
| --- | --- | --- |
| `timestamp_ns` | G1 | System time recorded when a complete USB frame has been received, before decoding. Used for cross-device time correlation. |
| `source_monotonic_ns` | G1 | Monotonic time recorded immediately after `timestamp_ns`. Used to measure frame intervals on G1. |
| `received_timestamp_ns` | PC | System time recorded immediately after a complete ZMQ message is received, before JSON decoding. Used for the PC reception timeline. |
| `received_monotonic_ns` | PC | Monotonic time recorded immediately after `received_timestamp_ns`. Used to measure reception intervals on the PC. |

