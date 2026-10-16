Install server dependencies:

```bash
pip install numpy onnxruntime pyserial pyzmq
```



timestamp_ns -- time after the serial port reads a complete frame, before decoding and model inference

received_timestamp_ns -- after client receives the message before writes into the csv file. 