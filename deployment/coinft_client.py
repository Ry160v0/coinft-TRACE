import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 3000)
socket.setsockopt(zmq.LINGER, 0)
socket.connect("tcp://127.0.0.1:5560")

try:
    while True:
        socket.send(b"get")
        print(socket.recv_json())
        time.sleep(0.1)
finally:
    socket.close()
    context.term()