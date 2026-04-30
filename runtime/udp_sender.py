import socket
import json


class UdpSender:
    def __init__(self, host="172.28.182.183", port=5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.sock.sendto(payload, self.addr)
        print(f"SEND UDP -> {self.addr}: {data}")