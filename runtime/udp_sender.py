import socket
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class UdpSender:
    """
    UDP-р JSON өгөгдөл илгээнэ.

    Socket алдаа гарвал дахин холбогдохыг оролдоно.
    """

    def __init__(self, host: str = "172.28.182.183", port: int = 5005):
        self.addr = (host, port)
        self._sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        """Socket үүсгэнэ."""
        try:
            if self._sock:
                self._sock.close()
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(1.0)
            logger.info(f"UDP socket ready -> {self.addr}")
        except OSError as e:
            logger.error(f"UDP socket үүсгэхэд алдаа: {e}")
            self._sock = None

    def send(self, data: Dict[str, Any]) -> bool:
        """
        Өгөгдлийг JSON болгон илгээнэ.
        Амжилттай бол True, алдаа гарвал False буцаана.
        """
        if self._sock is None:
            logger.warning("Socket алга, дахин холбогдож байна...")
            self._connect()
            if self._sock is None:
                return False

        try:
            payload = json.dumps(data).encode("utf-8")
            self._sock.sendto(payload, self.addr)
            logger.debug(f"UDP -> {self.addr}: {data}")
            return True
        except OSError as e:
            logger.error(f"UDP илгээлт амжилтгүй: {e}, дахин холбогдож байна...")
            self._connect()
            return False

    def close(self) -> None:
        """Socket-г хаана."""
        if self._sock:
            self._sock.close()
            self._sock = None