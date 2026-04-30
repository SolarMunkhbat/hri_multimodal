import socket
import json
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class UDPNode(Node):
    def __init__(self):
        super().__init__("udp_node")

        self.pub = self.create_publisher(Twist, "/model/vehicle_blue/cmd_vel", 10)

        self.host = "0.0.0.0"
        self.port = 5005
        self.last_packet_time = time.time()
        self.timeout_sec = 1.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info(f"UDP listener started on {self.host}:{self.port}")

    def publish_twist(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published -> linear_x={msg.linear.x:.2f}, angular_z={msg.angular.z:.2f}"
        )

    def stop_robot(self):
        self.publish_twist(0.0, 0.0)
        self.get_logger().warn("No UDP packet received recently. Robot stopped for safety.")

    def loop(self):
        got_packet = False

        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                got_packet = True
                self.last_packet_time = time.time()

                try:
                    decoded = data.decode("utf-8")
                    d = json.loads(decoded)

                    linear_x = d.get("linear_x", 0.0)
                    angular_z = d.get("angular_z", 0.0)

                    self.get_logger().info(
                        f"Received from {addr[0]}:{addr[1]} -> {decoded}"
                    )

                    self.publish_twist(linear_x, angular_z)

                except json.JSONDecodeError:
                    self.get_logger().error("Invalid JSON packet received.")
                except Exception as e:
                    self.get_logger().error(f"Packet processing error: {e}")

            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().error(f"UDP socket error: {e}")
                break

        if not got_packet:
            if (time.time() - self.last_packet_time) > self.timeout_sec:
                self.stop_robot()
                self.last_packet_time = time.time()


def main():
    rclpy.init()
    node = UDPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down UDP listener...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()