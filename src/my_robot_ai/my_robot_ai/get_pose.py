import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import sys

class PoseSnapshot(Node):
    def __init__(self):
        super().__init__('pose_snapshot')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.callback,
            10)
        self.get_logger().info('Esperando posición... (Detén el robot en la meta)')

    def callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        
        print("\n" + "="*30)
        print(f"COORDENADAS DE LA META:")
        print(f"X: {p.x:.4f}")
        print(f"Y: {p.y:.4f}")
        print(f"Yaw: {yaw:.4f}")
        print("="*30)
        
        # Salir después de recibir la primera posición
        sys.exit(0)

def main():
    rclpy.init()
    node = PoseSnapshot()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
