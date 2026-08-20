import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class WhereAmI(Node):
    def __init__(self):
        super().__init__('where_am_i')
        
        # Usamos la configuración por defecto (Reliable)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            10)
            
        self.get_logger().info('=== RASTREADOR V3 ===')
        self.get_logger().info('Escuchando en /odom...')

    def listener_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        
        # Usamos rclpy logger para asegurar que sale en la consola de ros2 run
        self.get_logger().info(f'POS -> X: {p.x:.2f} | Y: {p.y:.2f} | Yaw: {yaw:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = WhereAmI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
