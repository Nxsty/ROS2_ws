#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class CarStyleTeleop(Node):
    def __init__(self):
        super().__init__('car_style_teleop')
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # Parámetros de velocidad
        self.max_linear_speed = 0.8  # m/s
        self.max_angular_speed = 3.0 # rad/s
        
        self.get_logger().info('Control Estilo RC (Split-Stick) Activado')
        self.get_logger().info('Joy Izquierdo: Adelante/Atras | Joy Derecho: Direccion (Giro)')

    def joy_callback(self, msg):
        twist = Twist()
        
        # En la mayoría de los controles (Xbox/PS4):
        # Joystick Izquierdo Vertical es ejes[1]
        # Joystick Derecho Horizontal es ejes[3] (o ejes[2] en algunos drivers)
        
        # Acelerador (Joy Izquierdo)
        linear_val = msg.axes[1]
        
        # Dirección (Joy Derecho)
        # Nota: Usamos el eje 3 que suele ser el horizontal del stick derecho
        angular_val = msg.axes[3]
        
        # Aplicamos las velocidades
        twist.linear.x = linear_val * self.max_linear_speed
        twist.angular.z = angular_val * self.max_angular_speed
        
        # Botón de seguridad (Opcional): Botón A o X para frenado en seco
        if msg.buttons[0]:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = CarStyleTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
