#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class WorldOrientedTeleop(Node):
    def __init__(self):
        super().__init__('world_oriented_teleop')
        
        # Publicador de comandos de velocidad a mecanum_drive_controller
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Suscripción al nodo joy (mando de Xbox)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # Suscripción a la odometría para el modo orientado al mundo (World-Oriented)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Parámetros base de velocidad
        self.declare_parameter('max_linear_speed', 1.0)   # m/s
        self.declare_parameter('max_angular_speed', 2.0)  # rad/s
        self.declare_parameter('world_oriented_default', False) # False = Robot-Centric, True = World-Oriented
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.world_oriented = self.get_parameter('world_oriented_default').value
        
        self.current_yaw = 0.0
        self.last_toggle_btn = 0
        
        self.get_logger().info('🎮 Teleoperación Mecanum (Xbox / Joystick) Inicializada')
        self.get_logger().info('   🕹️ Stick Izquierdo: Adelante / Atras (X) + Desplazamiento Lateral Strafe (Y)')
        self.get_logger().info('   🕹️ Stick Derecho: Giro Yaw (Z)')
        self.get_logger().info('   🔘 Boton Y: Alternar Modo (Robot-Centric vs World-Oriented)')
        self.get_logger().info('   🔘 Boton A: Freno de emergencia')
        self.get_logger().info('   🔘 RB: Modo Turbo | LB: Modo Precision')
        self.get_logger().info(f'   👉 Modo Actual: {"WORLD-ORIENTED (Orientado al Mapa)" if self.world_oriented else "ROBOT-CENTRIC (Estilo Vehiculo)"}')

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def joy_callback(self, msg: Joy):
        twist = Twist()
        
        # 1. Alternar modo con botón Y (botón 3 en mando de Xbox)
        toggle_btn = msg.buttons[3] if len(msg.buttons) > 3 else 0
        if toggle_btn == 1 and self.last_toggle_btn == 0:
            self.world_oriented = not self.world_oriented
            self.get_logger().info(f'🔄 Modo cambiado a: {"WORLD-ORIENTED (Orientado al Mapa)" if self.world_oriented else "ROBOT-CENTRIC (Estilo Vehiculo)"}')
        self.last_toggle_btn = toggle_btn

        # 2. Botón de seguridad / freno (Botón A en Xbox: botón 0)
        if len(msg.buttons) > 0 and msg.buttons[0] == 1:
            self.publisher_.publish(twist)
            return

        # 3. Modificadores de velocidad (LB: Precisión 0.5x, RB: Turbo 1.4x)
        speed_mult = 1.0
        if len(msg.buttons) > 5 and msg.buttons[5] == 1: # RB
            speed_mult = 1.4
        elif len(msg.buttons) > 4 and msg.buttons[4] == 1: # LB
            speed_mult = 0.5

        # 4. Lectura de sticks analógicos
        # Stick Izquierdo:
        # axes[1]: Vertical (+1 arriba, -1 abajo) -> Forward / Backward
        # axes[0]: Horizontal (+1 izquierda, -1 derecha) -> Strafe Left / Right
        raw_vx = msg.axes[1] if len(msg.axes) > 1 else 0.0
        raw_vy = msg.axes[0] if len(msg.axes) > 0 else 0.0
        
        # Stick Derecho:
        # axes[3]: Horizontal (+1 izquierda, -1 derecha) -> Yaw counter-clockwise
        raw_wz = msg.axes[3] if len(msg.axes) > 3 else 0.0

        # Zona muerta para evitar derivas por sticks desgastados
        deadzone = 0.08
        if abs(raw_vx) < deadzone: raw_vx = 0.0
        if abs(raw_vy) < deadzone: raw_vy = 0.0
        if abs(raw_wz) < deadzone: raw_wz = 0.0

        vx_cmd = raw_vx * self.max_linear_speed * speed_mult
        vy_cmd = raw_vy * self.max_linear_speed * speed_mult
        wz_cmd = raw_wz * self.max_angular_speed * speed_mult

        # 5. Transformación World-Oriented (si está activa)
        # Transforma los comandos del marco global inercial al marco local del robot
        if self.world_oriented:
            # v_local = R(yaw)^T * v_world
            cos_y = math.cos(self.current_yaw)
            sin_y = math.sin(self.current_yaw)
            local_vx = cos_y * vx_cmd + sin_y * vy_cmd
            local_vy = -sin_y * vx_cmd + cos_y * vy_cmd
            twist.linear.x = local_vx
            twist.linear.y = local_vy
        else:
            # Modo Robot-Centric estándar
            twist.linear.x = vx_cmd
            twist.linear.y = vy_cmd

        twist.angular.z = wz_cmd
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WorldOrientedTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
