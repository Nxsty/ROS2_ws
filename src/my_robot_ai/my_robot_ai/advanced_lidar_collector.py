import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import h5py
import os
import time
from math import atan2, asin

def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return atan2(t3, t4)

class AdvancedDataCollector(Node):
    def __init__(self):
        super().__init__('advanced_data_collector')
        
        # Setup HDF5
        os.makedirs('datasets', exist_ok=True)
        now = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f'datasets/expert_dataset_{now}.h5'
        self.h5_file = h5py.File(self.filename, 'w')
        self.dset_lidar = self.h5_file.create_dataset('lidar', (0, 360), maxshape=(None, 360), dtype='f4')
        self.dset_pose = self.h5_file.create_dataset('pose', (0, 3), maxshape=(None, 3), dtype='f4')
        self.dset_vel = self.h5_file.create_dataset('velocity', (0, 2), maxshape=(None, 2), dtype='f4')
        
        # ROS 2
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.current_scan = None
        self.current_pose = None
        self.count = 0
        
        # Timer para el control (10Hz)
        self.timer = self.create_timer(0.1, self.control_and_record)
        self.get_logger().info(f'Iniciando Colector Experto. Guardando en {self.filename}')

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0)
        if len(ranges) != 360: ranges = np.resize(ranges, (360,))
        self.current_scan = ranges

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        yaw = euler_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = [pos.x, pos.y, yaw]

    def control_and_record(self):
        if self.current_scan is None or self.current_pose is None: return

        # --- Lógica de Navegación Experta (Campos Potenciales) ---
        # 1. Fuerza de avance (siempre queremos ir hacia adelante)
        v = 0.4 
        w = 0.0
        
        # 2. Fuerza repulsiva de obstáculos
        # Dividimos el lidar en zonas: Izquierda, Centro, Derecha
        # Nota: En tu LIDAR (360), 180 es el centro (atrás), 0 es adelante (si el offset es 0)
        # Vamos a detectar obstáculos en los 180 grados frontales (90 a 270)
        front_left = self.current_scan[200:270]
        front_right = self.current_scan[90:160]
        center = self.current_scan[160:200]
        
        min_dist = 0.8
        if np.min(center) < min_dist:
            v = 0.1 # Frenar
            w = 0.8 if np.mean(front_left) > np.mean(front_right) else -0.8
        elif np.min(front_left) < min_dist:
            w = -0.6 # Girar a la derecha
        elif np.min(front_right) < min_dist:
            w = 0.6 # Girar a la izquierda
        
        # Aplicar ruido aleatorio ocasional para explorar más (Data Augmentation en vivo)
        if np.random.rand() > 0.95:
            w += np.random.uniform(-0.5, 0.5)

        # Publicar comando
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        # --- Grabar datos ---
        new_size = self.count + 1
        self.dset_lidar.resize(new_size, axis=0)
        self.dset_pose.resize(new_size, axis=0)
        self.dset_vel.resize(new_size, axis=0)
        
        self.dset_lidar[self.count] = self.current_scan
        self.dset_pose[self.count] = self.current_pose
        self.dset_vel[self.count] = [v, w]
        
        self.count += 1
        if self.count % 500 == 0:
            self.get_logger().info(f'Recolectadas {self.count} muestras expertas...')

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedDataCollector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.h5_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
