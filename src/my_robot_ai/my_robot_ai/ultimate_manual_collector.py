import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import h5py
import os
import time
from math import atan2, sqrt

class UltimateManualCollector(Node):
    def __init__(self):
        super().__init__('ultimate_manual_collector')
        
        # 1. Configuración de Archivo HDF5 (Optimizado para Imitation Learning)
        os.makedirs('datasets', exist_ok=True)
        now = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f'datasets/trajectory_dataset_{now}.h5'
        self.h5_file = h5py.File(self.filename, 'w')
        
        # Datasets simplificados: Solo LIDAR y Comandos (Velocidad)
        # El objetivo es aprender qué velocidad aplicar dado un escaneo LIDAR
        self.dset_lidar = self.h5_file.create_dataset('lidar', (0, 360), maxshape=(None, 360), dtype='f4', compression="gzip")
        self.dset_vel = self.h5_file.create_dataset('velocity', (0, 2), maxshape=(None, 2), dtype='f4') # [v_cmd, w_cmd]
        self.dset_pose = self.h5_file.create_dataset('pose', (0, 3), maxshape=(None, 3), dtype='f4')    # Para referencia/análisis

        # 2. Suscripciones
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Estado Interno
        self.current_scan = None
        self.current_pose = None
        self.current_cmd = [0.0, 0.0]
        self.count = 0
        
        # 3. Loop de Grabación Constante (10Hz)
        self.record_timer = self.create_timer(0.1, self.record_step)
        self.print_timer = self.create_timer(1.0, self.print_status)
        
        self.get_logger().info(f'--- COLECTOR DE TRAYECTORIAS INICIADO ---')
        self.get_logger().info(f'Modo: Aprendizaje por Imitación Puro (Sin Meta)')
        self.get_logger().info(f'Maneja el robot por el circuito para grabar la trazada...')

    def print_status(self):
        if self.current_pose:
            x, y, _ = self.current_pose
            self.get_logger().info(f'Pos: ({x:.2f}, {y:.2f}) | Muestras grabadas: {self.count}')

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0)
        if len(ranges) != 360: ranges = np.resize(ranges, (360,))
        self.current_scan = ranges / 12.0 # Normalizado 0-1

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = [pos.x, pos.y, yaw]

    def cmd_callback(self, msg):
        self.current_cmd = [msg.linear.x, msg.angular.z]

    def record_step(self):
        if self.current_scan is None or self.current_pose is None: return
        
        # Filtro de movimiento: Solo grabar si realmente te estás moviendo
        if abs(self.current_cmd[0]) < 0.01 and abs(self.current_cmd[1]) < 0.01:
            return

        # Guardar datos
        new_size = self.count + 1
        self.dset_lidar.resize(new_size, axis=0)
        self.dset_vel.resize(new_size, axis=0)
        self.dset_pose.resize(new_size, axis=0)
        
        self.dset_lidar[self.count] = self.current_scan
        self.dset_vel[self.count] = self.current_cmd
        self.dset_pose[self.count] = self.current_pose
        
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = UltimateManualCollector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.get_logger().info(f'Grabación finalizada. Total muestras: {node.count}')
    node.h5_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
