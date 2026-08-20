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

class GoalDataCollector(Node):
    def __init__(self):
        super().__init__('goal_data_collector')
        
        os.makedirs('datasets', exist_ok=True)
        now = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f'datasets/goal_dataset_{now}.h5'
        self.h5_file = h5py.File(self.filename, 'w')
        
        # Datasets (Añadimos 'goal' como entrada)
        self.dset_lidar = self.h5_file.create_dataset('lidar', (0, 360), maxshape=(None, 360), dtype='f4')
        self.dset_goal = self.h5_file.create_dataset('goal', (0, 2), maxshape=(None, 2), dtype='f4') # [rel_dist, rel_angle]
        self.dset_vel = self.h5_file.create_dataset('velocity', (0, 2), maxshape=(None, 2), dtype='f4')
        self.dset_pose = self.h5_file.create_dataset('pose', (0, 3), maxshape=(None, 3), dtype='f4')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.current_scan = None
        self.current_pose = None
        self.target_goal = self.get_random_goal()
        self.count = 0
        
        self.timer = self.create_timer(0.1, self.control_and_record)
        self.get_logger().info(f'Iniciando Colector con Metas. Guardando en {self.filename}')

    def get_random_goal(self):
        # Generar una meta aleatoria en un rango más amplio (cubriendo el mapa)
        gx = np.random.uniform(-2.0, 10.0)
        gy = np.random.uniform(-5.0, 5.0)
        self.get_logger().info(f'Nueva Meta: x={gx:.2f}, y={gy:.2f}')
        return [gx, gy]

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0)
        if len(ranges) != 360: ranges = np.resize(ranges, (360,))
        self.current_scan = ranges

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        # Yaw
        q = msg.pose.pose.orientation
        yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = [pos.x, pos.y, yaw]

    def control_and_record(self):
        if self.current_scan is None or self.current_pose is None: return

        # 1. Calcular vector a la meta
        dx = self.target_goal[0] - self.current_pose[0]
        dy = self.target_goal[1] - self.current_pose[1]
        dist = sqrt(dx**2 + dy**2)
        angle_to_goal = atan2(dy, dx) - self.current_pose[2]
        angle_to_goal = atan2(np.sin(angle_to_goal), np.cos(angle_to_goal)) # Normalizar

        # Si llegamos a la meta, elegir otra
        if dist < 0.5:
            self.target_goal = self.get_random_goal()
            return

        # 2. Navegación: Ir a meta + Esquivar obstáculos
        v = 0.4
        w = angle_to_goal * 1.5 # Fuerza atractiva
        
        # Fuerza repulsiva (Lidar)
        front = self.current_scan[160:200]
        if np.min(front) < 1.0:
            v = 0.1
            w += 1.2 if np.mean(self.current_scan[200:270]) > np.mean(self.current_scan[90:160]) else -1.2

        # Publicar comando
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        # 3. Grabar (Lo más importante: guardamos el goal relativo)
        new_size = self.count + 1
        self.dset_lidar.resize(new_size, axis=0)
        self.dset_goal.resize(new_size, axis=0)
        self.dset_vel.resize(new_size, axis=0)
        self.dset_pose.resize(new_size, axis=0)
        
        self.dset_lidar[self.count] = self.current_scan
        self.dset_goal[self.count] = [dist, angle_to_goal] # Esto enseñará al robot a "donde ir"
        self.dset_vel[self.count] = [v, w]
        self.dset_pose[self.count] = self.current_pose
        
        self.count += 1
        if self.count % 500 == 0:
            self.get_logger().info(f'Muestras grabadas: {self.count}. Distancia meta: {dist:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = GoalDataCollector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.h5_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
