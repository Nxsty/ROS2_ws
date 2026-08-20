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

class GroundTruthManualCollector(Node):
    def __init__(self):
        super().__init__('ground_truth_manual_collector')
        
        # Meta final real
        self.target_x = 11.35
        self.target_y = 7.28
        
        # Setup HDF5
        os.makedirs('datasets', exist_ok=True)
        now = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f'datasets/advanced_manual_{now}.h5'
        self.h5_file = h5py.File(self.filename, 'w')
        
        self.dset_lidar = self.h5_file.create_dataset('lidar', (0, 360), maxshape=(None, 360), dtype='f4')
        self.dset_goal = self.h5_file.create_dataset('goal', (0, 2), maxshape=(None, 2), dtype='f4')
        self.dset_cmd = self.h5_file.create_dataset('cmd', (0, 2), maxshape=(None, 2), dtype='f4')
        self.dset_pose = self.h5_file.create_dataset('pose', (0, 3), maxshape=(None, 3), dtype='f4')

        # Suscripciones
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # IMPORTANTE: Suscribirse a la odometría de GAZEBO (Ground Truth)
        self.odom_sub = self.create_subscription(Odometry, '/model/my_robot/odometry', self.odom_callback, 10)
        
        self.current_scan = None
        self.current_pose = None
        self.current_cmd = [0.0, 0.0]
        self.count = 0
        
        self.print_timer = self.create_timer(1.0, self.print_status)
        self.get_logger().info(f'Colector GROUND TRUTH iniciado. Meta: ({self.target_x}, {self.target_y})')

    def print_status(self):
        if self.current_pose:
            x, y, yaw = self.current_pose
            dx, dy = self.target_x - x, self.target_y - y
            dist = sqrt(dx**2 + dy**2)
            self.get_logger().info(f'REAL POS: X={x:.2f}, Y={y:.2f} | Error Meta: {dist:.2f}m')

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = [pos.x, pos.y, yaw]

    def cmd_callback(self, msg):
        self.current_cmd = [msg.linear.x, msg.angular.z]

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0)
        if len(ranges) != 360: ranges = np.resize(ranges, (360,))
        self.current_scan = ranges
        self.record_step()

    def record_step(self):
        if self.current_scan is None or self.current_pose is None: return
        if abs(self.current_cmd[0]) < 0.01 and abs(self.current_cmd[1]) < 0.01: return

        dx, dy = self.target_x - self.current_pose[0], self.target_y - self.current_pose[1]
        dist = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx) - self.current_pose[2]
        angle = atan2(np.sin(angle), np.cos(angle))

        new_size = self.count + 1
        for dset, data in zip([self.dset_lidar, self.dset_goal, self.dset_cmd, self.dset_pose], 
                              [self.current_scan, [dist, angle], self.current_cmd, self.current_pose]):
            dset.resize(new_size, axis=0)
            dset[self.count] = data
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthManualCollector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.h5_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
