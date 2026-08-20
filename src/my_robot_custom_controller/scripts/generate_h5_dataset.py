#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import os
import h5py
import time
from math import atan2, asin

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

class H5DatasetGenerator(Node):
    def __init__(self):
        super().__init__('h5_dataset_generator')
        
        # Crear carpeta datasets si no existe
        self.dataset_dir = 'datasets'
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            self.get_logger().info(f'Creada carpeta: {self.dataset_dir}')

        # Get current time for default filename
        now = time.strftime("%Y%m%d_%H%M%S")
        default_filename = os.path.join(self.dataset_dir, f'robot_dataset_{now}.h5')
        
        self.declare_parameter('output_file', default_filename)
        self.output_file = self.get_parameter('output_file').get_parameter_value().string_value
        
        self.get_logger().info(f'Generando dataset HDF5 en: {self.output_file}')
        
        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # Current State
        self.current_scan = None
        self.current_pose = None # [x, y, yaw]
        self.current_vel = None  # [v, w]
        self.current_cmd = [0.0, 0.0]
        
        # HDF5 Setup (Mode 'w' to create new file with timestamp)
        self.h5_file = h5py.File(self.output_file, 'w')
        
        # Create resizable datasets
        self.dset_lidar = self.h5_file.create_dataset('lidar', (0, 360), maxshape=(None, 360), dtype='f4', chunks=(100, 360))
        self.dset_pose = self.h5_file.create_dataset('pose', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(100, 3))
        self.dset_vel = self.h5_file.create_dataset('velocity', (0, 2), maxshape=(None, 2), dtype='f4', chunks=(100, 2))
        self.dset_cmd = self.h5_file.create_dataset('cmd', (0, 2), maxshape=(None, 2), dtype='f4', chunks=(100, 2))
        self.dset_ts = self.h5_file.create_dataset('timestamp', (0,), maxshape=(None,), dtype='f8', chunks=(100,))
        
        self.count = 0
        self.start_time = time.time()

    def scan_callback(self, msg):
        # We use 360 samples as defined in URDF
        if len(msg.ranges) != 360:
            # If the lidar config changed, we might need to adjust, but let's assume 360
            ranges = np.array(msg.ranges)
            if len(ranges) > 360:
                ranges = ranges[:360]
            else:
                ranges = np.pad(ranges, (0, 360 - len(ranges)), 'constant', constant_values=msg.range_max)
            self.current_scan = ranges
        else:
            self.current_scan = np.array(msg.ranges)
            
        # Clean data (remove inf/nan)
        self.current_scan = np.nan_to_num(self.current_scan, nan=msg.range_max, posinf=msg.range_max, neginf=msg.range_min)
        
        self.record_step()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(ori)
        
        self.current_pose = [pos.x, pos.y, yaw]
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def cmd_callback(self, msg):
        self.current_cmd = [msg.linear.x, msg.angular.z]

    def record_step(self):
        if self.current_scan is None or self.current_pose is None:
            return
            
        # Resize datasets
        new_size = self.count + 1
        self.dset_lidar.resize(new_size, axis=0)
        self.dset_pose.resize(new_size, axis=0)
        self.dset_vel.resize(new_size, axis=0)
        self.dset_cmd.resize(new_size, axis=0)
        self.dset_ts.resize(new_size, axis=0)
        
        # Write data
        self.dset_lidar[self.count] = self.current_scan
        self.dset_pose[self.count] = self.current_pose
        self.dset_vel[self.count] = self.current_vel
        self.dset_cmd[self.count] = self.current_cmd
        self.dset_ts[self.count] = time.time() - self.start_time
        
        self.count += 1
        if self.count % 100 == 0:
            self.get_logger().info(f'Recorded {self.count} steps...')
            self.h5_file.flush()

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = H5DatasetGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Closing HDF5 file. Total samples: {node.count}')
        node.h5_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
