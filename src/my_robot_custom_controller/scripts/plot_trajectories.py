#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import time
import os


from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class TrajectoryPlotterNode(Node):
    """
    Live Trajectory Plotter & Quantitative Performance Evaluator.
    Compares:
      1. 🟢 Ground Truth (/model/my_robot/odometry)
      2. 🔴 Raw VIO Before Filter (/odom/raw_vio)
      3. 🔵 EKF Filtered Odometry (/odom)
    """

    def __init__(self):
        super().__init__('trajectory_plotter')

        # Storage for trajectories
        self.gt_x, self.gt_y = [], []
        self.raw_x, self.raw_y = [], []
        self.ekf_x, self.ekf_y = [], []

        # Current latest values
        self.current_gt = None
        self.current_raw = None
        self.current_ekf = None

        # Universal QoS Profile compatible with both Gazebo bridge and standard ROS nodes
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscriptions
        self.gt_sub = self.create_subscription(
            Odometry, '/model/my_robot/odometry', self.gt_callback, qos
        )
        self.raw_sub = self.create_subscription(
            Odometry, '/odom/raw_vio', self.raw_callback, qos
        )
        self.ekf_sub = self.create_subscription(
            Odometry, '/odom', self.ekf_callback, qos
        )

        # Matplotlib interactive figure setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Trajectory Comparison: Ground Truth vs Raw VIO vs EKF")
        
        # Setup static warehouse geometry
        self.setup_warehouse_plot()

        # Lines for plotting
        self.line_gt, = self.ax.plot([], [], 'g--', linewidth=2.0, label='Ground Truth (Gazebo)', zorder=4)
        self.line_raw, = self.ax.plot([], [], 'r:', linewidth=1.5, label='Raw VIO (Before Filter)', zorder=3)
        self.line_ekf, = self.ax.plot([], [], 'b-', linewidth=2.2, label='EKF Filtered (/odom)', zorder=5)

        # Current robot markers
        self.dot_gt, = self.ax.plot([], [], 'go', markersize=8, zorder=6)
        self.dot_raw, = self.ax.plot([], [], 'ro', markersize=6, zorder=6)
        self.dot_ekf, = self.ax.plot([], [], 'bo', markersize=8, zorder=6)

        self.info_text = self.ax.text(
            0.02, 0.96, 'Waiting for odometry data...',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        )

        self.ax.legend(loc='lower left', framealpha=0.9)
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # 5 Hz GUI Update Timer
        self.timer = self.create_timer(0.20, self.update_plot)
        self.get_logger().info('Trajectory Plotter Node Started. Listening to /model/my_robot/odometry, /odom/raw_vio, and /odom.')

    def setup_warehouse_plot(self):
        self.ax.set_title('Warehouse Trajectory Validation (EKF vs Raw vs Ground Truth)', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X Position (meters)', fontsize=11)
        self.ax.set_ylabel('Y Position (meters)', fontsize=11)
        self.ax.set_xlim(-15.0, 15.0)
        self.ax.set_ylim(-13.0, 13.0)
        self.ax.set_aspect('equal')

        # Outer walls
        wall = patches.Rectangle((-13.5, -11.0), 27.0, 22.0, fill=False, edgecolor='black', linewidth=2.0, linestyle='-')
        self.ax.add_patch(wall)

        # 4 Colored Shelves (2m x 1m)
        # Blue: (10, 8), Yellow: (10, -8), Green: (-10, -8), Red: (-10, 8)
        shelf_blue = patches.Rectangle((9.0, 7.5), 2.0, 1.0, facecolor='blue', edgecolor='darkblue', alpha=0.6, label='Blue Station')
        shelf_yellow = patches.Rectangle((9.0, -8.5), 2.0, 1.0, facecolor='yellow', edgecolor='orange', alpha=0.6, label='Yellow Station')
        shelf_green = patches.Rectangle((-11.0, -8.5), 2.0, 1.0, facecolor='green', edgecolor='darkgreen', alpha=0.6, label='Green Station')
        shelf_red = patches.Rectangle((-11.0, 7.5), 2.0, 1.0, facecolor='red', edgecolor='darkred', alpha=0.6, label='Red Station')

        self.ax.add_patch(shelf_blue)
        self.ax.add_patch(shelf_yellow)
        self.ax.add_patch(shelf_green)
        self.ax.add_patch(shelf_red)

        # Target Waypoint Markers (2m in front of each shelf)
        waypoints = [(10.0, 6.0), (10.0, -6.0), (-10.0, -6.0), (-10.0, 6.0), (0.0, 0.0)]
        labels = ['WP1 (Blue)', 'WP2 (Yellow)', 'WP3 (Green)', 'WP4 (Red)', 'WP5 (Home)']
        for (wx, wy), lbl in zip(waypoints, labels):
            self.ax.plot(wx, wy, 'kx', markersize=10, markeredgewidth=2)
            self.ax.text(wx + 0.3, wy + 0.3, lbl, fontsize=8, color='black', fontweight='bold')

    def gt_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.gt_x.append(p.x)
        self.gt_y.append(p.y)
        self.current_gt = (p.x, p.y)

    def raw_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.raw_x.append(p.x)
        self.raw_y.append(p.y)
        self.current_raw = (p.x, p.y)

    def ekf_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.ekf_x.append(p.x)
        self.ekf_y.append(p.y)
        self.current_ekf = (p.x, p.y)

    def update_plot(self):
        if not self.gt_x and not self.ekf_x and not self.raw_x:
            return

        # Update line data
        if self.gt_x:
            self.line_gt.set_data(self.gt_x, self.gt_y)
        if self.raw_x:
            self.line_raw.set_data(self.raw_x, self.raw_y)
        if self.ekf_x:
            self.line_ekf.set_data(self.ekf_x, self.ekf_y)

        # Update current dots
        if self.current_gt:
            self.dot_gt.set_data([self.current_gt[0]], [self.current_gt[1]])
        if self.current_raw:
            self.dot_raw.set_data([self.current_raw[0]], [self.current_raw[1]])
        if self.current_ekf:
            self.dot_ekf.set_data([self.current_ekf[0]], [self.current_ekf[1]])

        # Compute error statistics
        lines_info = []
        if self.current_gt:
            lines_info.append(f"Ground Truth: ({self.current_gt[0]:.2f}, {self.current_gt[1]:.2f})m")
        if self.current_ekf:
            ekf_err_str = ""
            if self.current_gt:
                ekf_err = math.sqrt((self.current_ekf[0] - self.current_gt[0])**2 + (self.current_ekf[1] - self.current_gt[1])**2)
                ekf_err_str = f" | Error: {ekf_err:.3f}m"
            lines_info.append(f"EKF Filtered: ({self.current_ekf[0]:.2f}, {self.current_ekf[1]:.2f})m{ekf_err_str}")
        if self.current_raw:
            raw_err_str = ""
            if self.current_gt:
                raw_err = math.sqrt((self.current_raw[0] - self.current_gt[0])**2 + (self.current_raw[1] - self.current_gt[1])**2)
                raw_err_str = f" | Error: {raw_err:.3f}m"
            lines_info.append(f"Raw VIO:      ({self.current_raw[0]:.2f}, {self.current_raw[1]:.2f})m{raw_err_str}")

        if lines_info:
            self.info_text.set_text("\n".join(lines_info))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_comparison_plot(self):
        output_path = os.path.expanduser('~/ros2_ws/trajectory_comparison.png')
        self.fig.savefig(output_path, dpi=200, bbox_inches='tight')
        self.get_logger().info(f'📊 Comparison plot saved to: {output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlotterNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_comparison_plot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
