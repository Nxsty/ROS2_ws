#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math


class TrajectoryManager(Node):
    def __init__(self):
        super().__init__('trajectory_manager')
        
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # 4 Colored Station Waypoints (Parking 2.0m directly in front of each shelf)
        self.waypoints = [
            # 1. 🔵 Blue Station Front (Shelf is at x=10, y=8)
            (10.0, 6.0),

            # 2. 🟡 Yellow Station Front (Shelf is at x=10, y=-8)
            (10.0, -6.0),

            # 3. 🟢 Green Station Front (Shelf is at x=-10, y=-8)
            (-10.0, -6.0),

            # 4. 🔴 Red Station Front (Shelf is at x=-10, y=8)
            (-10.0, 6.0),

            # 5. 🏁 Return Home (Warehouse Center)
            (0.0, 0.0)
        ]
        
        self.current_waypoint_idx = 0
        self.current_pose = None
        self.mission_started = False
        self.mission_finished = False
        
        # Non-blocking pause state
        self.is_paused = False
        self.pause_end_time = 0.0
        self.dist_tolerance = 0.35  # Synchronized with pose_controller tolerance

        # High-rate 20 Hz mission loop (no missed arrival windows)
        self.timer = self.create_timer(0.05, self.mission_loop)
        self.get_logger().info('Trajectory Manager Started (20 Hz loop). Waiting for first Odom...')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def send_goal(self, x, y):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.get_logger().info(f'=== Sending Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}: x={x:.1f}, y={y:.1f} ===')

    def mission_loop(self):
        if self.current_pose is None or self.mission_finished:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # Handle non-blocking pause at waypoint
        if self.is_paused:
            if now_sec >= self.pause_end_time:
                self.is_paused = False
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.waypoints):
                    self.send_goal(*self.waypoints[self.current_waypoint_idx])
                else:
                    self.get_logger().info('🏆 MISSION COMPLETED! Robot is back home.')
                    self.mission_finished = True
            return

        if not self.mission_started:
            self.send_goal(*self.waypoints[self.current_waypoint_idx])
            self.mission_started = True
            return

        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        curr_x = self.current_pose.position.x
        curr_y = self.current_pose.position.y
        
        dist = math.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)

        if dist < self.dist_tolerance:
            self.get_logger().info(f'✅ Reached Waypoint {self.current_waypoint_idx + 1} (dist={dist:.2f}m). Pausing 2.0s...')
            self.is_paused = True
            self.pause_end_time = now_sec + 2.0


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
