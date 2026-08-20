#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math
import time

class TrajectoryManager(Node):
    def __init__(self):
        super().__init__('trajectory_manager')
        
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.waypoints = [
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (0.0, 0.0)
        ]
        
        self.current_waypoint_idx = 0
        self.current_pose = None
        self.mission_started = False
        self.mission_finished = False
        
        self.timer = self.create_timer(0.5, self.mission_loop)
        self.get_logger().info('Trajectory Manager Started. Waiting for first Odom...')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def send_goal(self, x, y):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.get_logger().info(f'--- Sending Waypoint {self.current_waypoint_idx + 1}: x={x}, y={y} ---')

    def mission_loop(self):
        if self.current_pose is None or self.mission_finished:
            return

        if not self.mission_started:
            self.send_goal(*self.waypoints[self.current_waypoint_idx])
            self.mission_started = True
            return

        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        curr_x = self.current_pose.position.x
        curr_y = self.current_pose.position.y
        
        dist = math.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)

        if dist < 0.2:
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx + 1}. Stopping for 2 seconds...')
            time.sleep(2.0)
            
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.waypoints):
                self.send_goal(*self.waypoints[self.current_waypoint_idx])
            else:
                self.get_logger().info('MISSION COMPLETED! Robot is back home.')
                self.mission_finished = True

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryManager()
    try:
        while rclpy.ok() and not node.mission_finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
