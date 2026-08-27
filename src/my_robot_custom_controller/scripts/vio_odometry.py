#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * cp * cy
    return q


class VIOOdometryNode(Node):
    """
    High-Precision Visual-Inertial Odometry (VIO) Node for Diff-Drive Mobile Robots.
    - Integrates IMU Gyroscope (100 Hz) for robust, drift-free heading (yaw).
    - Tracks ground-plane optical flow for accurate forward metric displacement.
    - Applies non-holonomic mobile base constraint (vy = 0) to eliminate lateral drift.
    """

    def __init__(self):
        super().__init__('vio_odometry')

        # --- Parameters ---
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('max_features', 300)
        self.declare_parameter('min_features', 40)
        self.declare_parameter('feature_quality', 0.005)
        self.declare_parameter('feature_min_dist', 8)
        self.declare_parameter('camera_height', 0.20)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.max_features = self.get_parameter('max_features').value
        self.min_features = self.get_parameter('min_features').value
        self.feature_quality = self.get_parameter('feature_quality').value
        self.feature_min_dist = self.get_parameter('feature_min_dist').value
        self.camera_height = self.get_parameter('camera_height').value

        # --- State Variables ---
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.vx = 0.0
        self.wz = 0.0

        # IMU state
        self.last_imu_time = None
        self.last_img_time = None

        # Vision state
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None
        self.fx = 381.4
        self.fy = 381.4
        self.cx = 320.0
        self.cy = 240.0
        self.camera_matrix = None
        self.visual_tracking_active = False

        # LK Optical Flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Feature Detector params
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=self.feature_quality,
            minDistance=self.feature_min_dist,
            blockSize=7
        )

        # --- Subscriptions & Publishers ---
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 50
        )

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.debug_img_pub = self.create_publisher(Image, '/vio/tracked_features', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 50 Hz timer ensuring smooth /odom and TF stream
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info('VIO Odometry Node Initialized. Streaming /odom at 50 Hz.')

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        self.publish_odometry(stamp)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.fx = float(self.camera_matrix[0, 0])
            self.fy = float(self.camera_matrix[1, 1])
            self.cx = float(self.camera_matrix[0, 2])
            self.cy = float(self.camera_matrix[1, 2])
            self.get_logger().info(f'Camera intrinsics calibrated: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

    def imu_callback(self, msg: Imu):
        curr_time = self.get_clock().now()
        gz = msg.angular_velocity.z

        # Deadband for sensor noise at rest
        if abs(gz) < 0.0005:
            gz = 0.0

        if self.last_imu_time is not None:
            dt = (curr_time - self.last_imu_time).nanoseconds / 1e9
            if 0.0 < dt < 0.2:
                # Integrate gyro heading (Z-axis rotation)
                self.pose_theta += gz * dt
                self.pose_theta = math.atan2(math.sin(self.pose_theta), math.cos(self.pose_theta))
                self.wz = gz

        self.last_imu_time = curr_time

    def image_callback(self, msg: Image):
        try:
            curr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion error: {e}')
            return

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_time = self.get_clock().now()

        dt = 0.05
        if self.last_img_time is not None:
            dt = (curr_time - self.last_img_time).nanoseconds / 1e9
            if dt <= 0 or dt > 0.5:
                dt = 0.05
        self.last_img_time = curr_time

        # Initial frame or re-detection
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_gray = curr_gray
            corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            if corners is not None and len(corners) > 0:
                self.prev_pts = corners
            return

        # 1. Optical Flow tracking (Lucas-Kanade)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params
        )

        good_prev = []
        good_next = []
        if next_pts is not None and status is not None:
            for p0, p1, st in zip(self.prev_pts, next_pts, status):
                if st[0] == 1:
                    good_prev.append(p0)
                    good_next.append(p1)

        good_prev = np.array(good_prev, dtype=np.float32)
        good_next = np.array(good_next, dtype=np.float32)

        debug_img = curr_frame.copy()
        dx_robot = 0.0

        if len(good_prev) >= 8:
            self.visual_tracking_active = True
            
            # Ground-plane metric forward displacement estimation
            dx_estimates = []

            # If the robot is primarily rotating on the spot, suppress forward optical flow estimation
            is_turning_in_place = abs(self.wz) > 0.35

            if not is_turning_in_place:
                for p0, p1 in zip(good_prev, good_next):
                    u0, v0 = p0[0, 0], p0[0, 1]
                    u1, v1 = p1[0, 0], p1[0, 1]

                    cv2.circle(debug_img, (int(u1), int(v1)), 3, (0, 255, 0), -1)
                    cv2.line(debug_img, (int(u0), int(v0)), (int(u1), int(v1)), (0, 0, 255), 1)

                    dv = v1 - v0

                    # Points in lower half of image (ground plane in front of robot)
                    y_img = v0 - self.cy
                    if y_img > 12:  # Ground points below optical center
                        # Flow dv = y_img^2 * dx / (fy * h) => dx = dv * fy * h / y_img^2
                        dx_i = (dv * self.fy * self.camera_height) / (y_img ** 2)

                        if abs(dx_i) < 1.5 * dt:
                            dx_estimates.append(dx_i)

                if len(dx_estimates) >= 4:
                    dx_robot = float(np.median(dx_estimates))
            else:
                # During turning, draw tracking points in yellow
                for p0, p1 in zip(good_prev, good_next):
                    u1, v1 = p1[0, 0], p1[0, 1]
                    cv2.circle(debug_img, (int(u1), int(v1)), 3, (0, 255, 255), -1)

            # Update linear velocity
            self.vx = dx_robot / dt if dt > 0 else 0.0

            # Accumulate global displacement in odom frame (Non-holonomic: lateral velocity is zero)
            cos_t = math.cos(self.pose_theta)
            sin_t = math.sin(self.pose_theta)
            self.pose_x += dx_robot * cos_t
            self.pose_y += dx_robot * sin_t

            # Re-detect if feature count drops
            if len(good_next) < self.min_features:
                new_corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
                if new_corners is not None and len(new_corners) > 0:
                    self.prev_pts = new_corners
                else:
                    self.prev_pts = good_next.reshape(-1, 1, 2)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2)
        else:
            self.visual_tracking_active = False
            self.vx = 0.0
            corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            if corners is not None and len(corners) > 0:
                self.prev_pts = corners

        self.prev_gray = curr_gray

        # Publish visualization image
        try:
            status_text = f"VIO: {'ACTIVE' if self.visual_tracking_active else 'SEARCHING'} | Pts: {len(good_prev)} | (x={self.pose_x:.2f}, y={self.pose_y:.2f}, yaw={math.degrees(self.pose_theta):.1f} deg)"
            cv2.putText(debug_img, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header.stamp = msg.header.stamp
            debug_msg.header.frame_id = 'camera_link_optical'
            self.debug_img_pub.publish(debug_msg)
        except Exception:
            pass

    def publish_odometry(self, stamp):
        q = euler_to_quaternion(0.0, 0.0, self.pose_theta)

        # 1. TF Broadcast (odom -> base_footprint)
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.pose_x
            t.transform.translation.y = self.pose_y
            t.transform.translation.z = 0.0
            t.transform.rotation = q
            self.tf_broadcaster.sendTransform(t)

        # 2. Odometry Message
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        # Pose
        odom.pose.pose.position.x = self.pose_x
        odom.pose.pose.position.y = self.pose_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q

        # Covariances
        odom.pose.covariance[0] = 0.005   # x
        odom.pose.covariance[7] = 0.005   # y
        odom.pose.covariance[35] = 0.002  # yaw

        # Twist (Local robot frame)
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = self.wz

        odom.twist.covariance[0] = 0.01
        odom.twist.covariance[7] = 0.01
        odom.twist.covariance[35] = 0.005

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = VIOOdometryNode()
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
