#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, Quaternion, Twist, PoseStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


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


class EKFStateEstimator:
    """
    5-State Extended Kalman Filter for Differential Drive Mobile Robot.
    State: x = [x, y, theta, v, omega]^T
      - x, y: 2D position in world/odom frame (m)
      - theta: Heading in world/odom frame (rad)
      - v: Linear forward velocity in base frame (m/s)
      - omega: Yaw angular velocity in base frame (rad/s)
    """

    def __init__(self):
        # State vector: [x, y, theta, v, omega]
        self.x = np.zeros(5, dtype=np.float64)

        # State covariance P (5x5)
        self.P = np.diag([0.01, 0.01, 0.005, 0.05, 0.05])

        # Process noise Q (5x5)
        self.Q = np.diag([0.002, 0.002, 0.001, 0.05, 0.05])

    def predict(self, dt: float):
        """Kinematic motion model prediction step."""
        if dt <= 0.0 or dt > 0.5:
            return

        x, y, theta, v, omega = self.x
        if abs(v) < 0.02:
            v = 0.0
            self.x[3] = 0.0

        theta_mid = theta + 0.5 * omega * dt

        # State transition
        self.x[0] = x + v * math.cos(theta_mid) * dt
        self.x[1] = y + v * math.sin(theta_mid) * dt
        self.x[2] = normalize_angle(theta + omega * dt)

        # Jacobian F = df/dx
        F = np.eye(5, dtype=np.float64)
        F[0, 2] = -v * math.sin(theta_mid) * dt
        F[0, 3] = math.cos(theta_mid) * dt
        F[0, 4] = -0.5 * v * math.sin(theta_mid) * (dt ** 2)

        F[1, 2] = v * math.cos(theta_mid) * dt
        F[1, 3] = math.sin(theta_mid) * dt
        F[1, 4] = 0.5 * v * math.cos(theta_mid) * (dt ** 2)

        F[2, 4] = dt

        # Covariance update
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_imu(self, omega_imu: float, var_omega: float = 0.0005):
        """Measurement update from IMU Gyroscope."""
        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 4] = 1.0

        z = np.array([omega_imu], dtype=np.float64)
        z_hat = np.array([self.x[4]], dtype=np.float64)
        y = z - z_hat

        R = np.array([[var_omega]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(5, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_wheel_vel(self, v_wheel: float, omega_wheel: float, var_v: float = 0.005, var_omega: float = 0.005):
        """Measurement update from Wheel Encoders."""
        H = np.zeros((2, 5), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        z = np.array([v_wheel, omega_wheel], dtype=np.float64)
        z_hat = np.array([self.x[3], self.x[4]], dtype=np.float64)
        y = z - z_hat

        R = np.diag([var_v, var_omega])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(5, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_vio(self, v_vio: float, var_v: float = 0.015):
        """Measurement update from Visual Odometry (Optical Flow)."""
        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 3] = 1.0

        z = np.array([v_vio], dtype=np.float64)
        z_hat = np.array([self.x[3]], dtype=np.float64)
        y = z - z_hat

        R = np.array([[var_v]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(5, dtype=np.float64)
        self.P = (I - K @ H) @ self.P


class VIOOdometryNode(Node):
    """
    Visual-Inertial-Wheel Odometry with Extended Kalman Filter (EKF) Sensor Fusion.
    - Fuses Wheel Encoders (Forward Kinematics), 100 Hz IMU Gyro, and 20 Hz Monocular VIO Optical Flow.
    - Publishes:
        1. /odom (EKF Fused Odometry)
        2. /odom/raw_vio (Raw Open-loop VIO Odometry before filter)
        3. /path/ekf, /path/raw_vio, /path/ground_truth (Paths for RViz visualization)
    """

    def __init__(self):
        super().__init__('vio_odometry')

        # --- Parameters ---
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('max_features', 300)
        self.declare_parameter('min_features', 20)
        self.declare_parameter('feature_quality', 0.005)
        self.declare_parameter('feature_min_dist', 8)
        self.declare_parameter('camera_height', 0.1575)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.max_features = self.get_parameter('max_features').value
        self.min_features = self.get_parameter('min_features').value
        self.feature_quality = self.get_parameter('feature_quality').value
        self.feature_min_dist = self.get_parameter('feature_min_dist').value
        self.camera_height = self.get_parameter('camera_height').value

        # --- EKF Estimator ---
        self.ekf = EKFStateEstimator()

        # --- Raw Open-Loop VIO Integrator (Before Filter) ---
        self.raw_vio_x = 0.0
        self.raw_vio_y = 0.0
        self.raw_vio_theta = 0.0
        self.raw_vio_v = 0.0

        # --- Wheel velocity cache ---
        self.last_wheel_v = 0.0
        self.last_wheel_w = 0.0

        # --- Ground Truth Pose cache ---
        self.gt_pose = None

        # Timing
        self.last_predict_time = None
        self.last_img_time = None

        # Paths for RViz
        self.path_ekf = Path()
        self.path_ekf.header.frame_id = self.odom_frame
        self.path_raw_vio = Path()
        self.path_raw_vio.header.frame_id = self.odom_frame
        self.path_gt = Path()
        self.path_gt.header.frame_id = self.odom_frame

        # Vision State
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None
        self.fx = 381.36
        self.fy = 381.36
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

        # --- Subscriptions ---
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 50
        )
        self.wheel_vel_sub = self.create_subscription(
            Twist, '/robot_measured_vel', self.wheel_vel_callback, 50
        )
        self.gt_sub = self.create_subscription(
            Odometry, '/model/my_robot/odometry', self.gt_callback, 10
        )

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.raw_vio_pub = self.create_publisher(Odometry, '/odom/raw_vio', 10)
        self.debug_img_pub = self.create_publisher(Image, '/vio/tracked_features', 10)
        
        self.path_ekf_pub = self.create_publisher(Path, '/path/ekf', 10)
        self.path_raw_pub = self.create_publisher(Path, '/path/raw_vio', 10)
        self.path_gt_pub = self.create_publisher(Path, '/path/ground_truth', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # 50 Hz EKF Loop & Odometry Publisher Timer
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info(f'EKF Visual-Inertial Odometry Initialized (camera_height={self.camera_height:.4f}m).')

    def gt_callback(self, msg: Odometry):
        self.gt_pose = msg.pose.pose
        # Append to Ground Truth Path
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.odom_frame
        ps.pose = msg.pose.pose
        self.path_gt.poses.append(ps)
        if len(self.path_gt.poses) > 2000:
            self.path_gt.poses.pop(0)

    def timer_callback(self):
        curr_time = self.get_clock().now()
        stamp = curr_time.to_msg()

        # Prediction step
        if self.last_predict_time is not None:
            dt = (curr_time - self.last_predict_time).nanoseconds / 1e9
            if 0.0 < dt < 0.2:
                self.ekf.predict(dt)
        self.last_predict_time = curr_time

        # Publish Odometry, TF, and Paths
        self.publish_odometry(stamp)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.fx = float(self.camera_matrix[0, 0])
            self.fy = float(self.camera_matrix[1, 1])
            self.cx = float(self.camera_matrix[0, 2])
            self.cy = float(self.camera_matrix[1, 2])
            self.get_logger().info(f'Camera calibrated: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

    def imu_callback(self, msg: Imu):
        gz = msg.angular_velocity.z
        if abs(gz) < 0.0005:
            gz = 0.0

        # EKF Measurement Update for Gyroscope
        self.ekf.update_imu(gz, var_omega=0.0002)

        # Raw VIO yaw integration (open loop)
        dt = 0.01
        self.raw_vio_theta = normalize_angle(self.raw_vio_theta + gz * dt)

    def wheel_vel_callback(self, msg: Twist):
        vx = msg.linear.x
        wz = msg.angular.z
        self.last_wheel_v = vx
        self.last_wheel_w = wz

        # EKF Measurement Update for Wheel Encoders
        self.ekf.update_wheel_vel(vx, wz, var_v=0.002, var_omega=0.002)

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

        if len(good_prev) >= 4:
            self.visual_tracking_active = True
            dx_estimates = []
            current_wz = self.ekf.x[4]

            for p0, p1 in zip(good_prev, good_next):
                u0, v0 = p0[0, 0], p0[0, 1]
                u1, v1 = p1[0, 0], p1[0, 1]

                cv2.circle(debug_img, (int(u1), int(v1)), 3, (0, 255, 0), -1)
                cv2.line(debug_img, (int(u0), int(v0)), (int(u1), int(v1)), (0, 0, 255), 1)

                dv_observed = v1 - v0
                y_img = v0 - self.cy

                # Strict Floor Region Filtering:
                # y_img > 45 ensures points are on the ground in front of the robot (0.25m - 1.3m),
                # preventing vertical shelf features from distorting displacement.
                if y_img > 45:
                    # Derotate vertical optical flow by removing camera rotation component
                    dv_rot = ((u0 - self.cx) * (v0 - self.cy) / self.fx) * (current_wz * dt)
                    dv_trans = dv_observed - dv_rot

                    # Ground plane metric displacement
                    dx_i = (dv_trans * self.fy * self.camera_height) / (y_img ** 2)
                    
                    # Sanity check: individual point displacement should be reasonable
                    if -0.05 <= dx_i <= 0.20:
                        dx_estimates.append(dx_i)

            if len(dx_estimates) >= 4:
                dx_med = float(np.median(dx_estimates))
                v_vio_meas = dx_med / dt if dt > 0 else 0.0

                # 1. Update Raw VIO Open-Loop Odometry (Before Filter)
                self.raw_vio_v = v_vio_meas
                self.raw_vio_x += dx_med * math.cos(self.raw_vio_theta)
                self.raw_vio_y += dx_med * math.sin(self.raw_vio_theta)

                # 2. EKF Measurement Update with Outlier & Rotation Gating
                pred_v = self.ekf.x[3]
                # If robot is not turning fast and visual velocity is consistent with wheel/predicted velocity
                if abs(current_wz) < 0.25 and abs(v_vio_meas - pred_v) < 0.40:
                    self.ekf.update_vio(v_vio_meas, var_v=0.015)

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
            corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            if corners is not None and len(corners) > 0:
                self.prev_pts = corners

        self.prev_gray = curr_gray

        # Publish visualization image
        try:
            status_text = (
                f"EKF-VIO: {'ACTIVE' if self.visual_tracking_active else 'SEARCHING'} | "
                f"Pts: {len(good_prev)} | "
                f"Pose: ({self.ekf.x[0]:.2f}, {self.ekf.x[1]:.2f}, {math.degrees(self.ekf.x[2]):.1f}deg) | "
                f"v: {self.ekf.x[3]:.2f}m/s"
            )
            cv2.putText(debug_img, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header.stamp = msg.header.stamp
            debug_msg.header.frame_id = 'camera_link_optical'
            self.debug_img_pub.publish(debug_msg)
        except Exception:
            pass

    def publish_odometry(self, stamp):
        pose_x = float(self.ekf.x[0])
        pose_y = float(self.ekf.x[1])
        pose_theta = float(self.ekf.x[2])
        vx = float(self.ekf.x[3])
        wz = float(self.ekf.x[4])

        q = euler_to_quaternion(0.0, 0.0, pose_theta)

        # 1. TF Broadcast (odom -> base_footprint)
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = pose_x
            t.transform.translation.y = pose_y
            t.transform.translation.z = 0.0
            t.transform.rotation = q
            self.tf_broadcaster.sendTransform(t)

        # 2. Filtered Odometry Message (/odom)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = pose_x
        odom.pose.pose.position.y = pose_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q

        odom.pose.covariance[0] = float(self.ekf.P[0, 0])    # Var(x)
        odom.pose.covariance[1] = float(self.ekf.P[0, 1])    # Cov(x, y)
        odom.pose.covariance[7] = float(self.ekf.P[1, 1])    # Var(y)
        odom.pose.covariance[35] = float(self.ekf.P[2, 2])   # Var(theta)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = wz

        odom.twist.covariance[0] = float(self.ekf.P[3, 3])   # Var(vx)
        odom.twist.covariance[35] = float(self.ekf.P[4, 4])  # Var(wz)

        self.odom_pub.publish(odom)

        # 3. Raw VIO Odometry Message (/odom/raw_vio) - Before Filter
        raw_odom = Odometry()
        raw_odom.header.stamp = stamp
        raw_odom.header.frame_id = self.odom_frame
        raw_odom.child_frame_id = self.base_frame

        raw_odom.pose.pose.position.x = self.raw_vio_x
        raw_odom.pose.pose.position.y = self.raw_vio_y
        raw_odom.pose.pose.position.z = 0.0
        raw_odom.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, self.raw_vio_theta)
        raw_odom.twist.twist.linear.x = self.raw_vio_v
        raw_odom.twist.twist.angular.z = wz
        self.raw_vio_pub.publish(raw_odom)

        # 4. Update & Publish RViz Paths
        ps_ekf = PoseStamped()
        ps_ekf.header.stamp = stamp
        ps_ekf.header.frame_id = self.odom_frame
        ps_ekf.pose = odom.pose.pose
        self.path_ekf.poses.append(ps_ekf)
        if len(self.path_ekf.poses) > 2000:
            self.path_ekf.poses.pop(0)
        self.path_ekf_pub.publish(self.path_ekf)

        ps_raw = PoseStamped()
        ps_raw.header.stamp = stamp
        ps_raw.header.frame_id = self.odom_frame
        ps_raw.pose = raw_odom.pose.pose
        self.path_raw_vio.poses.append(ps_raw)
        if len(self.path_raw_vio.poses) > 2000:
            self.path_raw_vio.poses.pop(0)
        self.path_raw_pub.publish(self.path_raw_vio)

        if len(self.path_gt.poses) > 0:
            self.path_gt_pub.publish(self.path_gt)


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
