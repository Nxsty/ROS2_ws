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
    6-State Extended Kalman Filter for Omnidirectional / Mecanum Mobile Robot.
    State: x = [x, y, theta, vx, vy, omega]^T
      - x, y: 2D position in world/odom frame (m)
      - theta: Heading in world/odom frame (rad)
      - vx: Linear forward velocity in base frame (m/s)
      - vy: Linear lateral velocity in base frame (m/s)
      - omega: Yaw angular velocity in base frame (rad/s)
    """

    def __init__(self):
        # State vector: [x, y, theta, vx, vy, omega]
        self.x = np.zeros(6, dtype=np.float64)

        # State covariance P (6x6)
        self.P = np.diag([0.01, 0.01, 0.005, 0.05, 0.05, 0.05])

        # Process noise Q (6x6)
        self.Q = np.diag([0.002, 0.002, 0.001, 0.05, 0.05, 0.05])

    def predict(self, dt: float):
        """Kinematic motion model prediction step for Mecanum omnidirectional robot."""
        if dt <= 0.0 or dt > 0.5:
            return

        x, y, theta, vx, vy, omega = self.x
        if abs(vx) < 0.01:
            vx = 0.0
            self.x[3] = 0.0
        if abs(vy) < 0.01:
            vy = 0.0
            self.x[4] = 0.0
        if abs(omega) < 0.005:
            omega = 0.0
            self.x[5] = 0.0

        theta_mid = theta + 0.5 * omega * dt

        # State transition
        delta_x = (vx * math.cos(theta_mid) - vy * math.sin(theta_mid)) * dt
        delta_y = (vx * math.sin(theta_mid) + vy * math.cos(theta_mid)) * dt

        self.x[0] = x + delta_x
        self.x[1] = y + delta_y
        self.x[2] = normalize_angle(theta + omega * dt)

        # Jacobian F = df/dx (6x6)
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = (-vx * math.sin(theta_mid) - vy * math.cos(theta_mid)) * dt
        F[0, 3] = math.cos(theta_mid) * dt
        F[0, 4] = -math.sin(theta_mid) * dt
        F[0, 5] = -0.5 * (vx * math.sin(theta_mid) + vy * math.cos(theta_mid)) * (dt ** 2)

        F[1, 2] = (vx * math.cos(theta_mid) - vy * math.sin(theta_mid)) * dt
        F[1, 3] = math.sin(theta_mid) * dt
        F[1, 4] = math.cos(theta_mid) * dt
        F[1, 5] = 0.5 * (vx * math.cos(theta_mid) - vy * math.sin(theta_mid)) * (dt ** 2)

        F[2, 5] = dt

        # Covariance update
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_imu(self, omega_imu: float, var_omega: float = 0.0005):
        """Measurement update from IMU Gyroscope."""
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 5] = 1.0

        z = np.array([omega_imu], dtype=np.float64)
        z_hat = np.array([self.x[5]], dtype=np.float64)
        y = z - z_hat

        R = np.array([[var_omega]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_orientation(self, yaw_imu: float, var_yaw: float = 0.0005):
        """Measurement update from IMU Orientation (Quaternion)."""
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 2] = 1.0

        z = np.array([yaw_imu], dtype=np.float64)
        z_hat = np.array([self.x[2]], dtype=np.float64)
        y = normalize_angle(z[0] - z_hat[0])

        R = np.array([[var_yaw]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K.flatten() * y)
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_wheel_vel(self, vx: float, vy: float, omega: float, 
                         var_vx: float = 0.005, var_vy: float = 0.005, var_omega: float = 0.005):
        """Measurement update from Mecanum Wheel Encoders."""
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        z = np.array([vx, vy, omega], dtype=np.float64)
        z_hat = np.array([self.x[3], self.x[4], self.x[5]], dtype=np.float64)
        y = z - z_hat

        R = np.diag([var_vx, var_vy, var_omega])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_vio(self, vx_vio: float, vy_vio: float = 0.0, var_vx: float = 0.015, var_vy: float = 0.020):
        """Measurement update from Visual Odometry (2D Optical Flow: vx and vy)."""
        H = np.zeros((2, 6), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        z = np.array([vx_vio, vy_vio], dtype=np.float64)
        z_hat = np.array([self.x[3], self.x[4]], dtype=np.float64)
        y = z - z_hat

        R = np.diag([var_vx, var_vy])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(6, dtype=np.float64)
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
        self.declare_parameter('max_features', 60)
        self.declare_parameter('min_features', 15)
        self.declare_parameter('feature_quality', 0.005)
        self.declare_parameter('feature_min_dist', 12)
        self.declare_parameter('camera_height', 0.1575)
        self.declare_parameter('enable_debug_image', False)
        self.declare_parameter('enable_vision', True)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.max_features = self.get_parameter('max_features').value
        self.min_features = self.get_parameter('min_features').value
        self.feature_quality = self.get_parameter('feature_quality').value
        self.feature_min_dist = self.get_parameter('feature_min_dist').value
        self.camera_height = self.get_parameter('camera_height').value
        self.enable_debug_image = self.get_parameter('enable_debug_image').value
        self.enable_vision = bool(self.get_parameter('enable_vision').value)

        # --- EKF Estimator ---
        self.ekf = EKFStateEstimator()

        # --- Raw Open-Loop VIO Integrator (Before Filter) ---
        self.raw_vio_x = 0.0
        self.raw_vio_y = 0.0
        self.raw_vio_theta = 0.0
        self.raw_vio_v = 0.0
        self.raw_vio_vy = 0.0

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
        if self.enable_vision:
            self.cam_info_sub = self.create_subscription(
                CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
            )
            self.image_sub = self.create_subscription(
                Image, '/camera/image_raw', self.image_callback, 10
            )
        else:
            self.cam_info_sub = None
            self.image_sub = None
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

        # EKF Measurement Update for Absolute Yaw from IMU Orientation Quaternion
        q = msg.orientation
        if abs(q.w) > 1e-4 or abs(q.z) > 1e-4:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_imu = math.atan2(siny_cosp, cosy_cosp)
            self.ekf.update_orientation(yaw_imu, var_yaw=0.0005)
            self.raw_vio_theta = yaw_imu
        else:
            # Fallback to gyro integration if quaternion is uninitialized
            dt = 0.01
            self.raw_vio_theta = normalize_angle(self.raw_vio_theta + gz * dt)

    def wheel_vel_callback(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z
        self.last_wheel_v = vx
        self.last_wheel_vy = vy
        self.last_wheel_w = wz

        # EKF Measurement Update for Mecanum Wheel Encoders (Tolerancia realista a deslizamientos)
        self.ekf.update_wheel_vel(vx, vy, wz, var_vx=0.015, var_vy=0.015, var_omega=0.002)

    def image_callback(self, msg: Image):
        if not self.enable_vision:
            return

        try:
            curr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion error: {e}')
            return

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate dt using image header timestamp to be resilient to callback queueing
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.05
        if hasattr(self, '_last_img_stamp') and self._last_img_stamp is not None and msg_time > self._last_img_stamp:
            dt = msg_time - self._last_img_stamp
            dt = max(0.02, min(dt, 0.20))
        elif self.last_img_time is not None:
            curr_time = self.get_clock().now()
            dt = (curr_time - self.last_img_time).nanoseconds / 1e9
            if dt <= 0 or dt > 0.5:
                dt = 0.05
        self._last_img_stamp = msg_time
        self.last_img_time = self.get_clock().now()

        # Ground ROI mask (only look at floor in front of the robot, saving CPU and rejecting shelf/ceiling corners)
        h, w = curr_gray.shape
        floor_y = int(self.cy + 45)
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        if floor_y < h:
            floor_mask[floor_y:, :] = 255
        else:
            floor_mask = None

        # Initial frame or re-detection
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_gray = curr_gray
            corners = cv2.goodFeaturesToTrack(curr_gray, mask=floor_mask, **self.feature_params)
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

        should_debug = self.enable_debug_image and (self.debug_img_pub.get_subscription_count() > 0)
        debug_img = curr_frame.copy() if should_debug else None

        if len(good_prev) >= 4:
            self.visual_tracking_active = True
            dx_estimates = []
            dy_estimates = []
            current_wz = self.ekf.x[5]

            for p0, p1 in zip(good_prev, good_next):
                u0, v0 = p0[0, 0], p0[0, 1]
                u1, v1 = p1[0, 0], p1[0, 1]

                if debug_img is not None:
                    cv2.circle(debug_img, (int(u1), int(v1)), 3, (0, 255, 0), -1)
                    cv2.line(debug_img, (int(u0), int(v0)), (int(u1), int(v1)), (0, 0, 255), 1)

                dv_observed = v1 - v0
                du_observed = u1 - u0
                y_img = v0 - self.cy

                # Strict Floor Region Filtering:
                # y_img > 45 ensures points are on the ground in front of the robot (0.25m - 1.3m),
                # preventing vertical shelf features from distorting displacement.
                if y_img > 45:
                    # Derotate vertical optical flow by removing camera rotation component
                    dv_rot = ((u0 - self.cx) * (v0 - self.cy) / self.fx) * (current_wz * dt)
                    dv_trans = dv_observed - dv_rot

                    # Ground plane metric displacement along forward axis (X_robot)
                    dx_i = (dv_trans * self.fy * self.camera_height) / (y_img ** 2)
                    
                    if -0.10 <= dx_i <= 0.25:
                        dx_estimates.append(dx_i)

                    # Derotate horizontal optical flow
                    # When robot turns CCW (+wz), world features move to the right (+u in image).
                    # Standard optical flow: du_rot = +(fx + (u0-cx)^2 / fx) * wz * dt
                    du_rot = (self.fx + ((u0 - self.cx) ** 2) / self.fx) * (current_wz * dt)
                    du_trans = du_observed - du_rot

                    # Ground plane lateral metric displacement:
                    # When robot moves LEFT (+Y_robot), world features move to the right (+u, du_trans > 0).
                    # Standard ground projection: dy_i = +(du_trans * h * fy) / (y_img * fx)
                    dy_i = (du_trans * self.camera_height * self.fy) / (y_img * self.fx)
                    if -0.20 <= dy_i <= 0.20:
                        dy_estimates.append(dy_i)

            if len(dx_estimates) >= 4:
                dx_med = float(np.median(dx_estimates))
                v_vio_meas = dx_med / dt if dt > 0 else 0.0

                dy_med = float(np.median(dy_estimates)) if len(dy_estimates) >= 4 else 0.0
                vy_vio_meas = dy_med / dt if dt > 0 else 0.0

                # 1. Update Raw VIO Open-Loop Odometry (Before Filter)
                self.raw_vio_v = v_vio_meas
                self.raw_vio_vy = vy_vio_meas
                self.raw_vio_x += (dx_med * math.cos(self.raw_vio_theta) - dy_med * math.sin(self.raw_vio_theta))
                self.raw_vio_y += (dx_med * math.sin(self.raw_vio_theta) + dy_med * math.cos(self.raw_vio_theta))

                # 2. EKF Measurement Update with Outlier & Rotation Gating
                pred_vx = self.ekf.x[3]
                pred_vy = self.ekf.x[4]
                # Gate both forward and lateral velocities and avoid updating during sharp turns
                if abs(current_wz) < 0.35 and abs(v_vio_meas - pred_vx) < 0.40 and abs(vy_vio_meas - pred_vy) < 0.35:
                    self.ekf.update_vio(v_vio_meas, vy_vio_meas, var_vx=0.060, var_vy=0.080)

            # Re-detect if feature count drops
            if len(good_next) < self.min_features:
                new_corners = cv2.goodFeaturesToTrack(curr_gray, mask=floor_mask, **self.feature_params)
                if new_corners is not None and len(new_corners) > 0:
                    self.prev_pts = new_corners
                else:
                    self.prev_pts = good_next.reshape(-1, 1, 2)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2)
        else:
            self.visual_tracking_active = False
            corners = cv2.goodFeaturesToTrack(curr_gray, mask=floor_mask, **self.feature_params)
            if corners is not None and len(corners) > 0:
                self.prev_pts = corners

        self.prev_gray = curr_gray

        # Publish visualization image (only if enabled and subscribed)
        if debug_img is not None:
            try:
                status_text = (
                    f"EKF-VIO: {'ACTIVE' if self.visual_tracking_active else 'SEARCHING'} | "
                    f"Pts: {len(good_prev)} | "
                    f"Pose: ({self.ekf.x[0]:.2f}, {self.ekf.x[1]:.2f}, {math.degrees(self.ekf.x[2]):.1f}deg) | "
                    f"vx: {self.ekf.x[3]:.2f}, vy: {self.ekf.x[4]:.2f}m/s"
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
        vy = float(self.ekf.x[4])
        wz = float(self.ekf.x[5])

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

        # Covariances
        odom.pose.covariance[0] = float(self.ekf.P[0, 0])    # Var(x)
        odom.pose.covariance[1] = float(self.ekf.P[0, 1])    # Cov(x, y)
        odom.pose.covariance[7] = float(self.ekf.P[1, 1])    # Var(y)
        odom.pose.covariance[35] = float(self.ekf.P[2, 2])   # Var(theta)

        # Twist (Local robot frame)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        odom.twist.covariance[0] = float(self.ekf.P[3, 3])   # Var(vx)
        odom.twist.covariance[7] = float(self.ekf.P[4, 4])   # Var(vy)
        odom.twist.covariance[35] = float(self.ekf.P[5, 5])  # Var(wz)

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
        raw_odom.twist.twist.linear.y = self.raw_vio_vy
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
