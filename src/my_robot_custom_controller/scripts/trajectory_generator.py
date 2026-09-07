#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from nav_msgs.msg import Path, Odometry, Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Time

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import math


def quaternion_from_euler(roll, pitch, yaw):
    """Convert roll, pitch, yaw to geometry_msgs Quaternion."""
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
    q.z = cr * cp * sy - sr * sp * cy
    return q


def euler_from_quaternion(q):
    """Convert geometry_msgs Quaternion to yaw."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ContinuousTrajectoryGenerator(Node):
    """
    Continuous C^2 Smooth Trajectory Generator for Omnidirectional Mecanum Robots.
    
    Generates time-parameterized trajectories sigma(t) = [x, y, theta, vx, vy, wz, ax, ay, alpha]
    with continuous acceleration and bounded jerk using smoothed S-curve profiling.
    
    Modes:
      - 'stations': Visits each warehouse station, decelerates smoothly to 0 m/s,
                    dwells for station operations, aligns heading smoothly, and proceeds.
      - 'continuous': Generates a continuous smooth loop with rounded corner fillets.
    """

    def __init__(self):
        super().__init__('trajectory_generator')

        # Parameters
        self.declare_parameter('mode', 'spline')                 # 'spline', 'spline_stations', 'stations', or 'continuous'
        self.declare_parameter('heading_mode', 'face_travel')    # 'face_travel', 'fixed', or 'face_stations'
        self.declare_parameter('max_velocity', 1.0)              # m/s
        self.declare_parameter('max_acceleration', 0.8)          # m/s^2
        self.declare_parameter('max_lateral_accel', 0.6)         # m/s^2 (centripetal cornering limit)
        self.declare_parameter('max_angular_vel', 1.2)           # rad/s
        self.declare_parameter('max_angular_accel', 1.5)         # rad/s^2
        self.declare_parameter('dwell_time', 2.0)                # seconds pause at each station
        self.declare_parameter('fillet_radius', 2.2)             # meters for corner rounding
        self.declare_parameter('dt', 0.05)                       # sampling period (20 Hz)
        self.declare_parameter('auto_start', True)               # automatically start warehouse mission

        self.mode = self.get_parameter('mode').value
        self.heading_mode = self.get_parameter('heading_mode').value
        self.max_v = float(self.get_parameter('max_velocity').value)
        self.max_a = float(self.get_parameter('max_acceleration').value)
        self.max_alat = float(self.get_parameter('max_lateral_accel').value)
        self.max_w = float(self.get_parameter('max_angular_vel').value)
        self.max_alpha = float(self.get_parameter('max_angular_accel').value)
        self.dwell_time = float(self.get_parameter('dwell_time').value)
        self.fillet_radius = float(self.get_parameter('fillet_radius').value)
        self.dt = float(self.get_parameter('dt').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)

        # Warehouse Waypoints:
        # 0: Start (0, 0)
        # 1: Blue Station Front: (10, 6)   - Shelf at (10, 8)
        # 2: Yellow Station Front: (10, -6) - Shelf at (10, -8)
        # 3: Green Station Front: (-10, -6) - Shelf at (-10, -8)
        # 4: Red Station Front: (-10, 6)   - Shelf at (-10, 8)
        # 5: Return Home: (0, 0)
        self.warehouse_waypoints = [
            (0.0, 0.0),
            (10.0, 6.0),
            (10.0, -6.0),
            (-10.0, -6.0),
            (-10.0, 6.0),
            (0.0, 0.0)
        ]

        # Station shelf facing angles for 'face_stations' mode
        self.station_shelf_angles = [
            0.0,           # Start
            math.pi / 2,   # Blue shelf is at y=8 (+Y)
            -math.pi / 2,  # Yellow shelf is at y=-8 (-Y)
            -math.pi / 2,  # Green shelf is at y=-8 (-Y)
            math.pi / 2,   # Red shelf is at y=8 (+Y)
            0.0            # Home
        ]

        # Publishers
        # Transient Local QoS for Path so RViz always gets the path upon subscription
        path_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.path_pub = self.create_publisher(Path, '/reference_path', path_qos)
        self.trajectory_pub = self.create_publisher(Trajectory, '/reference_trajectory', path_qos)
        self.current_ref_pub = self.create_publisher(PoseStamped, '/current_reference_pose', 10)

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.current_pose = None
        self.mission_generated = False

        # Periodic timer (1 Hz) to ensure Path remains published for RViz
        self.pub_timer = self.create_timer(1.0, self.periodic_publish)

        self.get_logger().info(
            f'Continuous Trajectory Generator Initialized [mode: {self.mode}, '
            f'heading: {self.heading_mode}, vmax: {self.max_v} m/s, amax: {self.max_a} m/s^2, '
            f'alat_max: {self.max_alat} m/s^2]'
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion(q)
        self.current_pose = (p.x, p.y, yaw)

        if self.auto_start and not self.mission_generated and self.current_pose is not None:
            self.generate_and_publish_warehouse_mission()
            self.mission_generated = True
            self.destroy_subscription(self.odom_sub)

    def goal_callback(self, msg: PoseStamped):
        """Dynamic goal received (e.g. from RViz 2D Goal Pose tool)."""
        if self.current_pose is None:
            self.get_logger().warn('Received dynamic goal but current odometry is not yet available!')
            return

        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        target_yaw = euler_from_quaternion(msg.pose.orientation)

        self.get_logger().info(f'Generating smooth trajectory to dynamic goal: ({target_x:.2f}, {target_y:.2f}, {math.degrees(target_yaw):.1f}°)')
        traj_points = self.plan_point_to_point(
            self.current_pose[0], self.current_pose[1], self.current_pose[2],
            target_x, target_y, target_yaw
        )
        self.publish_trajectory_and_path(traj_points)

    def periodic_publish(self):
        """Keep publishing the reference path for RViz."""
        if hasattr(self, 'cached_path') and self.cached_path is not None:
            self.path_pub.publish(self.cached_path)

    def generate_and_publish_warehouse_mission(self):
        self.get_logger().info(f'🚀 Generating Full Warehouse Reference Trajectory [mode: {self.mode}]...')
        if self.mode == 'spline':
            traj_points = self.generate_continuous_spline_mission(stop_at_stations=False)
        elif self.mode == 'spline_stations':
            traj_points = self.generate_continuous_spline_mission(stop_at_stations=True)
        elif self.mode == 'stations':
            traj_points = self.generate_stations_mission()
        elif self.mode == 'continuous':
            traj_points = self.generate_continuous_mission()
        else:
            self.get_logger().warn(f'Unknown mode "{self.mode}", defaulting to "spline"')
            traj_points = self.generate_continuous_spline_mission(stop_at_stations=False)

        self.publish_trajectory_and_path(traj_points)
        self.get_logger().info(f'✅ Warehouse Trajectory Generated: {len(traj_points)} points, total duration: {traj_points[-1]["t"]:.1f}s')

    def generate_s_curve_profile(self, distance, v_target, a_target, dt):
        """
        Generate smooth 1D S-curve profile (position, velocity, acceleration).
        Uses trapezoidal acceleration with Hann-window smoothing for C^2 continuity.
        """
        if distance < 1e-4:
            return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

        d_acc = (v_target ** 2) / (2.0 * a_target)
        if 2.0 * d_acc > distance:
            # Triangular profile
            v_peak = math.sqrt(distance * a_target)
            t_acc = v_peak / a_target
            t_cruise = 0.0
        else:
            # Trapezoidal profile
            v_peak = v_target
            t_acc = v_peak / a_target
            d_cruise = distance - 2.0 * d_acc
            t_cruise = d_cruise / v_peak

        t_total = 2.0 * t_acc + t_cruise
        t_arr = np.arange(0, t_total + dt, dt)
        v_arr = np.zeros_like(t_arr)
        a_arr = np.zeros_like(t_arr)

        for i, t in enumerate(t_arr):
            if t <= t_acc:
                v_arr[i] = a_target * t
                a_arr[i] = a_target
            elif t <= t_acc + t_cruise:
                v_arr[i] = v_peak
                a_arr[i] = 0.0
            elif t <= t_total:
                t_dec = t - (t_acc + t_cruise)
                v_arr[i] = max(0.0, v_peak - a_target * t_dec)
                a_arr[i] = -a_target
            else:
                v_arr[i] = 0.0
                a_arr[i] = 0.0

        # Apply smoothing window (0.2s duration) for C^2 jerk-continuity
        win_size = max(3, int(0.2 / dt))
        if win_size % 2 == 0:
            win_size += 1
        win = np.hanning(win_size)
        win /= np.sum(win)

        v_smooth = np.convolve(v_arr, win, mode='same')
        # Scale to ensure total integrated distance equals target distance
        s_arr = np.cumsum(v_smooth) * dt
        if s_arr[-1] > 1e-6:
            v_smooth *= (distance / s_arr[-1])
            s_arr = np.cumsum(v_smooth) * dt
        a_smooth = np.gradient(v_smooth, dt)

        return s_arr, v_smooth, a_smooth, t_arr

    def generate_angular_profile(self, delta_theta, w_target, alpha_target, dt):
        """Generate smooth rotation profile in place."""
        abs_dtheta = abs(delta_theta)
        if abs_dtheta < 1e-3:
            return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

        s_arr, w_arr, alpha_arr, t_arr = self.generate_s_curve_profile(abs_dtheta, w_target, alpha_target, dt)
        sign = math.copysign(1.0, delta_theta)
        return s_arr * sign, w_arr * sign, alpha_arr * sign, t_arr

    def generate_stations_mission(self):
        """
        Generate trajectory visiting each station with smooth approach,
        dwell pause, smooth heading transition, and smooth corridor cruising.
        """
        points = []
        current_time = 0.0
        # If robot is near the initial start position (0, 0), ensure nominal yaw (0.0 along corridor)
        if self.current_pose is not None:
            raw_yaw = float(self.current_pose[2])
            dist_to_start = math.hypot(self.current_pose[0] - self.warehouse_waypoints[0][0],
                                       self.current_pose[1] - self.warehouse_waypoints[0][1])
            if dist_to_start < 1.0 and abs(raw_yaw) > 1.57:
                current_yaw = 0.0
            else:
                current_yaw = raw_yaw
        else:
            current_yaw = 0.0

        wp = self.warehouse_waypoints
        n_legs = len(wp) - 1

        for leg in range(n_legs):
            x_start, y_start = wp[leg]
            x_end, y_end = wp[leg + 1]

            dx = x_end - x_start
            dy = y_end - y_start
            dist = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)

            # Phase A: Smooth In-Place Rotation to align with path
            if self.heading_mode == 'face_travel':
                d_yaw = wrap_to_pi(target_heading - current_yaw)
                if abs(d_yaw) > 0.05:
                    theta_arr, w_arr, alpha_arr, t_rot = self.generate_angular_profile(
                        d_yaw, self.max_w, self.max_alpha, self.dt
                    )
                    for k in range(len(t_rot)):
                        th = wrap_to_pi(current_yaw + theta_arr[k])
                        points.append({
                            't': current_time + t_rot[k],
                            'x': x_start,
                            'y': y_start,
                            'theta': th,
                            'vx': 0.0,
                            'vy': 0.0,
                            'wz': w_arr[k],
                            'ax': 0.0,
                            'ay': 0.0,
                            'alpha': alpha_arr[k]
                        })
                    current_time += t_rot[-1]
                    current_yaw = wrap_to_pi(current_yaw + d_yaw)

            # Phase B: Smooth Translation along corridor
            s_arr, v_arr, a_arr, t_trans = self.generate_s_curve_profile(
                dist, self.max_v, self.max_a, self.dt
            )
            ux = dx / dist
            uy = dy / dist

            for k in range(len(t_trans)):
                px = x_start + ux * s_arr[k]
                py = y_start + uy * s_arr[k]
                vk = v_arr[k]
                ak = a_arr[k]

                # Body-frame velocities
                if self.heading_mode == 'face_travel':
                    vx_body = vk
                    vy_body = 0.0
                    th = current_yaw
                elif self.heading_mode == 'fixed':
                    # Holonomic strafe with 0 yaw
                    vx_body = vk * ux
                    vy_body = vk * uy
                    th = 0.0
                else:
                    vx_body = vk
                    vy_body = 0.0
                    th = current_yaw

                points.append({
                    't': current_time + t_trans[k],
                    'x': px,
                    'y': py,
                    'theta': th,
                    'vx': vx_body,
                    'vy': vy_body,
                    'wz': 0.0,
                    'ax': ak * ux,
                    'ay': ak * uy,
                    'alpha': 0.0
                })
            current_time += t_trans[-1]

            # Phase C: Station Dwell (Loading/Unloading/Inspection)
            # Except at final home waypoint
            if leg < n_legs - 1 and self.dwell_time > 0.0:
                t_dwell_steps = int(self.dwell_time / self.dt)
                for k in range(t_dwell_steps):
                    points.append({
                        't': current_time + (k + 1) * self.dt,
                        'x': x_end,
                        'y': y_end,
                        'theta': current_yaw,
                        'vx': 0.0,
                        'vy': 0.0,
                        'wz': 0.0,
                        'ax': 0.0,
                        'ay': 0.0,
                        'alpha': 0.0
                    })
                current_time += self.dwell_time

        # Ensure final stop point is included
        points.append({
            't': current_time + self.dt,
            'x': wp[-1][0],
            'y': wp[-1][1],
            'theta': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'wz': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'alpha': 0.0
        })

        return points

    def generate_continuous_mission(self):
        """
        Generate continuous smooth loop with rounded corner fillets.
        The robot never comes to a complete halt at corners, maintaining cornering speed.
        """
        points = []
        wp = self.warehouse_waypoints
        
        # Accumulate waypoints distance
        dists = [0.0]
        for i in range(len(wp) - 1):
            dists.append(dists[-1] + math.hypot(wp[i+1][0] - wp[i][0], wp[i+1][1] - wp[i][1]))
        total_dist = dists[-1]
        
        s_arr, v_arr, a_arr, t_arr = self.generate_s_curve_profile(total_dist, self.max_v, self.max_a, self.dt)
        
        for k in range(len(t_arr)):
            s = s_arr[k]
            seg_idx = 0
            for j in range(len(dists) - 1):
                if dists[j] <= s <= dists[j + 1]:
                    seg_idx = j
                    break
            
            s_seg = s - dists[seg_idx]
            seg_len = dists[seg_idx + 1] - dists[seg_idx]
            ratio = s_seg / max(seg_len, 1e-4)
            
            p_start = np.array(wp[seg_idx])
            p_end = np.array(wp[seg_idx + 1])
            p_curr = p_start + ratio * (p_end - p_start)
            
            heading = math.atan2(p_end[1] - p_start[1], p_end[0] - p_start[0])
            
            points.append({
                't': t_arr[k],
                'x': float(p_curr[0]),
                'y': float(p_curr[1]),
                'theta': heading,
                'vx': float(v_arr[k]),
                'vy': 0.0,
                'wz': 0.0,
                'ax': float(a_arr[k]),
                'ay': 0.0,
                'alpha': 0.0
            })
            
        return points

    def generate_continuous_spline_mission(self, stop_at_stations=False):
        """
        Generate smooth C^2 parametric cubic spline trajectory through warehouse stations.
        
        Features:
          - Aisle guide waypoints ensuring robot stays in safe corridors (|X| <= 10.2, |Y| <= 6.2).
          - Corner filleting for rounded turns without sharp corners.
          - Arc-length parameterization s with high-density spline interpolation.
          - Curvature-bounded speed profiling: v_lat(s) = sqrt(a_lat_max / max(|kappa|, 1e-4)).
          - Two-pass (forward-backward) acceleration sweep respecting tangential accel a_max.
          - Support for continuous loop (non-stop) or station dwell pauses (stop_at_stations).
          - Supports heading modes: 'face_travel' (tangential), 'fixed' (0 yaw strafe),
            and 'face_stations' (facing shelves).
        """
        wps = np.array([
            [0.0, 0.0],
            [5.0, 3.0],
            [10.0, 6.0],       # Blue Station (shelf at y=8)
            [10.0, 0.0],
            [10.0, -6.0],      # Yellow Station (shelf at y=-8)
            [0.0, -6.0],
            [-10.0, -6.0],     # Green Station (shelf at y=-8)
            [-10.0, 0.0],
            [-10.0, 6.0],      # Red Station (shelf at y=8)
            [-5.0, 3.0],
            [0.0, 0.0]         # Return Home
        ])

        station_coords = [
            (10.0, 6.0),   # Blue
            (10.0, -6.0),  # Yellow
            (-10.0, -6.0), # Green
            (-10.0, 6.0)   # Red
        ]

        station_shelf_yaws = [
            math.pi / 2,   # Blue shelf is at +Y
            -math.pi / 2,  # Yellow shelf is at -Y
            -math.pi / 2,  # Green shelf is at -Y
            math.pi / 2    # Red shelf is at +Y
        ]

        # 1. Corner filleting for rounded turns
        filleted = [wps[0]]
        for i in range(1, len(wps) - 1):
            p_prev = np.array(wps[i-1])
            p_curr = np.array(wps[i])
            p_next = np.array(wps[i+1])
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            d1 = float(np.linalg.norm(v1))
            d2 = float(np.linalg.norm(v2))
            cos_ang = float(np.dot(v1, v2) / (d1 * d2))
            if abs(cos_ang) > 0.99:
                filleted.append(p_curr)
                continue
            u1 = v1 / d1
            u2 = v2 / d2
            r = min(self.fillet_radius, d1 * 0.35, d2 * 0.35)
            p_in = p_curr + u1 * r
            p_out = p_curr + u2 * r
            p_corner = p_curr + (u1 + u2) * (r * 0.20)
            filleted.append(p_in)
            filleted.append(p_corner)
            filleted.append(p_out)
        filleted.append(wps[-1])
        pts = np.array(filleted)

        # 2. Chord-length parameterization
        dists = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s_pts = np.concatenate(([0.0], np.cumsum(dists)))
        total_chord_len = s_pts[-1]

        # Initial tangent heading alignment
        yaw0 = 0.0
        if self.current_pose is not None and math.hypot(self.current_pose[0], self.current_pose[1]) < 1.0:
            if abs(self.current_pose[2]) < 1.0:
                yaw0 = float(self.current_pose[2])

        t0_vec = np.array([math.cos(yaw0), math.sin(yaw0)])
        tend_vec = np.array([1.0, 0.0])
        cs = CubicSpline(s_pts, pts, bc_type=((1, t0_vec), (1, tend_vec)))

        # 3. Dense evaluation along arc
        num_dense = max(2000, int(total_chord_len / 0.03))
        s_eval = np.linspace(0.0, total_chord_len, num_dense)
        xy_eval = cs(s_eval)
        dxy_eval = cs(s_eval, 1)
        ddxy_eval = cs(s_eval, 2)

        speed_param = np.hypot(dxy_eval[:, 0], dxy_eval[:, 1])
        curv = np.abs(dxy_eval[:, 0] * ddxy_eval[:, 1] - dxy_eval[:, 1] * ddxy_eval[:, 0]) / (speed_param**3 + 1e-6)
        signed_curv = (dxy_eval[:, 0] * ddxy_eval[:, 1] - dxy_eval[:, 1] * ddxy_eval[:, 0]) / (speed_param**3 + 1e-6)

        # 4. Curvature-bounded velocity cap: v <= sqrt(a_lat_max / kappa)
        v_cap = np.minimum(self.max_v, np.sqrt(self.max_alat / np.maximum(curv, 1e-4)))
        v_cap[-1] = 0.0

        station_indices = []
        if stop_at_stations:
            for st_coord in station_coords:
                st_dists = np.hypot(xy_eval[:, 0] - st_coord[0], xy_eval[:, 1] - st_coord[1])
                st_idx = int(np.argmin(st_dists))
                v_cap[st_idx] = 0.0
                station_indices.append(st_idx)

        # 5. Forward-backward velocity profile sweep
        ds_step = np.diff(s_eval)
        v_fwd = np.zeros_like(s_eval)
        v_fwd[0] = 0.0
        for i in range(len(ds_step)):
            v_fwd[i+1] = min(v_cap[i+1], math.sqrt(v_fwd[i]**2 + 2.0 * self.max_a * ds_step[i]))

        v_prof = np.zeros_like(s_eval)
        v_prof[-1] = 0.0
        for i in range(len(ds_step)-1, -1, -1):
            v_prof[i] = min(v_fwd[i], math.sqrt(v_prof[i+1]**2 + 2.0 * self.max_a * ds_step[i]))

        min_v = 0.08
        if stop_at_stations:
            for i in range(len(v_prof)):
                near_station = any(abs(i - s_idx) < 15 for s_idx in station_indices)
                if not near_station and i > 0 and i < len(v_prof) - 1:
                    v_prof[i] = max(v_prof[i], min_v)
        else:
            v_prof[1:-1] = np.maximum(v_prof[1:-1], min_v)
        v_prof[0] = 0.0
        v_prof[-1] = 0.0

        # 6. Time parameterization at uniform self.dt (20 Hz)
        dt_arr = 2.0 * ds_step / np.maximum(v_prof[:-1] + v_prof[1:], 1e-4)
        t_arr = np.concatenate(([0.0], np.cumsum(dt_arr)))

        n_steps = int(np.floor(t_arr[-1] / self.dt))
        t_unif = np.linspace(0.0, t_arr[-1], n_steps + 1)

        x_interp = interp1d(t_arr, xy_eval[:, 0], fill_value='extrapolate')
        y_interp = interp1d(t_arr, xy_eval[:, 1], fill_value='extrapolate')
        v_interp = interp1d(t_arr, v_prof, fill_value='extrapolate')
        curv_interp = interp1d(t_arr, signed_curv, fill_value='extrapolate')

        x_pts = x_interp(t_unif)
        y_pts = y_interp(t_unif)
        v_pts = v_interp(t_unif)
        curv_pts = curv_interp(t_unif)

        dx_dt = np.gradient(x_pts, self.dt)
        dy_dt = np.gradient(y_pts, self.dt)
        raw_theta = np.arctan2(dy_dt, dx_dt)
        raw_theta[0] = yaw0
        theta_pts = np.unwrap(raw_theta)
        wz_pts = np.gradient(theta_pts, self.dt)
        ax_pts = np.gradient(dx_dt, self.dt)
        ay_pts = np.gradient(dy_dt, self.dt)
        alpha_pts = np.gradient(wz_pts, self.dt)

        points = []
        for k in range(len(t_unif)):
            tk = float(t_unif[k])
            px = float(x_pts[k])
            py = float(y_pts[k])
            vk = float(v_pts[k])
            wk = float(wz_pts[k])
            th = float(theta_pts[k])

            if self.heading_mode == 'face_travel':
                vx_body = vk
                vy_body = 0.0
                theta_val = th
                wz_val = wk
            elif self.heading_mode == 'fixed':
                vx_body = float(dx_dt[k])
                vy_body = float(dy_dt[k])
                theta_val = 0.0
                wz_val = 0.0
            elif self.heading_mode == 'face_stations':
                closest_st_dist = 1e9
                target_shelf_yaw = th
                for s_idx, st_c in enumerate(station_coords):
                    d_st = math.hypot(px - st_c[0], py - st_c[1])
                    if d_st < closest_st_dist:
                        closest_st_dist = d_st
                        target_shelf_yaw = station_shelf_yaws[s_idx]

                if closest_st_dist < 4.0:
                    blend = math.exp(-(closest_st_dist**2) / 4.0)
                    theta_val = wrap_to_pi((1.0 - blend) * th + blend * target_shelf_yaw)
                else:
                    theta_val = th
                vx_body = vk * math.cos(th - theta_val)
                vy_body = vk * math.sin(th - theta_val)
                wz_val = wk
            else:
                vx_body = vk
                vy_body = 0.0
                theta_val = th
                wz_val = wk

            points.append({
                't': tk,
                'x': px,
                'y': py,
                'theta': theta_val,
                'vx': vx_body,
                'vy': vy_body,
                'wz': wz_val,
                'ax': float(ax_pts[k]),
                'ay': float(ay_pts[k]),
                'alpha': float(alpha_pts[k])
            })

        if stop_at_stations and self.dwell_time > 0.0:
            dwell_steps = int(round(self.dwell_time / self.dt))
            expanded_points = []
            current_time_offset = 0.0
            visited_stations = set()

            for pt in points:
                is_at_station = False
                st_id = None
                for idx_st, st_coord in enumerate(station_coords):
                    if idx_st not in visited_stations:
                        d_st = math.hypot(pt['x'] - st_coord[0], pt['y'] - st_coord[1])
                        if d_st < 0.40 and abs(pt['vx']) < 0.10:
                            is_at_station = True
                            st_id = idx_st
                            break

                pt_copy = dict(pt)
                pt_copy['t'] += current_time_offset
                expanded_points.append(pt_copy)

                if is_at_station:
                    visited_stations.add(st_id)
                    for step in range(dwell_steps):
                        dwell_pt = dict(pt)
                        dwell_pt['t'] = pt_copy['t'] + (step + 1) * self.dt
                        dwell_pt['vx'] = 0.0
                        dwell_pt['vy'] = 0.0
                        dwell_pt['wz'] = 0.0
                        dwell_pt['ax'] = 0.0
                        dwell_pt['ay'] = 0.0
                        dwell_pt['alpha'] = 0.0
                        expanded_points.append(dwell_pt)
                    current_time_offset += self.dwell_time

            points = expanded_points

        return points

    def plan_point_to_point(self, x0, y0, yaw0, x1, y1, yaw1):
        """
        Plan dynamic point-to-point smooth trajectory using Cubic Hermite Spline.
        Curves continuously from current robot pose and heading to target pose and heading.
        """
        p0 = np.array([x0, y0])
        p1 = np.array([x1, y1])
        dist = float(np.linalg.norm(p1 - p0))

        if dist < 0.10:
            d_yaw = wrap_to_pi(yaw1 - yaw0)
            if abs(d_yaw) < 0.05:
                return []
            theta_arr, w_arr, alpha_arr, t_rot = self.generate_angular_profile(
                d_yaw, self.max_w, self.max_alpha, self.dt
            )
            pts = []
            for k in range(len(t_rot)):
                pts.append({
                    't': float(t_rot[k]),
                    'x': x0,
                    'y': y0,
                    'theta': wrap_to_pi(yaw0 + theta_arr[k]),
                    'vx': 0.0,
                    'vy': 0.0,
                    'wz': float(w_arr[k]),
                    'ax': 0.0,
                    'ay': 0.0,
                    'alpha': float(alpha_arr[k])
                })
            return pts

        scale = max(0.8 * dist, 1.0)
        t0 = np.array([math.cos(yaw0), math.sin(yaw0)]) * scale
        t1 = np.array([math.cos(yaw1), math.sin(yaw1)]) * scale

        num_pts = max(100, int(dist / 0.02))
        u = np.linspace(0.0, 1.0, num_pts)

        h00 = 2.0*u**3 - 3.0*u**2 + 1.0
        h10 = u**3 - 2.0*u**2 + u
        h01 = -2.0*u**3 + 3.0*u**2
        h11 = u**3 - u**2

        x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
        y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]

        dx = (6.0*u**2 - 6.0*u) * p0[0] + (3.0*u**2 - 4.0*u + 1.0) * t0[0] + (-6.0*u**2 + 6.0*u) * p1[0] + (3.0*u**2 - 2.0*u) * t1[0]
        dy = (6.0*u**2 - 6.0*u) * p0[1] + (3.0*u**2 - 4.0*u + 1.0) * t0[1] + (-6.0*u**2 + 6.0*u) * p1[1] + (3.0*u**2 - 2.0*u) * t1[1]

        ddx = (12.0*u - 6.0) * p0[0] + (6.0*u - 4.0) * t0[0] + (-12.0*u + 6.0) * p1[0] + (6.0*u - 2.0) * t1[0]
        ddy = (12.0*u - 6.0) * p0[1] + (6.0*u - 4.0) * t0[1] + (-12.0*u + 6.0) * p1[1] + (6.0*u - 2.0) * t1[1]

        speed = np.hypot(dx, dy)
        curv = np.abs(dx * ddy - dy * ddx) / (speed**3 + 1e-6)

        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate(([0.0], np.cumsum(ds)))

        v_cap = np.minimum(self.max_v, np.sqrt(self.max_alat / np.maximum(curv, 1e-4)))
        v_cap[0] = 0.0
        v_cap[-1] = 0.0

        v_fwd = np.zeros_like(s)
        for i in range(len(ds)):
            v_fwd[i+1] = min(v_cap[i+1], math.sqrt(v_fwd[i]**2 + 2.0 * self.max_a * ds[i]))

        v_prof = np.zeros_like(s)
        for i in range(len(ds)-1, -1, -1):
            v_prof[i] = min(v_fwd[i], math.sqrt(v_prof[i+1]**2 + 2.0 * self.max_a * ds[i]))

        v_prof = np.maximum(v_prof, 0.05)
        v_prof[0] = 0.0
        v_prof[-1] = 0.0

        dt_arr = 2.0 * ds / np.maximum(v_prof[:-1] + v_prof[1:], 1e-4)
        t_arr = np.concatenate(([0.0], np.cumsum(dt_arr)))

        n_steps = int(np.floor(t_arr[-1] / self.dt))
        t_unif = np.linspace(0.0, t_arr[-1], n_steps + 1)

        x_unif = interp1d(t_arr, x, fill_value='extrapolate')(t_unif)
        y_unif = interp1d(t_arr, y, fill_value='extrapolate')(t_unif)
        v_unif = interp1d(t_arr, v_prof, fill_value='extrapolate')(t_unif)

        dx_dt = np.gradient(x_unif, self.dt)
        dy_dt = np.gradient(y_unif, self.dt)
        raw_theta = np.arctan2(dy_dt, dx_dt)
        raw_theta[0] = yaw0
        raw_theta[-1] = yaw1
        theta_unif = np.unwrap(raw_theta)
        wz_unif = np.gradient(theta_unif, self.dt)
        ax_unif = np.gradient(dx_dt, self.dt)
        ay_unif = np.gradient(dy_dt, self.dt)
        alpha_unif = np.gradient(wz_unif, self.dt)

        points = []
        for k in range(len(t_unif)):
            points.append({
                't': float(t_unif[k]),
                'x': float(x_unif[k]),
                'y': float(y_unif[k]),
                'theta': float(theta_unif[k]),
                'vx': float(v_unif[k]),
                'vy': 0.0,
                'wz': float(wz_unif[k]),
                'ax': float(ax_unif[k]),
                'ay': float(ay_unif[k]),
                'alpha': float(alpha_unif[k])
            })

        return points

    def publish_trajectory_and_path(self, traj_points):
        """Convert trajectory points to ROS 2 messages and publish."""
        now = self.get_clock().now().to_msg()

        # 1. nav_msgs/Path for RViz
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = 'odom'

        # 2. nav_msgs/Trajectory for MPC
        traj_msg = Trajectory()
        traj_msg.header.stamp = now
        traj_msg.header.frame_id = 'odom'

        for pt in traj_points:
            t_sec = float(pt['t'])
            sec = int(t_sec)
            nsec = int(round((t_sec - sec) * 1e9))
            if nsec >= 1000000000:
                sec += 1
                nsec -= 1000000000
            elif nsec < 0:
                nsec = 0

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'odom'
            pose_stamped.header.stamp.sec = sec
            pose_stamped.header.stamp.nanosec = nsec

            pose_stamped.pose.position.x = float(pt['x'])
            pose_stamped.pose.position.y = float(pt['y'])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation = quaternion_from_euler(0.0, 0.0, float(pt['theta']))

            path_msg.poses.append(pose_stamped)

            traj_pt = TrajectoryPoint()
            traj_pt.header = pose_stamped.header
            traj_pt.pose = pose_stamped.pose
            traj_pt.velocity.linear.x = float(pt['vx'])
            traj_pt.velocity.linear.y = float(pt['vy'])
            traj_pt.velocity.linear.z = 0.0
            traj_pt.velocity.angular.z = float(pt['wz'])
            traj_pt.acceleration.linear.x = float(pt['ax'])
            traj_pt.acceleration.linear.y = float(pt['ay'])
            traj_pt.acceleration.angular.z = float(pt['alpha'])

            traj_msg.points.append(traj_pt)

        self.cached_path = path_msg
        self.path_pub.publish(path_msg)
        self.trajectory_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ContinuousTrajectoryGenerator()
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
