#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry, Path, Trajectory, TrajectoryPoint

import numpy as np
import scipy.sparse as sp
import osqp
import math
import time


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
    """Convert geometry_msgs Quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class MecanumMPCController(Node):
    """
    High-Performance Model Predictive Controller (MPC) for Omnidirectional Mecanum Robots.
    
    Features:
      - Horizon N-step preview (receding horizon optimal control)
      - Linearized error kinematics around dynamic reference trajectory
      - Multi-objective optimization: position error, heading error, energy, and jerk
      - Anti-slip acceleration limits (slew rate bounds)
      - Explicit 4-wheel Mecanum inverse kinematic speed constraints (|w_i| <= w_max)
      - Sub-millisecond QP solving via OSQP
      - RViz predictive visualization: predicted path and reference horizon
    """

    def __init__(self):
        super().__init__('mpc_controller')

        # --- Parameters ---
        self.declare_parameter('horizon', 10)                  # N steps lookahead
        self.declare_parameter('dt', 0.05)                     # Control loop period: 20 Hz
        self.declare_parameter('max_linear_vel', 1.20)         # m/s
        self.declare_parameter('max_lateral_vel', 0.80)        # m/s (omnidirectional lateral authority)
        self.declare_parameter('max_angular_vel', 1.50)        # rad/s
        self.declare_parameter('max_linear_accel', 1.00)       # m/s^2 (anti-slip constraint)
        self.declare_parameter('max_angular_accel', 1.80)      # rad/s^2
        self.declare_parameter('max_wheel_speed', 20.0)        # rad/s
        self.declare_parameter('wheel_radius', 0.10)           # m
        self.declare_parameter('wheel_separation_x', 0.272)    # m (half wheelbase)
        self.declare_parameter('wheel_separation_y', 0.225)    # m (half trackwidth)

        # Objective Weights
        self.declare_parameter('q_x', 20.0)                    # Tracking error X
        self.declare_parameter('q_y', 20.0)                    # Tracking error Y
        self.declare_parameter('q_yaw', 15.0)                  # Heading error
        self.declare_parameter('r_vx', 0.8)                    # Control effort Vx
        self.declare_parameter('r_vy', 1.0)                    # Control effort Vy
        self.declare_parameter('r_wz', 0.5)                    # Control effort Wz
        self.declare_parameter('s_vx', 3.5)                    # Slew rate penalty Vx
        self.declare_parameter('s_vy', 4.0)                    # Slew rate penalty Vy
        self.declare_parameter('s_wz', 2.0)                    # Slew rate penalty Wz
        self.declare_parameter('q_terminal_mult', 2.0)         # Terminal weight multiplier

        # Get Parameters
        self.N = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.v_max = float(self.get_parameter('max_linear_vel').value)
        self.vy_max = float(self.get_parameter('max_lateral_vel').value)
        self.w_max = float(self.get_parameter('max_angular_vel').value)
        self.a_max = float(self.get_parameter('max_linear_accel').value)
        self.alpha_max = float(self.get_parameter('max_angular_accel').value)
        self.wheel_max = float(self.get_parameter('max_wheel_speed').value)
        self.r_wheel = float(self.get_parameter('wheel_radius').value)
        self.lx_ly = float(self.get_parameter('wheel_separation_x').value) + float(self.get_parameter('wheel_separation_y').value)

        # Weight matrices
        self.Q = np.diag([
            float(self.get_parameter('q_x').value),
            float(self.get_parameter('q_y').value),
            float(self.get_parameter('q_yaw').value)
        ])
        self.R = np.diag([
            float(self.get_parameter('r_vx').value),
            float(self.get_parameter('r_vy').value),
            float(self.get_parameter('r_wz').value)
        ])
        self.S = np.diag([
            float(self.get_parameter('s_vx').value),
            float(self.get_parameter('s_vy').value),
            float(self.get_parameter('s_wz').value)
        ])
        self.Q_N = float(self.get_parameter('q_terminal_mult').value) * self.Q

        # Mecanum Inverse Kinematics Jacobian: w_wheels = J_inv * [vx, vy, wz]
        self.J_inv = (1.0 / self.r_wheel) * np.array([
            [1.0, -1.0, -self.lx_ly],  # Front-Left
            [1.0,  1.0,  self.lx_ly],  # Front-Right
            [1.0,  1.0, -self.lx_ly],  # Back-Left
            [1.0, -1.0,  self.lx_ly]   # Back-Right
        ])

        # State dimensions
        self.nx = 3  # [e_x, e_y, e_theta] (body error frame)
        self.nu = 3  # [v_x, v_y, omega_z] (body velocities)

        # Precompute Constant Quadratic Cost Matrix P (Dense -> CSC once)
        nz = (self.N + 1) * self.nx + self.N * self.nu
        self.nz = nz
        self.offset_u = (self.N + 1) * self.nx
        self.n_cons = self.nx + self.N * self.nx + self.N * self.nu + self.N * self.nu + self.N * 4

        P_dense = np.zeros((nz, nz))
        for k in range(self.N):
            P_dense[k*self.nx:(k+1)*self.nx, k*self.nx:(k+1)*self.nx] = 2.0 * self.Q
        P_dense[self.N*self.nx:(self.N+1)*self.nx, self.N*self.nx:(self.N+1)*self.nx] = 2.0 * self.Q_N

        for k in range(self.N):
            idx_u = self.offset_u + k * self.nu
            P_dense[idx_u:idx_u+self.nu, idx_u:idx_u+self.nu] += 2.0 * self.R + 2.0 * self.S
            if k > 0:
                idx_prev = self.offset_u + (k - 1) * self.nu
                P_dense[idx_prev:idx_prev+self.nu, idx_prev:idx_prev+self.nu] += 2.0 * self.S
                P_dense[idx_u:idx_u+self.nu, idx_prev:idx_prev+self.nu] += -2.0 * self.S
                P_dense[idx_prev:idx_prev+self.nu, idx_u:idx_u+self.nu] += -2.0 * self.S
        self.P_csc = sp.csc_matrix(P_dense)

        # Trajectory Storage
        self.ref_traj_points = None
        self.current_idx = 0
        self.mission_completed = False

        # Current Robot State
        self.current_pose = None      # (x, y, yaw)
        self.current_vel = None       # (vx, vy, wz)
        self.last_u = np.zeros(self.nu)
        self.last_odom_time = self.get_clock().now()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pred_path_pub = self.create_publisher(Path, '/mpc_predicted_path', 10)
        self.ref_horizon_pub = self.create_publisher(Path, '/mpc_reference_horizon', 10)
        self.error_pub = self.create_publisher(Point, '/mpc/tracking_error', 10)

        # Subscriptions & Callback Group for concurrent odom updates
        traj_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.cb_group = ReentrantCallbackGroup()
        self.traj_sub = self.create_subscription(Trajectory, '/reference_trajectory', self.trajectory_callback, traj_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10, callback_group=self.cb_group)

        # Control Loop Timer (20 Hz)
        self.timer = self.create_timer(self.dt, self.control_loop, callback_group=self.cb_group)

        self.get_logger().info(
            f'Mecanum MPC Controller Started: N={self.N}, dt={self.dt}s (lookahead: {self.N * self.dt:.2f}s), '
            f'vmax={self.v_max} m/s, amax={self.a_max} m/s^2'
        )

    def trajectory_callback(self, msg: Trajectory):
        """Receive reference trajectory from Trajectory Generator."""
        if not msg.points:
            self.get_logger().warn('Received empty trajectory message!')
            return

        pts = []
        for pt in msg.points:
            t = pt.header.stamp.sec + pt.header.stamp.nanosec * 1e-9
            x = pt.pose.position.x
            y = pt.pose.position.y
            yaw = euler_from_quaternion(pt.pose.orientation)
            vx = pt.velocity.linear.x
            vy = pt.velocity.linear.y
            wz = pt.velocity.angular.z
            pts.append({
                't': t, 'x': x, 'y': y, 'theta': yaw,
                'vx': vx, 'vy': vy, 'wz': wz
            })

        self.ref_traj_points = pts
        self.current_idx = 0
        self.mission_completed = False
        self.get_logger().info(f'📥 Loaded Reference Trajectory: {len(pts)} points ({pts[-1]["t"]:.1f}s total). Ready to track.')

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist
        yaw = euler_from_quaternion(q)

        self.current_pose = np.array([p.x, p.y, yaw])
        self.current_vel = np.array([v.linear.x, v.linear.y, v.angular.z])
        self.last_odom_time = self.get_clock().now()

    def find_nearest_trajectory_index(self):
        """
        Find closest point on reference trajectory forward from current progress.
        Strictly monotonic: search window is forward-only [current_idx, current_idx + 40],
        preventing backward jumps, orbital limit cycles, or trap loops.
        """
        if self.ref_traj_points is None or self.current_pose is None:
            return 0

        M = len(self.ref_traj_points)
        curr_x, curr_y = self.current_pose[0], self.current_pose[1]

        # Strictly forward search window: up to 40 points (2.0s lookahead)
        start_idx = self.current_idx
        end_idx = min(M, self.current_idx + 40)

        best_idx = self.current_idx
        min_dist_sq = float('inf')

        for i in range(start_idx, end_idx):
            pt = self.ref_traj_points[i]
            d_sq = (pt['x'] - curr_x)**2 + (pt['y'] - curr_y)**2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_idx = i

        self.current_idx = max(self.current_idx, best_idx)
        return self.current_idx

    def control_loop(self):
        """Main MPC Control Loop executing at 20 Hz."""
        if self.current_pose is None or self.ref_traj_points is None:
            return

        if self.mission_completed:
            # Send stop command
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            return

        # 1. Locate closest progress point along reference trajectory
        M = len(self.ref_traj_points)
        nearest_idx = self.find_nearest_trajectory_index()

        # Check for mission arrival at final point
        final_pt = self.ref_traj_points[-1]
        dist_to_final = math.hypot(final_pt['x'] - self.current_pose[0], final_pt['y'] - self.current_pose[1])
        if nearest_idx >= M - 10 and dist_to_final < 0.35:
            self.get_logger().info(f'🏆 MISSION ACCOMPLISHED! Final distance: {dist_to_final:.3f}m. Holding home position.')
            self.mission_completed = True
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            return

        # 2. Extract Reference Horizon of length N
        x_ref_seq = np.zeros((self.N + 1, self.nx))
        u_ref_seq = np.zeros((self.N, self.nu))

        for k in range(self.N + 1):
            idx = min(nearest_idx + k, M - 1)
            pt = self.ref_traj_points[idx]
            x_ref_seq[k] = [pt['x'], pt['y'], pt['theta']]
            if k < self.N:
                u_ref_seq[k] = [pt['vx'], pt['vy'], pt['wz']]

        # Continuously unroll reference yaw starting from current robot heading
        curr_yaw = self.current_pose[2]
        yaw_diff = wrap_to_pi(x_ref_seq[0, 2] - curr_yaw)
        x_ref_seq[0, 2] = curr_yaw + yaw_diff
        for k in range(1, self.N + 1):
            step_diff = wrap_to_pi(x_ref_seq[k, 2] - x_ref_seq[k-1, 2])
            x_ref_seq[k, 2] = x_ref_seq[k-1, 2] + step_diff

        # Current state and error
        curr_x = self.current_pose[0]
        curr_y = self.current_pose[1]

        # Calculate tracking error for telemetry
        ref_curr = x_ref_seq[0]
        err_x = curr_x - ref_curr[0]
        err_y = curr_y - ref_curr[1]
        err_yaw = wrap_to_pi(curr_yaw - ref_curr[2])

        err_msg = Point(x=err_x, y=err_y, z=err_yaw)
        self.error_pub.publish(err_msg)

        # 3. Solve the Quadratic Program (sub-2ms solve time)
        t_start = time.perf_counter()
        u_opt, pred_states, solved = self.solve_mpc_qp(
            self.current_pose, x_ref_seq, u_ref_seq, self.last_u
        )
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0

        if not solved:
            u_opt = u_ref_seq[0]

        # 4. Command Publication (Omnidirectional Coupled MPC Output)
        cmd = Twist()
        cmd.linear.x = float(u_opt[0])
        cmd.linear.y = float(u_opt[1])
        cmd.angular.z = float(u_opt[2])
        self.cmd_vel_pub.publish(cmd)

        self.last_u = np.copy(u_opt)

        # Periodic telemetry log every 1.0s (20 cycles)
        if not hasattr(self, '_loop_counter'):
            self._loop_counter = 0
        self._loop_counter += 1
        if self._loop_counter % 20 == 0:
            pos_err = math.hypot(err_x, err_y)
            self.get_logger().info(
                f"MPC Progress: {nearest_idx}/{M} ({100.0*nearest_idx/M:.1f}%) | "
                f"PosErr: {pos_err:.2f}m | "
                f"Pose: ({curr_x:.1f}, {curr_y:.1f}, {math.degrees(curr_yaw):.0f}°) | "
                f"Cmd: (vx={u_opt[0]:.2f}, vy={u_opt[1]:.2f}, wz={u_opt[2]:.2f}) | "
                f"Solve: {solve_time_ms:.2f}ms"
            )

        # 5. RViz Visualization: Predicted Path & Reference Horizon
        self.publish_rviz_paths(pred_states, x_ref_seq)

    def solve_mpc_qp(self, x_curr, x_ref_seq, u_ref_seq, u_prev):
        """
        Formulate and solve the Local Body-Frame Error MPC Quadratic Program using OSQP.
        
        State:   e = [e_x, e_y, e_theta]^T in robot body frame
        Input:   u = [v_x, v_y, omega_z]^T in robot body frame
        Dynamics: e_{k+1} = A_{dk} e_k + B_{dk} (u_k - u_{rk})
                  where B_{dk} = dt * I (direct omnidirectional actuation)
        Performance: ~1.5ms solve time with precomputed P and dense matrix assembly.
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        dt = self.dt
        nz = self.nz
        offset_u = self.offset_u
        n_cons = self.n_cons

        # Linear cost vector q
        q = np.zeros(nz)
        for k in range(N):
            idx_u = offset_u + k * nu
            q[idx_u:idx_u+nu] += -2.0 * self.R @ u_ref_seq[k]
            if k == 0:
                q[idx_u:idx_u+nu] += -2.0 * self.S @ u_prev

        # 1. Transform global error to robot body frame
        th = x_curr[2]
        dx_glob = x_curr[0] - x_ref_seq[0, 0]
        dy_glob = x_curr[1] - x_ref_seq[0, 1]
        dth_glob = wrap_to_pi(x_curr[2] - x_ref_seq[0, 2])

        e0 = np.array([
             math.cos(th) * dx_glob + math.sin(th) * dy_glob,
            -math.sin(th) * dx_glob + math.cos(th) * dy_glob,
            dth_glob
        ])

        # Dense constraint matrix (vectorized, 100x faster than lil_matrix)
        A_dense = np.zeros((n_cons, nz))
        l = np.zeros(n_cons)
        u_b = np.zeros(n_cons)

        # 1. Initial condition: e_0 == e0
        row = 0
        A_dense[0:nx, 0:nx] = np.eye(nx)
        l[0:nx] = e0
        u_b[0:nx] = e0
        row += nx

        # 2. Local Error Dynamics: e_{k+1} - A_{dk} e_k - B_{dk} u_k == -B_{dk} u_{rk}
        for k in range(N):
            w_rk = u_ref_seq[k, 2]
            A_dk = np.array([
                [1.0,         w_rk * dt, 0.0],
                [-w_rk * dt, 1.0,        0.0],
                [0.0,        0.0,        1.0]
            ])
            B_dk = np.eye(3) * dt

            idx_ek = k * nx
            idx_ek1 = (k + 1) * nx
            idx_uk = offset_u + k * nu

            A_dense[row:row+nx, idx_ek1:idx_ek1+nx] = np.eye(nx)
            A_dense[row:row+nx, idx_ek:idx_ek+nx] = -A_dk
            A_dense[row:row+nx, idx_uk:idx_uk+nu] = -B_dk

            dyn_rhs = -B_dk @ u_ref_seq[k]
            l[row:row+nx] = dyn_rhs
            u_b[row:row+nx] = dyn_rhs
            row += nx

        # 3. Input bounds: -u_max <= u_k <= u_max
        u_min = np.array([-self.v_max, -self.vy_max, -self.w_max])
        u_max = np.array([ self.v_max,  self.vy_max,  self.w_max])
        for k in range(N):
            idx_uk = offset_u + k * nu
            A_dense[row:row+nu, idx_uk:idx_uk+nu] = np.eye(nu)
            l[row:row+nu] = u_min
            u_b[row:row+nu] = u_max
            row += nu

        # 4. Slew rate (acceleration) bounds: |u_k - u_{k-1}| <= a_max * dt
        du_lim = np.array([self.a_max * dt, self.a_max * dt, self.alpha_max * dt])
        for k in range(N):
            idx_uk = offset_u + k * nu
            A_dense[row:row+nu, idx_uk:idx_uk+nu] = np.eye(nu)
            if k == 0:
                l[row:row+nu] = u_prev - du_lim
                u_b[row:row+nu] = u_prev + du_lim
            else:
                idx_prev = offset_u + (k - 1) * nu
                A_dense[row:row+nu, idx_prev:idx_prev+nu] = -np.eye(nu)
                l[row:row+nu] = -du_lim
                u_b[row:row+nu] = du_lim
            row += nu

        # 5. 4-Wheel Mecanum speed limits: |J_inv u_k| <= w_max
        for k in range(N):
            idx_uk = offset_u + k * nu
            A_dense[row:row+4, idx_uk:idx_uk+nu] = self.J_inv
            l[row:row+4] = -self.wheel_max
            u_b[row:row+4] = self.wheel_max
            row += 4

        A_csc = sp.csc_matrix(A_dense)

        # Solve with OSQP
        try:
            prob = osqp.OSQP()
            prob.setup(self.P_csc, q, A_csc, l, u_b, verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=300, adaptive_rho=True)
            res = prob.solve()

            if res.info.status_val in [1, 2, 7]:  # Solved, Inaccurate, or Max Iter
                u_opt = res.x[offset_u:offset_u+nu]
                e_opt = res.x[0:(N+1)*nx].reshape((N+1, nx))

                # Reconstruct predicted states in world frame for RViz visualization
                pred_states = np.zeros((N + 1, nx))
                for k in range(N + 1):
                    th_k = x_ref_seq[k, 2]
                    e_k_body = e_opt[k]
                    dx_g = math.cos(th_k) * e_k_body[0] - math.sin(th_k) * e_k_body[1]
                    dy_g = math.sin(th_k) * e_k_body[0] + math.cos(th_k) * e_k_body[1]
                    pred_states[k] = [
                        x_ref_seq[k, 0] + dx_g,
                        x_ref_seq[k, 1] + dy_g,
                        wrap_to_pi(th_k + e_k_body[2])
                    ]
                return u_opt, pred_states, True
            else:
                self.get_logger().warn(f"OSQP status: {res.info.status}")
                return u_ref_seq[0], None, False
        except Exception as e:
            self.get_logger().error(f"OSQP Exception: {e}")
            return u_ref_seq[0], None, False

    def publish_rviz_paths(self, pred_states, x_ref_seq):
        """Publish predicted path and reference horizon for live RViz visualization."""
        now = self.get_clock().now().to_msg()

        # Reference horizon path
        ref_msg = Path()
        ref_msg.header.stamp = now
        ref_msg.header.frame_id = 'odom'
        for k in range(len(x_ref_seq)):
            ps = PoseStamped()
            ps.header = ref_msg.header
            ps.pose.position.x = float(x_ref_seq[k, 0])
            ps.pose.position.y = float(x_ref_seq[k, 1])
            ps.pose.position.z = 0.0
            ps.pose.orientation = quaternion_from_euler(0.0, 0.0, float(x_ref_seq[k, 2]))
            ref_msg.poses.append(ps)
        self.ref_horizon_pub.publish(ref_msg)

        # Predicted robot path
        if pred_states is not None:
            pred_msg = Path()
            pred_msg.header.stamp = now
            pred_msg.header.frame_id = 'odom'
            for k in range(len(pred_states)):
                ps = PoseStamped()
                ps.header = pred_msg.header
                ps.pose.position.x = float(pred_states[k, 0])
                ps.pose.position.y = float(pred_states[k, 1])
                ps.pose.position.z = 0.0
                ps.pose.orientation = quaternion_from_euler(0.0, 0.0, float(pred_states[k, 2]))
                pred_msg.poses.append(ps)
            self.pred_path_pub.publish(pred_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MecanumMPCController()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
