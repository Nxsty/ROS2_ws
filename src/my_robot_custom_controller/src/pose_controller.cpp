#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

/**
 * @brief Controlador de Posición (x, y, yaw) para Robot Omnidireccional Mecanum
 * 
 * Implementa:
 * 1. Cinemática Inversa de Pose (Global -> Local):
 *    Transforma el error del marco global (odom) al marco local del robot (base_footprint)
 *    mediante la matriz de rotación transpuesta R(yaw)^T:
 *    [e_x]   = [ cos(yaw)   sin(yaw)  0 ] [ x_goal - x_curr ]
 *    [e_y]   = [-sin(yaw)   cos(yaw)  0 ] [ y_goal - y_curr ]
 *    [e_yaw] = [    0          0      1 ] [ yaw_goal - yaw_curr ]
 * 
 * 2. Ley de Control Holonómica en Marco Local:
 *    v_x_cmd   = Kp_x   * e_x
 *    v_y_cmd   = Kp_y   * e_y
 *    omega_cmd = Kp_yaw * e_yaw
 * 
 * 3. Cinemática Directa de Pose (Local -> Global):
 *    Calcula la derivada temporal de la pose en el marco inercial global:
 *    [X_dot]   = [cos(yaw) -sin(yaw)  0] [v_x]
 *    [Y_dot]   = [sin(yaw)  cos(yaw)  0] [v_y]
 *    [yaw_dot] = [   0         0      1] [omega_z]
 * 
 * 4. Cinemática de Ruedas Mecanum (Inversa y Directa) como funciones auxiliares.
 */
class PoseController : public rclcpp::Node
{
public:
    PoseController()
    : Node("pose_controller"),
      has_goal_(false),
      has_odom_(false),
      has_goal_yaw_(false),
      goal_yaw_(0.0)
    {
        // --- Parámetros de Control ---
        this->declare_parameter<double>("kp_x", 1.5);
        this->declare_parameter<double>("kp_y", 1.5);
        this->declare_parameter<double>("kp_yaw", 2.0);
        this->declare_parameter<double>("dist_tolerance", 0.35); // Sincronizado con trajectory_manager
        this->declare_parameter<double>("yaw_tolerance", 0.10);
        this->declare_parameter<double>("max_linear_vel", 1.20);
        this->declare_parameter<double>("max_angular_vel", 1.50);
        this->declare_parameter<bool>("control_yaw", true);
        this->declare_parameter<bool>("face_direction_of_travel", true);

        // Parámetros de aceleración y límites de rueda para prevenir patinamiento
        this->declare_parameter<double>("max_linear_accel", 1.0);  // m/s^2
        this->declare_parameter<double>("max_angular_accel", 1.8); // rad/s^2
        this->declare_parameter<double>("max_wheel_speed", 20.0);  // rad/s

        // Parámetros físicos del robot mecanum (para cinemática de ruedas)
        this->declare_parameter<double>("wheel_radius", 0.1);
        this->declare_parameter<double>("wheel_separation_x", 0.272);
        this->declare_parameter<double>("wheel_separation_y", 0.225);

        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation_x", wheel_separation_x_);
        this->get_parameter("wheel_separation_y", wheel_separation_y_);

        // --- Suscripciones ---
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&PoseController::goal_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PoseController::odom_callback, this, std::placeholders::_1));

        // --- Publicadores ---
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        global_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/pose_controller/global_vel_projected", 10);

        RCLCPP_INFO(this->get_logger(), 
            "==========================================================\n"
            "   Pose Controller Mecanum (Holonómico) Inicializado     \n"
            "   Cinemática Inversa y Directa Activas (SE(2))          \n"
            "==========================================================");
    }

    /**
     * @brief Cinemática Inversa de Pose: Transforma un vector de error global al marco local del robot.
     */
    static void inverse_pose_kinematics(
        double delta_x, double delta_y, double delta_yaw, double current_yaw,
        double &local_ex, double &local_ey, double &local_eyaw)
    {
        // R(yaw)^T * [delta_x, delta_y]^T
        local_ex =  std::cos(current_yaw) * delta_x + std::sin(current_yaw) * delta_y;
        local_ey = -std::sin(current_yaw) * delta_x + std::cos(current_yaw) * delta_y;
        local_eyaw = delta_yaw;
    }

    /**
     * @brief Cinemática Directa de Pose: Transforma velocidades locales del robot al marco global.
     */
    static void forward_pose_kinematics(
        double vx_local, double vy_local, double wz_local, double current_yaw,
        double &x_dot_global, double &y_dot_global, double &yaw_dot_global)
    {
        // R(yaw) * [vx_local, vy_local]^T
        x_dot_global   = std::cos(current_yaw) * vx_local - std::sin(current_yaw) * vy_local;
        y_dot_global   = std::sin(current_yaw) * vx_local + std::cos(current_yaw) * vy_local;
        yaw_dot_global = wz_local;
    }

    /**
     * @brief Cinemática Inversa de Ruedas: Convierte (vx, vy, wz) a velocidades de cada rueda mecanum.
     */
    static void inverse_wheel_kinematics(
        double vx, double vy, double wz, double R, double LxLy,
        double &w_fl, double &w_fr, double &w_bl, double &w_br)
    {
        w_fl = (vx - vy - LxLy * wz) / R;
        w_fr = (vx + vy + LxLy * wz) / R;
        w_bl = (vx + vy - LxLy * wz) / R;
        w_br = (vx - vy + LxLy * wz) / R;
    }

    /**
     * @brief Cinemática Directa de Ruedas: Convierte velocidades de 4 ruedas a (vx, vy, wz).
     */
    static void forward_wheel_kinematics(
        double w_fl, double w_fr, double w_bl, double w_br, double R, double LxLy,
        double &vx, double &vy, double &wz)
    {
        vx = (R / 4.0) * (w_fl + w_fr + w_bl + w_br);
        vy = (R / 4.0) * (-w_fl + w_fr + w_bl - w_br);
        wz = (R / (4.0 * LxLy)) * (-w_fl + w_fr - w_bl + w_br);
    }

private:
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        goal_pose_ = *msg;
        has_goal_ = true;

        tf2::Quaternion q(
            goal_pose_.pose.orientation.x,
            goal_pose_.pose.orientation.y,
            goal_pose_.pose.orientation.z,
            goal_pose_.pose.orientation.w);

        // Extraer ángulo yaw objetivo si el cuaternión es válido
        if (q.length2() > 0.01) {
            double roll_g, pitch_g;
            tf2::Matrix3x3(q).getRPY(roll_g, pitch_g, goal_yaw_);
            has_goal_yaw_ = true;
        } else {
            has_goal_yaw_ = false;
        }

        RCLCPP_INFO(this->get_logger(), 
            "🎯 Nuevo Objetivo Recibido: x=%.2f, y=%.2f, yaw=%.2f rad (%.1f°)", 
            msg->pose.position.x, msg->pose.position.y, 
            goal_yaw_, goal_yaw_ * 180.0 / M_PI);
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_odom_ = *msg;
        has_odom_ = true;
        
        compute_control_law();
    }

    void compute_control_law()
    {
        auto cmd = geometry_msgs::msg::Twist();

        if (!has_goal_) {
            prev_vx_ = 0.0;
            prev_vy_ = 0.0;
            prev_wz_ = 0.0;
            cmd.linear.x = 0.0;
            cmd.linear.y = 0.0;
            cmd.angular.z = 0.0;
            cmd_vel_pub_->publish(cmd);
            return;
        }

        // 1. Pose actual del robot obtenida de la odometría
        double curr_x = current_odom_.pose.pose.position.x;
        double curr_y = current_odom_.pose.pose.position.y;
        
        tf2::Quaternion q(
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z,
            current_odom_.pose.pose.orientation.w);
        double roll, pitch, curr_yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, curr_yaw);

        // 2. Pose objetivo
        double goal_x = goal_pose_.pose.position.x;
        double goal_y = goal_pose_.pose.position.y;

        // 3. Error en el marco global (Inercial)
        double dx_global = goal_x - curr_x;
        double dy_global = goal_y - curr_y;
        double distance_error = std::sqrt(dx_global * dx_global + dy_global * dy_global);

        // Parámetros
        double dist_tol    = this->get_parameter("dist_tolerance").as_double();
        double yaw_tol     = this->get_parameter("yaw_tolerance").as_double();
        double kp_x        = this->get_parameter("kp_x").as_double();
        double kp_y        = this->get_parameter("kp_y").as_double();
        double kp_yaw      = this->get_parameter("kp_yaw").as_double();
        double max_v       = this->get_parameter("max_linear_vel").as_double();
        double max_w       = this->get_parameter("max_angular_vel").as_double();
        bool ctrl_yaw      = this->get_parameter("control_yaw").as_bool();
        bool face_travel   = this->get_parameter("face_direction_of_travel").as_bool();

        // Determinar orientación deseada
        double target_yaw = curr_yaw;
        if (ctrl_yaw) {
            if (face_travel && distance_error > dist_tol) {
                // Orientar hacia el frente de la trayectoria para visión/LiDAR
                target_yaw = std::atan2(dy_global, dx_global);
            } else if (has_goal_yaw_) {
                target_yaw = goal_yaw_;
            }
        }

        double dyaw_global = target_yaw - curr_yaw;
        // Normalización angular a [-pi, pi]
        dyaw_global = std::atan2(std::sin(dyaw_global), std::cos(dyaw_global));

        // 4. Verificación de llegada al objetivo
        bool pos_reached = (distance_error <= dist_tol);
        bool yaw_reached = (!ctrl_yaw || face_travel || !has_goal_yaw_ || std::abs(dyaw_global) <= yaw_tol);

        if (pos_reached && yaw_reached) {
            RCLCPP_INFO(this->get_logger(), 
                "✅ ¡Objetivo Alcanzado! Error pos: %.3fm <= %.3fm. Manteniendo posición.",
                distance_error, dist_tol);
            has_goal_ = false;
            prev_vx_ = 0.0;
            prev_vy_ = 0.0;
            prev_wz_ = 0.0;
            cmd.linear.x = 0.0;
            cmd.linear.y = 0.0;
            cmd.angular.z = 0.0;
            cmd_vel_pub_->publish(cmd);
            return;
        }

        // =========================================================================
        // CINEMÁTICA INVERSA DE POSE (Marco Global -> Marco Local del Robot)
        // e_local = R(curr_yaw)^T * delta_global
        // =========================================================================
        double local_ex = 0.0, local_ey = 0.0, local_eyaw = 0.0;
        inverse_pose_kinematics(dx_global, dy_global, dyaw_global, curr_yaw,
                                local_ex, local_ey, local_eyaw);

        // =========================================================================
        // LEY DE CONTROL HOLONÓMICA (Generación de velocidades en marco del robot)
        // =========================================================================
        double vx_cmd = kp_x * local_ex;
        double vy_cmd = kp_y * local_ey;
        double wz_cmd = ctrl_yaw ? (kp_yaw * local_eyaw) : 0.0;

        // Coordinación de orientación: Cuando se orienta hacia la trayectoria,
        // si el error de orientación es mayor a ~25° (0.45 rad), se prioriza rotar en el sitio
        // para encarar el pasillo antes de acelerar, evitando trayectorias en arco o derrapes.
        if (face_travel && distance_error > dist_tol) {
            if (std::abs(local_eyaw) > 0.45) {
                // Rotar en el sitio de forma limpia
                vx_cmd = 0.0;
                vy_cmd = 0.0;
            } else {
                // Alineado con el pasillo: modular avance con el coseno del error angular
                double heading_align = std::cos(local_eyaw);
                vx_cmd *= heading_align;
                vy_cmd *= heading_align;
            }
        }

        // Saturación vectorial de velocidad lineal (mantiene dirección de traslación pura)
        double v_mag = std::sqrt(vx_cmd * vx_cmd + vy_cmd * vy_cmd);
        if (v_mag > max_v) {
            vx_cmd = (vx_cmd / v_mag) * max_v;
            vy_cmd = (vy_cmd / v_mag) * max_v;
        }

        // Saturación de velocidad angular
        wz_cmd = std::clamp(wz_cmd, -max_w, max_w);

        // Desaturación cinemática de ruedas: asegura que ninguna rueda exceda su límite físico
        double max_wheel_speed = this->get_parameter("max_wheel_speed").as_double();
        double w_fl = 0.0, w_fr = 0.0, w_bl = 0.0, w_br = 0.0;
        inverse_wheel_kinematics(vx_cmd, vy_cmd, wz_cmd, wheel_radius_, 
                                wheel_separation_x_ + wheel_separation_y_,
                                w_fl, w_fr, w_bl, w_br);
        double max_w_req = std::max({std::abs(w_fl), std::abs(w_fr), std::abs(w_bl), std::abs(w_br)});
        if (max_w_req > max_wheel_speed && max_w_req > 1e-3) {
            double scale = max_wheel_speed / max_w_req;
            vx_cmd *= scale;
            vy_cmd *= scale;
            wz_cmd *= scale;
        }

        // Limitador de aceleración (Slew Rate Limiter) para evitar patinamiento y pérdidas de tracción
        double max_lin_accel = this->get_parameter("max_linear_accel").as_double();
        double max_ang_accel = this->get_parameter("max_angular_accel").as_double();
        rclcpp::Time current_time = this->now();
        if (last_control_time_.nanoseconds() > 0) {
            double dt = (current_time - last_control_time_).seconds();
            if (dt > 0.0 && dt < 0.5) {
                double max_dv = max_lin_accel * dt;
                double max_dw = max_ang_accel * dt;
                vx_cmd = std::clamp(vx_cmd, prev_vx_ - max_dv, prev_vx_ + max_dv);
                vy_cmd = std::clamp(vy_cmd, prev_vy_ - max_dv, prev_vy_ + max_dv);
                wz_cmd = std::clamp(wz_cmd, prev_wz_ - max_dw, prev_wz_ + max_dw);
            }
        }
        last_control_time_ = current_time;
        prev_vx_ = vx_cmd;
        prev_vy_ = vy_cmd;
        prev_wz_ = wz_cmd;

        // Asignar al comando Twist
        cmd.linear.x  = vx_cmd;
        cmd.linear.y  = vy_cmd;
        cmd.linear.z  = 0.0;
        cmd.angular.x = 0.0;
        cmd.angular.y = 0.0;
        cmd.angular.z = wz_cmd;

        // Publicar velocidades locales al controlador mecanum (/cmd_vel)
        cmd_vel_pub_->publish(cmd);

        // =========================================================================
        // CINEMÁTICA DIRECTA DE POSE (Marco Local -> Marco Global)
        // Proyección de velocidad en el espacio inercial global para monitoreo/validación
        // =========================================================================
        double x_dot = 0.0, y_dot = 0.0, yaw_dot = 0.0;
        forward_pose_kinematics(cmd.linear.x, cmd.linear.y, cmd.angular.z, curr_yaw,
                                x_dot, y_dot, yaw_dot);

        geometry_msgs::msg::Twist global_cmd;
        global_cmd.linear.x  = x_dot;
        global_cmd.linear.y  = y_dot;
        global_cmd.angular.z = yaw_dot;
        global_vel_pub_->publish(global_cmd);
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr global_vel_pub_;

    geometry_msgs::msg::PoseStamped goal_pose_;
    nav_msgs::msg::Odometry current_odom_;
    bool has_goal_;
    bool has_odom_;
    bool has_goal_yaw_;
    double goal_yaw_;

    double prev_vx_ = 0.0;
    double prev_vy_ = 0.0;
    double prev_wz_ = 0.0;
    rclcpp::Time last_control_time_{0, 0, RCL_ROS_TIME};

    double wheel_radius_ = 0.1;
    double wheel_separation_x_ = 0.272;
    double wheel_separation_y_ = 0.225;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
