
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

class DiffDriveController : public rclcpp::Node
{
public:
    DiffDriveController()
    : Node("diff_drive_controller"),
      last_linear_vel_(0.0),
      last_angular_vel_(0.0),
      pose_x_(0.0),
      pose_y_(0.0),
      pose_theta_(0.0)
    {
        // --- Parámetros ---
        this->declare_parameter<double>("wheel_radius", 0.1);
        this->declare_parameter<double>("wheel_separation", 0.47);
        this->declare_parameter<std::string>("left_front_wheel_joint", "base_left_front_wheel_joint");
        this->declare_parameter<std::string>("left_back_wheel_joint", "base_left_back_wheel_joint");
        this->declare_parameter<std::string>("right_front_wheel_joint", "base_right_front_wheel_joint");
        this->declare_parameter<std::string>("right_back_wheel_joint", "base_right_back_wheel_joint");
        this->declare_parameter<std::string>("left_front_wheel_topic", "/base_left_front_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("left_back_wheel_topic", "/base_left_back_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("right_front_wheel_topic", "/base_right_front_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("right_back_wheel_topic", "/base_right_back_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "base_footprint");
        this->declare_parameter<bool>("publish_odom", true);
        this->declare_parameter<bool>("publish_tf", true);

        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation", wheel_separation_);
        this->get_parameter("left_front_wheel_joint", lf_joint_);
        this->get_parameter("left_back_wheel_joint", lb_joint_);
        this->get_parameter("right_front_wheel_joint", rf_joint_);
        this->get_parameter("right_back_wheel_joint", rb_joint_);
        this->get_parameter("odom_frame", odom_frame_);
        this->get_parameter("base_frame", base_frame_);
        this->get_parameter("publish_odom", publish_odom_);
        this->get_parameter("publish_tf", publish_tf_);

        // --- Suscripciones ---
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&DiffDriveController::cmd_vel_callback, this, std::placeholders::_1));

        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&DiffDriveController::joint_states_callback, this, std::placeholders::_1));

        // --- Publicadores ---
        fl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            this->get_parameter("left_front_wheel_topic").as_string(), 10);
        fr_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            this->get_parameter("right_front_wheel_topic").as_string(), 10);
        bl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            this->get_parameter("left_back_wheel_topic").as_string(), 10);
        br_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            this->get_parameter("right_back_wheel_topic").as_string(), 10);
        
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        measured_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/robot_measured_vel", 10);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // --- Timer del control loop (50 Hz) ---
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&DiffDriveController::control_loop, this));

        last_cmd_time_ = this->now();
        last_odom_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "DiffDriveController: Tasks 1.1 (Forward Kinematics) and 1.2 (Dead-reckoning) Ready.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        last_linear_vel_ = msg->linear.x;
        last_angular_vel_ = msg->angular.z;
        last_cmd_time_ = this->now();
    }

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        double w_lf=0, w_lb=0, w_rf=0, w_rb=0;
        bool f_lf=false, f_lb=false, f_rf=false, f_rb=false;

        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i].find(lf_joint_) != std::string::npos && msg->velocity.size() > i) { w_lf = msg->velocity[i]; f_lf = true; }
            else if (msg->name[i].find(lb_joint_) != std::string::npos && msg->velocity.size() > i) { w_lb = msg->velocity[i]; f_lb = true; }
            else if (msg->name[i].find(rf_joint_) != std::string::npos && msg->velocity.size() > i) { w_rf = msg->velocity[i]; f_rf = true; }
            else if (msg->name[i].find(rb_joint_) != std::string::npos && msg->velocity.size() > i) { w_rb = msg->velocity[i]; f_rb = true; }
        }

        if (f_lf && f_lb && f_rf && f_rb) {
            // --- Tarea 1.1: Cinemática Directa (Forward Kinematics) ---
            // Promedio de velocidades por lado (Skid-Steer model)
            double left_avg = (w_lf + w_lb) / 2.0;
            double right_avg = (w_rf + w_rb) / 2.0;

            // Velocidades del robot (v_x, omega_z)
            double vx = (wheel_radius_ / 2.0) * (right_avg + left_avg);
            double wz = (wheel_radius_ / wheel_separation_) * (right_avg - left_avg);

            // Publicar velocidad medida
            geometry_msgs::msg::Twist m_vel;
            m_vel.linear.x = vx;
            m_vel.angular.z = wz;
            measured_vel_pub_->publish(m_vel);

            // --- Tarea 1.2: Dead-reckoning (Odometer Integration) ---
            if (publish_odom_ || publish_tf_) {
                rclcpp::Time current_time = this->now();
                double dt = (current_time - last_odom_time_).seconds();
                last_odom_time_ = current_time;

                if (dt > 0.0) {
                    // Integración de Pose (Euler)
                    pose_x_ += vx * std::cos(pose_theta_) * dt;
                    pose_y_ += vx * std::sin(pose_theta_) * dt;
                    pose_theta_ += wz * dt;

                    // Normalización de ángulo (-pi a pi)
                    pose_theta_ = atan2(sin(pose_theta_), cos(pose_theta_));

                    tf2::Quaternion q;
                    q.setRPY(0, 0, pose_theta_);

                    // Publicar TF (Odom -> Base Footprint)
                    if (publish_tf_) {
                        geometry_msgs::msg::TransformStamped t;
                        t.header.stamp = current_time;
                        t.header.frame_id = odom_frame_;
                        t.child_frame_id = base_frame_;
                        t.transform.translation.x = pose_x_;
                        t.transform.translation.y = pose_y_;
                        t.transform.translation.z = 0.0;
                        t.transform.rotation.x = q.x();
                        t.transform.rotation.y = q.y();
                        t.transform.rotation.z = q.z();
                        t.transform.rotation.w = q.w();
                        tf_broadcaster_->sendTransform(t);
                    }

                    // Publicar Mensaje Odometry
                    if (publish_odom_) {
                        nav_msgs::msg::Odometry odom;
                        odom.header.stamp = current_time;
                        odom.header.frame_id = odom_frame_;
                        odom.child_frame_id = base_frame_;
                        
                        // Pose
                        odom.pose.pose.position.x = pose_x_;
                        odom.pose.pose.position.y = pose_y_;
                        odom.pose.pose.position.z = 0.0;
                        odom.pose.pose.orientation.x = q.x();
                        odom.pose.pose.orientation.y = q.y();
                        odom.pose.pose.orientation.z = q.z();
                        odom.pose.pose.orientation.w = q.w();

                        // Añadir Covarianzas (Incertidumbre) - Obligatorio para Tarea 1.2
                        // Pose Covariance
                        odom.pose.covariance[0]  = 0.001; // x
                        odom.pose.covariance[7]  = 0.001; // y
                        odom.pose.covariance[35] = 0.001; // yaw
                        
                        // Twist (Velocidad en el frame local base_link)
                        odom.twist.twist.linear.x = vx;
                        odom.twist.twist.angular.z = wz;
                        
                        // Twist Covariance
                        odom.twist.covariance[0]  = 0.001; // vx
                        odom.twist.covariance[35] = 0.001; // wz

                        odom_pub_->publish(odom);
                    }
                }
            }
        }
    }

    void control_loop()
    {
        double time_since_cmd = (this->now() - last_cmd_time_).seconds();
        
        // Timeout de seguridad: 0.4s
        if (time_since_cmd > 0.4) {
            last_linear_vel_ = 0.0;
            last_angular_vel_ = 0.0;
        }

        // --- Cinemática Inversa (Inverse Kinematics) ---
        double w_l = (last_linear_vel_ - last_angular_vel_ * wheel_separation_ / 2.0) / wheel_radius_;
        double w_r = (last_linear_vel_ + last_angular_vel_ * wheel_separation_ / 2.0) / wheel_radius_;

        std_msgs::msg::Float64 l_msg, r_msg;
        l_msg.data = w_l;
        r_msg.data = w_r;

        // Publicar velocidades a las 4 ruedas
        fl_wheel_pub_->publish(l_msg);
        bl_wheel_pub_->publish(l_msg);
        fr_wheel_pub_->publish(r_msg);
        br_wheel_pub_->publish(r_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fr_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr bl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr br_wheel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr measured_vel_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    double last_linear_vel_, last_angular_vel_;
    rclcpp::Time last_cmd_time_, last_odom_time_;
    double pose_x_, pose_y_, pose_theta_;

    double wheel_radius_, wheel_separation_;
    std::string lf_joint_, lb_joint_, rf_joint_, rb_joint_;
    std::string odom_frame_, base_frame_;
    bool publish_odom_ = true;
    bool publish_tf_ = true;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DiffDriveController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
