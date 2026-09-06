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
#include <algorithm>
#include <cmath>

class MecanumDriveController : public rclcpp::Node
{
public:
    MecanumDriveController()
    : Node("mecanum_drive_controller"),
      last_linear_x_vel_(0.0),
      last_linear_y_vel_(0.0),
      last_angular_vel_(0.0),
      pose_x_(0.0),
      pose_y_(0.0),
      pose_theta_(0.0)
    {
        // --- Parameters ---
        this->declare_parameter<double>("wheel_radius", 0.1);
        this->declare_parameter<double>("wheel_separation_x", 0.272); // Half wheelbase (Lx)
        this->declare_parameter<double>("wheel_separation_y", 0.225); // Half trackwidth (Ly)

        this->declare_parameter<std::string>("left_front_wheel_joint", "base_left_front_wheel_joint");
        this->declare_parameter<std::string>("right_front_wheel_joint", "base_right_front_wheel_joint");
        this->declare_parameter<std::string>("left_back_wheel_joint", "base_left_back_wheel_joint");
        this->declare_parameter<std::string>("right_back_wheel_joint", "base_right_back_wheel_joint");

        this->declare_parameter<std::string>("left_front_wheel_topic", "/base_left_front_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("right_front_wheel_topic", "/base_right_front_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("left_back_wheel_topic", "/base_left_back_wheel_joint/cmd_vel");
        this->declare_parameter<std::string>("right_back_wheel_topic", "/base_right_back_wheel_joint/cmd_vel");

        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "base_footprint");
        this->declare_parameter<bool>("publish_odom", true);
        this->declare_parameter<bool>("publish_tf", true);
        this->declare_parameter<double>("cmd_vel_timeout", 0.4);

        // Get parameters
        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation_x", wheel_separation_x_);
        this->get_parameter("wheel_separation_y", wheel_separation_y_);
        this->get_parameter("left_front_wheel_joint", lf_joint_);
        this->get_parameter("right_front_wheel_joint", rf_joint_);
        this->get_parameter("left_back_wheel_joint", lb_joint_);
        this->get_parameter("right_back_wheel_joint", rb_joint_);
        this->get_parameter("odom_frame", odom_frame_);
        this->get_parameter("base_frame", base_frame_);
        this->get_parameter("publish_odom", publish_odom_);
        this->get_parameter("publish_tf", publish_tf_);
        this->get_parameter("cmd_vel_timeout", cmd_vel_timeout_);

        std::string lf_topic, rf_topic, lb_topic, rb_topic;
        this->get_parameter("left_front_wheel_topic", lf_topic);
        this->get_parameter("right_front_wheel_topic", rf_topic);
        this->get_parameter("left_back_wheel_topic", lb_topic);
        this->get_parameter("right_back_wheel_topic", rb_topic);

        // --- Subscriptions ---
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&MecanumDriveController::cmd_vel_callback, this, std::placeholders::_1));

        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&MecanumDriveController::joint_states_callback, this, std::placeholders::_1));

        // --- Publishers ---
        fl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(lf_topic, 10);
        fr_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(rf_topic, 10);
        bl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(lb_topic, 10);
        br_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(rb_topic, 10);

        measured_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/robot_measured_vel", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // --- Control loop timer (50 Hz = 20 ms) ---
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&MecanumDriveController::control_loop, this));

        last_cmd_time_ = this->now();
        last_odom_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), 
            "MecanumDriveController Initialized: R=%.3f, Lx=%.3f, Ly=%.3f, base_frame=%s, odom_frame=%s",
            wheel_radius_, wheel_separation_x_, wheel_separation_y_,
            base_frame_.c_str(), odom_frame_.c_str());
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        last_linear_x_vel_ = msg->linear.x;
        last_linear_y_vel_ = msg->linear.y;
        last_angular_vel_ = msg->angular.z;
        last_cmd_time_ = this->now();
    }

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        double w_fl = 0.0, w_fr = 0.0, w_bl = 0.0, w_br = 0.0;
        bool found_fl = false, found_fr = false, found_bl = false, found_br = false;

        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->velocity.size() > i) {
                if (msg->name[i].find(lf_joint_) != std::string::npos) { w_fl = msg->velocity[i]; found_fl = true; }
                else if (msg->name[i].find(rf_joint_) != std::string::npos) { w_fr = msg->velocity[i]; found_fr = true; }
                else if (msg->name[i].find(lb_joint_) != std::string::npos) { w_bl = msg->velocity[i]; found_bl = true; }
                else if (msg->name[i].find(rb_joint_) != std::string::npos) { w_br = msg->velocity[i]; found_br = true; }
            }
        }

        if (found_fl && found_fr && found_bl && found_br) {
            // --- Cinemática Directa (Forward Kinematics) ---
            // Transforma velocidades de las 4 ruedas en velocidades del cuerpo del robot (vx, vy, wz)
            double R = wheel_radius_;
            double LxLy = wheel_separation_x_ + wheel_separation_y_;

            double vx = (R / 4.0) * (w_fl + w_fr + w_bl + w_br);
            double vy = (R / 4.0) * (-w_fl + w_fr + w_bl - w_br);
            double wz = (R / (4.0 * LxLy)) * (-w_fl + w_fr - w_bl + w_br);

            // Publicar velocidad medida local
            geometry_msgs::msg::Twist measured_vel;
            measured_vel.linear.x = vx;
            measured_vel.linear.y = vy;
            measured_vel.angular.z = wz;
            measured_vel_pub_->publish(measured_vel);

            // --- Odometría Dead-reckoning con integración Runge-Kutta 2 / Midpoint ---
            if (publish_odom_ || publish_tf_) {
                rclcpp::Time current_time = this->now();
                double dt = (current_time - last_odom_time_).seconds();
                last_odom_time_ = current_time;

                if (dt > 0.0 && dt < 1.0) {
                    double delta_theta = wz * dt;
                    double mid_theta = pose_theta_ + delta_theta / 2.0;

                    // Integración en el marco global
                    double delta_x = (vx * std::cos(mid_theta) - vy * std::sin(mid_theta)) * dt;
                    double delta_y = (vx * std::sin(mid_theta) + vy * std::cos(mid_theta)) * dt;

                    pose_x_ += delta_x;
                    pose_y_ += delta_y;
                    pose_theta_ += delta_theta;
                    pose_theta_ = std::atan2(std::sin(pose_theta_), std::cos(pose_theta_));

                    tf2::Quaternion q;
                    q.setRPY(0.0, 0.0, pose_theta_);

                    // Publicar TF (odom -> base_footprint)
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

                    // Publicar mensaje Odometry
                    if (publish_odom_) {
                        nav_msgs::msg::Odometry odom;
                        odom.header.stamp = current_time;
                        odom.header.frame_id = odom_frame_;
                        odom.child_frame_id = base_frame_;

                        odom.pose.pose.position.x = pose_x_;
                        odom.pose.pose.position.y = pose_y_;
                        odom.pose.pose.position.z = 0.0;
                        odom.pose.pose.orientation.x = q.x();
                        odom.pose.pose.orientation.y = q.y();
                        odom.pose.pose.orientation.z = q.z();
                        odom.pose.pose.orientation.w = q.w();

                        // Covarianzas de Pose
                        odom.pose.covariance[0]  = 0.001; // x
                        odom.pose.covariance[7]  = 0.001; // y
                        odom.pose.covariance[35] = 0.001; // yaw

                        // Twist en marco local del robot
                        odom.twist.twist.linear.x = vx;
                        odom.twist.twist.linear.y = vy;
                        odom.twist.twist.angular.z = wz;

                        // Covarianzas de Twist
                        odom.twist.covariance[0]  = 0.001; // vx
                        odom.twist.covariance[7]  = 0.001; // vy
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

        // Safety timeout: si no hay comandos nuevos en cmd_vel_timeout_, detener
        if (time_since_cmd > cmd_vel_timeout_) {
            last_linear_x_vel_ = 0.0;
            last_linear_y_vel_ = 0.0;
            last_angular_vel_ = 0.0;
        }

        // --- Cinemática Inversa (Inverse Kinematics) ---
        // Transforma velocidades deseadas del robot (vx, vy, wz) en velocidades de cada rueda
        double vx = last_linear_x_vel_;
        double vy = last_linear_y_vel_;
        double wz = last_angular_vel_;

        double R = wheel_radius_;
        double LxLy = wheel_separation_x_ + wheel_separation_y_;

        // Rueda delantera izquierda (FL)
        double w_fl = (vx - vy - LxLy * wz) / R;
        // Rueda delantera derecha (FR)
        double w_fr = (vx + vy + LxLy * wz) / R;
        // Rueda trasera izquierda (BL)
        double w_bl = (vx + vy - LxLy * wz) / R;
        // Rueda trasera derecha (BR)
        double w_br = (vx - vy + LxLy * wz) / R;

        // Publicar comandos de velocidad a los controladores de articulación en Gazebo
        std_msgs::msg::Float64 fl_msg, fr_msg, bl_msg, br_msg;
        fl_msg.data = w_fl;
        fr_msg.data = w_fr;
        bl_msg.data = w_bl;
        br_msg.data = w_br;

        fl_wheel_pub_->publish(fl_msg);
        fr_wheel_pub_->publish(fr_msg);
        bl_wheel_pub_->publish(bl_msg);
        br_wheel_pub_->publish(br_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fr_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr bl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr br_wheel_pub_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr measured_vel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    double last_linear_x_vel_;
    double last_linear_y_vel_;
    double last_angular_vel_;
    rclcpp::Time last_cmd_time_;
    rclcpp::Time last_odom_time_;

    double pose_x_, pose_y_, pose_theta_;

    double wheel_radius_;
    double wheel_separation_x_;
    double wheel_separation_y_;
    double cmd_vel_timeout_;

    std::string lf_joint_, rf_joint_, lb_joint_, rb_joint_;
    std::string odom_frame_, base_frame_;
    bool publish_odom_ = true;
    bool publish_tf_ = true;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MecanumDriveController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
