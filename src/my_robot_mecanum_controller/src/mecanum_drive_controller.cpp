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
        // Declare parameters
        this->declare_parameter<double>("wheel_radius", 0.029);
        this->declare_parameter<double>("wheel_separation_x", 0.112); // Half of wheelbase (Lx)
        this->declare_parameter<double>("wheel_separation_y", 0.1375); // Half of track width (Ly)
        this->declare_parameter<std::string>("fl_wheel_joint", "rim_left_front_joint");
        this->declare_parameter<std::string>("fr_wheel_joint", "rim_right_front_joint");
        this->declare_parameter<std::string>("bl_wheel_joint", "rim_left_back_joint");
        this->declare_parameter<std::string>("br_wheel_joint", "rim_right_back_joint");
        this->declare_parameter<std::string>("fl_wheel_topic", "/rim_left_front_joint/cmd_vel");
        this->declare_parameter<std::string>("fr_wheel_topic", "/rim_right_front_joint/cmd_vel");
        this->declare_parameter<std::string>("bl_wheel_topic", "/rim_left_back_joint/cmd_vel");
        this->declare_parameter<std::string>("br_wheel_topic", "/rim_right_back_joint/cmd_vel");
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "base_link");

        // Get parameters
        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation_x", wheel_separation_x_);
        this->get_parameter("wheel_separation_y", wheel_separation_y_);
        this->get_parameter("fl_wheel_joint", fl_joint_name_);
        this->get_parameter("fr_wheel_joint", fr_joint_name_);
        this->get_parameter("bl_wheel_joint", bl_joint_name_);
        this->get_parameter("br_wheel_joint", br_joint_name_);
        this->get_parameter("odom_frame", odom_frame_);
        this->get_parameter("base_frame", base_frame_);

        std::string fl_wheel_topic, fr_wheel_topic, bl_wheel_topic, br_wheel_topic;
        this->get_parameter("fl_wheel_topic", fl_wheel_topic);
        this->get_parameter("fr_wheel_topic", fr_wheel_topic);
        this->get_parameter("bl_wheel_topic", bl_wheel_topic);
        this->get_parameter("br_wheel_topic", br_wheel_topic);

        // --- Subscriptions ---
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&MecanumDriveController::cmd_vel_callback, this, std::placeholders::_1));

        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&MecanumDriveController::joint_states_callback, this, std::placeholders::_1));

        // --- Publishers ---
        fl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(fl_wheel_topic, 10);
        fr_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(fr_wheel_topic, 10);
        bl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(bl_wheel_topic, 10);
        br_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(br_wheel_topic, 10);

        measured_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/robot_measured_vel", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        // TF Broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // --- Control loop timer (50 Hz = 20 ms) ---
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&MecanumDriveController::control_loop, this));

        last_cmd_time_ = this->now();
        last_odom_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "MecanumDriveController (Odometry Task 1.2) has been started.");
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
        double w_fl = 0, w_fr = 0, w_bl = 0, w_br = 0;
        bool found_fl = false, found_fr = false, found_bl = false, found_br = false;

        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i] == fl_joint_name_) { w_fl = msg->velocity[i]; found_fl = true; }
            else if (msg->name[i] == fr_joint_name_) { w_fr = msg->velocity[i]; found_fr = true; }
            else if (msg->name[i] == bl_joint_name_) { w_bl = msg->velocity[i]; found_bl = true; }
            else if (msg->name[i] == br_joint_name_) { w_br = msg->velocity[i]; found_br = true; }
        }

        if (found_fl && found_fr && found_bl && found_br) {
            // --- Tarea 1.1: Cinemática Directa (Local velocities) ---
            double R = wheel_radius_;
            double LxLy = wheel_separation_x_ + wheel_separation_y_;

            double vx = (R / 4.0) * (w_fl + w_fr + w_bl + w_br);
            double vy = (R / 4.0) * (-w_fl + w_fr + w_bl - w_br);
            double wz = (R / (4.0 * LxLy)) * (-w_fl + w_fr - w_bl + w_br);

            // Publish measured velocities
            geometry_msgs::msg::Twist measured_vel;
            measured_vel.linear.x = vx;
            measured_vel.linear.y = vy;
            measured_vel.angular.z = wz;
            measured_vel_pub_->publish(measured_vel);

            // --- Tarea 1.2: Odometría (Dead-reckoning) ---
            rclcpp::Time current_time = this->now();
            double dt = (current_time - last_odom_time_).seconds();
            last_odom_time_ = current_time;

            if (dt > 0.0) {
                // RK2 or Simple Euler integration?
                // For pose, we need to transform local velocities to global frame
                double delta_x = (vx * std::cos(pose_theta_) - vy * std::sin(pose_theta_)) * dt;
                double delta_y = (vx * std::sin(pose_theta_) + vy * std::cos(pose_theta_)) * dt;
                double delta_theta = wz * dt;

                pose_x_ += delta_x;
                pose_y_ += delta_y;
                pose_theta_ += delta_theta;

                // Create Quaternion from Yaw
                tf2::Quaternion q;
                q.setRPY(0, 0, pose_theta_);

                // Publish TF (odom -> base_link)
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

                // Publish Odometry message
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
                odom.twist.twist.linear.x = vx;
                odom.twist.twist.linear.y = vy;
                odom.twist.twist.angular.z = wz;
                odom_pub_->publish(odom);
            }
        }
    }

    void control_loop()
    {
        // --- Deceleration ramp if no new commands ---
        double time_since_cmd = (this->now() - last_cmd_time_).seconds();
        if (time_since_cmd > 0.5) {
            last_linear_x_vel_ *= 0.9;
            last_linear_y_vel_ *= 0.9;
            last_angular_vel_ *= 0.9;
        }

        // --- Mecanum kinematics: Inverse Kinematics ---
        double vx = last_linear_x_vel_;
        double vy = last_linear_y_vel_;
        double wz = last_angular_vel_;

        double R = wheel_radius_;
        double LxLy = wheel_separation_x_ + wheel_separation_y_;

        // Front Left wheel
        double w_front_left = (vx - vy - LxLy * wz) / R;
        // Front Right wheel
        double w_front_right = (vx + vy + LxLy * wz) / R;
        // Back Left wheel
        double w_back_left = (vx + vy - LxLy * wz) / R;
        // Back Right wheel
        double w_back_right = (vx - vy + LxLy * wz) / R;

        // --- Publish wheel velocities ---
        std_msgs::msg::Float64 fl_msg, fr_msg, bl_msg, br_msg;
        fl_msg.data = w_front_left;
        fr_msg.data = w_front_right;
        bl_msg.data = w_back_left;
        br_msg.data = w_back_right;

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
    std::string fl_joint_name_, fr_joint_name_, bl_joint_name_, br_joint_name_;
    std::string odom_frame_, base_frame_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MecanumDriveController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}