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

class PoseController : public rclcpp::Node
{
public:
    PoseController()
    : Node("pose_controller")
    {
        // --- Parameters ---
        this->declare_parameter<double>("kp_linear", 1.2);
        this->declare_parameter<double>("kp_angular", 2.2);
        this->declare_parameter<double>("dist_tolerance", 0.35);
        this->declare_parameter<double>("max_linear_vel", 1.20);
        this->declare_parameter<double>("max_angular_vel", 1.50);

        // --- Subscriptions ---
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&PoseController::goal_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PoseController::odom_callback, this, std::placeholders::_1));

        // --- Publishers ---
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(), "Pose Controller Initialized (Smooth Continuous Unicycle Control).");
    }

private:
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        goal_pose_ = *msg;
        has_goal_ = true;
        RCLCPP_INFO(this->get_logger(), "🎯 New Target Received: x=%.2f, y=%.2f", 
                    msg->pose.position.x, msg->pose.position.y);
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
            // Stop robot when no active goal
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            cmd_vel_pub_->publish(cmd);
            return;
        }

        // 1. Current pose
        double curr_x = current_odom_.pose.pose.position.x;
        double curr_y = current_odom_.pose.pose.position.y;
        
        tf2::Quaternion q(
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z,
            current_odom_.pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // 2. Goal position
        double goal_x = goal_pose_.pose.position.x;
        double goal_y = goal_pose_.pose.position.y;

        // 3. Error computation
        double dx = goal_x - curr_x;
        double dy = goal_y - curr_y;
        double distance_error = std::sqrt(dx*dx + dy*dy);
        
        double angle_to_goal = std::atan2(dy, dx);
        double angle_error = angle_to_goal - yaw;
        // Normalize angle to [-pi, pi]
        angle_error = std::atan2(std::sin(angle_error), std::cos(angle_error));

        double dist_tol = this->get_parameter("dist_tolerance").as_double();
        double kp_lin = this->get_parameter("kp_linear").as_double();
        double kp_ang = this->get_parameter("kp_angular").as_double();
        double max_v = this->get_parameter("max_linear_vel").as_double();
        double max_w = this->get_parameter("max_angular_vel").as_double();

        if (distance_error > dist_tol) {
            // Phase 1: In-place turn if heading deviates by more than ~15 degrees (0.26 rad)
            if (std::abs(angle_error) > 0.26) {
                cmd.linear.x = 0.0;
                double wz = kp_ang * angle_error;
                cmd.angular.z = std::clamp(wz, -max_w, max_w);
            } else {
                // Phase 2: Drive straight towards waypoint with gentle heading tracking
                // Smooth deceleration profile as it approaches target
                double v_scaled = kp_lin * (distance_error - 0.15);
                double align_factor = std::max(0.0, std::cos(angle_error));
                double vx = v_scaled * align_factor;
                cmd.linear.x = std::clamp(vx, 0.0, max_v);

                // Micro-corrections of heading while driving
                double wz = 1.5 * angle_error;
                cmd.angular.z = std::clamp(wz, -0.40, 0.40);
            }
        } else {
            // Target reached!
            RCLCPP_INFO(this->get_logger(), "✅ Target Reached (error: %.2fm <= %.2fm). Holding position.",
                        distance_error, dist_tol);
            has_goal_ = false;
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
        }

        cmd_vel_pub_->publish(cmd);
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

    geometry_msgs::msg::PoseStamped goal_pose_;
    nav_msgs::msg::Odometry current_odom_;
    bool has_goal_ = false;
    bool has_odom_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
