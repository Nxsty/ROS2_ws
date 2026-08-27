
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/utils.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <memory>
#include <cmath>

class PoseController : public rclcpp::Node
{
public:
    PoseController()
    : Node("pose_controller")
    {
        // --- Parámetros de Control (Ganancias) ---
        this->declare_parameter<double>("kp_linear", 1.0);
        this->declare_parameter<double>("kp_angular", 1.8);
        this->declare_parameter<double>("dist_tolerance", 0.15);

        // --- Suscripciones ---
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&PoseController::goal_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PoseController::odom_callback, this, std::placeholders::_1));

        // --- Publicadores ---
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(), "Pose Controller (Tasks 2.2 & 2.3) Started.");
    }

private:
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        goal_pose_ = *msg;
        has_goal_ = true;
        RCLCPP_INFO(this->get_logger(), "Moving to: x=%.2f, y=%.2f", 
                    msg->pose.position.x, msg->pose.position.y);
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_odom_ = *msg;
        has_odom_ = true;
        
        if (has_goal_) {
            compute_control_law();
        }
    }

    void compute_control_law()
    {
        // 1. Obtener posición actual
        double curr_x = current_odom_.pose.pose.position.x;
        double curr_y = current_odom_.pose.pose.position.y;
        
        tf2::Quaternion q(
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z,
            current_odom_.pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // 2. Obtener posición objetivo
        double goal_x = goal_pose_.pose.position.x;
        double goal_y = goal_pose_.pose.position.y;

        // --- Tarea 2.2: Cálculo de Error ---
        double dx = goal_x - curr_x;
        double dy = goal_y - curr_y;
        double distance_error = std::sqrt(dx*dx + dy*dy);
        
        // Ángulo hacia el objetivo
        double angle_to_goal = std::atan2(dy, dx);
        double angle_error = angle_to_goal - yaw;

        // Normalizar ángulo (-pi a pi)
        angle_error = std::atan2(std::sin(angle_error), std::cos(angle_error));

        // --- Tarea 2.3: Ley de Control (P) ---
        auto cmd = geometry_msgs::msg::Twist();

        if (distance_error > this->get_parameter("dist_tolerance").as_double()) {
            // Si el ángulo es muy grande, primero girar sobre el sitio suavemente
            if (std::abs(angle_error) > 0.4) {
                cmd.linear.x = 0.0;
                cmd.angular.z = this->get_parameter("kp_angular").as_double() * angle_error;
            } else {
                // Si está más o menos alineado, avanzar y corregir rumbo
                cmd.linear.x = this->get_parameter("kp_linear").as_double() * distance_error;
                cmd.angular.z = this->get_parameter("kp_angular").as_double() * angle_error;
            }
            
            // Limitar velocidades máximas para estabilidad visual
            if (cmd.linear.x > 0.8) cmd.linear.x = 0.8;
            if (cmd.angular.z > 0.8) cmd.angular.z = 0.8;
            if (cmd.angular.z < -0.8) cmd.angular.z = -0.8;

        } else {
            // Objetivo alcanzado
            RCLCPP_INFO(this->get_logger(), "Goal Reached!");
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

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
