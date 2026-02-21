
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"

#include <memory>
#include <chrono>

class DiffDriveController : public rclcpp::Node
{
public:
    DiffDriveController()
    : Node("diff_drive_controller"),
      last_linear_vel_(0.0),
      last_angular_vel_(0.0)
    {
        // --- Parámetros físicos del robot ---
        wheel_radius_ = 0.1;       // metros
        wheel_separation_ = 0.45;  // metros

        // --- Suscripciones ---
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&DiffDriveController::cmd_vel_callback, this, std::placeholders::_1));

        // --- Publicadores ---
        left_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/left_wheel_controller/cmd_vel", 10);
        right_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_wheel_controller/cmd_vel", 10);

        // --- Timer del control loop (50 Hz = 20 ms) ---
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&DiffDriveController::control_loop, this));

        last_cmd_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "DiffDriveController (Velocity Control) has been started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        last_linear_vel_ = msg->linear.x;
        last_angular_vel_ = msg->angular.z;
        last_cmd_time_ = this->now();
    }

    void control_loop()
    {
        // --- Rampa de desaceleración si no hay nuevos comandos ---
        double time_since_cmd = (this->now() - last_cmd_time_).seconds();
        if (time_since_cmd > 0.5) {
            last_linear_vel_ *= 0.9;
            last_angular_vel_ *= 0.9;
        }

        // --- Cinemática diferencial: velocidades deseadas ---
        double desired_left_wheel_vel =
            (last_linear_vel_ - last_angular_vel_ * wheel_separation_ / 2.0) / wheel_radius_;
        double desired_right_wheel_vel =
            (last_linear_vel_ + last_angular_vel_ * wheel_separation_ / 2.0) / wheel_radius_;

        // --- Publicar velocidades ---
        std_msgs::msg::Float64 left_msg;
        std_msgs::msg::Float64 right_msg;
        left_msg.data = desired_left_wheel_vel;
        right_msg.data = desired_right_wheel_vel;

        left_wheel_pub_->publish(left_msg);
        right_wheel_pub_->publish(right_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_wheel_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Último comando recibido
    double last_linear_vel_;
    double last_angular_vel_;
    rclcpp::Time last_cmd_time_;

    // Parámetros físicos del robot
    double wheel_radius_;
    double wheel_separation_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DiffDriveController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
