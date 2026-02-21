#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"

#include <memory>
#include <chrono>

class MecanumDriveController : public rclcpp::Node
{
public:
    MecanumDriveController()
    : Node("mecanum_drive_controller"),
      last_linear_x_vel_(0.0),
      last_linear_y_vel_(0.0),
      last_angular_vel_(0.0)
    {
        // Declare parameters
        this->declare_parameter<double>("wheel_radius", 0.029);
        this->declare_parameter<double>("wheel_separation_x", 0.272);
        this->declare_parameter<double>("wheel_separation_y", 0.19);
        this->declare_parameter<std::string>("fl_wheel_topic", "/rim_left_front_joint/cmd_vel");
        this->declare_parameter<std::string>("fr_wheel_topic", "/rim_right_front_joint/cmd_vel");
        this->declare_parameter<std::string>("bl_wheel_topic", "/rim_left_back_joint/cmd_vel");
        this->declare_parameter<std::string>("br_wheel_topic", "/rim_right_back_joint/cmd_vel");

        // Get parameters
        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation_x", wheel_separation_x_);
        this->get_parameter("wheel_separation_y", wheel_separation_y_);
        std::string fl_wheel_topic, fr_wheel_topic, bl_wheel_topic, br_wheel_topic;
        this->get_parameter("fl_wheel_topic", fl_wheel_topic);
        this->get_parameter("fr_wheel_topic", fr_wheel_topic);
        this->get_parameter("bl_wheel_topic", bl_wheel_topic);
        this->get_parameter("br_wheel_topic", br_wheel_topic);

        // --- Subscriptions ---
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&MecanumDriveController::cmd_vel_callback, this, std::placeholders::_1));

        // --- Publishers ---
        fl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(fl_wheel_topic, 10);
        fr_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(fr_wheel_topic, 10);
        bl_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(bl_wheel_topic, 10);
        br_wheel_pub_ = this->create_publisher<std_msgs::msg::Float64>(br_wheel_topic, 10);

        // --- Control loop timer (50 Hz = 20 ms) ---
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&MecanumDriveController::control_loop, this));

        last_cmd_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "MecanumDriveController (Velocity Control) has been started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        //RCLCPP_INFO(this->get_logger(), "Received cmd_vel: linear.x=%.2f, angular.z=%.2f", msg->linear.x, msg->angular.z);
        last_linear_x_vel_ = msg->linear.x;
        last_linear_y_vel_ = msg->linear.y;
        last_angular_vel_ = msg->angular.z;
        last_cmd_time_ = this->now();
    }

    void control_loop()
    {
        // --- Deceleration ramp if no new commands ---
        double time_since_cmd = (this->now() - last_cmd_time_).seconds();
        if (time_since_cmd > 0.5) {
            last_linear_x_vel_ *= 0.9;
            last_angular_vel_ *= 0.9;
        }

        // --- Mecanum kinematics: desired velocities ---
        double vx = last_linear_x_vel_;
        double vy = last_linear_y_vel_;
        double wz = last_angular_vel_;

        double R = wheel_radius_;
        double Lx = wheel_separation_x_; // Half of wheelbase
        double Ly = wheel_separation_y_; // Half of track width

        // Inverse kinematics for Mecanum wheels
        // Front Left wheel
        double w_front_left = (vx - vy - (Lx + Ly) * wz) / R;
        // Front Right wheel
        double w_front_right = (vx + vy + (Lx + Ly) * wz) / R;
        // Back Left wheel
        double w_back_left = (vx + vy - (Lx + Ly) * wz) / R;
        // Back Right wheel
        double w_back_right = (vx - vy + (Lx + Ly) * wz) / R;

        //RCLCPP_INFO(this->get_logger(), "Publishing wheel velocities: FL=%.2f, FR=%.2f, BL=%.2f, BR=%.2f", w_front_left, w_front_right, w_back_left, w_back_right);

        // --- Publish velocities ---
        std_msgs::msg::Float64 fl_msg, fr_msg, bl_msg, br_msg;
        fl_msg.data = w_front_left;  // Front left
        fr_msg.data = w_front_right; // Front right
        bl_msg.data = w_back_left;   // Back left
        br_msg.data = w_back_right;  // Back right

        fl_wheel_pub_->publish(fl_msg);
        fr_wheel_pub_->publish(fr_msg);
        bl_wheel_pub_->publish(bl_msg);
        br_wheel_pub_->publish(br_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fr_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr bl_wheel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr br_wheel_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Last received command
    double last_linear_x_vel_;
    double last_linear_y_vel_;
    double last_angular_vel_;
    rclcpp::Time last_cmd_time_;

    // Robot physical parameters
    double wheel_radius_;
    double wheel_separation_x_;
    double wheel_separation_y_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MecanumDriveController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}