#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>

#include "nav_policy_node/inference_backend.hpp"
#include "nav_policy_node/preprocess.hpp"

namespace nav_policy_node
{

/// Goal-conditioned navigation policy running on the Ventuno Q.
///
/// Sensor handling follows the reference project's robot_tracker.py rather than
/// message_filters: each subscription caches only the latest message (queue
/// depth 1), and a decoupled control timer takes whatever is current and
/// rejects anything older than a per-sensor max_age. An approximate-time
/// synchronizer would stall the control loop whenever one sensor hiccuped,
/// which for a moving robot is strictly worse than acting on a slightly stale
/// scan -- and worse still than deliberately stopping.
///
/// This node does NOT publish to /cmd_vel. It publishes its raw policy output
/// and lets nav_safety own the motors, so the safety layer keeps running (and
/// keeps stopping the robot) even if this node crashes or stalls.
class NavPolicyNode : public rclcpp::Node
{
public:
  explicit NavPolicyNode(const rclcpp::NodeOptions & options =
                         rclcpp::NodeOptions());

private:
  // ── Callbacks ─────────────────────────────────────────────────────────────
  void onScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void onGoal(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg);
  void onTimer();

  // ── Helpers ───────────────────────────────────────────────────────────────
  void verifyModelContract(const std::string & model_path);
  bool buildObservation(Observation & obs, std::string & reason);
  double ageOf(const rclcpp::Time & stamp) const;

  // ── Configuration ─────────────────────────────────────────────────────────
  double lidar_max_age_{};
  double depth_max_age_{};
  double odom_max_age_{};
  bool require_depth_{};
  bool goal_is_relative_{};

  // ── Inference ─────────────────────────────────────────────────────────────
  std::unique_ptr<InferenceBackend> backend_;
  Observation observation_;

  // ── Cached sensor state (latest message only) ─────────────────────────────
  std::mutex mutex_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  rclcpp::Time scan_stamp_;
  rclcpp::Time depth_stamp_;
  rclcpp::Time odom_stamp_;

  // ── Goal ──────────────────────────────────────────────────────────────────
  bool has_goal_{false};
  bool goal_anchored_{false};
  double goal_world_x_{0.0};
  double goal_world_y_{0.0};

  // ── Policy state ──────────────────────────────────────────────────────────
  float last_v_{0.0f};
  float last_omega_{0.0f};
  std::size_t inference_count_{0};
  double inference_time_sum_{0.0};
  double inference_time_max_{0.0};

  // ── ROS interfaces ────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace nav_policy_node
