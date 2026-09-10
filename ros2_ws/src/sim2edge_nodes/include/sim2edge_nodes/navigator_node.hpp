// The deployable navigation stack, as one ROS 2 node.
//
// **Why one node and not four.** Deskew, binning, localization and inference
// are separate concerns and it is tempting to give each its own node. Doing so
// would push a 720-point sweep and a 731-float observation through DDS twice
// per control tick, and buy nothing: the stages are not independently useful,
// nothing else subscribes to a half-processed sweep, and no stage can run
// while another is waiting. sim2edge::Navigator already owns the tick as one
// code path -- the same one the parity harness runs -- so this node is a thin
// adapter around it, and the whole chain stays in one address space with no
// serialization at all.
//
// **What is deliberately a separate node.** The safety layer, which owns
// /cmd_vel. This node publishes a *proposal*. If it crashes, stalls, or is
// killed, the safety node keeps running and keeps stopping the robot -- which
// it cannot do if the thing that crashed was also the only publisher.
//
// **Sensor handling follows the reference project's pattern, not
// message_filters.** Each subscription caches only the latest message at queue
// depth 1, and a decoupled control timer takes whatever is current and rejects
// anything older than a per-sensor max age. An approximate-time synchronizer
// would stall the control loop whenever one sensor hiccuped, which for a moving
// robot is strictly worse than acting on a slightly stale scan -- and worse
// still than deliberately holding.

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <sim2edge/actor.hpp>
#include <sim2edge/navigator.hpp>
#include <sim2edge_msgs/msg/localizer_state.hpp>
#include <sim2edge_msgs/msg/nav_status.hpp>

namespace sim2edge_nodes
{

class NavigatorNode : public rclcpp::Node
{
public:
  explicit NavigatorNode(const rclcpp::NodeOptions & options =
                           rclcpp::NodeOptions());

private:
  void onScan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void onImu(sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void onOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void onGoal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg);
  void onControlTimer();

  void loadActor();
  double ageOf(const rclcpp::Time & stamp) const;

  /// This robot's yaw rate, pulled out of whichever gyro component carries it.
  double yawRateOf(const sensor_msgs::msg::Imu & msg) const;

  // ── The stack ─────────────────────────────────────────────────────────────
  sim2edge::Navigator navigator_;
  std::shared_ptr<sim2edge::Actor> actor_;
  std::string backend_name_{"none"};

  // ── Configuration ─────────────────────────────────────────────────────────
  double scan_max_age_{};
  double imu_max_age_{};
  double odom_max_age_{};
  bool require_odom_{};
  bool goal_is_body_frame_{};
  double control_period_{};
  /// Optional heartbeat timeout for a continuously refreshed high-level goal.
  /// Zero preserves the point-goal node's historical one-shot behavior.
  double goal_max_age_{};
  /// Which angular_velocity component is yaw, and its sign. Measured on the
  /// hardware, not assumed -- the OAK-D's IMU has its vertical axis on y.
  std::string gyro_yaw_axis_{"y"};
  double gyro_yaw_sign_{1.0};

  // ── Cached sensor state, latest message only ──────────────────────────────
  std::mutex mutex_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  rclcpp::Time scan_stamp_;
  rclcpp::Time imu_stamp_;
  rclcpp::Time odom_stamp_;

  // Yaw is *integrated* in the IMU callback rather than sampled at the control
  // tick. The OAK-D's IMU runs far faster than 20 Hz, and multiplying one
  // sampled rate by the control period throws away every sample in between --
  // which is exactly the rotation deskew needs to be right about.
  double gyro_accumulator_{0.0};
  rclcpp::Time last_imu_stamp_;
  bool have_imu_{false};

  // Previous odometry pose, for the body-frame increment prior.
  bool have_previous_odom_{false};
  double previous_odom_x_{0.0};
  double previous_odom_y_{0.0};
  double previous_odom_yaw_{0.0};
  bool have_goal_update_{false};
  std::chrono::steady_clock::time_point last_goal_update_{};

  // ── Scratch, reused so a 20 Hz loop allocates nothing in steady state ─────
  std::vector<float> bearings_;
  std::vector<float> ages_;

  std::size_t tick_count_{0};
  std::size_t hold_count_{0};

  // Rolling latency, summarized in the node itself.
  //
  // Measuring this from outside -- `ros2 topic echo` on nav/status -- costs a
  // Python process subscribing at 20 Hz, and on an eight-core board that is
  // enough load to move the number being measured. A ring buffer and a
  // nth_element every five seconds is free by comparison.
  static constexpr std::size_t kLatencyWindow = 2048;
  std::vector<double> latency_us_;
  std::size_t latency_head_{0};
  std::size_t latency_filled_{0};
  std::size_t overrun_count_{0};

  // ── ROS interfaces ────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr command_pub_;
  rclcpp::Publisher<sim2edge_msgs::msg::NavStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<sim2edge_msgs::msg::LocalizerState>::SharedPtr localizer_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace sim2edge_nodes
