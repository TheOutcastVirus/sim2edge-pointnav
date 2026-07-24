// Classical safety layer. Owns /cmd_vel; the policy never touches the motors.
//
// This is a SEPARATE NODE from nav_policy_node on purpose. If the safety logic
// lived inside the inference node, then a crash, a stall, or an ExecuTorch
// deadlock in that node would take the watchdog down with it and leave the
// robot driving on its last command. Here, the policy node going silent is
// precisely the condition this node is built to detect, and it responds by
// publishing a zero Twist.
//
// Everything in this file is classical and deterministic. Nothing learned, and
// nothing quantized, has stopping authority.
//
// Constants come from spec_generated.hpp so that the simulator's SafetyLayer
// (sim2edge/safety.py) and this node apply the same thresholds, which is what
// makes intervention counts comparable between sim and hardware.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include "nav_policy_node/spec_generated.hpp"

namespace nav_safety
{

namespace spec = nav_policy_node::spec;

class NavSafetyNode : public rclcpp::Node
{
public:
  NavSafetyNode()
  : Node("nav_safety"),
    cmd_stamp_(0, 0, RCL_ROS_TIME),
    scan_stamp_(0, 0, RCL_ROS_TIME)
  {
    declare_parameter("cmd_in_topic", std::string("/nav_policy/cmd_raw"));
    declare_parameter("cmd_out_topic", std::string("/cmd_vel"));
    declare_parameter("scan_topic", std::string("/scan"));
    declare_parameter("status_topic", std::string("~/status"));
    declare_parameter("interventions_topic", std::string("~/interventions"));
    declare_parameter("stop_distance", static_cast<double>(spec::kStopDistanceM));
    declare_parameter("slow_distance", static_cast<double>(spec::kSlowDistanceM));
    declare_parameter("forward_sector",
                      static_cast<double>(spec::kForwardSectorRad));
    declare_parameter("policy_timeout",
                      static_cast<double>(spec::kPolicyTimeoutS));
    declare_parameter("lidar_max_age", static_cast<double>(spec::kLidarMaxAgeS));
    declare_parameter("control_rate", static_cast<double>(spec::kControlRateHz));
    declare_parameter("publish_cmd_vel", true);

    stop_distance_ = get_parameter("stop_distance").as_double();
    slow_distance_ = get_parameter("slow_distance").as_double();
    forward_sector_ = get_parameter("forward_sector").as_double();
    policy_timeout_ = get_parameter("policy_timeout").as_double();
    lidar_max_age_ = get_parameter("lidar_max_age").as_double();
    publish_cmd_vel_ = get_parameter("publish_cmd_vel").as_bool();
    const double rate = get_parameter("control_rate").as_double();

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      get_parameter("cmd_in_topic").as_string(), rclcpp::QoS(1),
      [this](const geometry_msgs::msg::Twist::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        cmd_ = *msg;
        cmd_stamp_ = now();
      });
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      get_parameter("scan_topic").as_string(),
      rclcpp::SensorDataQoS().keep_last(1),
      [this](const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        scan_ = msg;
        scan_stamp_ = now();
      });

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      get_parameter("cmd_out_topic").as_string(), rclcpp::QoS(1));
    status_pub_ = create_publisher<std_msgs::msg::String>(
      get_parameter("status_topic").as_string(), rclcpp::QoS(1));
    interventions_pub_ = create_publisher<std_msgs::msg::UInt32>(
      get_parameter("interventions_topic").as_string(), rclcpp::QoS(1));

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate),
      std::bind(&NavSafetyNode::onTimer, this));

    RCLCPP_INFO(get_logger(),
                "nav_safety up: stop %.2f m, slow %.2f m, sector +/-%.0f deg, "
                "watchdog %.2f s%s",
                stop_distance_, slow_distance_, forward_sector_ * 180.0 / M_PI,
                policy_timeout_,
                publish_cmd_vel_ ? "" : "  [DRY RUN: not publishing /cmd_vel]");
  }

  /// Publish a single zero Twist so the robot halts on shutdown.
  void stop()
  {
    try {
      if (publish_cmd_vel_) {
        cmd_pub_->publish(geometry_msgs::msg::Twist());
      }
    } catch (...) {
      // Shutdown path: a failure to publish here must not mask the original
      // reason we are shutting down.
    }
  }

private:
  /// Minimum valid range within +/-sector of straight ahead.
  /// Returns infinity when the forward sector holds no valid return at all --
  /// the caller must treat that as sensor loss, NOT as clear space.
  double forwardMinRange(const sensor_msgs::msg::LaserScan & scan) const
  {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
      const double angle = scan.angle_min + scan.angle_increment * i;
      double wrapped = std::fmod(angle + M_PI, 2.0 * M_PI);
      if (wrapped < 0.0) {
        wrapped += 2.0 * M_PI;
      }
      wrapped -= M_PI;
      if (std::abs(wrapped) > forward_sector_) {
        continue;
      }
      const double r = scan.ranges[i];
      if (std::isfinite(r) && r >= scan.range_min) {
        best = std::min(best, r);
      }
    }
    return best;
  }

  void onTimer()
  {
    geometry_msgs::msg::Twist cmd;
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
    double cmd_age, scan_age;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cmd = cmd_;
      scan = scan_;
      cmd_age = cmd_stamp_.nanoseconds() == 0
        ? std::numeric_limits<double>::infinity()
        : (now() - cmd_stamp_).seconds();
      scan_age = scan_stamp_.nanoseconds() == 0
        ? std::numeric_limits<double>::infinity()
        : (now() - scan_stamp_).seconds();
    }

    double v = cmd.linear.x;
    double omega = cmd.angular.z;
    const double v_in = v;
    const double omega_in = omega;
    const char * reason = "ok";

    // Hard clamps first, so everything downstream reasons about achievable
    // commands.
    v = std::min(std::max(v, static_cast<double>(spec::kVMin)),
                 static_cast<double>(spec::kVMax));
    omega = std::min(std::max(omega, -static_cast<double>(spec::kOmegaMax)),
                     static_cast<double>(spec::kOmegaMax));

    if (cmd_age > policy_timeout_) {
      // The policy has gone silent: crashed, stalled, or deliberately
      // withholding a command because its sensors are stale. All three mean
      // stop.
      v = 0.0;
      omega = 0.0;
      reason = "policy timeout";
    } else if (!scan || scan_age > lidar_max_age_) {
      v = 0.0;
      omega = 0.0;
      reason = "lidar stale";
    } else {
      const double d = forwardMinRange(*scan);
      if (!std::isfinite(d)) {
        v = 0.0;  // no valid forward returns: do not drive blind
        reason = "no forward returns";
      } else if (d < stop_distance_) {
        // Rotation is deliberately still permitted. A robot stopped 25 cm from
        // a wall must be able to turn away from it; freezing both axes turns a
        // recoverable state into a deadlock.
        v = 0.0;
        reason = "bubble stop";
      } else if (d < slow_distance_) {
        const double span = std::max(1e-6, slow_distance_ - stop_distance_);
        v *= (d - stop_distance_) / span;
        reason = "bubble slow";
      }
    }

    // Rate limiting last, so the bubble's decision is what gets ramped rather
    // than being ramped away.
    const double dt = 1.0 / static_cast<double>(spec::kControlRateHz);
    v = std::min(std::max(v, prev_v_ - spec::kAccelMax * dt),
                 prev_v_ + spec::kAccelMax * dt);
    omega = std::min(std::max(omega, prev_omega_ - spec::kAlphaMax * dt),
                     prev_omega_ + spec::kAlphaMax * dt);
    prev_v_ = v;
    prev_omega_ = omega;

    const bool intervened =
      std::abs(v - v_in) > 1e-3 || std::abs(omega - omega_in) > 1e-3;
    if (intervened) {
      ++interventions_;
      // Interventions are an evaluation metric, not just a log line: a run that
      // succeeds only because the watchdog kept saving it is a finding.
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "intervention #%u (%s): (%.3f, %+.3f) -> (%.3f, %+.3f)",
                           interventions_, reason, v_in, omega_in, v, omega);
    }

    geometry_msgs::msg::Twist out;
    out.linear.x = v;
    out.angular.z = omega;
    if (publish_cmd_vel_) {
      cmd_pub_->publish(out);
    }

    std_msgs::msg::UInt32 count;
    count.data = interventions_;
    interventions_pub_->publish(count);

    if (status_pub_->get_subscription_count() > 0) {
      std_msgs::msg::String status;
      status.data = std::string(reason) + " | in (" + std::to_string(v_in) +
        ", " + std::to_string(omega_in) + ") out (" + std::to_string(v) + ", " +
        std::to_string(omega) + ") | interventions " +
        std::to_string(interventions_);
      status_pub_->publish(status);
    }
  }

  double stop_distance_{};
  double slow_distance_{};
  double forward_sector_{};
  double policy_timeout_{};
  double lidar_max_age_{};
  bool publish_cmd_vel_{true};

  double prev_v_{0.0};
  double prev_omega_{0.0};
  std::uint32_t interventions_{0};

  std::mutex mutex_;
  geometry_msgs::msg::Twist cmd_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_;
  rclcpp::Time cmd_stamp_;
  rclcpp::Time scan_stamp_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr interventions_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace nav_safety

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<nav_safety::NavSafetyNode>();
  try {
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "fatal: %s", e.what());
  }
  // Zero Twist on the way out, verbatim from the reference controllers: without
  // it, Ctrl-C leaves the robot driving at its last commanded velocity.
  node->stop();
  rclcpp::shutdown();
  return 0;
}
