// The layer that owns /cmd_vel, and the only one.
//
// It exists as a separate process from the navigator for one reason: so that it
// keeps running when the navigator does not. A policy node that published
// straight to the motors would, on a crash or a stall, leave the base holding
// its last command -- and the Create 3 will happily keep driving on a Twist
// nobody is refreshing. Here the watchdog is in a process whose only job is to
// notice, and whose code is simple enough to trust with that job.
//
// Three independent reasons to override, in order of authority:
//
//   1. **Watchdog.** No proposal within `command_timeout_s` means the thing
//      upstream is gone. Zero, and keep publishing zero.
//   2. **Forward clearance.** Measured from the raw scan, not from anything the
//      navigator computed, so a bug in the perception stack cannot talk the
//      safety layer out of stopping. Inside `stop_distance_m` the robot stops;
//      between there and `slow_distance_m` forward speed is scaled down.
//   3. **Envelope.** Clamp to the Create 3's real limits, and rate-limit
//      acceleration. A command the base cannot track is a command whose
//      resulting motion the policy never predicted.
//
// Rotation is deliberately NOT gated by clearance. A robot that has driven into
// a corner has to turn to get out, and a safety layer that forbids the only
// recovery available is not safer -- it is stuck.

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>

#include <sim2edge/preprocess.hpp>
#include <sim2edge/spec_generated.hpp>

namespace sim2edge_nodes
{

class SafetyNode : public rclcpp::Node
{
public:
  explicit SafetyNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sim2edge_safety", options)
  {
    command_timeout_ = declare_parameter("command_timeout_s", 0.30);
    scan_timeout_ = declare_parameter("scan_timeout_s", 0.50);
    stop_distance_ = declare_parameter("stop_distance_m", 0.25);
    slow_distance_ = declare_parameter("slow_distance_m", 0.55);
    // Half-angle of the sector that counts as "ahead". 35 degrees covers the
    // Create 3's body plus a margin at its width, without letting a wall
    // alongside a doorway read as an obstacle in front.
    forward_half_angle_ = declare_parameter("forward_half_angle_deg", 35.0) *
                          M_PI / 180.0;
    v_max_ = declare_parameter("v_max", static_cast<double>(sim2edge::spec::kVMax));
    omega_max_ = declare_parameter("omega_max",
                                   static_cast<double>(sim2edge::spec::kOmegaMax));
    accel_max_ = declare_parameter("accel_max",
                                   static_cast<double>(sim2edge::spec::kAccelMax));
    alpha_max_ = declare_parameter("alpha_max",
                                  static_cast<double>(sim2edge::spec::kAlphaMax));
    require_scan_ = declare_parameter("require_scan", true);
    period_ = declare_parameter("period_s",
                                static_cast<double>(sim2edge::spec::kDt));

    command_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "nav/command", 1,
      [this](geometry_msgs::msg::Twist::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        proposal_ = *msg;
        last_command_ = now();
      });
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        clearance_ = forwardClearance(*msg);
        last_scan_ = now();
      });

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
    reason_pub_ = create_publisher<std_msgs::msg::String>("safety/reason", 10);

    timer_ = create_wall_timer(std::chrono::duration<double>(period_),
                               std::bind(&SafetyNode::onTimer, this));

    RCLCPP_INFO(get_logger(),
      "safety up: stop %.2f m, slow %.2f m, forward sector +/-%.0f deg, "
      "watchdog %.2f s",
      stop_distance_, slow_distance_, forward_half_angle_ * 180.0 / M_PI,
      command_timeout_);
  }

private:
  /// Nearest valid return inside the forward sector, in metres.
  ///
  /// Computed from the raw message with its own validity limits, and binned by
  /// bearing rather than index for the same reason the policy's preprocessing
  /// is: the driver's start angle is not stable across versions, and a safety
  /// layer that looks in the wrong direction is worse than none.
  double forwardClearance(const sensor_msgs::msg::LaserScan & scan) const
  {
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
      const float r = scan.ranges[i];
      if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) {
        continue;
      }
      const double angle = static_cast<double>(scan.angle_min) +
                           static_cast<double>(i) *
                             static_cast<double>(scan.angle_increment);
      // Into base_link, exactly once -- the same quarter turn the policy path
      // applies. If this and the policy disagreed about which way is forward,
      // the failure would be a robot that stops when nothing is in front of it
      // and drives into what is.
      const double bearing =
        sim2edge::wrapAngle(angle + sim2edge::spec::kLidarMountYaw);
      if (std::abs(bearing) <= forward_half_angle_) {
        nearest = std::min(nearest, static_cast<double>(r));
      }
    }
    return nearest;
  }

  void onTimer()
  {
    geometry_msgs::msg::Twist proposal;
    double clearance = std::numeric_limits<double>::infinity();
    double command_age = 0.0;
    double scan_age = 0.0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      proposal = proposal_;
      clearance = clearance_;
      command_age = last_command_.nanoseconds() == 0
                      ? std::numeric_limits<double>::infinity()
                      : (now() - last_command_).seconds();
      scan_age = last_scan_.nanoseconds() == 0
                   ? std::numeric_limits<double>::infinity()
                   : (now() - last_scan_).seconds();
    }

    std::string reason;
    double v = proposal.linear.x;
    double omega = proposal.angular.z;

    if (command_age > command_timeout_) {
      v = omega = 0.0;
      reason = "no command from the navigator";
    } else if (require_scan_ && scan_age > scan_timeout_) {
      v = omega = 0.0;
      reason = "no scan";
    } else if (clearance <= stop_distance_) {
      // Forward only. Turning in place is how a cornered robot recovers, and
      // forbidding it would trade a stop for a permanent stop.
      v = std::min(v, 0.0);
      reason = "obstacle at " + std::to_string(clearance) + " m";
    } else if (clearance < slow_distance_) {
      const double scale = (clearance - stop_distance_) /
                           (slow_distance_ - stop_distance_);
      if (v > 0.0) {
        v *= scale;
        reason = "slowing near an obstacle";
      }
    }

    v = std::clamp(v, 0.0, v_max_);
    omega = std::clamp(omega, -omega_max_, omega_max_);

    // Rate limit. The oracle the policy was cloned from respected these
    // limits, so a step the base cannot track produces motion the policy never
    // predicted -- and the mismatch looks like a perception failure while
    // actually being dynamics.
    const double dv = accel_max_ * period_;
    const double dw = alpha_max_ * period_;
    v = std::clamp(v, last_v_ - dv, last_v_ + dv);
    omega = std::clamp(omega, last_omega_ - dw, last_omega_ + dw);
    last_v_ = v;
    last_omega_ = omega;

    geometry_msgs::msg::Twist out;
    out.linear.x = v;
    out.angular.z = omega;
    cmd_pub_->publish(out);

    if (reason != last_reason_) {
      std_msgs::msg::String message;
      message.data = reason.empty() ? "clear" : reason;
      reason_pub_->publish(message);
      if (!reason.empty()) {
        RCLCPP_WARN(get_logger(), "%s", reason.c_str());
      }
      last_reason_ = reason;
    }
  }

  double command_timeout_{};
  double scan_timeout_{};
  double stop_distance_{};
  double slow_distance_{};
  double forward_half_angle_{};
  double v_max_{};
  double omega_max_{};
  double accel_max_{};
  double alpha_max_{};
  bool require_scan_{};
  double period_{};

  std::mutex mutex_;
  geometry_msgs::msg::Twist proposal_;
  double clearance_{std::numeric_limits<double>::infinity()};
  rclcpp::Time last_command_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_scan_{0, 0, RCL_ROS_TIME};

  double last_v_{0.0};
  double last_omega_{0.0};
  std::string last_reason_{"__init__"};

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr reason_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace sim2edge_nodes

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Exceptions in the constructor (a missing replay file, an unreadable model)
  // are reported and exited on rather than allowed to abort: an abort gives a
  // core dump where a message gives the reason.
  try {
    rclcpp::spin(std::make_shared<sim2edge_nodes::SafetyNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("safety_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
