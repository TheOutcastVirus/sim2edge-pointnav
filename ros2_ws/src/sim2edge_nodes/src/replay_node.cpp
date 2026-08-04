// Publish a recorded corpus as real sensor topics, so the on-board test drives
// the actual node graph rather than a harness that resembles it.
//
// sim2edge_parity already replays through sim2edge::Navigator directly, and
// that is the right tool for numeric agreement. It cannot answer a different
// set of questions: does the node subscribe to the right topics with the right
// QoS, does the 20 Hz timer hold under real DDS load, does the safety layer
// see the scan it thinks it sees, does the whole graph come up on the board at
// all. Those need messages on the wire.
//
// Two things this deliberately does not do:
//
//   **It publishes the pre-deskew sweep.** The recorded scan is what a driver
//   would hand over; deskew belongs to the node under test. Publishing the
//   corrected sweep would test the easy half.
//
//   **It does not close the loop.** The recorded corpus was driven by the
//   oracle, so the robot's pose does not respond to what the policy commands.
//   That makes this an input/output test of the node graph, not a navigation
//   trial, and the difference is worth keeping visible.

#include <cmath>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <sim2edge/replay.hpp>
#include <sim2edge/spec_generated.hpp>

namespace sim2edge_nodes
{

class ReplayNode : public rclcpp::Node
{
public:
  explicit ReplayNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sim2edge_replay", options)
  {
    const std::string path = declare_parameter("replay_path", std::string(""));
    loop_ = declare_parameter("loop", false);
    rate_scale_ = declare_parameter("rate_scale", 1.0);
    frame_id_ = declare_parameter("frame_id", std::string("laser_frame"));
    odom_frame_ = declare_parameter("odom_frame", std::string("odom"));
    // The IMU is published faster than the control loop, as the OAK-D's is,
    // so the navigator's integrator sees the multi-sample-per-tick case it
    // will see on hardware rather than a convenient one-to-one.
    imu_substeps_ = static_cast<int>(declare_parameter("imu_substeps", 5));
    startup_wait_ = declare_parameter("startup_wait_s", 10.0);

    if (path.empty()) {
      RCLCPP_FATAL(get_logger(), "replay_path is required");
      throw std::runtime_error("replay_path is required");
    }

    std::string error;
    if (!replay_.open(path, error)) {
      RCLCPP_FATAL(get_logger(), "%s", error.c_str());
      throw std::runtime_error(error);
    }
    if (replay_.observationFingerprint() !=
        sim2edge::spec::kObservationFingerprint)
    {
      RCLCPP_WARN(get_logger(),
        "replay was exported at observation fingerprint %s but this build is "
        "%s -- the navigator will be fed inputs from a different contract",
        replay_.observationFingerprint().c_str(),
        sim2edge::spec::kObservationFingerprint);
    }

    scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS().keep_last(1));
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
      "imu", rclcpp::SensorDataQoS().keep_last(1));
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      "odom", rclcpp::SensorDataQoS().keep_last(1));
    goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "goal_pose", rclcpp::QoS(1).transient_local());

    reset_client_ = create_client<std_srvs::srv::Trigger>("nav/reset");

    const double period = sim2edge::spec::kDt / std::max(rate_scale_, 1.0e-3);
    timer_ = create_wall_timer(std::chrono::duration<double>(period),
                               std::bind(&ReplayNode::onTimer, this));

    RCLCPP_INFO(get_logger(),
      "replaying %s: %zu ticks in %zu episodes, %u beams, at %.1fx",
      path.c_str(), replay_.tickCount(), replay_.episodeCount(),
      replay_.header().beam_count, rate_scale_);
  }

private:
  /// Reconstruct the LaserScan a driver would have produced for this tick.
  ///
  /// The replay stores per-beam bearings explicitly, because the corpus's beam
  /// layout carries a per-episode mount bias and is not exactly uniform. A
  /// LaserScan cannot express that -- it has one angle_min and one increment --
  /// so the message is built from the first and last recorded bearings, and the
  /// residual is the small bias the sensor model injected. That is a real and
  /// unavoidable lossy step in this direction only: the parity harness reads
  /// the exact bearings and is the tool that answers numeric questions.
  void publishScan(const sim2edge::ReplayTick & tick, const rclcpp::Time & stamp)
  {
    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp = stamp;
    scan.header.frame_id = frame_id_;
    const std::size_t beams = tick.beams;

    // Undo the mount rotation: beamGeometry in the node will apply it again,
    // and the driver's message is in the lidar's own frame.
    const double first = static_cast<double>(tick.bearings[0]) -
                         sim2edge::spec::kLidarMountYaw;
    const double last = static_cast<double>(tick.bearings[beams - 1]) -
                        sim2edge::spec::kLidarMountYaw;
    scan.angle_min = static_cast<float>(first);
    scan.angle_max = static_cast<float>(last);
    scan.angle_increment = static_cast<float>(
      (last - first) / static_cast<double>(beams - 1));
    scan.time_increment = static_cast<float>(
      sim2edge::spec::kSpinPeriodS / static_cast<double>(beams));
    scan.scan_time = static_cast<float>(sim2edge::spec::kSpinPeriodS);
    scan.range_min = sim2edge::spec::kLidarMinM;
    scan.range_max = sim2edge::spec::kLidarMaxM;
    scan.ranges.assign(tick.ranges, tick.ranges + beams);
    scan_pub_->publish(scan);
  }

  void publishGoal(double x, double y)
  {
    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp = now();
    goal.header.frame_id = "base_link";
    goal.pose.position.x = x;
    goal.pose.position.y = y;
    goal.pose.orientation.w = 1.0;
    goal_pub_->publish(goal);
  }

  void onTimer()
  {
    // Do not start until the navigator is discoverable.
    //
    // Launch starts all three processes at once, and DDS discovery takes a few
    // hundred milliseconds. Publishing into that window means the first
    // episode's reset never arrives, so it runs with no goal and reports a
    // believed distance of zero for its whole length -- a result that looks
    // like a localization failure and is actually a startup race. Waiting
    // costs a second and removes the whole class of confusion.
    if (!started_) {
      if (reset_client_->service_is_ready()) {
        started_ = true;
        RCLCPP_INFO(get_logger(), "navigator found after %.1f s; starting",
                    static_cast<double>(startup_ticks_) * sim2edge::spec::kDt);
      } else if (static_cast<double>(startup_ticks_) * sim2edge::spec::kDt >=
                 startup_wait_) {
        started_ = true;
        RCLCPP_WARN(get_logger(),
          "nav/reset did not appear within %.1f s; replaying anyway. Episode "
          "boundaries will not reset the tracker, so the submap will be "
          "stitched across teleports and the pose estimate is meaningless "
          "after the first episode.", startup_wait_);
      } else {
        ++startup_ticks_;
        return;
      }
    }

    if (cursor_ >= replay_.tickCount()) {
      if (!loop_) {
        RCLCPP_INFO_ONCE(get_logger(), "replay finished: %zu ticks published",
                         cursor_);
        return;
      }
      cursor_ = 0;
      episode_ = 0;
    }

    // Episode boundary: reset the navigator and re-issue the goal, the same
    // rule the offline harness uses. A submap stitched across a teleport is
    // geometry the tracker never sees on the robot.
    if (episode_ < replay_.episodeCount() &&
        cursor_ == replay_.episode(episode_).first_tick)
    {
      const auto & episode = replay_.episode(episode_);
      RCLCPP_INFO(get_logger(), "episode %zu: %u ticks, goal (%+.2f, %+.2f) m",
                  episode_, episode.tick_count, episode.goal_body_x,
                  episode.goal_body_y);

      // The goal must be published *after* the reset has been applied, not
      // merely after it was sent. A reset clears the goal -- correctly, since
      // on the robot it means "forget everything" -- and a service request and
      // a topic message have no ordering guarantee between them, so firing
      // both together lets the reset land second and silently discard the
      // goal. The symptom is a believed goal distance pinned at zero and a
      // robot that thinks it has already arrived.
      const double goal_x = episode.goal_body_x;
      const double goal_y = episode.goal_body_y;
      if (reset_client_->service_is_ready()) {
        reset_client_->async_send_request(
          std::make_shared<std_srvs::srv::Trigger::Request>(),
          [this, goal_x, goal_y](
            rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture) {
            publishGoal(goal_x, goal_y);
          });
      } else {
        RCLCPP_WARN(get_logger(),
          "nav/reset is not available; episode %zu will start with the "
          "previous episode's submap and its pose estimate", episode_);
        publishGoal(goal_x, goal_y);
      }
      ++episode_;
    }

    const sim2edge::ReplayTick tick = replay_.tick(cursor_);
    const rclcpp::Time stamp = now();

    publishScan(tick, stamp);

    // The gyro increment is spread over several IMU messages so the node's
    // integrator has to do its job. Their sum is exactly the recorded
    // increment, so nothing is added or lost.
    const double rate = tick.gyro_delta / sim2edge::spec::kDt;
    const double substep = sim2edge::spec::kDt /
                           static_cast<double>(std::max(imu_substeps_, 1));
    for (int i = 0; i < std::max(imu_substeps_, 1); ++i) {
      sensor_msgs::msg::Imu imu;
      imu.header.stamp = stamp + rclcpp::Duration::from_seconds(
        substep * static_cast<double>(i));
      imu.header.frame_id = "imu_link";
      imu.angular_velocity.z = rate;
      imu.orientation.w = 1.0;
      // Negative covariance[0] is the REP-145 way of saying "this field is
      // absent". The replay has no attitude to offer and must not imply one.
      imu.orientation_covariance[0] = -1.0;
      imu_pub_->publish(imu);
    }

    // Integrate the recorded body-frame increment into a plausible odom pose.
    // The navigator differences consecutive poses back into an increment, so
    // what matters is that the difference is right, not where the origin is.
    const double c = std::cos(odom_yaw_);
    const double s = std::sin(odom_yaw_);
    odom_x_ += c * tick.prior_delta_body[0] - s * tick.prior_delta_body[1];
    odom_y_ += s * tick.prior_delta_body[0] + c * tick.prior_delta_body[1];
    odom_yaw_ += tick.prior_delta_body[2];

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = "base_link";
    odom.pose.pose.position.x = odom_x_;
    odom.pose.pose.position.y = odom_y_;
    odom.pose.pose.orientation.z = std::sin(0.5 * odom_yaw_);
    odom.pose.pose.orientation.w = std::cos(0.5 * odom_yaw_);
    odom.twist.twist.linear.x = tick.speed;
    odom.twist.twist.angular.z = rate;
    odom_pub_->publish(odom);

    ++cursor_;
  }

  sim2edge::ReplayReader replay_;
  std::size_t cursor_{0};
  std::size_t episode_{0};
  bool loop_{false};
  double rate_scale_{1.0};
  int imu_substeps_{5};
  double startup_wait_{10.0};
  bool started_{false};
  std::size_t startup_ticks_{0};
  std::string frame_id_;
  std::string odom_frame_;

  double odom_x_{0.0};
  double odom_y_{0.0};
  double odom_yaw_{0.0};

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
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
    rclcpp::spin(std::make_shared<sim2edge_nodes::ReplayNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("replay_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
