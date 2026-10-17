#include "sim2edge_nodes/navigator_node.hpp"

#include <algorithm>
#include <cmath>

#include <sim2edge/deskew.hpp>
#include <sim2edge/preprocess.hpp>
#include <sim2edge/spec_generated.hpp>

#include <rclcpp/executors/multi_threaded_executor.hpp>

namespace sim2edge_nodes
{
namespace
{

/// Yaw from a quaternion, planar robot assumed.
double yawOf(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

}  // namespace

NavigatorNode::NavigatorNode(const rclcpp::NodeOptions & options)
: Node("sim2edge_navigator", options)
{
  // ── Parameters ────────────────────────────────────────────────────────────
  sim2edge::Navigator::Config config;
  config.localizer.max_submap_scans =
    static_cast<int>(declare_parameter("localizer.max_submap_scans", 128));
  config.localizer.point_stride =
    static_cast<int>(declare_parameter("localizer.point_stride", 2));
  config.localizer.correction_gain =
    declare_parameter("localizer.correction_gain", 0.10);
  config.localizer.yaw_correction_gain =
    declare_parameter("localizer.yaw_correction_gain", 0.05);
  config.localizer.max_correction_m =
    declare_parameter("localizer.max_correction_m", 0.12);
  config.localizer.max_correction_yaw_rad =
    declare_parameter("localizer.max_correction_yaw_deg", 4.0) * M_PI / 180.0;
  config.localizer.motion_scale_update_rate =
    declare_parameter("localizer.motion_scale_update_rate", 0.02);
  config.localizer.min_points =
    static_cast<int>(declare_parameter("localizer.min_points", 60));
  config.localizer.max_iterations =
    static_cast<int>(declare_parameter("localizer.max_iterations", 6));

  config.arrival_radius_m = declare_parameter("arrival.radius_m", 0.15);
  config.arrival_ticks =
    static_cast<int>(declare_parameter("arrival.ticks", 3));
  config.travel_sigma_per_m = declare_parameter("arrival.travel_sigma_per_m",
                                                0.01);
  // Default false: the self-contained arm, whose three inputs are the lidar,
  // the gyro and the base's velocity. Setting this true adds the Create 3's own
  // odometry increment as the prior, which is stronger but no longer
  // lidar/IMU-only -- the localizer stamps which one it used so the distinction
  // survives into the log.
  config.use_odometry_prior = declare_parameter("use_odometry_prior", false);

  navigator_ = sim2edge::Navigator(config);

  // Which component of angular_velocity is this robot's yaw rate, and with
  // what sign. See gyro_yaw_axis in navigator.yaml -- on this build it is 'y',
  // measured, not 'z' as the code assumed for its whole life before hardware.
  gyro_yaw_axis_ = declare_parameter("gyro_yaw_axis", std::string("y"));
  gyro_yaw_sign_ = declare_parameter("gyro_yaw_sign", 1.0);
  if (gyro_yaw_axis_ != "x" && gyro_yaw_axis_ != "y" && gyro_yaw_axis_ != "z") {
    throw std::runtime_error(
      "gyro_yaw_axis must be one of x, y, z -- got '" + gyro_yaw_axis_ + "'");
  }
  RCLCPP_INFO(get_logger(), "gyro yaw rate = %sangular_velocity.%s",
              gyro_yaw_sign_ < 0 ? "-" : "", gyro_yaw_axis_.c_str());

  scan_max_age_ = declare_parameter("scan_max_age_s", 0.25);
  imu_max_age_ = declare_parameter("imu_max_age_s", 0.15);
  odom_max_age_ = declare_parameter("odom_max_age_s", 0.25);
  require_odom_ = declare_parameter("require_odom", true);
  goal_is_body_frame_ = declare_parameter("goal_is_body_frame", true);
  goal_max_age_ = declare_parameter("goal_max_age_s", 0.0);
  control_period_ = declare_parameter("control_period_s",
                                      static_cast<double>(sim2edge::spec::kDt));
  latency_us_.assign(kLatencyWindow, 0.0);

  loadActor();

  // ── Interfaces ────────────────────────────────────────────────────────────
  // Depth 1, best effort on the sensors: a queued scan is a stale scan, and the
  // control loop wants the newest one or none.
  const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
  sensor_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions sensor_options;
  sensor_options.callback_group = sensor_callback_group_;
  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    "scan", sensor_qos,
    std::bind(&NavigatorNode::onScan, this, std::placeholders::_1),
    sensor_options);
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "imu", sensor_qos,
    std::bind(&NavigatorNode::onImu, this, std::placeholders::_1),
    sensor_options);
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "odom", sensor_qos,
    std::bind(&NavigatorNode::onOdom, this, std::placeholders::_1),
    sensor_options);
  goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
    "goal_pose", 1,
    std::bind(&NavigatorNode::onGoal, this, std::placeholders::_1));

  command_pub_ = create_publisher<geometry_msgs::msg::Twist>("nav/command", 1);
  status_pub_ = create_publisher<sim2edge_msgs::msg::NavStatus>("nav/status", 10);
  localizer_pub_ = create_publisher<sim2edge_msgs::msg::LocalizerState>(
    "localizer/state", 10);
  deskewed_scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
    "nav/deskewed_scan", sensor_qos);

  reset_srv_ = create_service<std_srvs::srv::Trigger>(
    "nav/reset",
    [this](const std_srvs::srv::Trigger::Request::SharedPtr,
           std_srvs::srv::Trigger::Response::SharedPtr response) {
      std::lock_guard<std::mutex> lock(mutex_);
      navigator_.reset();
      if (async_actor_) async_actor_->invalidate();
      have_goal_update_ = false;
      have_previous_odom_ = false;
      have_imu_ = false;
      gyro_accumulator_ = 0.0;
      response->success = true;
      response->message = "tracker, history and goal cleared";
      RCLCPP_INFO(get_logger(), "navigator reset");
    });

  timer_ = create_wall_timer(
    std::chrono::duration<double>(control_period_),
    std::bind(&NavigatorNode::onControlTimer, this));

  RCLCPP_INFO(get_logger(),
    "sim2edge navigator up: %.0f Hz, contract %s/%s, backend %s, "
    "submap %d stride %d, prior %s",
    1.0 / control_period_, sim2edge::spec::kSensorContractVersion,
    sim2edge::spec::kObservationFingerprint, backend_name_.c_str(),
    config.localizer.max_submap_scans, config.localizer.point_stride,
    config.use_odometry_prior ? "create3-odom" : "command-gyro");
}

void NavigatorNode::loadActor()
{
  const std::string path = declare_parameter("model_path", std::string(""));
  const std::string backend = declare_parameter("backend", std::string("qnn"));
  const std::string qnn_lib_dir = declare_parameter(
    "qnn_lib_dir", std::string(""));

  if (path.empty()) {
    RCLCPP_WARN(get_logger(),
      "no model_path: perception and localization will run and publish, but "
      "the command will stay zero. This is the configuration the preprocessing "
      "bring-up check uses.");
    return;
  }

  std::string error;
  auto loaded = sim2edge::loadActor(
    path, backend == "cpu" ? sim2edge::ActorBackend::kCpu
                           : sim2edge::ActorBackend::kQnn,
    qnn_lib_dir, error);
  if (!loaded) {
    // Not fatal, and deliberately so: a node that exits takes its diagnostics
    // with it, and "the NPU did not come up" is exactly when they are wanted.
    // The robot does not move -- the command stays zero -- which is the safe
    // failure and is visible in nav/status as a hold reason.
    RCLCPP_ERROR(get_logger(), "could not load the actor: %s", error.c_str());
    return;
  }

  RCLCPP_INFO(get_logger(), "actor %s on %s, loaded in %.1f ms",
              path.c_str(), backend.c_str(), loaded->info().load_ms);
  backend_name_ = backend;
  actor_ = std::move(loaded);
  const double max_age = declare_parameter("actor_max_age_s", 0.5);
  if (max_age <= 0.0) throw std::runtime_error("actor_max_age_s must be positive");
  async_actor_ = std::make_shared<sim2edge::AsyncActor>(actor_, max_age);
  navigator_.setActor(async_actor_);
}

void NavigatorNode::onScan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  scan_ = std::move(msg);
  scan_stamp_ = scan_->header.stamp;
}

double NavigatorNode::yawRateOf(const sensor_msgs::msg::Imu & msg) const
{
  const auto & w = msg.angular_velocity;
  const double raw = gyro_yaw_axis_ == "x" ? w.x
                   : gyro_yaw_axis_ == "y" ? w.y
                                           : w.z;
  return gyro_yaw_sign_ * raw;
}

void NavigatorNode::onImu(sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const rclcpp::Time stamp(msg->header.stamp);
  if (have_imu_) {
    // Trapezoid over the real inter-sample interval. Integrating here rather
    // than sampling the rate at the control tick is what makes the deskew
    // rotation right: at 100+ Hz the IMU sees several samples per tick, and a
    // hard turn is exactly when they differ.
    const double dt = (stamp - last_imu_stamp_).seconds();
    if (dt > 0.0 && dt < 0.5) {
      gyro_accumulator_ += yawRateOf(*msg) * dt;
    }
  }
  last_imu_stamp_ = stamp;
  // Freshness asks whether the sensor is still delivering, so use receipt
  // time rather than the DepthAI-converted header clock.  The RGB-D driver
  // has been measured continuously delivering 100+ Hz while its converted
  // stamps wander 0.25-0.7 s behind ROS time, which otherwise pins the
  // navigator on "imu stale" despite a live gyro.  Keep `stamp` above for the
  // integration interval: capture-time deltas, unlike the absolute epoch,
  // remain the right clock for deskew.
  imu_stamp_ = now();
  have_imu_ = true;
}

void NavigatorNode::onOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  odom_ = std::move(msg);
  // Arrival time, NOT odom_->header.stamp, and this one is deliberate.
  //
  // The Create 3 has no RTC and does not sync, so it stamps its messages from
  // its own unanchored clock -- measured 3.73 years behind this board, which
  // makes every stamp look 1.2e8 seconds old and pins the loop on "odometry
  // stale" forever.
  //
  // Substituting arrival time costs nothing real here, because the base's
  // clock never enters the arithmetic: the tick reads twist.linear.x, a
  // scalar, and the optional pose prior is a difference of consecutive poses.
  // Neither is a function of time. The freshness check is therefore the only
  // consumer of the stamp, and what it is actually asking -- "is the base
  // still talking to me" -- is answered at least as well by arrival.
  //
  // It is still a weaker check than the scan's and the IMU's, which do use
  // device stamps: a message delayed in transport reads as fresh here. If the
  // base is ever clock-synced, revert this to the header stamp.
  odom_stamp_ = now();
}

void NavigatorNode::onGoal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  double x = msg->pose.position.x;
  double y = msg->pose.position.y;

  if (!goal_is_body_frame_) {
    // An odom-frame goal has to become a body-frame offset before it enters the
    // tracker, because the tracker has no odom frame -- it has its own, which
    // starts wherever it was last reset.
    if (!odom_) {
      RCLCPP_WARN(get_logger(),
        "goal given in the odom frame but no odometry has arrived; ignoring");
      return;
    }
    const double rx = odom_->pose.pose.position.x;
    const double ry = odom_->pose.pose.position.y;
    const double ryaw = yawOf(odom_->pose.pose.orientation);
    sim2edge::toBodyFrame(x - rx, y - ry, ryaw, x, y);
  }

  if (std::hypot(x, y) < 1e-6) {
    navigator_.clearGoal();
    if (async_actor_) async_actor_->invalidate();
  } else {
    navigator_.setGoalBody(x, y);
  }
  have_goal_update_ = true;
  last_goal_update_ = std::chrono::steady_clock::now();
  RCLCPP_DEBUG(get_logger(), "goal set: (%+.2f, %+.2f) m in the body frame, "
               "%.2f m away", x, y, std::sqrt(x * x + y * y));
}

double NavigatorNode::ageOf(const rclcpp::Time & stamp) const
{
  if (stamp.nanoseconds() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  return (now() - stamp).seconds();
}

void NavigatorNode::onControlTimer()
{
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
  nav_msgs::msg::Odometry::ConstSharedPtr odom;
  double gyro_delta = 0.0;
  double scan_age = 0.0;
  double imu_age = 0.0;
  double odom_age = 0.0;
  double goal_age = 0.0;
  bool have_goal_update = false;
  std::string hold;
  std::string goal_hold;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    scan = scan_;
    odom = odom_;
    scan_age = ageOf(scan_stamp_);
    imu_age = ageOf(imu_stamp_);
    odom_age = ageOf(odom_stamp_);
    have_goal_update = have_goal_update_;
    goal_age = have_goal_update
      ? std::chrono::duration<double>(
          std::chrono::steady_clock::now() - last_goal_update_).count()
      : 0.0;

    // Consume the integral: whatever rotation accumulated since the last tick
    // is this tick's increment, and it must not be counted twice.
    gyro_delta = gyro_accumulator_;
    gyro_accumulator_ = 0.0;
  }

  if (!scan) {
    hold = "no scan yet";
  } else if (scan_age > scan_max_age_) {
    hold = "scan stale";
  } else if (imu_age > imu_max_age_) {
    hold = "imu stale";
  } else if (require_odom_ && (!odom || odom_age > odom_max_age_)) {
    hold = odom ? "odometry stale" : "no odometry yet";
  }
  if (goal_max_age_ > 0.0 && !have_goal_update) {
    goal_hold = "no goal yet";
  } else if (goal_max_age_ > 0.0 && goal_age > goal_max_age_) {
    goal_hold = "goal stale";
  }

  if (!hold.empty()) {
    if (async_actor_) async_actor_->invalidate();
    // Hold means publish zero, every tick, on the same schedule. Publishing
    // nothing would let the safety node's watchdog decide the difference
    // between "stopped on purpose" and "the navigator died", and those want
    // different responses.
    ++hold_count_;
    geometry_msgs::msg::Twist stop;
    command_pub_->publish(stop);

    sim2edge_msgs::msg::NavStatus status;
    status.header.stamp = now();
    status.scan_age_s = scan_age;
    status.imu_age_s = imu_age;
    status.odom_age_s = odom_age;
    status.hold_reason = hold;
    status.backend = backend_name_;
    status_pub_->publish(status);

    // The gyro integral is not put back: it belongs to a tick that was not
    // served, and re-applying it later would attribute rotation to the wrong
    // interval. Deskew under-corrects for one tick instead, which is the
    // honest failure.
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "holding: %s (scan %.3f s, imu %.3f s, odom %.3f s)", hold.c_str(),
      scan_age, imu_age, odom_age);
    return;
  }

  // A missing goal is a command gate, not a sensor-processing gate.  Keep
  // deskew and localization running while stopped so an object-goal planner
  // can consume /localizer/state and /nav/deskewed_scan to create its first
  // body-frame goal.  Returning above for "no goal" creates a bootstrap
  // deadlock: the planner cannot publish a goal until these streams exist,
  // while these streams would not exist until the planner published a goal.
  if (!goal_hold.empty()) {
    if (async_actor_) async_actor_->invalidate();
    ++hold_count_;
    if (goal_hold == "goal stale") {
      navigator_.clearGoal();
    }
  }

  const double speed = odom ? odom->twist.twist.linear.x : 0.0;

  // The trailing age is the delay between the last beam of the revolution and
  // now: the driver stamps the scan at its start, so the sweep finished
  // scan_time after that, and the control tick is being served later still.
  const double trailing = std::max(
    0.0, scan_age - static_cast<double>(scan->scan_time));
  sim2edge::beamGeometry(scan->ranges.size(), scan->angle_min,
                         scan->angle_increment, scan->time_increment, trailing,
                         bearings_, ages_);

  // The odometry prior, as a body-frame increment.
  double prior[3] = {0.0, 0.0, 0.0};
  bool have_prior = false;
  if (odom) {
    const double x = odom->pose.pose.position.x;
    const double y = odom->pose.pose.position.y;
    const double yaw = yawOf(odom->pose.pose.orientation);
    if (have_previous_odom_) {
      sim2edge::toBodyFrame(x - previous_odom_x_, y - previous_odom_y_,
                            previous_odom_yaw_, prior[0], prior[1]);
      prior[2] = sim2edge::wrapAngle(yaw - previous_odom_yaw_);
      have_prior = true;
    }
    previous_odom_x_ = x;
    previous_odom_y_ = y;
    previous_odom_yaw_ = yaw;
    have_previous_odom_ = true;
  }

  const sim2edge::Navigator::Result * result = nullptr;
  try {
    result = &navigator_.tick(scan->ranges.data(), bearings_.data(),
                              ages_.data(), scan->ranges.size(), gyro_delta,
                              speed, control_period_,
                              have_prior ? prior : nullptr);
  } catch (const std::exception & e) {
    if (async_actor_) async_actor_->invalidate();
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "tick failed: %s -- commanding zero", e.what());
    geometry_msgs::msg::Twist stop;
    command_pub_->publish(stop);
    return;
  }
  ++tick_count_;
  if (result->arrived && async_actor_) async_actor_->invalidate();

  geometry_msgs::msg::Twist command;
  command.linear.x = result->command.v;
  command.angular.z = result->command.omega;
  command_pub_->publish(command);

  sim2edge_msgs::msg::NavStatus status;
  status.header.stamp = now();
  status.v = result->command.v;
  status.omega = result->command.omega;
  status.has_goal = result->has_goal;
  status.arrived = result->arrived;
  status.arrival_streak = result->arrival_streak;
  status.goal_body_x = result->goal_body_x;
  status.goal_body_y = result->goal_body_y;
  status.belief_distance_m = result->belief_distance_m;
  status.sigma_m = result->sigma_m;
  status.travelled_m = result->travelled_m;
  status.scan_age_s = scan_age;
  status.imu_age_s = imu_age;
  status.odom_age_s = odom_age;
  status.deskew_us = result->deskew_us;
  status.bin_us = result->bin_us;
  status.localize_us = result->localize_us;
  status.infer_us = result->infer_us;
  status.total_us = result->total_us;
  status.backend = backend_name_;
  if (!goal_hold.empty()) {
    status.hold_reason = goal_hold;
  } else if (!result->has_goal) {
    status.hold_reason = "no goal";
  } else if (result->arrived) {
    status.hold_reason = "arrived";
  } else if (!actor_) {
    status.hold_reason = "no actor loaded";
  } else if (async_actor_ && !async_actor_->stats().valid) {
    status.hold_reason = "actor pending or stale";
  }
  if (async_actor_) {
    const auto stats = async_actor_->stats();
    status.infer_us = stats.latency_ms * 1000.0;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
      "actor worker: updates %lu, latency %.1f ms, input age %.3f s, valid %d",
      stats.updates, stats.latency_ms, stats.age_s, stats.valid);
  }
  status_pub_->publish(status);

  const auto & estimate = navigator_.localizer().estimate();
  const auto & timings = navigator_.localizer().timings();
  sim2edge_msgs::msg::LocalizerState state;
  state.header.stamp = status.header.stamp;
  state.x = estimate.pose.x;
  state.y = estimate.pose.y;
  state.yaw = estimate.pose.yaw;
  std::copy(std::begin(estimate.covariance), std::end(estimate.covariance),
            state.covariance.begin());
  state.motion_scale = estimate.motion_scale;
  state.observable = estimate.observable;
  state.correction_x = estimate.correction.x;
  state.correction_y = estimate.correction.y;
  state.correction_yaw = estimate.correction.yaw;
  state.points_used = estimate.points_used;
  state.iterations = estimate.iterations;
  state.submap_points = static_cast<int32_t>(navigator_.localizer().submapSize());
  state.motion_prior_source = estimate.motion_prior_source;
  state.extract_us = timings.extract_us;
  state.submap_us = timings.submap_us;
  state.icp_us = timings.icp_us;
  state.total_us = timings.total_us;
  localizer_pub_->publish(state);

  // Publish the exact deskew result consumed by the actor and localizer. A
  // LaserScan has one uniform angle increment, so rasterize the corrected
  // per-return bearings at the raw scan's angular resolution with a minimum
  // reduction; a thin obstacle must survive if two returns share a bin. Give
  // it the localizer stamp and publish after the pose so the object mapper can
  // associate the two without treating an old pose as current geometry.
  sim2edge::rasterizeLidar(
    navigator_.sweep(), scan->ranges.size(), deskewed_ranges_);
  if (!deskewed_ranges_.empty()) {
    sensor_msgs::msg::LaserScan deskewed;
    deskewed.header.stamp = state.header.stamp;
    deskewed.header.frame_id = "base_link";
    deskewed.angle_min = static_cast<float>(-M_PI);
    deskewed.angle_increment = static_cast<float>(
      2.0 * M_PI / static_cast<double>(deskewed_ranges_.size()));
    deskewed.angle_max = deskewed.angle_min + deskewed.angle_increment *
      static_cast<float>(deskewed_ranges_.size() - 1);
    deskewed.time_increment = 0.0f;  // every return refers to header.stamp
    deskewed.scan_time = static_cast<float>(control_period_);
    deskewed.range_min = sim2edge::spec::kLidarMinM;
    deskewed.range_max = sim2edge::spec::kLidarMaxM;
    deskewed.ranges = deskewed_ranges_;
    deskewed_scan_pub_->publish(deskewed);
  }

  latency_us_[latency_head_] = result->total_us;
  latency_head_ = (latency_head_ + 1) % kLatencyWindow;
  if (latency_filled_ < kLatencyWindow) {
    ++latency_filled_;
  }
  if (result->total_us > control_period_ * 1.0e6) {
    ++overrun_count_;
  }

  // One line every five seconds. Enough to notice a budget problem from an ssh
  // session without drowning the log at 20 Hz. The percentiles are over the
  // rolling window, the overrun count is since startup: a tick that misses the
  // period is the number that matters and it must not age out of view.
  double median = 0.0;
  double p95 = 0.0;
  double worst = 0.0;
  {
    std::vector<double> window(latency_us_.begin(),
                               latency_us_.begin() +
                                 static_cast<std::ptrdiff_t>(latency_filled_));
    if (!window.empty()) {
      const std::size_t mid = window.size() / 2;
      const std::size_t hi = static_cast<std::size_t>(
        0.95 * static_cast<double>(window.size() - 1));
      std::nth_element(window.begin(),
                       window.begin() + static_cast<std::ptrdiff_t>(mid),
                       window.end());
      median = window[mid];
      std::nth_element(window.begin(),
                       window.begin() + static_cast<std::ptrdiff_t>(hi),
                       window.end());
      p95 = window[hi];
      worst = *std::max_element(window.begin(), window.end());
    }
  }

  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
    "tick %zu | now %.1f ms (localizer %.1f, actor %.2f) | window median %.1f "
    "p95 %.1f max %.1f ms | over %.0f ms: %zu | v %.3f w %+.3f | goal %.2f m "
    "sigma %.3f m | %s | held %zu",
    tick_count_, result->total_us / 1000.0, result->localize_us / 1000.0,
    result->infer_us / 1000.0, median / 1000.0, p95 / 1000.0, worst / 1000.0,
    control_period_ * 1000.0, overrun_count_, result->command.v,
    result->command.omega, result->belief_distance_m, result->sigma_m,
    estimate.observable ? "observable" : "prior-only", hold_count_);
}

}  // namespace sim2edge_nodes

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Exceptions in the constructor (a missing replay file, an unreadable model)
  // are reported and exited on rather than allowed to abort: an abort gives a
  // core dump where a message gives the reason.
  try {
    auto node = std::make_shared<sim2edge_nodes::NavigatorNode>();
    // Keep sensor receipt independent of localization. Accelerator inference
    // runs in AsyncActor's worker and never owns the Navigator instance.
    // The default group still serializes goal, reset, and control callbacks
    // around the Navigator instance.
    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("navigator_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
