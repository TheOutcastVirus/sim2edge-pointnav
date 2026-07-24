#include "nav_policy_node/nav_policy_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include "nav_policy_node/cpu_backend.hpp"
#include "nav_policy_node/spec_generated.hpp"
#ifdef HAVE_QNN_BACKEND
#include "nav_policy_node/qnn_backend.hpp"
#endif

namespace nav_policy_node
{

namespace
{

/// Yaw from a quaternion. Only the z-axis rotation matters for a ground robot.
double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny, cosy);
}

/// Read the spec fingerprint out of the .pte's JSON sidecar.
///
/// Deliberately a hand-rolled scan rather than a JSON dependency: the sidecar
/// has exactly one field this node cares about, and adding nlohmann/json to the
/// board build to read it would not be a good trade.
std::string readSidecarFingerprint(const std::string & model_path)
{
  const auto dot = model_path.find_last_of('.');
  const std::string side =
    (dot == std::string::npos ? model_path : model_path.substr(0, dot)) +
    ".json";
  std::ifstream in(side);
  if (!in) {
    return {};
  }
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  const std::string key = "\"spec_fingerprint\"";
  auto pos = text.find(key);
  if (pos == std::string::npos) {
    return {};
  }
  pos = text.find('"', text.find(':', pos + key.size()));
  if (pos == std::string::npos) {
    return {};
  }
  const auto end = text.find('"', pos + 1);
  return end == std::string::npos ? std::string{}
         : text.substr(pos + 1, end - pos - 1);
}

}  // namespace

NavPolicyNode::NavPolicyNode(const rclcpp::NodeOptions & options)
: Node("nav_policy", options),
  scan_stamp_(0, 0, RCL_ROS_TIME),
  depth_stamp_(0, 0, RCL_ROS_TIME),
  odom_stamp_(0, 0, RCL_ROS_TIME)
{
  // ── Parameters ────────────────────────────────────────────────────────────
  declare_parameter("backend", std::string("npu"));
  declare_parameter("model_path", std::string(""));
  declare_parameter("qnn_lib_dir", std::string(""));
  declare_parameter("scan_topic", std::string("/scan"));
  declare_parameter("depth_topic", std::string("/oak/stereo/image_raw"));
  declare_parameter("odom_topic", std::string("/odom"));
  declare_parameter("goal_topic", std::string("~/goal"));
  declare_parameter("cmd_topic", std::string("~/cmd_raw"));
  declare_parameter("debug_topic", std::string("~/debug"));
  // Goal as (x, y) relative to the robot's pose at startup -- the deployment
  // task definition. Set via parameter or overridden live on goal_topic.
  declare_parameter("goal_x", 0.0);
  declare_parameter("goal_y", 0.0);
  declare_parameter("goal_is_relative", true);
  declare_parameter("require_depth", true);
  declare_parameter("lidar_max_age", static_cast<double>(spec::kLidarMaxAgeS));
  declare_parameter("depth_max_age", static_cast<double>(spec::kDepthMaxAgeS));
  declare_parameter("odom_max_age", static_cast<double>(spec::kOdomMaxAgeS));
  declare_parameter("control_rate", static_cast<double>(spec::kControlRateHz));
  declare_parameter("strict_fingerprint", true);

  const auto backend_name = get_parameter("backend").as_string();
  auto model_path = get_parameter("model_path").as_string();
  const auto qnn_lib_dir = get_parameter("qnn_lib_dir").as_string();
  lidar_max_age_ = get_parameter("lidar_max_age").as_double();
  depth_max_age_ = get_parameter("depth_max_age").as_double();
  odom_max_age_ = get_parameter("odom_max_age").as_double();
  require_depth_ = get_parameter("require_depth").as_bool();
  goal_is_relative_ = get_parameter("goal_is_relative").as_bool();
  const double control_rate = get_parameter("control_rate").as_double();

  if (model_path.empty()) {
    model_path = (backend_name == "cpu")
      ? "models/nav_policy_xnnpack.pte"
      : "models/nav_policy_qnn.pte";
  }

  // ── Backend ───────────────────────────────────────────────────────────────
  if (backend_name == "cpu") {
    backend_ = std::make_unique<CpuBackend>();
  } else if (backend_name == "npu") {
#ifdef HAVE_QNN_BACKEND
    backend_ = std::make_unique<QnnBackend>(qnn_lib_dir);
#else
    throw std::runtime_error(
            "nav_policy_node was compiled without QNN support. Rebuild with "
            "-DBUILD_QNN_BACKEND=ON, or run with backend:=cpu.");
#endif
  } else {
    throw std::invalid_argument(
            "unknown backend '" + backend_name + "'; choose 'cpu' or 'npu'");
  }

  verifyModelContract(model_path);
  if (!backend_->load(model_path)) {
    throw std::runtime_error("failed to load model: " + model_path);
  }
  RCLCPP_INFO(get_logger(), "Loaded '%s' on %s backend (spec %s)",
              model_path.c_str(), backend_->name(), spec::kSpecFingerprint);

  // ── Goal ──────────────────────────────────────────────────────────────────
  const double gx = get_parameter("goal_x").as_double();
  const double gy = get_parameter("goal_y").as_double();
  if (gx != 0.0 || gy != 0.0) {
    has_goal_ = true;
    goal_world_x_ = gx;
    goal_world_y_ = gy;
    // A relative goal cannot be anchored until the first odometry message
    // arrives and defines the starting pose.
    goal_anchored_ = !goal_is_relative_;
    RCLCPP_INFO(get_logger(), "Goal (%.2f, %.2f) %s", gx, gy,
                goal_is_relative_ ? "relative to start pose" : "in odom frame");
  } else {
    RCLCPP_INFO(get_logger(), "No goal set; waiting on '%s'",
                get_parameter("goal_topic").as_string().c_str());
  }

  // ── ROS interfaces ────────────────────────────────────────────────────────
  // Queue depth 1 everywhere: this is a control loop, and a queued sensor
  // message is by definition one the controller already missed its chance to
  // act on. SensorDataQoS is best-effort, matching the drivers.
  const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    get_parameter("scan_topic").as_string(), sensor_qos,
    std::bind(&NavPolicyNode::onScan, this, std::placeholders::_1));
  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    get_parameter("depth_topic").as_string(), sensor_qos,
    std::bind(&NavPolicyNode::onDepth, this, std::placeholders::_1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    get_parameter("odom_topic").as_string(), rclcpp::QoS(1),
    std::bind(&NavPolicyNode::onOdom, this, std::placeholders::_1));
  goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
    get_parameter("goal_topic").as_string(), rclcpp::QoS(1),
    std::bind(&NavPolicyNode::onGoal, this, std::placeholders::_1));

  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
    get_parameter("cmd_topic").as_string(), rclcpp::QoS(1));
  debug_pub_ = create_publisher<std_msgs::msg::String>(
    get_parameter("debug_topic").as_string(), rclcpp::QoS(1));

  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / control_rate),
    std::bind(&NavPolicyNode::onTimer, this));

  RCLCPP_INFO(get_logger(),
              "nav_policy up at %.1f Hz. Publishing raw commands on '%s' -- "
              "nav_safety owns /cmd_vel.",
              control_rate, get_parameter("cmd_topic").as_string().c_str());
}

void NavPolicyNode::verifyModelContract(const std::string & model_path)
{
  const std::string got = readSidecarFingerprint(model_path);
  const std::string want = spec::kSpecFingerprint;
  if (got == want) {
    return;
  }

  // A preprocessing-contract mismatch does not crash and does not log -- it
  // silently feeds the network inputs unlike the ones it was quantized on, and
  // the robot drives confidently and wrongly. Refusing to start is the correct
  // response.
  const std::string message =
    got.empty()
    ? ("no spec fingerprint sidecar found next to '" + model_path +
       "'. Export with tools/export_qnn.py, which writes one.")
    : ("model was exported under spec fingerprint '" + got +
       "' but this node was built against '" + want +
       "'. Re-run tools/gen_spec_header.py and rebuild, or re-export the "
       "model.");

  if (get_parameter("strict_fingerprint").as_bool()) {
    throw std::runtime_error("spec contract mismatch: " + message);
  }
  RCLCPP_ERROR(get_logger(), "SPEC CONTRACT MISMATCH (continuing anyway "
               "because strict_fingerprint:=false): %s", message.c_str());
}

// ── Callbacks ───────────────────────────────────────────────────────────────

void NavPolicyNode::onScan(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  scan_ = msg;
  scan_stamp_ = now();
}

void NavPolicyNode::onDepth(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (msg->encoding != "16UC1" && msg->encoding != "mono16") {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "expected 16UC1 depth, got '%s'", msg->encoding.c_str());
    return;
  }
  if (msg->step < msg->width * 2 ||
      msg->data.size() < static_cast<std::size_t>(msg->step) * msg->height)
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                          "malformed depth image");
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  depth_ = msg;
  depth_stamp_ = now();
}

void NavPolicyNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_goal_ && !goal_anchored_) {
    // Anchor the relative goal to the first pose we ever see. From here on the
    // goal lives in the odom frame and is transformed into the robot frame each
    // tick, so accumulated drift moves the *apparent* goal -- which is exactly
    // the corruption the odometry-drift randomization trained against.
    const double yaw = yawFromQuaternion(msg->pose.pose.orientation);
    const double gx = goal_world_x_;
    const double gy = goal_world_y_;
    goal_world_x_ = msg->pose.pose.position.x +
      std::cos(yaw) * gx - std::sin(yaw) * gy;
    goal_world_y_ = msg->pose.pose.position.y +
      std::sin(yaw) * gx + std::cos(yaw) * gy;
    goal_anchored_ = true;
    RCLCPP_INFO(get_logger(), "Goal anchored to odom frame at (%.2f, %.2f)",
                goal_world_x_, goal_world_y_);
  }
  odom_ = msg;
  odom_stamp_ = now();
}

void NavPolicyNode::onGoal(
  const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  goal_world_x_ = msg->pose.position.x;
  goal_world_y_ = msg->pose.position.y;
  has_goal_ = true;
  goal_anchored_ = true;  // a published goal is already in the odom frame
  RCLCPP_INFO(get_logger(), "New goal: (%.2f, %.2f)", goal_world_x_,
              goal_world_y_);
}

double NavPolicyNode::ageOf(const rclcpp::Time & stamp) const
{
  if (stamp.nanoseconds() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  return (now() - stamp).seconds();
}

// ── Observation assembly ────────────────────────────────────────────────────

bool NavPolicyNode::buildObservation(Observation & obs, std::string & reason)
{
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
  sensor_msgs::msg::Image::ConstSharedPtr depth;
  nav_msgs::msg::Odometry::ConstSharedPtr odom;
  double scan_age, depth_age, odom_age;
  double goal_x, goal_y;
  bool ready;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    scan = scan_;
    depth = depth_;
    odom = odom_;
    scan_age = ageOf(scan_stamp_);
    depth_age = ageOf(depth_stamp_);
    odom_age = ageOf(odom_stamp_);
    goal_x = goal_world_x_;
    goal_y = goal_world_y_;
    ready = has_goal_ && goal_anchored_;
  }

  if (!ready) {
    reason = "no goal";
    return false;
  }
  if (!scan || scan_age > lidar_max_age_) {
    reason = "lidar stale/missing (" + std::to_string(scan_age) + " s)";
    return false;
  }
  if (!odom || odom_age > odom_max_age_) {
    reason = "odom stale/missing (" + std::to_string(odom_age) + " s)";
    return false;
  }

  preprocessLidar(scan->ranges.data(), scan->ranges.size(), scan->angle_min,
                  scan->angle_increment, scan->range_min, scan->range_max,
                  obs.lidar);

  // A stale or missing depth frame is not automatically fatal. With
  // require_depth:=false the node substitutes the "no return" image and keeps
  // driving on lidar alone -- a graceful degradation to the Gate-B policy's
  // behaviour rather than a stop. It is opt-in because it is only safe if the
  // policy was trained with the depth channel sometimes uninformative.
  const bool depth_ok = depth && depth_age <= depth_max_age_;
  if (!depth_ok && require_depth_) {
    reason = "depth stale/missing (" + std::to_string(depth_age) + " s)";
    return false;
  }
  if (depth_ok) {
    preprocessDepthMillimetres(
      reinterpret_cast<const uint16_t *>(depth->data.data()),
      static_cast<int>(depth->height), static_cast<int>(depth->width),
      obs.depth);
  } else {
    std::fill(obs.depth.begin(), obs.depth.end(), spec::kDepthInvalidFill);
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "running lidar-only: depth is stale (%.3f s)",
                         depth_age);
  }

  const double yaw = yawFromQuaternion(odom->pose.pose.orientation);
  float rel_x = 0.0f;
  float rel_y = 0.0f;
  worldGoalToRobotFrame(
    static_cast<float>(goal_x), static_cast<float>(goal_y),
    static_cast<float>(odom->pose.pose.position.x),
    static_cast<float>(odom->pose.pose.position.y),
    static_cast<float>(yaw), rel_x, rel_y);

  preprocessGoal(rel_x, rel_y, obs.goal);
  preprocessLastAction(last_v_, last_omega_, obs.last_action);
  return true;
}

// ── Control loop ────────────────────────────────────────────────────────────

void NavPolicyNode::onTimer()
{
  std::string reason;
  if (!buildObservation(observation_, reason)) {
    // Publish nothing. nav_safety's watchdog sees the silence and stops the
    // robot -- a positive stop rather than a stale command being repeated.
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "no command: %s", reason.c_str());
    last_v_ = 0.0f;
    last_omega_ = 0.0f;
    return;
  }

  std::vector<float> action;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    action = backend_->infer(observation_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                          "inference failed: %s", e.what());
    return;
  }
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();

  if (action.size() != 2) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                          "model returned %zu outputs, expected 2",
                          action.size());
    return;
  }

  ++inference_count_;
  inference_time_sum_ += ms;
  inference_time_max_ = std::max(inference_time_max_, ms);
  if (inference_count_ % 200 == 0) {
    RCLCPP_INFO(get_logger(), "inference: mean %.2f ms, max %.2f ms over %zu "
                "(budget %.1f ms at %.0f Hz)",
                inference_time_sum_ / inference_count_, inference_time_max_,
                inference_count_, 1000.0 / spec::kControlRateHz,
                spec::kControlRateHz);
  }

  // The squash and scaling are inside the graph, so these are already SI units
  // within the trained limits. Clamping here anyway is cheap insurance against
  // a quantization artifact at the tanh saturation point.
  const float v = std::min(std::max(action[0], spec::kVMin), spec::kVMax);
  const float omega =
    std::min(std::max(action[1], -spec::kOmegaMax), spec::kOmegaMax);

  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = v;
  cmd.angular.z = omega;
  cmd_pub_->publish(cmd);

  last_v_ = v;
  last_omega_ = omega;

  // Debug is built only when someone is listening, per the reference pattern:
  // formatting a string every tick at 20 Hz for nobody is pure waste on a board
  // that is also running the NPU.
  if (debug_pub_->get_subscription_count() > 0) {
    const auto lidar_min = *std::min_element(observation_.lidar.begin(),
                                             observation_.lidar.end());
    const auto depth_min = *std::min_element(observation_.depth.begin(),
                                             observation_.depth.end());
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "v=%.3f w=%+.3f | goal d=%.3f sin=%+.3f cos=%+.3f | "
                  "lidar_min=%.3f depth_min=%.3f | infer=%.2fms",
                  v, omega, observation_.goal[0], observation_.goal[1],
                  observation_.goal[2], lidar_min, depth_min, ms);
    std_msgs::msg::String msg;
    msg.data = buf;
    debug_pub_->publish(msg);
  }
}

}  // namespace nav_policy_node
