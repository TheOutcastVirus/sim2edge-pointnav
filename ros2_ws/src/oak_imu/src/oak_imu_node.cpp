#include "oak_imu/oak_imu_node.hpp"

// depthai included ONLY in this translation unit -- never in the header. Its
// bundled dependencies conflict with rclcpp's when both reach one TU. Same
// containment as the reference project's oak_camera.
#include <depthai/depthai.hpp>

#include <chrono>
#include <functional>
#include <stdexcept>

namespace oak_imu
{

// In depthai v3, Pipeline owns the implicit Device and the whole node graph,
// and queues come from Node::Output rather than XLinkOut.
struct OakImuNode::DaiResources
{
  dai::Pipeline pipeline;
  std::shared_ptr<dai::MessageQueue> imu_queue;
};

OakImuNode::OakImuNode(const rclcpp::NodeOptions & options)
: Node("oak_imu", options)
{
  declareParameters();

  try {
    dai_ = std::make_unique<DaiResources>();
    buildPipeline();
    dai_->pipeline.start();
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string("could not bring up the OAK IMU: ") +
                             e.what());
  }

  // SensorDataQoS, depth deliberately generous: unlike a scan, a *dropped* IMU
  // sample is not recoverable. The consumer integrates these, so a lost sample
  // is rotation that silently never happened, and deskew under-corrects by
  // exactly that much. Queue rather than drop.
  imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
    "imu", rclcpp::SensorDataQoS().keep_last(100));

  // Polled at 200 Hz against a gyro at `gyro_hz_`; the queue holds anything
  // that arrives between polls, so this rate bounds latency, not throughput.
  timer_ = create_wall_timer(std::chrono::milliseconds(5),
                             std::bind(&OakImuNode::pollLoop, this));

  RCLCPP_INFO(get_logger(),
    "OAK IMU up: gyro %d Hz, accel %d Hz (%s), frame '%s', %s timestamps",
    gyro_hz_, accel_hz_, publish_accel_ ? "published" : "gyro only",
    frame_id_.c_str(), use_device_time_ ? "device" : "host");
}

OakImuNode::~OakImuNode()
{
  if (timer_) {
    timer_->cancel();
  }
  // dai_ is RAII: the Pipeline destructor closes the device.
}

void OakImuNode::declareParameters()
{
  frame_id_ = declare_parameter("frame_id", std::string("oak_imu_frame"));
  // 200 Hz gives ten samples per 50 ms control period and roughly thirty-six
  // across one lidar revolution, which is what makes the deskew rotation a
  // measured integral rather than an extrapolation from a single rate.
  gyro_hz_ = static_cast<int>(declare_parameter("gyro_hz", 200));
  accel_hz_ = static_cast<int>(declare_parameter("accel_hz", 200));
  publish_accel_ = declare_parameter("publish_accel", true);
  batch_size_ = static_cast<int>(declare_parameter("batch_report_threshold", 1));
  max_batch_size_ = static_cast<int>(declare_parameter("max_batch_reports", 10));
  use_device_time_ = declare_parameter("use_device_time", true);
}

void OakImuNode::buildPipeline()
{
  auto imu = dai_->pipeline.create<dai::node::IMU>();

  // ACCELEROMETER_RAW/GYROSCOPE_RAW rather than the fused ROTATION_VECTOR.
  // The stack wants a rate to integrate over known intervals, and the BNO's
  // fused attitude applies its own filtering with an unspecified delay -- which
  // would show up as a deskew correction that lags the motion producing it.
  imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, gyro_hz_);
  if (publish_accel_) {
    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, accel_hz_);
  }
  imu->setBatchReportThreshold(batch_size_);
  imu->setMaxBatchReports(max_batch_size_);

  // depth 30, blocking=false: hold recent packets, drop the oldest under
  // overrun rather than stalling the device.
  dai_->imu_queue = imu->out.createOutputQueue(30, false);
}

void OakImuNode::pollLoop()
{
  if (!dai_ || !dai_->imu_queue) {
    return;
  }

  // Drain everything waiting. Each IMU packet carries several reports and the
  // publisher emits one message per report, so no sample is merged away.
  for (;;) {
    std::shared_ptr<dai::IMUData> packet;
    try {
      packet = dai_->imu_queue->tryGet<dai::IMUData>();
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "IMU queue error: %s", e.what());
      return;
    }
    if (!packet) {
      return;
    }

    for (const auto & report : packet->packets) {
      sensor_msgs::msg::Imu message;
      message.header.frame_id = frame_id_;

      if (use_device_time_) {
        const auto device_time = report.gyroscope.getTimestampDevice();
        const auto since_epoch = device_time.time_since_epoch();
        const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
          since_epoch).count();
        message.header.stamp = rclcpp::Time(nanos, RCL_ROS_TIME);
      } else {
        message.header.stamp = now();
      }

      message.angular_velocity.x = report.gyroscope.x;
      message.angular_velocity.y = report.gyroscope.y;
      message.angular_velocity.z = report.gyroscope.z;

      if (publish_accel_) {
        message.linear_acceleration.x = report.acceleroMeter.x;
        message.linear_acceleration.y = report.acceleroMeter.y;
        message.linear_acceleration.z = report.acceleroMeter.z;
      } else {
        message.linear_acceleration_covariance[0] = -1.0;
      }

      // REP-145: a leading -1 declares the field absent. This node reports raw
      // rates and has no attitude to offer; implying one with an identity
      // quaternion would be a lie that something downstream would eventually
      // believe.
      message.orientation_covariance[0] = -1.0;

      imu_pub_->publish(message);
      last_stamp_ = message.header.stamp;
      ++published_;
    }
  }
}

}  // namespace oak_imu

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<oak_imu::OakImuNode>());
  } catch (const std::exception & e) {
    // A missing or unplugged OAK is the common case and deserves a sentence,
    // not a stack trace.
    RCLCPP_FATAL(rclcpp::get_logger("oak_imu"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
