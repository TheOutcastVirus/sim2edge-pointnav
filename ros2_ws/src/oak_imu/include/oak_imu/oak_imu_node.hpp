// The OAK-D's IMU, published as sensor_msgs/Imu.
//
// This is the robot's gyro, and the gyro is one of the three inputs the
// deployable stack has: the lidar sweep, this, and the Create 3's velocity.
// It does two jobs, and the second is the demanding one:
//
//   1. It supplies the yaw increment the localizer's motion prior integrates.
//   2. It supplies the rotation that **deskews the lidar sweep**. One A1
//      revolution spans 182 ms, so the beams in a scan were acquired across
//      three or four control periods, and undoing that smear needs the
//      rotation over each of those intervals -- not a single rate sampled once
//      per tick. That is why this node publishes at the sensor's own rate and
//      lets the consumer integrate, rather than decimating to 20 Hz here.
//
// A camera driver is deliberately not part of this package. The navigation
// policy is lidar-only; the OAK-D is present for its IMU and, in the sibling
// ventuno projects, for vision. Bringing up the image pipeline would put a USB2
// bandwidth consumer next to a 20 Hz control loop for no benefit to this stack.
//
// DepthAI headers are included in the .cpp only, never here: its bundled
// dependencies conflict with rclcpp's if both are pulled into one translation
// unit. Same containment the reference project's oak_camera uses.

#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace oak_imu
{

class OakImuNode : public rclcpp::Node
{
public:
  explicit OakImuNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OakImuNode() override;

private:
  /// Opaque holder for every DepthAI type, defined in the .cpp.
  struct DaiResources;

  void declareParameters();
  void buildPipeline();
  void pollLoop();

  std::unique_ptr<DaiResources> dai_;

  std::string frame_id_;
  int gyro_hz_{};
  int accel_hz_{};
  int batch_size_{};
  int max_batch_size_{};
  bool publish_accel_{};
  /// Report the device's own timestamps instead of arrival time. The whole
  /// point of this node is the integral of rotation over precise intervals, and
  /// host arrival time carries USB scheduling jitter that the device clock does
  /// not.
  bool use_device_time_{};

  std::size_t published_{0};
  rclcpp::Time last_stamp_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace oak_imu
