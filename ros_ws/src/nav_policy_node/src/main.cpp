#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "nav_policy_node/nav_policy_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<nav_policy_node::NavPolicyNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("nav_policy"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
