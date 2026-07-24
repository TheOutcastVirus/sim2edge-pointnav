#pragma once

#include <string>
#include <vector>

#include "nav_policy_node/preprocess.hpp"

namespace nav_policy_node
{

/// Strategy interface over the two ExecuTorch backends.
///
/// Cloned from the reference project's collision_classifier pattern, with one
/// change forced by this model: inference takes four input tensors rather than
/// one. The Observation struct carries them in ObservationSpec.actor_inputs
/// order, which is the order the exported graph binds them -- an implementation
/// must not reorder.
class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  /// Load a .pte. Returns false on failure rather than throwing, so the node
  /// can report a clean error instead of unwinding through rclcpp.
  virtual bool load(const std::string & model_path) = 0;

  /// Run the policy. Returns {v, omega} in SI units -- the squash and scaling
  /// are inside the graph, so no denormalization happens here. Throws on
  /// inference failure; the caller treats that as a policy timeout and lets the
  /// safety layer stop the robot.
  virtual std::vector<float> infer(const Observation & obs) = 0;

  /// Human-readable backend name, for logging.
  virtual const char * name() const = 0;
};

}  // namespace nav_policy_node
