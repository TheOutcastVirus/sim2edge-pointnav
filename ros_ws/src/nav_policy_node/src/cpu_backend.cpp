#include "nav_policy_node/cpu_backend.hpp"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <stdexcept>

#include "nav_policy_node/spec_generated.hpp"

using executorch::aten::SizesType;
using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::runtime::EValue;
using executorch::runtime::Error;

namespace nav_policy_node
{

CpuBackend::CpuBackend() = default;
CpuBackend::~CpuBackend() = default;

bool CpuBackend::load(const std::string & model_path)
{
  module_ = std::make_unique<Module>(model_path, Module::LoadMode::MmapUseMlock);
  return module_->load() == Error::Ok;
}

std::vector<float> CpuBackend::infer(const Observation & obs)
{
  // Same binding order as the QNN backend, for the same reason: it is the
  // exported graph's forward() signature.
  std::vector<SizesType> depth_shape = {1, 1, spec::kDepthHeight,
    spec::kDepthWidth};
  std::vector<SizesType> lidar_shape = {1, spec::kLidarBins};
  std::vector<SizesType> goal_shape = {1, spec::kGoalDim};
  std::vector<SizesType> action_shape = {1, spec::kActionDim};

  auto depth = from_blob(const_cast<float *>(obs.depth.data()), depth_shape);
  auto lidar = from_blob(const_cast<float *>(obs.lidar.data()), lidar_shape);
  auto goal = from_blob(const_cast<float *>(obs.goal.data()), goal_shape);
  auto last = from_blob(const_cast<float *>(obs.last_action.data()),
                        action_shape);

  auto result = module_->execute(
    "forward", {EValue(depth), EValue(lidar), EValue(goal), EValue(last)});
  if (!result.ok()) {
    throw std::runtime_error("ExecuTorch XNNPACK inference failed");
  }

  const auto & out = result->at(0).toTensor();
  const float * data = out.const_data_ptr<float>();
  return std::vector<float>(data, data + out.numel());
}

}  // namespace nav_policy_node
