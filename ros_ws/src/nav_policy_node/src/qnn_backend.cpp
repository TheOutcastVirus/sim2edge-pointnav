#include "nav_policy_node/qnn_backend.hpp"

#include <dlfcn.h>  // dlopen / RTLD_GLOBAL -- QNN libs must be pre-loaded

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

namespace
{

/// Pre-load the QNN shared libraries so ExecuTorch's backend can resolve them.
///
/// RTLD_GLOBAL is required, not merely convenient: the QNN backend resolves
/// symbols from these libraries lazily at Module load, and without RTLD_GLOBAL
/// they are invisible to it. This is copied from the reference project, where
/// it was established the hard way.
void preloadQnnLibs(const std::string & lib_dir)
{
  static const char * kLibs[] = {
    "libQnnSystem.so",
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
    nullptr,
  };
  for (int i = 0; kLibs[i] != nullptr; ++i) {
    const std::string path =
      lib_dir.empty() ? kLibs[i] : (lib_dir + "/" + kLibs[i]);
    if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
      throw std::runtime_error(
              std::string("failed to load QNN library '") + path + "': " +
              dlerror() +
              "\nCheck QNN_SDK_ROOT and that ADSP_LIBRARY_PATH is set "
              "(note: it is SEMICOLON-separated, a FastRPC quirk).");
    }
  }
}

}  // namespace

QnnBackend::QnnBackend(const std::string & qnn_lib_dir)
: qnn_lib_dir_(qnn_lib_dir) {}

QnnBackend::~QnnBackend() = default;

bool QnnBackend::load(const std::string & model_path)
{
  preloadQnnLibs(qnn_lib_dir_);
  // MmapUseMlock pins the model in memory, which removes a large first-
  // inference latency spike -- unacceptable in a 20 Hz control loop.
  module_ = std::make_unique<Module>(model_path, Module::LoadMode::MmapUseMlock);
  return module_->load() == Error::Ok;
}

std::vector<float> QnnBackend::infer(const Observation & obs)
{
  // Input order is fixed by ObservationSpec.actor_inputs and must match the
  // exported graph's forward() signature exactly. Reordering these silently
  // feeds the lidar encoder a goal vector.
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
    throw std::runtime_error("ExecuTorch QNN inference failed");
  }

  const auto & out = result->at(0).toTensor();
  const float * data = out.const_data_ptr<float>();
  return std::vector<float>(data, data + out.numel());
}

}  // namespace nav_policy_node
