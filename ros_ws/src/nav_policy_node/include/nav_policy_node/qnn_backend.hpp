#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nav_policy_node/inference_backend.hpp"

namespace executorch::extension { class Module; }

namespace nav_policy_node
{

/// Qualcomm QNN / HTP V75 NPU backend.
///
/// Two details are inherited verbatim from the reference project and are both
/// load-bearing:
///   * the QNN shared libraries must be dlopen'd with RTLD_GLOBAL *before* the
///     Module is constructed, or ExecuTorch cannot resolve them;
///   * the Module is opened with LoadMode::MmapUseMlock, which pins the model
///     and removes a first-inference latency spike.
class QnnBackend : public InferenceBackend
{
public:
  explicit QnnBackend(const std::string & qnn_lib_dir);
  ~QnnBackend() override;

  bool load(const std::string & model_path) override;
  std::vector<float> infer(const Observation & obs) override;
  const char * name() const override {return "qnn-htp";}

private:
  std::unique_ptr<executorch::extension::Module> module_;
  std::string qnn_lib_dir_;
};

}  // namespace nav_policy_node
