#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nav_policy_node/inference_backend.hpp"

namespace executorch::extension { class Module; }

namespace nav_policy_node
{

/// XNNPACK CPU backend. The Gate-A fallback: this policy is ~2.7M parameters,
/// small enough that CPU inference has a real chance of holding 20 Hz, so a
/// QNN problem never has to block the rest of the project.
///
/// Requires ExecuTorch built with EXECUTORCH_BUILD_XNNPACK=ON. The reference
/// board build used -DBUILD_XNNPACK_BACKEND=OFF, which silently makes
/// backend:=cpu a dead option.
class CpuBackend : public InferenceBackend
{
public:
  CpuBackend();
  ~CpuBackend() override;

  bool load(const std::string & model_path) override;
  std::vector<float> infer(const Observation & obs) override;
  const char * name() const override {return "xnnpack-cpu";}

private:
  std::unique_ptr<executorch::extension::Module> module_;
};

}  // namespace nav_policy_node
