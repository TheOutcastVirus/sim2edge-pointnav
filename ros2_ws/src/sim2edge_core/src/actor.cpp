#include "sim2edge/actor.hpp"

#include <chrono>

#include "sim2edge/spec_generated.hpp"

#ifdef SIM2EDGE_HAVE_EXECUTORCH

#include <dlfcn.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

using executorch::aten::SizesType;
using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::runtime::EValue;
using executorch::runtime::Error;

#endif  // SIM2EDGE_HAVE_EXECUTORCH

namespace sim2edge
{

bool actorRuntimeAvailable()
{
#ifdef SIM2EDGE_HAVE_EXECUTORCH
  return true;
#else
  return false;
#endif
}

#ifdef SIM2EDGE_HAVE_EXECUTORCH

namespace
{

/// Pre-load the QNN shared libraries with RTLD_GLOBAL.
///
/// ExecuTorch's QNN delegate resolves these lazily, so without the pre-load a
/// missing or wrong-variant library surfaces as an opaque delegate init
/// failure rather than a name. Note that the *right* variant matters as much as
/// the presence: on this board `aarch64-ubuntu-gcc9.4` ships no V75 stub and
/// rejects the silicon ("Unsupported SoC model (SnapdragonModel): 82"), while
/// `aarch64-oe-linux-gcc11.2` supports it. That directory must come first on
/// LD_LIBRARY_PATH; see docs/RUNBOOK.md.
bool preloadQnn(const std::string & lib_dir, std::string & error)
{
  static const char * kLibraries[] = {
    "libQnnSystem.so",
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
  };
  for (const char * name : kLibraries) {
    const std::string path = lib_dir.empty()
                               ? std::string(name)
                               : lib_dir + "/" + name;
    if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
      const char * why = dlerror();
      error = "failed to load " + path + ": " + (why ? why : "unknown error");
      return false;
    }
  }
  return true;
}

class ExecuTorchActor : public Actor
{
public:
  ExecuTorchActor(std::unique_ptr<Module> module, ActorInfo info)
  : module_(std::move(module)), info_(std::move(info)) {}

  Command run(const Observation & obs) override
  {
    // Wrapped in place. The observation buffer is owned by the caller and
    // reused every tick, so a 731-float copy per control period is avoidable
    // and therefore avoided.
    std::vector<SizesType> shape = {1, static_cast<SizesType>(spec::kObsDim)};
    auto tensor = from_blob(const_cast<float *>(obs.data.data()), shape);

    auto result = module_->execute("forward", {EValue(tensor)});
    if (!result.ok()) {
      // Zero is the safe command and the safety node will hold it, but a
      // silently-zero policy is indistinguishable from a policy that decided to
      // stop -- so this is reported, not swallowed. The node logs it.
      throw std::runtime_error("actor inference failed");
    }

    auto & out = result->at(0).toTensor();
    const float * data = out.const_data_ptr<float>();
    Command command;
    command.v = data[0];
    command.omega = data[1];
    return command;
  }

  const ActorInfo & info() const override { return info_; }

private:
  std::unique_ptr<Module> module_;
  ActorInfo info_;
};

}  // namespace

std::unique_ptr<Actor> loadActor(const std::string & pte_path,
                                 ActorBackend backend,
                                 const std::string & qnn_lib_dir,
                                 std::string & error)
{
  const auto started = std::chrono::steady_clock::now();

  if (backend == ActorBackend::kQnn && !preloadQnn(qnn_lib_dir, error)) {
    return nullptr;
  }

  // MmapUseMlock keeps the graph resident, which matters for warm latency on a
  // loop that must not miss a 50 ms deadline to a page fault.
  auto module = std::make_unique<Module>(pte_path, Module::LoadMode::MmapUseMlock);
  if (module->load() != Error::Ok) {
    error = "could not load " + pte_path;
    return nullptr;
  }
  if (module->load_method("forward") != Error::Ok) {
    error = "no 'forward' method in " + pte_path;
    return nullptr;
  }

  ActorInfo info;
  info.path = pte_path;
  info.backend = backend;
  info.input_dim = spec::kObsDim;
  info.output_dim = spec::kActionDim;

  // Prove the contract with a real forward pass rather than trusting metadata.
  // A graph exported under a different observation layout has plausible shapes
  // right up until it doesn't, and the failure downstream is silent.
  Observation probe;
  std::vector<SizesType> shape = {1, static_cast<SizesType>(spec::kObsDim)};
  auto tensor = from_blob(probe.data.data(), shape);
  auto result = module->execute("forward", {EValue(tensor)});
  if (!result.ok()) {
    error = "the graph in " + pte_path + " rejected a " +
            std::to_string(spec::kObsDim) +
            "-wide observation; it was exported under a different contract";
    return nullptr;
  }
  const auto numel = result->at(0).toTensor().numel();
  if (numel != spec::kActionDim) {
    error = "expected " + std::to_string(spec::kActionDim) +
            " outputs from " + pte_path + ", got " + std::to_string(numel);
    return nullptr;
  }

  info.load_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  return std::make_unique<ExecuTorchActor>(std::move(module), std::move(info));
}

#else  // !SIM2EDGE_HAVE_EXECUTORCH

std::unique_ptr<Actor> loadActor(const std::string & pte_path, ActorBackend,
                                 const std::string &, std::string & error)
{
  error = "this build has no ExecuTorch runtime, so " + pte_path +
          " cannot be executed; rebuild with -DEXECUTORCH_INSTALL_DIR=/opt/executorch";
  return nullptr;
}

#endif  // SIM2EDGE_HAVE_EXECUTORCH

}  // namespace sim2edge
