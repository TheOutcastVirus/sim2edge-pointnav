// The exported navigation actor, running on the Hexagon NPU or on CPU.
//
// The graph is small and the measurements say so: 0.15-0.18 ms fully delegated
// to HTP V75, 5.15 ms on portable CPU kernels, against a 50 ms control period
// and a localizer that costs 21-27 ms. The NPU is not the constraint and never
// was -- which is exactly why the CPU path here is a real, supported fallback
// rather than a dead configuration value. Losing the NPU costs 10% of the
// period, not the robot.
//
// `physical_action` is inside the exported graph, so `run` returns SI units
// directly: v in [0, kVMax] and omega in [-kOmegaMax, kOmegaMax]. Nothing
// downstream re-applies a scaling convention, which is the whole reason the
// squash was kept in-model.

#pragma once

#include <memory>
#include <string>

#include "sim2edge/types.hpp"

namespace sim2edge
{

/// How the actor should be executed.
enum class ActorBackend
{
  /// Delegate to QNN/HTP. Requires the QNN runtime libraries on
  /// LD_LIBRARY_PATH and a `.pte` lowered for this SoC.
  kQnn,
  /// Portable/XNNPACK CPU kernels. Runs any `.pte`, including a QNN-lowered
  /// one only if the delegate is present, so pair it with the CPU export.
  kCpu,
};

/// Loaded-model facts worth logging once and asserting against the contract.
struct ActorInfo
{
  std::string path;
  ActorBackend backend{ActorBackend::kCpu};
  int input_dim{0};
  int output_dim{0};
  double load_ms{0.0};
};

class Actor
{
public:
  virtual ~Actor() = default;

  /// One forward pass. `obs.data` is wrapped in place -- no copy -- so the
  /// caller must keep it alive for the duration of the call.
  virtual Command run(const Observation & obs) = 0;

  virtual const ActorInfo & info() const = 0;
};

/// Load an exported actor.
///
/// `qnn_lib_dir` is the directory holding libQnnSystem.so / libQnnHtp.so; the
/// QNN backend dlopens them with RTLD_GLOBAL before the delegate initializes,
/// because ExecuTorch's delegate resolves them lazily and an empty
/// LD_LIBRARY_PATH otherwise surfaces as an opaque init failure. Ignored for
/// the CPU backend.
///
/// Returns nullptr and fills `error` on failure. Failure modes here are
/// environmental (missing libraries, wrong SoC stub directory, absent file),
/// so the message matters more than the type.
std::unique_ptr<Actor> loadActor(const std::string & pte_path,
                                 ActorBackend backend,
                                 const std::string & qnn_lib_dir,
                                 std::string & error);

/// True when this build linked ExecuTorch. The parity harness and the
/// preprocessing stages build and run on a laptop without it.
bool actorRuntimeAvailable();

}  // namespace sim2edge
