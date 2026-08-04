// Replay a recorded corpus through the production code path and report how far
// the C++ has drifted from the Python it was transcribed from.
//
// The gate is deliberately asymmetric, because the two halves of this pipeline
// have different contracts:
//
//   **Preprocessing must agree tightly.** Deskew, binning and the goal features
//   are a literal transcription of sim2edge/preprocess.py and
//   sim2edge/sensors.py. There is no licence to differ: a divergence there
//   feeds the network inputs that differ from the ones the quantizer was
//   calibrated on, and the failure is silent. This is a pass/fail gate.
//
//   **The localizer is measured, not gated.** It was ported speed-first -- a
//   bucket grid instead of a KD-tree, memoized normals, float point clouds --
//   so its trajectory will not track Python's to the last bit and is not
//   supposed to. What matters is whether the difference is small against the
//   quantity being estimated, so the divergence is reported as a distribution
//   next to the drift signal it has to resolve. Pass `--pose-tolerance` to turn
//   it into a gate once a number has been agreed.
//
// No ROS, no simulator, and no Python: this runs on a laptop and on the board.
// A parity gate that can only be executed on the robot is a parity gate nobody
// runs.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sim2edge/actor.hpp"
#include "sim2edge/navigator.hpp"
#include "sim2edge/replay.hpp"
#include "sim2edge/spec_generated.hpp"

namespace
{

using sim2edge::spec::kLidarBins;
using sim2edge::spec::kObsDim;

struct Stat
{
  double max = 0.0;
  double sum = 0.0;
  std::size_t count = 0;
  std::vector<double> samples;

  void add(double value)
  {
    max = std::max(max, value);
    sum += value;
    ++count;
    samples.push_back(value);
  }

  double mean() const { return count ? sum / static_cast<double>(count) : 0.0; }

  double percentile(double q)
  {
    if (samples.empty()) {
      return 0.0;
    }
    const std::size_t k = std::min(
      samples.size() - 1,
      static_cast<std::size_t>(q * static_cast<double>(samples.size() - 1)));
    std::nth_element(samples.begin(),
                     samples.begin() + static_cast<std::ptrdiff_t>(k),
                     samples.end());
    return samples[k];
  }
};

void usage(const char * argv0)
{
  std::fprintf(stderr,
    "usage: %s --replay FILE [options]\n"
    "\n"
    "  --replay FILE            replay written by tools/export_replay.py\n"
    "  --model FILE.pte         actor to run; omitted skips action parity\n"
    "  --backend qnn|cpu        inference backend (default qnn when a model\n"
    "                           is given)\n"
    "  --qnn-lib-dir DIR        directory holding libQnnHtp.so\n"
    "  --submap-scans N         localizer submap size (default 128)\n"
    "  --point-stride N         localizer point stride (default 2)\n"
    "  --odometry-prior         feed the recorded odometry increment as the\n"
    "                           motion prior instead of gyro + velocity\n"
    "  --tolerance F            preprocessing gate, default 1e-4\n"
    "  --pose-tolerance F       gate the localizer too (default: report only)\n"
    "  --max-ticks N            stop early; useful for a smoke run\n",
    argv0);
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string replay_path;
  std::string model_path;
  std::string backend_name = "qnn";
  std::string qnn_lib_dir;
  int submap_scans = 128;
  int point_stride = 2;
  bool odometry_prior = false;
  double tolerance = 1.0e-4;
  double pose_tolerance = -1.0;
  long max_ticks = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char * name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--replay") {
      replay_path = next("--replay");
    } else if (arg == "--model") {
      model_path = next("--model");
    } else if (arg == "--backend") {
      backend_name = next("--backend");
    } else if (arg == "--qnn-lib-dir") {
      qnn_lib_dir = next("--qnn-lib-dir");
    } else if (arg == "--submap-scans") {
      submap_scans = std::stoi(next("--submap-scans"));
    } else if (arg == "--point-stride") {
      point_stride = std::stoi(next("--point-stride"));
    } else if (arg == "--odometry-prior") {
      odometry_prior = true;
    } else if (arg == "--tolerance") {
      tolerance = std::stod(next("--tolerance"));
    } else if (arg == "--pose-tolerance") {
      pose_tolerance = std::stod(next("--pose-tolerance"));
    } else if (arg == "--max-ticks") {
      max_ticks = std::stol(next("--max-ticks"));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  if (replay_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  sim2edge::ReplayReader replay;
  std::string error;
  if (!replay.open(replay_path, error)) {
    std::fprintf(stderr, "replay: %s\n", error.c_str());
    return 1;
  }

  // The fingerprint check is the reason this is not merely a numeric
  // comparison. A replay exported under a different observation contract would
  // compare cleanly against a C++ build that shares the mistake, and prove
  // nothing at all.
  if (replay.observationFingerprint() != sim2edge::spec::kObservationFingerprint) {
    std::fprintf(stderr,
      "contract mismatch: replay was exported at fingerprint %s, this build is "
      "%s.\nRegenerate the header (tools/gen_spec_header.py) or re-export the "
      "replay -- comparing across contracts measures nothing.\n",
      replay.observationFingerprint().c_str(),
      sim2edge::spec::kObservationFingerprint);
    return 1;
  }

  std::printf("replay          %s\n", replay_path.c_str());
  std::printf("  ticks         %zu in %zu episodes, %u beams\n",
              replay.tickCount(), replay.episodeCount(),
              replay.header().beam_count);
  std::printf("  contract      %s / %s\n", replay.sensorContract().c_str(),
              replay.observationFingerprint().c_str());
  const bool reference_is_actor =
    (replay.header().flags & sim2edge::kFlagReferenceActionFromActor) != 0;
  std::printf("  reference     %s\n", reference_is_actor
                ? "FP32 actor -- an action comparison measures quantization"
                : "stored oracle command -- an action comparison measures how "
                  "far the policy is from its teacher, NOT quantization");

  sim2edge::Navigator::Config config;
  config.localizer.max_submap_scans = submap_scans;
  config.localizer.point_stride = point_stride;
  config.use_odometry_prior = odometry_prior;
  sim2edge::Navigator navigator(config);

  std::shared_ptr<sim2edge::Actor> actor;
  if (!model_path.empty()) {
    const auto backend = backend_name == "cpu" ? sim2edge::ActorBackend::kCpu
                                               : sim2edge::ActorBackend::kQnn;
    auto loaded = sim2edge::loadActor(model_path, backend, qnn_lib_dir, error);
    if (!loaded) {
      std::fprintf(stderr, "actor: %s\n", error.c_str());
      return 1;
    }
    std::printf("  actor         %s on %s, loaded in %.1f ms\n",
                model_path.c_str(), backend_name.c_str(),
                loaded->info().load_ms);
    actor = std::move(loaded);
    navigator.setActor(actor);
  } else {
    std::printf("  actor         (none: preprocessing and localizer only)\n");
  }

  Stat deskew_range_error;
  Stat deskew_bearing_error;
  std::size_t total_beams = 0;
  std::size_t total_crossings = 0;
  Stat bins_error;
  Stat goal_feature_error;
  Stat goal_error;
  Stat obs_error;
  Stat pose_xy_error;
  Stat pose_yaw_error;
  Stat action_error;
  Stat deskew_us;
  Stat bin_us;
  Stat localize_us;
  Stat infer_us;
  Stat total_us;
  std::size_t observable_agree = 0;
  std::size_t observable_total = 0;

  std::size_t processed = 0;
  for (std::size_t e = 0; e < replay.episodeCount(); ++e) {
    const sim2edge::ReplayEpisode & episode = replay.episode(e);
    // Reset between episodes. A submap stitched across a teleport is geometry
    // the tracker never sees on the robot, and it would make the
    // correspondence search look artificially cheap as well as wrong.
    navigator.reset();
    navigator.setGoalBody(episode.goal_body_x, episode.goal_body_y);

    for (uint32_t t = 0; t < episode.tick_count; ++t) {
      if (max_ticks > 0 && static_cast<long>(processed) >= max_ticks) {
        break;
      }
      const sim2edge::ReplayTick tick = replay.tick(episode.first_tick + t);
      // Feed the recorded previous command rather than our own last output.
      // Without this the action block of the observation diverges after one
      // tick and every later comparison measures closed-loop drift instead of
      // whether the two implementations agree.
      navigator.overrideLastCommand(
        sim2edge::Command{static_cast<float>(tick.prev_v),
                          static_cast<float>(tick.prev_omega)});
      const auto & result = navigator.tick(
        tick.ranges, tick.bearings, tick.ages, tick.beams, tick.gyro_delta,
        tick.speed, tick.dt, tick.prior_delta_body);
      ++processed;

      // ── Deskew, compared per beam before binning ───────────────────────
      // Reported separately because a bin difference is a step function: one
      // beam crossing a bin edge changes that bin by whatever the two surfaces
      // differ by, which says nothing about whether the geometry is wrong.
      double worst_range = 0.0;
      double worst_bearing = 0.0;
      std::size_t edge_crossings = 0;
      for (std::size_t b = 0; b < tick.beams; ++b) {
        const float mine = navigator.sweep().ranges[b];
        const float theirs = tick.ref_deskewed_ranges[b];
        if (std::isfinite(mine) && std::isfinite(theirs)) {
          worst_range = std::max(worst_range,
                                 std::abs(static_cast<double>(mine - theirs)));
        }
        const double db = std::abs(sim2edge::wrapAngle(
          static_cast<double>(navigator.sweep().bearings[b]) -
          static_cast<double>(tick.ref_deskewed_bearings[b])));
        worst_bearing = std::max(worst_bearing, db);

        // Did this beam land in a different bin than Python put it in? That is
        // the mechanism behind every large bin difference, and counting it
        // separates a boundary coincidence from a real disagreement.
        auto binOf = [](double bearing) {
          const double wrapped = sim2edge::wrapAngle(bearing);
          const double kPi = 3.141592653589793;
          int idx = static_cast<int>(
            std::floor((wrapped + kPi) / sim2edge::spec::kBinWidthRad));
          return std::min(std::max(idx, 0), kLidarBins - 1);
        };
        if (binOf(navigator.sweep().bearings[b]) !=
            binOf(tick.ref_deskewed_bearings[b]))
        {
          ++edge_crossings;
        }
      }
      deskew_range_error.add(worst_range);
      deskew_bearing_error.add(worst_bearing);
      total_beams += tick.beams;
      total_crossings += edge_crossings;

      // ── Preprocessing: the gated half ──────────────────────────────────
      double worst_bin = 0.0;
      for (int b = 0; b < kLidarBins; ++b) {
        worst_bin = std::max(worst_bin,
          std::abs(static_cast<double>(navigator.bins()[b]) -
                   static_cast<double>(tick.ref_bins[b])));
      }
      bins_error.add(worst_bin);

      // The observation is split, because its 731 numbers do not all come from
      // the same place. The lidar and action blocks are pure preprocessing and
      // are gated; the three goal features are a *function of the localizer's
      // pose*, so their difference is a consequence of the deliberately
      // speed-first tracker and belongs with it. Gating them at the
      // preprocessing tolerance would fail the transcription for a divergence
      // that is not in the transcription.
      double worst_obs = 0.0;
      for (int k = 0; k < kObsDim; ++k) {
        const double difference =
          std::abs(static_cast<double>(navigator.observation().data[k]) -
                   static_cast<double>(tick.ref_observation[k]));
        const bool is_goal =
          k >= static_cast<int>(sim2edge::spec::kGoalOffset) &&
          k < static_cast<int>(sim2edge::spec::kActionOffset);
        if (is_goal) {
          goal_feature_error.add(difference);
        } else {
          worst_obs = std::max(worst_obs, difference);
        }
      }
      obs_error.add(worst_obs);

      const double dgx = result.goal_body_x - tick.ref_goal_body[0];
      const double dgy = result.goal_body_y - tick.ref_goal_body[1];
      goal_error.add(std::sqrt(dgx * dgx + dgy * dgy));

      // ── Localizer: the measured half ───────────────────────────────────
      const double dx = result.pose.x - tick.ref_pose[0];
      const double dy = result.pose.y - tick.ref_pose[1];
      pose_xy_error.add(std::sqrt(dx * dx + dy * dy));
      pose_yaw_error.add(
        std::abs(sim2edge::wrapAngle(result.pose.yaw - tick.ref_pose[2])));

      ++observable_total;
      if (result.observable == tick.ref_observable) {
        ++observable_agree;
      }

      if (actor && result.has_goal && !result.arrived) {
        const double dv = result.command.v - tick.ref_action[0];
        const double dw = result.command.omega - tick.ref_action[1];
        action_error.add(std::max(std::abs(dv), std::abs(dw)));
      }

      deskew_us.add(result.deskew_us);
      bin_us.add(result.bin_us);
      localize_us.add(result.localize_us);
      infer_us.add(result.infer_us);
      total_us.add(result.total_us);
    }
    if (max_ticks > 0 && static_cast<long>(processed) >= max_ticks) {
      break;
    }
  }

  std::printf("\n── preprocessing parity (gated at %.1e) ─────────────────\n",
              tolerance);
  std::printf("  %-22s mean %10.3e m max %10.3e m\n", "deskewed range",
              deskew_range_error.mean(), deskew_range_error.max);
  std::printf("  %-22s mean %10.3e   max %10.3e   (rad)\n", "deskewed bearing",
              deskew_bearing_error.mean(), deskew_bearing_error.max);
  std::printf("  %-22s %zu of %zu beams (%.4f%%)\n", "bin-edge crossings",
              total_crossings, total_beams,
              total_beams ? 100.0 * static_cast<double>(total_crossings) /
                              static_cast<double>(total_beams)
                          : 0.0);
  std::printf("  %-22s mean %10.3e   p99 %10.3e   max %10.3e\n", "lidar bins",
              bins_error.mean(), bins_error.percentile(0.99), bins_error.max);
  std::printf("  %-22s mean %10.3e   p99 %10.3e   max %10.3e\n",
              "obs, lidar+action", obs_error.mean(),
              obs_error.percentile(0.99), obs_error.max);

  std::printf("\n── localizer divergence (measured, not gated) ───────────\n");
  std::printf("  %-22s mean %10.3e   max %10.3e  (obs goal block)\n",
              "goal features", goal_feature_error.mean(),
              goal_feature_error.max);
  std::printf("  %-22s mean %8.4f cm  p95 %8.4f cm  max %8.4f cm\n",
              "pose translation", pose_xy_error.mean() * 100.0,
              pose_xy_error.percentile(0.95) * 100.0,
              pose_xy_error.max * 100.0);
  std::printf("  %-22s mean %8.4f deg p95 %8.4f deg max %8.4f deg\n",
              "pose yaw", pose_yaw_error.mean() * 57.29577951308232,
              pose_yaw_error.percentile(0.95) * 57.29577951308232,
              pose_yaw_error.max * 57.29577951308232);
  std::printf("  %-22s mean %8.4f cm  max %8.4f cm\n", "believed goal offset",
              goal_error.mean() * 100.0, goal_error.max * 100.0);
  std::printf("  %-22s %zu / %zu ticks agree\n", "observability flag",
              observable_agree, observable_total);
  if (action_error.count > 0) {
    // Reported under a name that says what it is. Against the FP32 actor this
    // is the INT8 lowering's error and belongs beside the quantization gate;
    // against the oracle it is the policy's imitation error and says nothing
    // about the lowering at all.
    std::printf("  %-22s mean %10.3e   max %10.3e  (v, omega in SI, %.2f%% / "
                "%.2f%% of the omega limit)\n",
                reference_is_actor ? "INT8 vs FP32 actor" : "policy vs oracle",
                action_error.mean(), action_error.max,
                100.0 * action_error.mean() / sim2edge::spec::kOmegaMax,
                100.0 * action_error.max / sim2edge::spec::kOmegaMax);
  }

  std::printf("\n── latency, %.0f ms control period ──────────────────────\n",
              sim2edge::spec::kDt * 1000.0);
  auto row = [&](const char * name, Stat & s) {
    std::printf("  %-22s median %7.3f ms  p95 %7.3f ms  max %7.3f ms\n", name,
                s.percentile(0.50) / 1000.0, s.percentile(0.95) / 1000.0,
                s.max / 1000.0);
  };
  row("deskew", deskew_us);
  row("bin + history", bin_us);
  row("localizer", localize_us);
  if (infer_us.count > 0) {
    row("actor", infer_us);
  }
  row("whole tick", total_us);
  std::printf("  %-22s %.1f%% of the period at the median\n", "budget",
              100.0 * total_us.percentile(0.50) / 1000.0 /
                (sim2edge::spec::kDt * 1000.0));

  // ── The gate ─────────────────────────────────────────────────────────────
  //
  // Gating on the maximum bin difference is the obvious choice and it is the
  // wrong one, because a bin difference is not a continuous quantity. Binning
  // is a floor(): two implementations that agree on a bearing to 5e-7 rad still
  // put a beam in different bins if that bearing sits within 5e-7 of an edge,
  // and the bin then differs by however much the two surfaces differ -- which
  // can be the full 0-to-1 range with nothing wrong anywhere.
  //
  // What can honestly be required is that the *geometry* agrees to float32
  // precision and that edge crossings stay at the rate rounding alone explains.
  // Both are gated. The bin and observation figures are reported because they
  // are what the network sees, and their tail is dominated by those crossings.
  const double crossing_rate =
    total_beams ? static_cast<double>(total_crossings) /
                    static_cast<double>(total_beams)
                : 0.0;
  const double kRangeGate = 1.0e-4;      // metres; float32 on ~10 m ranges
  const double kBearingGate = 1.0e-5;    // radians; 2e-2 of a bin width
  const double kCrossingGate = 1.0e-4;   // one beam in ten thousand

  int status = 0;
  if (deskew_range_error.max > kRangeGate ||
      deskew_bearing_error.max > kBearingGate)
  {
    std::printf("\nFAIL deskew geometry diverged: %.3e m / %.3e rad against "
                "gates of %.1e m / %.1e rad.\n",
                deskew_range_error.max, deskew_bearing_error.max, kRangeGate,
                kBearingGate);
    status = 1;
  } else if (crossing_rate > kCrossingGate) {
    std::printf("\nFAIL %.4f%% of beams landed in a different bin than Python "
                "chose, above the %.4f%% rounding allowance. The geometry "
                "agrees, so look for a bin-index or wrap convention "
                "difference rather than an arithmetic one.\n",
                100.0 * crossing_rate, 100.0 * kCrossingGate);
    status = 1;
  } else {
    std::printf("\nPASS deskew agrees to %.3e m and %.3e rad; %zu of %zu beams "
                "(%.5f%%) fell on the far side of a bin edge, which is "
                "rounding.\n",
                deskew_range_error.max, deskew_bearing_error.max,
                total_crossings, total_beams, 100.0 * crossing_rate);
    std::printf("     Worst resulting lidar/action observation difference: "
                "%.3e.\n", obs_error.max);
  }
  // The p99 rather than the max: the tail is bin crossings, whose *rate* is
  // already gated above, and the body is arithmetic, which is what must be
  // tight. A max-based gate here would just re-fail on the same single beam.
  if (tolerance > 0.0 && obs_error.percentile(0.99) > tolerance) {
    std::printf("FAIL the observation vector's p99 difference is %.3e, above "
                "the %.1e gate -- that is the arithmetic, not the tail.\n",
                obs_error.percentile(0.99), tolerance);
    status = 1;
  }
  if (pose_tolerance > 0.0 && pose_xy_error.max > pose_tolerance) {
    std::printf("FAIL localizer translation diverged by %.4f cm, above the "
                "%.4f cm gate.\n",
                pose_xy_error.max * 100.0, pose_tolerance * 100.0);
    status = 1;
  }
  return status;
}
