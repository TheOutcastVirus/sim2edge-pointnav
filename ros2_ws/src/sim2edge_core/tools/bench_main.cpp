// Latency of the C++ stack, swept over the localizer knobs that decide it.
//
// The counterpart of scripts/bench_localizer.py, and it exists to be compared
// against that script's board numbers directly: 21-27 ms median for the Python
// tracker at the production 128 scans / stride 2, against a 50 ms period. The
// same corpus, the same configurations, the same reset-per-episode rule.
//
// Steady state is what is being measured. The submap fills to its cap over the
// first seconds of an episode and the correspondence search grows with it, so
// an average over a short run understates the cost that actually matters --
// which is why occupancy is reported next to the timings rather than left
// implicit.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sim2edge/actor.hpp"
#include "sim2edge/navigator.hpp"
#include "sim2edge/replay.hpp"

namespace
{

double percentile(std::vector<double> & values, double q)
{
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t k = std::min(
    values.size() - 1,
    static_cast<std::size_t>(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(),
                   values.begin() + static_cast<std::ptrdiff_t>(k),
                   values.end());
  return values[k];
}

struct Row
{
  int submap;
  int stride;
  double median_ms;
  double p95_ms;
  double max_ms;
  double submap_points;
};

void usage(const char * argv0)
{
  std::fprintf(stderr,
    "usage: %s --replay FILE [options]\n"
    "\n"
    "  --replay FILE       replay written by tools/export_replay.py\n"
    "  --model FILE.pte    also time the actor inside the loop\n"
    "  --backend qnn|cpu   inference backend (default qnn)\n"
    "  --qnn-lib-dir DIR   directory holding libQnnHtp.so\n"
    "  --submap N,N,...    submap sizes to sweep (default 128,64,32)\n"
    "  --stride N,N,...    point strides to sweep (default 2,4)\n"
    "  --warmup N          ticks to discard per episode so the submap has\n"
    "                      filled (default 40)\n"
    "  --max-ticks N       stop early\n",
    argv0);
}

std::vector<int> parseList(const std::string & text)
{
  std::vector<int> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string piece = text.substr(
      start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!piece.empty()) {
      out.push_back(std::stoi(piece));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string replay_path;
  std::string model_path;
  std::string backend_name = "qnn";
  std::string qnn_lib_dir;
  std::vector<int> submaps{128, 64, 32};
  std::vector<int> strides{2, 4};
  int warmup = 40;
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
    } else if (arg == "--submap") {
      submaps = parseList(next("--submap"));
    } else if (arg == "--stride") {
      strides = parseList(next("--stride"));
    } else if (arg == "--warmup") {
      warmup = std::stoi(next("--warmup"));
    } else if (arg == "--max-ticks") {
      max_ticks = std::stol(next("--max-ticks"));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
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

  std::shared_ptr<sim2edge::Actor> actor;
  if (!model_path.empty()) {
    const auto backend = backend_name == "cpu" ? sim2edge::ActorBackend::kCpu
                                               : sim2edge::ActorBackend::kQnn;
    auto loaded = sim2edge::loadActor(model_path, backend, qnn_lib_dir, error);
    if (!loaded) {
      std::fprintf(stderr, "actor: %s\n", error.c_str());
      return 1;
    }
    std::printf("actor %s on %s, load %.1f ms\n", model_path.c_str(),
                backend_name.c_str(), loaded->info().load_ms);
    actor = std::move(loaded);
  }

  std::printf("replay %s: %zu ticks, %zu episodes, %u beams\n\n",
              replay_path.c_str(), replay.tickCount(), replay.episodeCount(),
              replay.header().beam_count);

  std::vector<Row> rows;
  for (int submap : submaps) {
    for (int stride : strides) {
      sim2edge::Navigator::Config config;
      config.localizer.max_submap_scans = submap;
      config.localizer.point_stride = stride;
      sim2edge::Navigator navigator(config);
      if (actor) {
        navigator.setActor(actor);
      }

      std::vector<double> localizer_ms;
      std::vector<double> total_ms;
      double occupancy_sum = 0.0;
      std::size_t occupancy_count = 0;
      long processed = 0;

      for (std::size_t e = 0; e < replay.episodeCount(); ++e) {
        const sim2edge::ReplayEpisode & episode = replay.episode(e);
        navigator.reset();
        navigator.setGoalBody(episode.goal_body_x, episode.goal_body_y);

        for (uint32_t t = 0; t < episode.tick_count; ++t) {
          if (max_ticks > 0 && processed >= max_ticks) {
            break;
          }
          const sim2edge::ReplayTick tick = replay.tick(episode.first_tick + t);
          const auto & result = navigator.tick(
            tick.ranges, tick.bearings, tick.ages, tick.beams, tick.gyro_delta,
            tick.speed, tick.dt, tick.prior_delta_body);
          ++processed;

          // Discard the fill transient: the number that matters is the one the
          // loop pays once the submap is at its cap.
          if (static_cast<int>(t) < warmup) {
            continue;
          }
          localizer_ms.push_back(result.localize_us / 1000.0);
          total_ms.push_back(result.total_us / 1000.0);
          occupancy_sum += static_cast<double>(navigator.localizer().submapSize());
          ++occupancy_count;
        }
        if (max_ticks > 0 && processed >= max_ticks) {
          break;
        }
      }

      if (localizer_ms.empty()) {
        std::fprintf(stderr,
          "no steady-state ticks at submap %d stride %d -- episodes are "
          "shorter than the %d-tick warmup\n", submap, stride, warmup);
        continue;
      }

      Row row;
      row.submap = submap;
      row.stride = stride;
      row.median_ms = percentile(localizer_ms, 0.50);
      row.p95_ms = percentile(localizer_ms, 0.95);
      row.max_ms = *std::max_element(localizer_ms.begin(), localizer_ms.end());
      row.submap_points = occupancy_count
                            ? occupancy_sum / static_cast<double>(occupancy_count)
                            : 0.0;
      rows.push_back(row);

      std::printf("submap %3d  stride %d | localizer median %7.3f ms  "
                  "p95 %7.3f ms  max %7.3f ms | tick median %7.3f ms | "
                  "submap %.0f pts\n",
                  submap, stride, row.median_ms, row.p95_ms, row.max_ms,
                  percentile(total_ms, 0.50), row.submap_points);
    }
  }

  if (!rows.empty()) {
    const Row * production = nullptr;
    for (const Row & row : rows) {
      if (row.submap == 128 && row.stride == 2) {
        production = &row;
      }
    }
    if (production != nullptr) {
      const double period_ms = sim2edge::spec::kDt * 1000.0;
      std::printf("\nproduction (128 scans, stride 2): %.3f ms median, "
                  "%.1f%% of the %.0f ms period\n",
                  production->median_ms,
                  100.0 * production->median_ms / period_ms, period_ms);
      std::printf("Python on this board measured 21-27 ms at the same "
                  "configuration (docs/daily/2026_08_04_log.md).\n");
    }
  }
  return 0;
}
