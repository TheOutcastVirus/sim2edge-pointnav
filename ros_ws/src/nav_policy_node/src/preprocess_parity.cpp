// Standalone parity harness: reads raw sensor samples, writes preprocessed
// observations, so tools/parity_check.py can diff this against the Python
// reference implementation.
//
// Deliberately depends on nothing but the C++ standard library and
// preprocess.cpp -- no ROS, no OpenCV, no ExecuTorch. That means the parity
// gate can be run on a laptop during development, not only on the board, and
// there is no chance of the test exercising a different code path than the node
// does, because it calls the very same functions.
//
//   python3 tools/parity_check.py --emit-inputs /tmp/in.bin
//   ./preprocess_parity /tmp/in.bin /tmp/cpp.bin
//   python3 tools/parity_check.py --emit-inputs /tmp/in.bin --cpp-outputs /tmp/cpp.bin
//
// Binary format is mirrored in tools/parity_check.py; keep the two in sync.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "nav_policy_node/preprocess.hpp"

namespace
{

constexpr char kInMagic[4] = {'S', '2', 'E', 'P'};
constexpr char kOutMagic[4] = {'S', '2', 'E', 'O'};
constexpr int32_t kFormatVersion = 1;

template<typename T>
bool readPod(std::istream & in, T & value)
{
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  return static_cast<bool>(in);
}

bool readArray(std::istream & in, std::vector<float> & out, std::size_t count)
{
  out.resize(count);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<bool>(in);
}

void writeArray(std::ostream & out, const std::vector<float> & values)
{
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 3) {
    std::cerr << "usage: preprocess_parity <input.bin> <output.bin>\n";
    return 2;
  }

  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::cerr << "ERROR: cannot open " << argv[1] << "\n";
    return 1;
  }

  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, kInMagic, 4) != 0) {
    std::cerr << "ERROR: " << argv[1] << " is not a parity input file\n";
    return 1;
  }

  int32_t version = 0;
  int32_t n_samples = 0;
  if (!readPod(in, version) || !readPod(in, n_samples)) {
    std::cerr << "ERROR: truncated header\n";
    return 1;
  }
  if (version != kFormatVersion) {
    std::cerr << "ERROR: format version " << version << ", expected "
              << kFormatVersion << "\n";
    return 1;
  }

  std::ofstream out(argv[2], std::ios::binary);
  if (!out) {
    std::cerr << "ERROR: cannot write " << argv[2] << "\n";
    return 1;
  }
  out.write(kOutMagic, 4);
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  out.write(reinterpret_cast<const char *>(&n_samples), sizeof(n_samples));

  std::vector<float> depth_raw;
  std::vector<float> ranges;
  nav_policy_node::Observation obs;

  for (int32_t i = 0; i < n_samples; ++i) {
    int32_t depth_h = 0;
    int32_t depth_w = 0;
    if (!readPod(in, depth_h) || !readPod(in, depth_w)) {
      std::cerr << "ERROR: truncated at sample " << i << " (depth header)\n";
      return 1;
    }
    if (!readArray(in, depth_raw,
                   static_cast<std::size_t>(depth_h) * depth_w)) {
      std::cerr << "ERROR: truncated at sample " << i << " (depth data)\n";
      return 1;
    }

    int32_t n_beams = 0;
    float angle_min = 0.0f;
    float angle_increment = 0.0f;
    float range_min = 0.0f;
    float range_max = 0.0f;
    if (!readPod(in, n_beams) || !readPod(in, angle_min) ||
        !readPod(in, angle_increment) || !readPod(in, range_min) ||
        !readPod(in, range_max))
    {
      std::cerr << "ERROR: truncated at sample " << i << " (scan header)\n";
      return 1;
    }
    if (!readArray(in, ranges, static_cast<std::size_t>(n_beams))) {
      std::cerr << "ERROR: truncated at sample " << i << " (scan data)\n";
      return 1;
    }

    float goal_x = 0.0f;
    float goal_y = 0.0f;
    float last_v = 0.0f;
    float last_omega = 0.0f;
    if (!readPod(in, goal_x) || !readPod(in, goal_y) ||
        !readPod(in, last_v) || !readPod(in, last_omega))
    {
      std::cerr << "ERROR: truncated at sample " << i << " (goal/action)\n";
      return 1;
    }

    nav_policy_node::preprocessDepth(depth_raw.data(), depth_h, depth_w,
                                     obs.depth);
    nav_policy_node::preprocessLidar(ranges.data(), ranges.size(), angle_min,
                                     angle_increment, range_min, range_max,
                                     obs.lidar);
    nav_policy_node::preprocessGoal(goal_x, goal_y, obs.goal);
    nav_policy_node::preprocessLastAction(last_v, last_omega, obs.last_action);

    writeArray(out, obs.depth);
    writeArray(out, obs.lidar);
    writeArray(out, obs.goal);
    writeArray(out, obs.last_action);
  }

  std::cout << "Preprocessed " << n_samples << " samples -> " << argv[2]
            << "\nSpec fingerprint (compiled in): "
            << nav_policy_node::spec::kSpecFingerprint << "\n";
  return 0;
}
