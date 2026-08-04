#include "sim2edge/replay.hpp"

#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sim2edge/spec_generated.hpp"

namespace sim2edge
{
namespace
{

constexpr uint32_t kVersion = 1;

/// Thirteen doubles lead every tick: gyro_delta, speed, dt,
/// prior_delta_body[3], prev_v, prev_omega, ref_pose[3], ref_goal_body[2].
constexpr std::size_t kScalarCount = 13;

/// Bytes one tick occupies. Must match tools/export_replay.py exactly; the
/// header's own counts are cross-checked against the file size at open, which
/// turns a silent disagreement here into a refusal to load rather than a read
/// of structured garbage.
///
/// Doubles come first and the stride is rounded to eight bytes, so every tick
/// starts 8-aligned and the scalar block is 8-aligned within it whatever the
/// beam count happens to be. The alternative -- floats first -- makes alignment
/// depend on the parity of `beams`, which is a bug that waits for a driver to
/// change its beam count.
std::size_t tickStride(std::size_t beams)
{
  const std::size_t floats = beams * 3                             // sweep in
                             + beams * 2                           // deskew ref
                             + static_cast<std::size_t>(spec::kLidarBins)
                             + static_cast<std::size_t>(spec::kObsDim)
                             + 2;                                  // ref_action
  const std::size_t bytes = kScalarCount * sizeof(double) +
                            floats * sizeof(float) +
                            1;                                     // observable
  return (bytes + 7) & ~std::size_t{7};
}

std::string fromFixed(const char * field, std::size_t capacity)
{
  const std::size_t length = ::strnlen(field, capacity);
  return std::string(field, length);
}

}  // namespace

ReplayReader::~ReplayReader()
{
  if (mapping_ != nullptr) {
    ::munmap(mapping_, mapping_size_);
  }
}

bool ReplayReader::open(const std::string & path, std::string & error)
{
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    error = "cannot open " + path;
    return false;
  }
  struct stat info {};
  if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
    ::close(fd);
    error = "cannot stat " + path;
    return false;
  }
  mapping_size_ = static_cast<std::size_t>(info.st_size);
  mapping_ = ::mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping_ == MAP_FAILED) {
    mapping_ = nullptr;
    error = "cannot mmap " + path;
    return false;
  }

  if (mapping_size_ < sizeof(ReplayHeader)) {
    error = path + " is too small to hold a header";
    return false;
  }
  header_ = static_cast<const ReplayHeader *>(mapping_);
  if (std::memcmp(header_->magic, "S2EREPL", 7) != 0) {
    error = path + " is not a sim2edge replay";
    return false;
  }
  if (header_->version != kVersion) {
    error = path + " is replay version " + std::to_string(header_->version) +
            ", this build reads version " + std::to_string(kVersion);
    return false;
  }
  if (header_->beam_count == 0 || header_->tick_count == 0) {
    error = path + " contains no ticks";
    return false;
  }

  const std::size_t episodes_bytes =
    static_cast<std::size_t>(header_->episode_count) * sizeof(ReplayEpisode);
  stride_ = tickStride(header_->beam_count);
  const std::size_t expected = sizeof(ReplayHeader) + episodes_bytes +
                               static_cast<std::size_t>(header_->tick_count) *
                                 stride_;
  if (mapping_size_ != expected) {
    // A stride disagreement between exporter and reader would otherwise read
    // structured garbage: plausible floats, wrong ones. Refuse instead.
    error = path + " is " + std::to_string(mapping_size_) +
            " bytes but its header describes " + std::to_string(expected) +
            "; exporter and reader disagree on the tick layout";
    return false;
  }

  const uint8_t * base = static_cast<const uint8_t *>(mapping_);
  const ReplayEpisode * episodes = reinterpret_cast<const ReplayEpisode *>(
    base + sizeof(ReplayHeader));
  episodes_.assign(episodes, episodes + header_->episode_count);
  ticks_ = base + sizeof(ReplayHeader) + episodes_bytes;
  return true;
}

ReplayTick ReplayReader::tick(std::size_t index) const
{
  const uint8_t * p = ticks_ + index * stride_;
  const std::size_t beams = header_->beam_count;

  ReplayTick out;
  out.beams = beams;

  const double * scalars = reinterpret_cast<const double *>(p);
  out.gyro_delta = scalars[0];
  out.speed = scalars[1];
  out.dt = scalars[2];
  out.prior_delta_body[0] = scalars[3];
  out.prior_delta_body[1] = scalars[4];
  out.prior_delta_body[2] = scalars[5];
  out.prev_v = scalars[6];
  out.prev_omega = scalars[7];
  out.ref_pose = scalars + 8;
  out.ref_goal_body = scalars + 11;
  p += kScalarCount * sizeof(double);

  out.ranges = reinterpret_cast<const float *>(p);
  p += beams * sizeof(float);
  out.bearings = reinterpret_cast<const float *>(p);
  p += beams * sizeof(float);
  out.ages = reinterpret_cast<const float *>(p);
  p += beams * sizeof(float);

  out.ref_deskewed_ranges = reinterpret_cast<const float *>(p);
  p += beams * sizeof(float);
  out.ref_deskewed_bearings = reinterpret_cast<const float *>(p);
  p += beams * sizeof(float);

  out.ref_bins = reinterpret_cast<const float *>(p);
  p += static_cast<std::size_t>(spec::kLidarBins) * sizeof(float);
  out.ref_observation = reinterpret_cast<const float *>(p);
  p += static_cast<std::size_t>(spec::kObsDim) * sizeof(float);
  out.ref_action = reinterpret_cast<const float *>(p);
  p += 2 * sizeof(float);
  out.ref_observable = *p != 0;
  return out;
}

std::string ReplayReader::observationFingerprint() const
{
  return fromFixed(header_->observation_fingerprint,
                   sizeof(header_->observation_fingerprint));
}

std::string ReplayReader::sensorContract() const
{
  return fromFixed(header_->sensor_contract, sizeof(header_->sensor_contract));
}

}  // namespace sim2edge
