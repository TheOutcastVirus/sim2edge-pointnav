// A recorded episode: the raw inputs one control tick needs, plus what the
// Python reference produced from them.
//
// This is the file the parity gate is built on. `tools/export_replay.py` writes
// it on the training host, where the corpus and torch live; the board reads it
// and replays it through the production code path. Because the reference
// outputs travel *with* the inputs, the board does not need Python, numpy, the
// dataset, or a network connection to answer "does the C++ agree" -- which
// matters, because this board has none of those.
//
// The layout is fixed-stride on purpose: every tick occupies the same number of
// bytes, so a tick is an offset multiply rather than a parse, and a 300 MB
// replay streams from page cache instead of being deserialized into RAM.
//
// Endianness is not negotiated. Host and board are both little-endian aarch64/
// x86-64, and a byte-order field that is never exercised is a field that is
// wrong the first time it matters. The magic and version catch the real risk,
// which is a format change.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sim2edge
{

/// On-disk header. Packed, fixed size, written verbatim.
/// Header flag: the reference action came from an actor checkpoint rather than
/// the corpus's stored oracle command. Against the actor an action comparison
/// measures *quantization* error; against the oracle it measures how far the
/// policy is from its teacher. Reporting either under one label would be a
/// straightforward way to mistake one for the other.
inline constexpr uint32_t kFlagReferenceActionFromActor = 1u << 0;

struct ReplayHeader
{
  char magic[8];            ///< "S2EREPL\0"
  uint32_t version;
  uint32_t flags;
  char observation_fingerprint[24];
  char sensor_contract[24];
  uint32_t episode_count;
  uint32_t tick_count;
  uint32_t beam_count;
  uint32_t reserved;
};

struct ReplayEpisode
{
  uint32_t first_tick;
  uint32_t tick_count;
  double goal_body_x;   ///< goal offset in the body frame at the episode's start
  double goal_body_y;
};

/// One tick's inputs and the Python reference outputs for them.
///
/// `ranges` / `bearings` / `ages` are the sweep **before** deskew -- what a
/// driver hands over -- so that the C++ deskew is part of what gets compared
/// rather than being assumed correct.
struct ReplayTick
{
  const float * ranges;
  const float * bearings;
  const float * ages;
  std::size_t beams;

  double gyro_delta;
  double speed;
  double dt;
  double prior_delta_body[3];

  /// The physical command the recorded run had issued on the previous tick.
  ///
  /// The observation's action block carries the previous command, so a replay
  /// that let the C++ use its *own* previous output would diverge from the
  /// reference after a single tick and the comparison would measure
  /// accumulated closed-loop drift rather than agreement. Forcing the recorded
  /// value keeps every tick a like-for-like test of one deterministic
  /// transformation. See Navigator::overrideLastCommand.
  double prev_v;
  double prev_omega;

  const double * ref_pose;       ///< 3: tracker-frame x, y, yaw
  const double * ref_goal_body;  ///< 2
  /// Deskew output, per beam. Carried so the parity report can attribute a
  /// divergence to a stage instead of only reporting that the bins disagree.
  /// A bin difference is a step function -- a beam that crossed a bin edge
  /// changes a bin by whatever the two surfaces differ by -- so the bins alone
  /// cannot distinguish "the geometry is wrong" from "a boundary case landed
  /// on the other side".
  const float * ref_deskewed_ranges;
  const float * ref_deskewed_bearings;
  const float * ref_bins;        ///< kLidarBins
  const float * ref_observation; ///< kObsDim
  const float * ref_action;      ///< 2, physical (v, omega)
  bool ref_observable;
};

/// Memory-maps a replay file and hands out zero-copy views of its ticks.
class ReplayReader
{
public:
  ReplayReader() = default;
  ~ReplayReader();

  ReplayReader(const ReplayReader &) = delete;
  ReplayReader & operator=(const ReplayReader &) = delete;

  /// Returns false and fills `error` on a missing file, a bad magic, an
  /// unsupported version, or a size that disagrees with the header's own
  /// counts -- the last of which is the one that would otherwise read past the
  /// mapping and crash somewhere unrelated.
  bool open(const std::string & path, std::string & error);

  const ReplayHeader & header() const { return *header_; }
  std::size_t episodeCount() const { return episodes_.size(); }
  const ReplayEpisode & episode(std::size_t i) const { return episodes_[i]; }
  std::size_t tickCount() const { return header_->tick_count; }

  ReplayTick tick(std::size_t index) const;

  /// The fingerprint the exporter recorded, as a string.
  std::string observationFingerprint() const;
  std::string sensorContract() const;

private:
  void * mapping_{nullptr};
  std::size_t mapping_size_{0};
  const ReplayHeader * header_{nullptr};
  std::vector<ReplayEpisode> episodes_;
  const uint8_t * ticks_{nullptr};
  std::size_t stride_{0};
};

}  // namespace sim2edge
