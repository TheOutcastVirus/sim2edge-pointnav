"""Demonstration storage. Raw in, observations out.

The shard holds **raw sensor readings and true poses**, never preprocessed
observations and never noisy ones. That single decision is what makes the
dataset survive design changes: retuning the noise model, changing the bin
count, or renormalizing the goal all become loader-side edits instead of a
recollection run. Only the raw quantities -- what the sensor measured, where
the robot was, what the oracle commanded -- are actually expensive to produce.

The observation is assembled per batch, on device, with fresh noise. Doing it
per sample in a ``Dataset.__getitem__`` would run 180-element torch ops one at a
time and spend more wall clock in kernel launches than in arithmetic.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import noise as noise_mod
from .preprocess import (bin_lidar, build_observation, goal_features,
                         rasterize_lidar, wrap_angle)
from .sensors import deskew_history, sweep_scan

#: Reconstruct each sweep at sub-tick resolution rather than rounding every
#: beam's acquisition time to the nearest stored scan. Off by default so every
#: existing corpus reads back bit-identically; see ``sensors.sweep_scan`` for
#: the 0.665 cm / 7.8 cm frame inconsistency that motivates turning it on, and
#: ``scripts/probe_sensor_consistency.py`` for the acceptance measurement.
#: Set by SIM2EDGE_SUBTICK_SWEEP=1 so a probe can flip it without a config
#: migration on every checkpoint.
SUBTICK_SWEEP = os.environ.get("SIM2EDGE_SUBTICK_SWEEP", "0") == "1"
from .spec import (DESKEW_WITH_GYRO, DT, HISTORY_FRAMES, LIDAR_SPIN_HZ,
                   MOTION_HISTORY_FRAMES, OMEGA_MAX, SPIN_TICKS, V_MAX,
                   fingerprint)

__all__ = ["Shard", "SequenceBatch", "save_shard", "load_shard"]


@dataclass
class SequenceBatch:
    """One time-major training window, everything the belief policy consumes.

    Time-major rather than batch-major because the training loop is a Python
    loop over timesteps -- ``rollout`` indexes ``[t]`` once per step, and a
    batch-major layout would make every one of those a strided gather.
    """

    lidar: torch.Tensor          # (T, N, H, B) binned, deskewed, corrupted
    raw_lidar: torch.Tensor      # (T, N, H, R) metric, uniform-angle ranges
    raw_valid: torch.Tensor      # (T, N, H, R) valid-return mask
    gyro: torch.Tensor           # (T, N, H) measured yaw increment, radians
    previous_action: torch.Tensor  # (T, N, H, 2) normalized commands
    action: torch.Tensor         # (T, N, 2) oracle label, SI units
    belief: torch.Tensor         # (T, N, 2) TRUE goal offset, body frame
    initial_belief: torch.Tensor  # (N, 2) truth at the window's first step
    #: (T, N, 2) TRUE body-frame displacement over the *preceding* tick -- the
    #: same backward-looking convention ``update_belief`` integrates in. Index 0
    #: is a placeholder, because no tick precedes the window's first step.
    #:
    #: Supervising this directly as well as the belief is worth the extra
    #: column: the belief loss has to assign credit through the whole
    #: recurrence, while this one lands on the step that produced the error.
    translation: torch.Tensor
    valid_steps: torch.Tensor     # (T, N) false only for padded episode tails

    def to(self, device) -> "SequenceBatch":
        for name in ("lidar", "raw_lidar", "raw_valid", "gyro",
                     "previous_action", "action", "belief",
                     "initial_belief", "translation", "valid_steps"):
            setattr(self, name, getattr(self, name).to(device))
        return self


@dataclass
class Shard:
    """One collection run's raw data, resident on whichever device it was loaded to.

    ``action`` is the oracle's *commanded* (v, omega) -- see ``oracle.py`` for
    why it must not be the achieved velocity. ``prev_action`` is stored rather
    than derived, because deriving it means special-casing episode boundaries
    and getting that wrong silently leaks the previous episode's last command
    into the first observation of the next one.
    """

    ranges: torch.Tensor        # (T, B) float16 raw metres, inf where no return
    angles: torch.Tensor        # (B,)   float32 beam bearings, robot frame
    pose: torch.Tensor          # (T, 3) float32 true world x, y, yaw
    goal: torch.Tensor          # (T, 2) float32 world goal
    action: torch.Tensor        # (T, 2) float32 oracle command
    prev_action: torch.Tensor   # (T, 2) float32 command issued the tick before
    episode: torch.Tensor       # (T,)   int32   episode id, for splitting
    layout: torch.Tensor | None = None      # (T,) int32, stable within layout_seed
    start_id: torch.Tensor | None = None    # (T,) int32 candidate within layout
    path_length: torch.Tensor | None = None # (T,) float32 geodesic start->goal
    outcome: torch.Tensor | None = None      # (T,) uint8: reached/collision/timeout
    steps_to_end: torch.Tensor | None = None # (T,) int32, zero at final sample
    driver: str = "oracle"                    # oracle or policy (DAgger)
    motion_model_version: int = 1
    #: Lazily derived, never loaded. See the ``episode_start`` property.
    _episode_start: torch.Tensor | None = field(
        default=None, repr=False, compare=False)
    _episode_end: torch.Tensor | None = field(
        default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return self.ranges.shape[0]

    def to(self, device) -> "Shard":
        for f in ("ranges", "angles", "pose", "goal", "action", "prev_action",
                  "episode", "layout", "start_id", "path_length", "outcome",
                  "steps_to_end"):
            value = getattr(self, f)
            if value is not None:
                setattr(self, f, value.to(device))
        # Derived indices belong to the device they were built on. Recomputing
        # is a single segment scan; silently indexing a CPU tensor into GPU data
        # is a device-mismatch crash three call frames away from the cause.
        self._episode_start = None
        self._episode_end = None
        return self

    def batches(self, batch_size: int, noise_scale: float = 1.0,
                shuffle: bool = True, generator: torch.Generator | None = None,
                drop_last: bool = False):
        """Yield ``(observation, action)`` with a fresh noise draw every pass.

        Because corruption happens here and not in the simulator, two epochs
        over the same shard show the policy two different corruptions of the
        same geometry. That is free regularization; a dataset with noise baked
        in gets exactly one sample per state, forever.
        """
        for sel in self._batch_indices(batch_size, shuffle, generator, drop_last):
            yield self._assemble_batch(sel, noise_scale, generator)

    def weighted_batches(
            self, batch_size: int, noise_scale: float = 1.0,
            shuffle: bool = True, generator: torch.Generator | None = None,
            drop_last: bool = False, *, collision_tail_steps: int = 20,
            collision_tail_weight: float = 5.0,
            near_obstacle_m: float = 0.30,
            near_obstacle_weight: float = 3.0,
            policy_data_weight: float = 2.0):
        """Yield observations, labels and bounded safety-importance weights.

        Outcome and distance-to-termination are training metadata, never model
        inputs.  Taking the maximum rather than multiplying overlapping rules
        keeps the objective bounded: a near-obstacle sample in a collision tail
        receives 5x weight, not an unstable 15x weight.
        """
        for sel in self._batch_indices(batch_size, shuffle, generator, drop_last):
            observation, action = self._assemble_batch(
                sel, noise_scale, generator)
            base_weight = policy_data_weight if self.driver == "policy" else 1.0
            weight = torch.full(
                (sel.numel(),), base_weight, device=self.ranges.device)
            if (self.outcome is not None and self.steps_to_end is not None
                    and collision_tail_steps > 0):
                tail = ((self.outcome[sel] == 1)
                        & (self.steps_to_end[sel] < collision_tail_steps))
                weight = torch.where(
                    tail, weight.new_tensor(collision_tail_weight), weight)
            if near_obstacle_weight > 1.0:
                minimum_range = self.ranges[sel].float().amin(dim=1)
                near = minimum_range < near_obstacle_m
                weight = torch.maximum(
                    weight,
                    torch.where(near, weight.new_tensor(near_obstacle_weight),
                                weight.new_tensor(1.0)),
                )
            yield observation, action, weight

    def _drift(self, noise_scale, generator) -> torch.Tensor:
        """(T, 3) odometry random walk over the whole shard, one draw per pass.

        Computed for the shard rather than per batch, because the walk is only
        meaningful along an episode and a batch is a scatter of unrelated rows.
        One segmented cumsum over 200k samples is cheaper than the observation
        assembly it feeds, so it is redrawn every epoch like the other noise.
        """
        return noise_mod.odometry_drift(
            self.episode, noise_scale, generator,
            yaw_increment=self.action[:, 1] * DT)

    # ── sequence access, for training through the belief recurrence ──────────

    @property
    def episode_start(self) -> torch.Tensor:
        """(T,) index of the first sample of each sample's own episode.

        Cached because every sequence batch clamps against it, and recomputing a
        segment scan per batch would dominate the assembly cost.
        """
        if self._episode_start is None:
            first = noise_mod._episode_starts(self.episode)
            ordinal = torch.cumsum(first.long(), dim=0) - 1
            starts = torch.nonzero(first, as_tuple=False).squeeze(1)
            self._episode_start = starts[ordinal]
        return self._episode_start

    @property
    def episode_end(self) -> torch.Tensor:
        """(T,) index of the final sample of each sample's own episode."""
        if self._episode_end is None:
            first = noise_mod._episode_starts(self.episode)
            starts = torch.nonzero(first, as_tuple=False).squeeze(1)
            ends = torch.cat((starts[1:] - 1, starts.new_tensor([len(self) - 1])))
            ordinal = torch.cumsum(first.long(), dim=0) - 1
            self._episode_end = ends[ordinal]
        return self._episode_end

    def sequence_starts(self, seq_len: int) -> torch.Tensor:
        """Indices where ``seq_len`` further samples stay inside one episode.

        Windows never straddle an episode boundary. A belief carried across one
        would be integrating motion between two unrelated rooms, and the
        auxiliary loss would be supervising it toward a goal that moved.
        """
        length = len(self)
        index = torch.arange(length, device=self.episode.device)
        last = index + seq_len - 1
        valid = last < length
        same = torch.zeros_like(valid)
        same[valid] = (self.episode[last[valid]] == self.episode[index[valid]])
        return index[valid & same]

    def sequence_batches(self, batch_size: int, seq_len: int,
                         noise_scale: float = 1.0, shuffle: bool = True,
                         generator: torch.Generator | None = None):
        """Yield :class:`SequenceBatch` windows with a fresh sensor draw each pass.

        This is where the spinning lidar actually gets simulated. For every
        output timestep the sweep is stitched from :data:`SPIN_TICKS` stored
        instantaneous scans, so a single observation contains beams up to 182 ms
        old -- which is what the hardware delivers and what the old
        instantaneous model never showed the policy.
        """
        starts = self.sequence_starts(seq_len)
        if starts.numel() == 0:
            return
        order = (starts[torch.randperm(starts.numel(), generator=generator,
                                       device=starts.device)]
                 if shuffle else starts)
        for i in range(0, order.numel(), batch_size):
            selection = order[i:i + batch_size]
            if selection.numel() == 0:
                break
            yield self._assemble_sequence(selection, seq_len, noise_scale,
                                          generator)

    def episode_batches(self, batch_size: int, noise_scale: float = 1.0,
                        shuffle: bool = True,
                        generator: torch.Generator | None = None):
        """Yield padded complete episodes, initializing belief only at reset.

        Unlike overlapping fixed windows, this path never resets a belief in the
        middle of a trajectory. Padding repeats the terminal sensor sample and is
        excluded from every loss by ``valid_steps``.
        """
        starts = torch.nonzero(noise_mod._episode_starts(self.episode),
                               as_tuple=False).squeeze(1)
        lengths = self.episode_end[starts] - starts + 1
        if shuffle:
            # Randomize ties, sort by length to minimize padded timesteps, then
            # randomize whole batches so optimization does not see a short-to-
            # long curriculum. On the production corpus this removes ~20% of
            # sensor/model work while preserving every episode exactly once.
            tie_order = torch.randperm(starts.numel(), generator=generator,
                                       device=starts.device)
            starts, lengths = starts[tie_order], lengths[tie_order]
            length_order = torch.argsort(lengths, stable=True)
            starts, lengths = starts[length_order], lengths[length_order]
            groups = list(range(0, starts.numel(), batch_size))
            group_order = torch.randperm(len(groups), generator=generator,
                                         device=starts.device).cpu().tolist()
            groups = [groups[index] for index in group_order]
        else:
            groups = list(range(0, starts.numel(), batch_size))
        for i in groups:
            selected = starts[i:i + batch_size]
            selected_lengths = lengths[i:i + batch_size]
            yield self._assemble_sequence(
                selected, int(selected_lengths.max()), noise_scale, generator,
                lengths=selected_lengths)

    def _assemble_sequence(self, starts, seq_len, noise_scale, generator,
                           lengths=None):
        device = self.ranges.device
        n = starts.numel()
        # Raw frames needed per window: the sequence itself, plus the actor's
        # history, plus one revolution of scan history behind the oldest of
        # those. Clamped at the episode's first sample, which repeats -- the
        # same convention `_history_indices` uses for the non-sequence path.
        pad = (MOTION_HISTORY_FRAMES - 1) + (SPIN_TICKS - 1)
        width = pad + seq_len
        offsets = torch.arange(width, device=device)
        raw = (starts[:, None] - pad + offsets[None, :])
        raw = torch.maximum(raw, self.episode_start[starts][:, None])
        raw = torch.minimum(raw, self.episode_end[starts][:, None])

        ranges = self.ranges[raw].float()                     # (n, W, B)
        beams = ranges.shape[-1]
        kw = {"generator": generator, "device": device, "dtype": torch.float32}
        # One mount error per window, not per frame: see `corrupt_lidar`.
        bias = torch.randn((n, 1), **kw) * (
            noise_mod.BEARING_BIAS_RAD * noise_scale)
        noisy, angles = noise_mod.corrupt_lidar(
            ranges.reshape(n * width, beams), self.angles, noise_scale,
            generator, bearing_bias=bias.repeat_interleave(width, dim=0))
        noisy = noisy.reshape(n, width, beams)
        angles = angles.reshape(n, width, beams)
        # A clamped episode-boundary frame is one physical scan repeated in the
        # FIFO, not several independent range-noise draws of that scan. Match
        # the live reset path before assembling overlapping sweeps.
        repeated = raw[:, 1:] == raw[:, :-1]
        for frame in range(1, width):
            same = repeated[:, frame - 1]
            noisy[:, frame] = torch.where(
                same[:, None], noisy[:, frame - 1], noisy[:, frame])
            angles[:, frame] = torch.where(
                same[:, None], angles[:, frame - 1], angles[:, frame])

        gyro_all = self._sequence_gyro(raw, noise_scale, generator)  # (n, W)

        # The sweep phase is a property of the motor, so it advances across the
        # window rather than being redrawn: consecutive observations must not
        # share a freshest bearing.
        phase0 = torch.rand((n,), **kw)
        # Assemble every output time in one batched sensor pass. The previous
        # implementation looped over ~150 timesteps and launched several tiny
        # CUDA kernels for each one, leaving a 4090 around 40% utilized. Time is
        # just another batch dimension here; flattening it removes that launch
        # bottleneck without changing any sensor arithmetic.
        outputs = width - SPIN_TICKS + 1
        windows = noisy.unfold(1, SPIN_TICKS, 1).permute(0, 1, 3, 2)
        flat_windows = windows.reshape(n * outputs, SPIN_TICKS, beams)
        j = torch.arange(SPIN_TICKS - 1, width, device=device)
        phase = ((phase0[:, None] + LIDAR_SPIN_HZ * DT * j[None]) % 1.0)
        swept, ages = sweep_scan(flat_windows, phase.reshape(-1),
                                 subtick=SUBTICK_SWEEP)
        frame_angles = angles[:, SPIN_TICKS - 1:].reshape(n * outputs, beams)
        if DESKEW_WITH_GYRO:
            gyro_windows = gyro_all.unfold(1, SPIN_TICKS, 1)
            raw_windows = raw.unfold(1, SPIN_TICKS, 1)
            speed_windows = self.prev_action[
                raw_windows.reshape(-1), 0
            ].reshape(n * outputs, SPIN_TICKS)
            swept, frame_angles = deskew_history(
                swept, frame_angles, ages,
                gyro_windows.reshape(n * outputs, SPIN_TICKS), speed_windows,
                subtick=SUBTICK_SWEEP)
        binned = bin_lidar(swept, frame_angles).reshape(n, outputs, -1)
        metric, metric_valid = rasterize_lidar(swept, frame_angles, beams)
        metric = metric.reshape(n, outputs, beams)
        metric_valid = metric_valid.reshape(n, outputs, beams)

        # Motion estimation gets a one-second metric/IMU/action baseline while
        # the warm-started actor retains its original four-frame observation.
        step = torch.arange(seq_len, device=device)
        frame = torch.arange(HISTORY_FRAMES, device=device)
        history = (step[:, None] + (MOTION_HISTORY_FRAMES - HISTORY_FRAMES)
                   + frame[None, :])                           # (T, actor H)
        motion_frame = torch.arange(MOTION_HISTORY_FRAMES, device=device)
        motion_history = step[:, None] + motion_frame[None, :] # (T, motion H)
        lidar = binned[:, history]                             # (n, T, actor H, bins)
        raw_lidar = metric[:, motion_history]
        raw_valid = metric_valid[:, motion_history]

        out = raw[:, pad:]                                     # (n, T) sample ids
        gyro_history = gyro_all[:, motion_history + (SPIN_TICKS - 1)]
        commands = self.prev_action[raw[:, SPIN_TICKS - 1:].reshape(-1)].clone()
        commands = commands.reshape(
            n, seq_len + MOTION_HISTORY_FRAMES - 1, 2)
        if self.motion_model_version < 2:
            commands[..., 0] = commands[..., 0] * self._legacy_command_scale(
                commands, noise_scale, generator)
        previous_history = _normalize_action(
            commands.reshape(-1, 2)
        ).reshape(n, seq_len + MOTION_HISTORY_FRAMES - 1, 2)[:, motion_history]

        belief = self._true_belief(out.reshape(-1)).reshape(n, seq_len, 2)
        translation = self._true_translation(out)
        valid_steps = torch.ones(n, seq_len, dtype=torch.bool, device=device)
        if lengths is not None:
            valid_steps = (torch.arange(seq_len, device=device)[None]
                           < lengths[:, None])
        transpose = lambda value: value.transpose(0, 1).contiguous()
        return SequenceBatch(
            lidar=transpose(lidar),
            raw_lidar=transpose(raw_lidar),
            raw_valid=transpose(raw_valid),
            gyro=transpose(gyro_history),
            previous_action=transpose(previous_history),
            action=transpose(self.action[out.reshape(-1)].reshape(
                n, seq_len, 2)),
            belief=transpose(belief),
            initial_belief=belief[:, 0].contiguous(),
            translation=transpose(translation),
            valid_steps=transpose(valid_steps),
        )

    def _legacy_command_scale(self, commands: torch.Tensor, noise_scale: float,
                              generator) -> torch.Tensor:
        """(n, W) multiplier turning true motion into a *believed* command.

        The corpus was collected against a simulator that tracked commands
        exactly, so the stored pose is simultaneously the truth and the
        dead-reckoned estimate, and their difference -- the only thing the
        odometry head can learn from -- is identically zero.

        Rather than discard 104 GB, the gap is introduced at load time in the
        direction the physics allows: the stored pose stays truth, and the
        command the model dead-reckons from is inflated by the slip that would
        have produced that motion. A robot commanding ``v`` and slipping 3%
        travels ``0.97 v``; presented with true motion ``v``, the equivalent
        belief is a command of ``v / 0.97``.

        This is a stand-in, and an honest one only because it is the *same*
        multiplicative structure real slip has. The real fix is a simulator that
        does not write the commanded pose straight to the root body; until
        collection is rerun against one, this is what makes the head trainable.
        """
        n, width, _ = commands.shape
        kw = {"generator": generator, "device": commands.device,
              "dtype": commands.dtype}
        systematic = torch.randn((n, 1), **kw) * (
            noise_mod.WHEEL_SLIP_SCALE * noise_scale)
        factor = noise_mod.wheel_motion_factor(
            systematic, commands[..., 1], noise_scale=noise_scale,
            generator=generator)
        return factor.reciprocal()

    def _true_translation(self, out: torch.Tensor) -> torch.Tensor:
        """(n, T, 2) exact body-frame displacement over each preceding tick.

        Clamped at the episode's first sample, where the displacement is defined
        as zero: there is no preceding tick, and differencing across the boundary
        would hand the model the previous episode's motion as a label.
        """
        previous = torch.maximum(out - 1, self.episode_start[out])
        delta = self.pose[out][..., :2] - self.pose[previous][..., :2]
        yaw = self.pose[previous][..., 2]
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([cos * delta[..., 0] + sin * delta[..., 1],
                            -sin * delta[..., 0] + cos * delta[..., 1]], dim=-1)

    def _sequence_gyro(self, raw, noise_scale, generator):
        """(n, W) measured yaw increment for every frame in every window.

        Synthesized from the stored true yaw rather than collected, for the same
        reason the lidar noise is: a derivable quantity kept out of the shard is
        a training knob, and a quantity written into the shard is a property of
        it forever.
        """
        yaw = self.pose[raw, 2]                                # (n, W)
        increment = torch.zeros_like(yaw)
        increment[:, 1:] = wrap_angle(yaw[:, 1:] - yaw[:, :-1])
        n, width = yaw.shape
        kw = {"generator": generator, "device": yaw.device, "dtype": yaw.dtype}
        bias0 = torch.randn((n, 1), **kw) * (
            noise_mod.GYRO_BIAS_RAD_S * noise_scale)
        scale_error = torch.randn((n, 1), **kw) * (
            noise_mod.GYRO_SCALE_ERROR * noise_scale)
        walk = torch.cumsum(
            torch.randn((n, width), **kw) * (
                noise_mod.GYRO_BIAS_WALK_RAD_S * noise_scale * math.sqrt(DT)),
            dim=1)
        white = torch.randn((n, width), **kw) * (
            noise_mod.GYRO_ARW_RAD_PER_SQRT_S * noise_scale * math.sqrt(DT))
        measured = (increment * (1.0 + scale_error)
                    + (bias0 + walk) * DT + white)
        # Clamped history repeats the first sample of an episode. No motion
        # interval precedes that sample, so neither gyro noise nor the previous
        # episode is allowed to invent one. This matches synthesize_gyro and the
        # live reset path.
        measured[raw == self.episode_start[raw]] = 0.0
        return measured

    def _true_belief(self, index: torch.Tensor) -> torch.Tensor:
        """(M, 2) exact goal offset in the body frame -- the supervision target.

        Ground truth is used *only* as a label here. It never enters an
        observation, which is the whole point of the design: the model has to
        reconstruct this quantity from lidar and gyro alone.
        """
        pose, goal = self.pose[index], self.goal[index]
        d = goal - pose[:, :2]
        cos, sin = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        return torch.stack([cos * d[:, 0] + sin * d[:, 1],
                            -sin * d[:, 0] + cos * d[:, 1]], dim=1)

    def _batch_indices(self, batch_size, shuffle, generator, drop_last):
        n = len(self)
        order = (torch.randperm(n, generator=generator, device=self.ranges.device)
                 if shuffle else torch.arange(n, device=self.ranges.device))
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            if drop_last and sel.numel() < batch_size:
                break
            yield sel

    def _assemble_batch(self, sel, noise_scale, generator):
        # Each actor frame is a full motor sweep assembled from several stored
        # instantaneous scans. The previous path binned four instantaneous
        # scans directly, so BC trained on a different sensor than Isaac and
        # hardware even after the spinning-lidar contract was introduced.
        n = sel.numel()
        width = HISTORY_FRAMES + SPIN_TICKS - 1
        offsets = torch.arange(width, device=sel.device)
        raw = sel[:, None] - (width - 1) + offsets[None]
        raw = torch.maximum(raw, self.episode_start[sel][:, None])

        ranges = self.ranges[raw].float()                    # (n, W, beams)
        beams = ranges.shape[-1]
        kw = {"generator": generator, "device": ranges.device,
              "dtype": torch.float32}
        bearing_bias = torch.randn((n, 1), **kw) * (
            noise_mod.BEARING_BIAS_RAD * noise_scale)
        noisy, angles = noise_mod.corrupt_lidar(
            ranges.reshape(n * width, beams), self.angles, noise_scale,
            generator,
            bearing_bias=bearing_bias.repeat_interleave(width, dim=0))
        noisy = noisy.reshape(n, width, beams)
        angles = angles.reshape(n, width, beams)
        repeated = raw[:, 1:] == raw[:, :-1]
        for frame in range(1, width):
            same = repeated[:, frame - 1]
            noisy[:, frame] = torch.where(
                same[:, None], noisy[:, frame - 1], noisy[:, frame])
            angles[:, frame] = torch.where(
                same[:, None], angles[:, frame - 1], angles[:, frame])

        windows = noisy.unfold(1, SPIN_TICKS, 1).permute(0, 1, 3, 2)
        flat_windows = windows.reshape(n * HISTORY_FRAMES, SPIN_TICKS, beams)
        # Free-running phase derived from the stable sample/episode identity:
        # deterministic clean validation, varied bearing freshness across the
        # corpus, and exactly LIDAR_SPIN_HZ * DT advance between history frames.
        history = raw[:, SPIN_TICKS - 1:]
        phase = (
            history.float() * (LIDAR_SPIN_HZ * DT)
            + self.episode[history].float() * 0.61803398875
        ).remainder(1.0)
        swept, ages = sweep_scan(flat_windows, phase.reshape(-1),
                                 subtick=SUBTICK_SWEEP)
        frame_angles = angles[:, SPIN_TICKS - 1:].reshape(
            n * HISTORY_FRAMES, beams)
        if DESKEW_WITH_GYRO:
            gyro = self._sequence_gyro(raw, noise_scale, generator)
            gyro_windows = gyro.unfold(1, SPIN_TICKS, 1)
            raw_windows = raw.unfold(1, SPIN_TICKS, 1)
            speed = self.prev_action[
                raw_windows.reshape(-1), 0
            ].reshape(n * HISTORY_FRAMES, SPIN_TICKS)
            swept, frame_angles = deskew_history(
                swept, frame_angles, ages,
                gyro_windows.reshape(n * HISTORY_FRAMES, SPIN_TICKS), speed)
        lidar = bin_lidar(swept, frame_angles).reshape(
            n, HISTORY_FRAMES, -1)
        # ObservationHistory pushes one reset scan and repeats that *processed*
        # frame across its actor FIFO. Recomputing the same raw scan at several
        # synthetic phases would instead create different deskewed frames.
        for frame in range(1, HISTORY_FRAMES):
            same = history[:, frame] == history[:, frame - 1]
            lidar[:, frame] = torch.where(
                same[:, None], lidar[:, frame - 1], lidar[:, frame])
        # Integrated drift first, then the instantaneous residual on top. Only
        # the second of those existed before; a policy evaluated without the
        # first has never been asked to cope with an error that does not come
        # back, which is the only kind that actually threatens it.
        drift = self._drift(noise_scale, generator)
        pose, goal = noise_mod.corrupt_goal(self.pose[sel] + drift[sel],
                                            self.goal[sel],
                                            noise_scale, generator)
        observation = build_observation(
            lidar,
            goal_features(pose, goal),
            _normalize_action(self.prev_action[history].flatten(0, 1)).reshape(
                sel.numel(), HISTORY_FRAMES, 2),
        )
        return observation, self.action[sel]


def _history_indices(sel: torch.Tensor, episode: torch.Tensor,
                     frames: int) -> torch.Tensor:
    """Oldest-to-current indices, repeating the episode's first sample."""
    offsets = torch.arange(frames, device=sel.device)
    backward = (sel[:, None] - offsets).clamp_min(0)  # current to oldest
    same_episode = episode[backward] == episode[sel, None]
    contiguous = torch.cumprod(same_episode.to(torch.int8), dim=1).bool()
    earliest = torch.where(contiguous, backward, sel[:, None]).amin(dim=1)
    chronological = sel[:, None] - torch.arange(
        frames - 1, -1, -1, device=sel.device)
    return torch.maximum(chronological, earliest[:, None])


def _normalize_action(a: torch.Tensor) -> torch.Tensor:
    """Physical (v, omega) -> [-1, 1], the scale the network sees.

    The *label* stays in SI units -- the exported graph squashes and rescales
    in-graph so the C++ node receives m/s directly -- but the last-action input
    is an activation and has to be conditioned like one.
    """
    return torch.stack([a[:, 0] / V_MAX, a[:, 1] / OMEGA_MAX], dim=1)


def save_shard(path, ranges, angles, pose, goal, action, prev_action, episode,
               **meta) -> None:
    """Write a shard, stamped with the spec fingerprint.

    Ranges go to float16: the A1's own error is 2 cm and float16 resolves better
    than 6 mm out at 12 m, so the storage format is two orders of magnitude
    finer than the sensor. It halves the shard, which is the only part of this
    that is large.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        fingerprint=np.array(fingerprint()),
        ranges=np.asarray(ranges, dtype=np.float16),
        angles=np.asarray(angles, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        prev_action=np.asarray(prev_action, dtype=np.float32),
        episode=np.asarray(episode, dtype=np.int32),
        **meta,
    )
    # The file remains an ordinary np.load-compatible NPZ.  Deflate level 1 is
    # 1.7x faster than NumPy's default level 6 on representative BC scans, for
    # only ~8% more disk, so synchronous shard writes stall simulation less.
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value),
                                          allow_pickle=False)


def load_shard(path, device="cpu") -> Shard:
    """Load and verify. A fingerprint mismatch raises rather than warns.

    A shard collected at 180 bins fed to a policy expecting 120 has compatible
    looking shapes right until it doesn't, and the resulting failure presents as
    a training problem. Refusing here is the cheapest place to catch it.
    """
    # ``np.load`` owns a ZipFile. Keeping it alive until garbage collection is
    # harmless for one shard and exhausts file descriptors in a 623-shard
    # training pass, so every array is materialized inside the context manager.
    with np.load(Path(path), allow_pickle=False) as z:
        got, want = str(z["fingerprint"]), fingerprint()
        if got != want:
            raise ValueError(
                f"shard {path} was collected under spec {got}, this process is "
                f"{want}. The observation contract changed; recollect or check "
                f"out the spec that produced it."
            )
        t = lambda k, d=torch.float32: torch.from_numpy(z[k]).to(
            device=device, dtype=d)
        optional = lambda k, d: (t(k, d) if k in z.files else None)
        episode_array = np.asarray(z["episode"], dtype=np.int32)
        return Shard(
            ranges=t("ranges", torch.float16), angles=t("angles"), pose=t("pose"),
            goal=t("goal"), action=t("action"), prev_action=t("prev_action"),
            episode=torch.from_numpy(episode_array).to(device=device),
            layout=optional("layout", torch.int32),
            start_id=optional("start_id", torch.int32),
            path_length=optional("path_length", torch.float32),
            outcome=optional("outcome", torch.uint8),
            steps_to_end=torch.from_numpy(
                _steps_to_episode_end(episode_array)).to(device=device),
            driver=str(z["driver"]) if "driver" in z.files else "oracle",
            motion_model_version=(int(z["motion_model_version"])
                                  if "motion_model_version" in z.files else 1),
        )


def _steps_to_episode_end(episode: np.ndarray) -> np.ndarray:
    """Return remaining stored samples within each contiguous episode."""
    result = np.empty(episode.shape, dtype=np.int32)
    if episode.size == 0:
        return result
    start = 0
    for end in np.flatnonzero(np.r_[episode[1:] != episode[:-1], True]):
        result[start:end + 1] = np.arange(end - start, -1, -1,
                                          dtype=np.int32)
        start = int(end) + 1
    return result
