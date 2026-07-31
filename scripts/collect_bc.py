"""Collect behaviour-cloning demonstrations in Isaac Lab.

    ~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 8192 \
        --episodes 65536 --num_layouts 8192 --out data/bc_diverse_v1

Bring it up in the order in ``docs/RUNBOOK.md`` -- each step isolates a
different failure. ``--num_envs 4 --gui --check`` is the first one.

What gets written
-----------------
Raw measurements only: ranges as the sensor reported them, beam bearings in
base_link, true poses, and the oracle's **commanded** ``(v, omega)``. Noise and
binning happen in the loader, so retuning either does not invalidate a run that
took hours.

By default, whole layouts are routed to separate ``train/`` and ``validation/``
directories. Each layout has several feasible start positions and every reset
gets a fresh yaw. This keeps geometry construction affordable without repeating
the same fixed navigation task hundreds of times.

Two label conventions that look like details and are not:

**The label is the command, never the achieved velocity.** Reading back what
physics produced would teach the policy to emit a value the actuator lags a
second time -- a systematic underactuation that is invisible in sim.

**The label is the *unperturbed* oracle command; only execution is perturbed.**
That is what puts the robot in a tube of states around the expert path while
still labelling each one with what the expert would do there. Without it every
state in the dataset is one the oracle put itself in, and the policy has no
recovery behaviour at all.

Passing ``--driver-checkpoint`` switches execution to a learned policy while
the oracle continues to provide every stored label.  This is the minimal DAgger
operation: collect the states caused by the policy's systematic errors, without
changing the durable observation or label format.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=2000,
                    help="episode prefixes to retain, not episodes attempted")
    ap.add_argument("--num_layouts", type=int, default=512)
    ap.add_argument("--starts_per_layout", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--shard_steps", type=int, default=200_000,
                    help="steps per shard; ~610 B each, so 200k is ~120 MB")
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="execution noise as a fraction of each axis limit. "
                         "0 collects pure on-policy expert data, which is the "
                         "distribution BC fails on -- see the module docstring")
    ap.add_argument("--driver-checkpoint", type=Path,
                    help="execute this policy while retaining oracle labels (DAgger)")
    ap.add_argument("--validation_fraction", type=float, default=0.05,
                    help="fraction of layout IDs routed to validation/; 0 keeps "
                         "the legacy single-directory format")
    ap.add_argument("--writer_queue", type=int, default=2,
                    help="completed shards allowed in the asynchronous writer; "
                         "each costs roughly 300 MB decompressed")
    filtering = ap.add_mutually_exclusive_group()
    filtering.add_argument(
        "--filter-failures", action="store_true",
        help="drop episodes shorter than 20 steps or with under 0.5 m net "
             "Euclidean progress. Off by default: pre-collision states have "
             "valid oracle labels regardless of the rollout outcome")
    filtering.add_argument(
        "--keep_failures", action="store_false", dest="filter_failures",
        help=argparse.SUPPRESS)
    ap.set_defaults(filter_failures=False)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--dump-scene", action="store_true",
                    help="print env origins, camera pose and sample prim world "
                         "positions, then exit -- for when the viewport is "
                         "unreadable and you need numbers instead of pixels")
    ap.add_argument("--check", action="store_true",
                    help="run the bring-up assertions and exit without writing")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in [0, 1)")
    if args.starts_per_layout < 1 or args.writer_queue < 1:
        raise ValueError("starts_per_layout and writer_queue must be positive")
    if args.driver_checkpoint is not None and not args.driver_checkpoint.is_file():
        raise FileNotFoundError(args.driver_checkpoint)

    # Line-buffer stdout. Redirected to a file it is block-buffered, so progress
    # lines sit in the buffer until exit -- and a run whose shutdown wedges never
    # shows them at all, making a completed run look like it stalled at reset.
    sys.stdout.reconfigure(line_buffering=True)

    # The app must launch before anything under isaaclab.sim is imported -- the
    # USD bindings live in Isaac's kit runtime and are not on sys.path until
    # then. nav_task swallows that ImportError to stay importable in CI, so
    # importing it too early fails later and much less clearly.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=not args.gui).app

    import torch

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv, normalize_action
    from sim2edge.dataset import save_shard
    from sim2edge.oracle import perturb_action
    from sim2edge.preprocess import ObservationHistory, bin_lidar, goal_features
    from sim2edge.spec import LIDAR_BINS, OMEGA_MAX, V_MAX
    from sim2edge.training import load_policy_checkpoint

    cfg = NavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.num_layouts = args.num_layouts
    cfg.starts_per_layout = args.starts_per_layout
    cfg.layout_seed = args.seed
    cfg.seed = args.seed
    cfg.collection_mode = True

    # Progress markers. Isaac Lab initializes sensors lazily on first `.data`
    # access, so a stall inside reset() is really a stall inside sensor setup --
    # and without these it is indistinguishable from a stall anywhere else.
    print("[collect] building env ...", flush=True)
    env = Sim2EdgeNavEnv(cfg)
    print(f"[collect] env ready: {env.num_envs} envs on {env.device}", flush=True)
    device = env.device
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print("[collect] first reset (initializes the ray caster) ...", flush=True)
    env.reset()
    print("[collect] reset done", flush=True)

    if args.gui:
        frame_env0(env)

    if args.dump_scene:
        dump_scene(env, torch)
        _finish(simulation_app)

    if args.check:
        run_checks(env, torch)
        _finish(simulation_app)

    # Do not rely on a separate manual --check invocation. Every collection is
    # gated on physical command tracking before a single sample can be written.
    print("[collect] drive-response preflight ...", flush=True)
    run_drive_checks(env, torch)
    env.reset()
    print("[collect] drive-response preflight passed", flush=True)

    driver_policy = None
    driver_history = None
    reset_history = None
    driver = "oracle"
    if args.driver_checkpoint is not None:
        driver_policy = load_policy_checkpoint(args.driver_checkpoint, device=device)
        driver_history = ObservationHistory(
            env.num_envs, frames=driver_policy.config.history_frames,
            lidar_bins=driver_policy.config.lidar_bins, device=device)
        driver = "policy"
        print(f"[collect] DAgger driver: {args.driver_checkpoint} ",
              f"({driver_policy.parameter_count:,} parameters)", flush=True)

    collector = Collector(args.out, args.shard_steps, args.num_envs,
                          max_steps=int(env.max_episode_length),
                          n_beams=int(env._beam_angles.shape[0]),
                          device=device, layout_seed=args.seed,
                          num_layouts=args.num_layouts,
                          starts_per_layout=args.starts_per_layout,
                          sigma=(0.0 if driver_policy is not None else args.sigma),
                          validation_fraction=args.validation_fraction,
                          writer_queue=args.writer_queue, driver=driver)
    step = 0
    # Wall-clock per control step, split so the bottleneck is attributable
    # rather than inferred. `sim` covers observation + oracle + env.step;
    # `rec` is the host-side recorder, which is O(num_envs) Python and the
    # first thing to suspect when scaling up.
    t_sim = t_rec = 0.0
    t0 = time.perf_counter()

    # Ctrl-C should land the episodes already buffered, not discard them. Data
    # sitting in `pending` is otherwise lost, and there can be a full shard's
    # worth (shard_steps) of finished episodes in there.
    stopping = {"now": False}

    def _on_interrupt(signum, frame):
        if stopping["now"]:                          # second Ctrl-C: give up
            os._exit(130)
        stopping["now"] = True
        print("\n[collect] interrupt: finishing the current step, then writing "
              "buffered episodes. Ctrl-C again to abandon them.", flush=True)

    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    while collector.episodes_kept < args.episodes and not stopping["now"]:
        pose = env._robot_pose()
        ranges, angles = env.raw_scan()

        label = env.oracle_command(pose)             # unperturbed: this is the label
        if driver_policy is None:
            executed = perturb_action(label, args.sigma, generator=gen,
                                      prev_action=env._action)
        else:
            last_action = torch.stack((env._action[:, 0] / V_MAX,
                                       env._action[:, 1] / OMEGA_MAX), dim=1)
            driver_history.push(bin_lidar(ranges, angles), last_action,
                                reset_history)
            observation = driver_history.observation(goal_features(
                pose, env._goal_w - env.scene.env_origins[:, :2]))
            with torch.inference_mode():
                executed = driver_policy(observation)

        _t = time.perf_counter()
        collector.record(
            ranges, pose, env._goal_w - env.scene.env_origins[:, :2],
            label, env._action, env._layout, env._start_id,
            env._optimal_path_m)
        t_rec += time.perf_counter() - _t

        _, _, terminated, truncated, _ = env.step(normalize_action(executed))

        done = terminated | truncated
        reset_history = done
        if done.any():
            collector.flush(done, env._reached, env._collided, angles,
                            filter_failures=args.filter_failures)

        step += 1
        if step % 50 == 0:
            wall = time.perf_counter() - t0
            t_sim = wall - t_rec - collector.write_wait_seconds
            useful = collector.steps_kept / wall
            print(f"step {step:6d}  kept {collector.episodes_kept:5d} "
                  f"(full {collector.kept_complete:5d} "
                  f"trunc {collector.kept_truncated:5d} "
                  f"part {collector.kept_partial:5d})  "
                  f"dropped {collector.episodes_dropped:5d}  "
                  f"| {wall / step * 1000:6.1f} ms/step "
                  f"(sim {t_sim / step * 1000:6.1f}  "
                  f"rec {t_rec / step * 1000:5.1f}  "
                  f"writer-wait {collector.write_wait_seconds / step * 1000:5.1f})  "
                  f"| {args.num_envs * step / wall:7.0f} env-steps/s  "
                  f"{useful:7.0f} kept-samples/s")

    collector.close(angles)
    print(f"[collect] async writer busy {collector.writer.worker_seconds:.1f}s; "
          f"simulation blocked on it for {collector.write_wait_seconds:.1f}s",
          flush=True)
    if stopping["now"]:
        print("[collect] stopped early by request")
    attempts = collector.episodes_kept + collector.episodes_dropped
    yield_pct = 100.0 * collector.episodes_kept / max(1, attempts)
    print(f"\n{collector.episodes_kept} episodes, {collector.total_steps} steps, "
          f"{collector.shards_written} shard(s) under {args.out}")
    print(f"yield {yield_pct:.0f}%  ({collector.episodes_kept} kept of {attempts} "
          f"attempts)")
    print(f"  kept, reached the goal   : {collector.kept_complete}")
    print(f"  kept, truncated at graze : {collector.kept_truncated}")
    print(f"  kept, ran out of clock   : {collector.kept_partial}")
    print(f"  dropped, too short       : {collector.dropped_short}")
    print(f"  dropped, no net progress : {collector.dropped_no_progress}")
    print(f"  samples kept / filtered  : {collector.steps_kept} / "
          f"{collector.steps_dropped}")
    print(f"  retained diagnostic only : {collector.kept_short} short, "
          f"{collector.kept_no_progress} under 0.5 m net progress")
    if yield_pct < 60:
        print("  NOTE: a low yield both wastes compute and biases the set -- the "
              "filter\n        correlates with difficulty, so survivors skew easy.")
    _finish(simulation_app)


def _finish(simulation_app) -> None:
    """Shut down and exit, without letting a wedged Kit shutdown hold the run.

    ``simulation_app.close()`` can hang indefinitely -- observed here keeping the
    process alive long after the shard was written and every print completed, so
    a finished run never returns and anything downstream never starts.

    ``os._exit`` is normally the wrong tool. It is right here: the shards are
    already on disk and stdout is flushed just below, so the only thing
    interpreter cleanup can still do is fail to return.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    # This is a standalone worker process and all durable state is already on
    # disk.  Kit's close() has repeatedly remained in native cleanup for many
    # minutes while retaining CPU, RAM and GPU memory; calling it before
    # os._exit() therefore defeated the bounded-shutdown guarantee entirely.
    # The OS releases the process-owned resources more reliably here.
    os._exit(0)


class AsyncShardWriter:
    """One compression worker with bounded, failure-propagating backpressure.

    NPZ deflate is CPU work and used to stop the simulation for roughly a
    second every shard. A single worker overlaps that work with GPU simulation;
    the bounded future queue caps transient RAM. If storage falls behind, only
    ``submit`` blocks, and that wait is reported separately from writer work.
    """

    def __init__(self, max_pending: int = 2):
        if max_pending < 1:
            raise ValueError("writer_queue must be positive")
        self.max_pending = max_pending
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="shard-writer")
        self._futures = deque()
        self.wait_seconds = 0.0
        self.worker_seconds = 0.0

    @staticmethod
    def _write(path, episodes, angles, metadata):
        from sim2edge.dataset import save_shard

        started = time.perf_counter()
        cat = lambda key: np.concatenate([episode[key] for episode in episodes])
        save_shard(
            path, ranges=cat("ranges"), angles=angles,
            pose=cat("pose"), goal=cat("goal"), action=cat("action"),
            prev_action=cat("prev_action"), episode=cat("episode"),
            layout=cat("layout"), start_id=cat("start_id"),
            path_length=cat("path_length"), outcome=cat("outcome"),
            **metadata,
        )
        return path, sum(len(episode["episode"]) for episode in episodes), \
            time.perf_counter() - started

    def _reap(self, *, block: bool) -> None:
        while self._futures and (block or self._futures[0].done()):
            future = self._futures.popleft()
            started = time.perf_counter()
            path, steps, elapsed = future.result()  # re-raises worker failures
            self.wait_seconds += time.perf_counter() - started
            self.worker_seconds += elapsed
            print(f"  wrote {path} ({steps} steps in {elapsed:.1f} s, async)",
                  flush=True)
            block = False

    def submit(self, path, episodes, angles, metadata) -> None:
        self._reap(block=False)
        if len(self._futures) >= self.max_pending:
            self._reap(block=True)
        self._futures.append(self._pool.submit(
            self._write, path, episodes, angles, metadata))

    def close(self) -> None:
        try:
            while self._futures:
                self._reap(block=True)
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)


class Collector:
    """Fixed-size per-environment buffers, flushed to shards on episode end.

    Memory is a **ceiling, not a pile.** The previous version appended
    ``ranges[i]`` -- a *view* -- into per-env Python lists. Two compounding
    problems: one 1.4 KB row kept the whole 2.9 MB batch frame alive, so nothing
    was ever released while any environment was mid-episode; and it created
    ``num_envs x steps x 5`` tiny tensor objects, several million of them, whose
    growing count made every GC pass slower. That is what produced both the
    climbing RSS and the climbing ms/step.

    Here each field is one preallocated array of shape ``(num_envs, max_steps,
    ...)``, written by a single vectorized scatter per step. No per-env Python
    loop, no views, no per-step allocation, and a hard memory bound known up
    front.
    """

    #: Shortest episode worth keeping. Below this there is not enough context
    #: for the last-action input to mean anything.
    MIN_STEPS = 20

    #: An episode must have closed at least this much distance to be evidence of
    #: goal-seeking rather than of a robot milling around.
    MIN_PROGRESS_M = 0.5

    def __init__(self, out: Path, shard_steps: int, num_envs: int,
                 max_steps: int, n_beams: int, device, *, layout_seed: int = 0,
                 num_layouts: int = 0, starts_per_layout: int = 1,
                 sigma: float = 0.0, validation_fraction: float = 0.0,
                 writer_queue: int = 2, driver: str = "oracle"):
        self.out, self.shard_steps, self.device = out, shard_steps, device
        self.num_envs, self.max_steps = num_envs, max_steps
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if validation_fraction and num_layouts < 2:
            raise ValueError("layout validation split needs at least two layouts")
        self.layout_seed = layout_seed
        self.num_layouts = num_layouts
        self.starts_per_layout = starts_per_layout
        self.sigma = sigma
        if driver not in ("oracle", "policy"):
            raise ValueError("driver must be 'oracle' or 'policy'")
        self.driver = driver
        self.validation_fraction = validation_fraction

        self.b_ranges = np.zeros((num_envs, max_steps, n_beams), np.float16)
        self.b_pose = np.zeros((num_envs, max_steps, 3), np.float32)
        self.b_goal = np.zeros((num_envs, max_steps, 2), np.float32)
        self.b_action = np.zeros((num_envs, max_steps, 2), np.float32)
        self.b_prev = np.zeros((num_envs, max_steps, 2), np.float32)
        self.b_layout = np.zeros((num_envs, max_steps), np.int32)
        self.b_start_id = np.zeros((num_envs, max_steps), np.uint16)
        self.b_path_length = np.zeros((num_envs, max_steps), np.float32)
        nbytes = sum(a.nbytes for a in (self.b_ranges, self.b_pose, self.b_goal,
                                        self.b_action, self.b_prev, self.b_layout,
                                        self.b_start_id, self.b_path_length))
        print(f"[collect] recorder buffers: {nbytes / 1e9:.2f} GB "
              f"({num_envs} envs x {max_steps} steps x {n_beams} beams), fixed",
              flush=True)

        self.t = np.zeros(num_envs, np.int32)          # steps buffered per env
        self.env_idx = np.arange(num_envs)

        self.episode_id = 0
        self.total_steps = 0
        self.kept_complete = self.kept_truncated = self.kept_partial = 0
        self.dropped_short = self.dropped_no_progress = 0
        self.steps_kept = self.steps_dropped = 0
        self.kept_short = self.kept_no_progress = 0
        self.filter_failures = False
        self.writer = AsyncShardWriter(writer_queue)

        if validation_fraction:
            count = min(num_layouts - 1,
                        max(1, round(num_layouts * validation_fraction)))
            rng = np.random.default_rng(layout_seed ^ 0xA11CE)
            self.validation_layout_ids = np.sort(
                rng.choice(num_layouts, size=count, replace=False).astype(np.int32))
            self._is_validation = np.zeros(num_layouts, dtype=np.bool_)
            self._is_validation[self.validation_layout_ids] = True
            self._stream_names = ("train", "validation")
            print(f"[collect] layout split: {num_layouts - count} train / "
                  f"{count} validation layouts", flush=True)
        else:
            self.validation_layout_ids = np.empty(0, np.int32)
            self._is_validation = np.zeros(max(1, num_layouts), dtype=np.bool_)
            self._stream_names = ("",)
        self._pending = {name: [] for name in self._stream_names}
        self._pending_steps = {name: 0 for name in self._stream_names}

        # Resume rather than clobber: a fresh run used to restart numbering at
        # bc_000 and overwrite an earlier run's shards in place.
        out.mkdir(parents=True, exist_ok=True)
        self._next_shard = {}
        for name in self._stream_names:
            directory = out / name if name else out
            directory.mkdir(parents=True, exist_ok=True)
            existing = sorted(directory.glob("bc_*.npz"))
            next_index = (1 + max(int(f.stem.split("_")[1]) for f in existing)
                          if existing else 0)
            self._next_shard[name] = next_index
            if existing:
                print(f"[collect] {len(existing)} shard(s) already in "
                      f"{directory}; appending from bc_{next_index:03d}",
                      flush=True)

    # ── per-step, O(1) Python ───────────────────────────────────────────────

    @property
    def pending(self):
        """Legacy single-stream view retained for recorder unit tests."""
        return self._pending[self._stream_names[0]]

    @property
    def pending_steps(self) -> int:
        return sum(self._pending_steps.values())

    @property
    def shards_written(self) -> int:
        return sum(self._next_shard.values())

    @property
    def write_wait_seconds(self) -> float:
        return self.writer.wait_seconds

    def record(self, ranges, pose, goal, label, prev_action, layout=None,
               start_id=None, path_length=None):
        """Two host transfers total, then one vectorized scatter."""
        t = self.t
        room = t < self.max_steps                    # ignore overflow, if any
        i, j = self.env_idx[room], t[room]

        torch = __import__("torch")
        if layout is None:
            layout = torch.zeros(self.num_envs, device=pose.device)
        if start_id is None:
            start_id = torch.zeros(self.num_envs, device=pose.device)
        if path_length is None:
            path_length = torch.zeros(self.num_envs, device=pose.device)

        # .half() is a Tensor method, so this needs no module-level torch
        # (which main() imports only after AppLauncher). Casting on device
        # also halves the host transfer.
        self.b_ranges[i, j] = ranges.half().cpu().numpy()[room]
        # Pack the four small tensors into one transfer.  Calling .cpu() on
        # each field separately introduced four CUDA synchronizations per
        # control tick; the scan transfer is the only other synchronization.
        features = np.asarray(
            torch.cat((pose, goal, label, prev_action,
                       layout[:, None].to(pose.dtype),
                       start_id[:, None].to(pose.dtype),
                       path_length[:, None].to(pose.dtype)), dim=1).cpu())
        self.b_pose[i, j] = features[room, :3]
        self.b_goal[i, j] = features[room, 3:5]
        self.b_action[i, j] = features[room, 5:7]
        self.b_prev[i, j] = features[room, 7:9]
        self.b_layout[i, j] = features[room, 9].astype(np.int32)
        self.b_start_id[i, j] = features[room, 10].astype(np.uint16)
        self.b_path_length[i, j] = features[room, 11]
        self.t[room] += 1

    # ── episode end ─────────────────────────────────────────────────────────

    def flush(self, done, reached, collided, angles, *, filter_failures: bool):
        self.filter_failures = filter_failures
        reached = reached.cpu().numpy()
        collided = collided.cpu().numpy()
        for i in np.flatnonzero(done.cpu().numpy()):
            upto = int(self.t[i])
            d = np.linalg.norm(
                self.b_goal[i, :upto] - self.b_pose[i, :upto, :2], axis=1)
            progress = float(d[0] - d[-1]) if upto else 0.0
            too_short = upto < self.MIN_STEPS
            no_progress = progress < self.MIN_PROGRESS_M

            if filter_failures and too_short:
                self.dropped_short += 1
                self.steps_dropped += upto
            elif filter_failures and no_progress:
                self.dropped_no_progress += 1
                self.steps_dropped += upto
            elif upto:
                outcome = 1 if collided[i] else (0 if reached[i] else 2)
                stream = self._emit(i, upto, outcome)
                self.steps_kept += upto
                self.kept_short += int(too_short)
                self.kept_no_progress += int(no_progress)
                if collided[i]:
                    self.kept_truncated += 1
                elif reached[i]:
                    self.kept_complete += 1
                else:
                    # Timed out, but every label is the oracle's action at the
                    # state it was in, so the supervision is still correct.
                    self.kept_partial += 1

                # Check after each episode, not after an entire synchronized
                # reset wave.  At 8192 envs the old placement overshot a 200k
                # target by millions of samples and caused a large copy spike.
                if self._pending_steps[stream] >= self.shard_steps:
                    self._queue_write(stream, angles)

            self.t[i] = 0

    def _emit(self, i: int, upto: int, outcome: int) -> str:
        layout_id = int(self.b_layout[i, 0])
        stream = ("validation" if self.validation_fraction
                  and self._is_validation[layout_id] else
                  ("train" if self.validation_fraction else ""))
        self._pending[stream].append({
            "ranges": self.b_ranges[i, :upto].copy(),
            "pose": self.b_pose[i, :upto].copy(),
            "goal": self.b_goal[i, :upto].copy(),
            "action": self.b_action[i, :upto].copy(),
            "prev_action": self.b_prev[i, :upto].copy(),
            "layout": self.b_layout[i, :upto].copy(),
            "start_id": self.b_start_id[i, :upto].copy(),
            "path_length": self.b_path_length[i, :upto].copy(),
            "episode": np.full(upto, self.episode_id, np.int32),
            # 0 reached, 1 collision, 2 timeout.  Keeping this per sample is a
            # small cost and makes every independently loaded shard self-
            # describing even when an episode straddles no external index.
            "outcome": np.full(upto, outcome, np.uint8),
        })
        self._pending_steps[stream] += upto
        self.episode_id += 1
        return stream

    @property
    def episodes_kept(self) -> int:
        return self.kept_complete + self.kept_truncated + self.kept_partial

    @property
    def episodes_dropped(self) -> int:
        return self.dropped_short + self.dropped_no_progress

    def _queue_write(self, stream: str, angles) -> None:
        episodes = self._pending[stream]
        steps = self._pending_steps[stream]
        if not episodes:
            return
        directory = self.out / stream if stream else self.out
        path = directory / f"bc_{self._next_shard[stream]:03d}.npz"
        metadata = {
            "filter_failures": np.array(self.filter_failures, dtype=np.bool_),
            "layout_seed": np.array(self.layout_seed, dtype=np.int64),
            "num_layouts": np.array(self.num_layouts, dtype=np.int32),
            "starts_per_layout": np.array(self.starts_per_layout, dtype=np.int32),
            "sigma": np.array(self.sigma, dtype=np.float32),
            "driver": np.array(self.driver),
            "validation_fraction": np.array(
                self.validation_fraction, dtype=np.float32),
            "validation_layout_ids": self.validation_layout_ids,
            "split": np.array(stream or "unsplit"),
        }
        angle_array = angles.detach().cpu().numpy().copy()
        self.writer.submit(path, episodes, angle_array, metadata)
        self.total_steps += steps
        self._next_shard[stream] += 1
        self._pending[stream] = []
        self._pending_steps[stream] = 0

    def close(self, angles):
        for stream in self._stream_names:
            self._queue_write(stream, angles)
        self.writer.close()


def frame_env0(env) -> None:
    """Point the viewport at environment 0's *actual* arena.

    The cfg default is a guess fixed before any layout exists, and arenas range
    4-10 m -- so one eye position either clips a large room or renders a small
    one as a speck. eye/lookat are offsets from the env origin.

    Safe to do once: the viewer's tracking callback only re-applies for
    origin_type "asset_root"/"asset_body", so this does not fight manual
    navigation afterwards (right-drag to orbit, middle-drag to pan, F to frame
    a selected prim).
    """
    ctrl = getattr(env, "viewport_camera_controller", None)
    if ctrl is None:
        return
    room = env._rooms[int(env._layout[0])]
    w, h = float(room.size_xy[0]), float(room.size_xy[1])
    span = max(w, h)
    ctrl.update_view_location(eye=(w / 2, h / 2 - 1.2 * span, 0.9 * span),
                              lookat=(w / 2, h / 2, 0.0))
    origin = env.scene.env_origins[0].tolist()
    print(f"[collect] framed env 0: {w:.1f} x {h:.1f} m arena at "
          f"({origin[0]:.1f}, {origin[1]:.1f})", flush=True)


def dump_scene(env, torch) -> None:
    """Where everything actually is, in world coordinates."""
    print("\n--- scene dump ---")
    print(f"env_spacing        {env.cfg.scene.env_spacing}")
    print(f"env_origins        {env.scene.env_origins.tolist()}")

    room = env._rooms[int(env._layout[0])]
    print(f"env 0 layout       {int(env._layout[0])}  arena "
          f"{room.size_xy[0]:.2f} x {room.size_xy[1]:.2f} m  "
          f"{int(room.valid.sum())} prims placed")

    pose = env._robot_pose()
    print(f"env 0 robot (local) {pose[0, :2].tolist()}  yaw {pose[0, 2]:.2f}")
    print(f"env 0 goal  (local) {(env._goal_w[0] - env.scene.env_origins[0, :2]).tolist()}")

    obs_w = env.obstacles.data.object_pos_w[0]
    print(f"env 0 obstacle world z range  [{obs_w[:, 2].min():.2f}, "
          f"{obs_w[:, 2].max():.2f}]   (parked prims sit at {PARKED_Z_HINT})")
    for slot in (0, 1, 24, 25, 26, 27):
        print(f"  slot {slot:2d}  world {[round(v, 2) for v in obs_w[slot].tolist()]}")

    ctrl = getattr(env, "viewport_camera_controller", None)
    if ctrl is not None:
        print(f"camera eye offset  {ctrl.default_cam_eye.tolist()}")
        print(f"camera lookat off  {ctrl.default_cam_lookat.tolist()}")
        print(f"camera origin      {ctrl.viewer_origin.tolist()}")
    else:
        print("camera             (headless, no viewport controller)")
    print("--- end dump ---\n")


PARKED_Z_HINT = -10.0


def run_checks(env, torch) -> None:
    """Bring-up assertions. Cheap, and each one catches a specific wiring bug."""
    from sim2edge.spec import INVALID_FILL, LIDAR_BINS

    print("\n--- bring-up checks ---")

    ranges, angles = env.raw_scan()
    print(f"scan shape          {tuple(ranges.shape)}  angles {tuple(angles.shape)}")

    finite = torch.isfinite(ranges) & (ranges > 0)
    frac = finite.float().mean().item()
    print(f"finite returns      {frac:.1%}")
    assert frac > 0.05, (
        "almost no rays hit anything -- MultiMeshRayCaster targets are probably "
        "not matching /World/envs/env_*/Obstacle_*, or every obstacle is parked"
    )

    from sim2edge.preprocess import bin_lidar
    binned = bin_lidar(ranges, env._beam_angles)
    occupied = (binned < INVALID_FILL).float().mean().item()
    print(f"bins with a return  {occupied:.1%}")
    assert occupied > 0.05, "every bin reads INVALID_FILL"

    # The arena is walled, so from anywhere inside it the nearest return in any
    # direction is finite. If the forward half is systematically emptier than
    # the rest, the mount yaw is being applied the wrong number of times.
    fwd = binned[:, LIDAR_BINS // 2 - 15:LIDAR_BINS // 2 + 15].min(dim=1).values
    print(f"nearest ahead       {fwd.mean().item():.3f} (normalized)")

    pose = env._robot_pose()
    in_free = env.pool.is_free(env._layout, pose[:, :2])
    print(f"spawned in free     {in_free.float().mean().item():.1%}")
    assert in_free.all(), (
        "a robot spawned in collision -- layouts are landing in the wrong "
        "environments, i.e. the prim-view permutation is wrong"
    )

    # Cloning replicates env_0. If the clones turned out to share transforms
    # rather than each owning one, every environment would silently show the
    # same layout -- and the policy would train on one room.
    if env.num_envs > 1:
        pos = env.obstacles.data.object_pos_w                 # (envs, slots, 3)
        local = pos[:, :, :2] - env.scene.env_origins[:, None, :2]
        spread = (local - local[0]).abs().max().item()
        print(f"layout spread       {spread:.3f} m across envs")
        assert spread > 1e-3, (
            "every environment has identical obstacle positions -- the clones "
            "share transforms, so per-env layouts are not taking effect"
        )

    cmd = env.oracle_command()
    print(f"oracle v range      [{cmd[:, 0].min():.3f}, {cmd[:, 0].max():.3f}]")
    print(f"oracle omega range  [{cmd[:, 1].min():.3f}, {cmd[:, 1].max():.3f}]")
    spinning = ((cmd[:, 0] == 0) & (cmd[:, 1].abs() > 1.8)).float().mean().item()
    print(f"spinning in place   {spinning:.1%}")
    assert spinning < 0.5, "most robots are pinned at max spin; oracle is stranded"

    run_drive_checks(env, torch)

    print("--- all checks passed ---\n")


def run_drive_checks(env, torch) -> None:
    """Prove that fixed physical commands produce the requested base motion."""
    # A valid oracle is irrelevant unless the base tracks the commands used as
    # labels. Measure pose deltas after the drive has settled, rather than only
    # checking that the oracle emits varied commands. The old free-suspension
    # failure passed every check above while achieving almost no commanded yaw.
    spin = _measure_drive_response(env, torch, (0.0, 1.0))
    env.reset()
    straight = _measure_drive_response(env, torch, (0.3, 0.0))

    _assert_tracks("angular velocity", spin["yaw_rate"], 1.0)
    assert spin["planar_speed"] < 0.05, (
        f"spin command translated at {spin['planar_speed']:.3f} m/s; check "
        "suspension and wheel contact")
    _assert_tracks("forward velocity", straight["forward_speed"], 0.3)
    assert abs(straight["lateral_speed"]) < 0.05, (
        f"straight command slipped laterally at "
        f"{straight['lateral_speed']:.3f} m/s; check suspension and wheel contact")
    assert abs(straight["yaw_rate"]) < 0.10, (
        f"straight command yawed at {straight['yaw_rate']:.3f} rad/s; check "
        "wheel signs and left/right drive gains")


def _measure_drive_response(env, torch, command: tuple[float, float],
                            *, settle_steps: int = 10,
                            command_steps: int = 50,
                            measure_steps: int = 20) -> dict[str, float]:
    """Return median achieved base velocities for one fixed physical command.

    The command runs for 50 control ticks. Only the final 20 are measured so
    motor/contact transients cannot make a healthy drive fail. Environments that
    terminate during the probe are excluded: their post-step pose has already
    been reset by ``DirectRLEnv`` and is not a drive-response measurement.
    """
    from isaaclab_env.nav_task import normalize_action
    from sim2edge.spec import DT

    assert 0 < measure_steps <= command_steps
    cmd = torch.tensor(command, device=env.device).expand(env.num_envs, -1)
    zero = torch.zeros_like(cmd)
    for _ in range(settle_steps):
        env.step(normalize_action(zero))

    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    forward, lateral, planar, yaw_rate = [], [], [], []
    for tick in range(command_steps):
        before = env._robot_pose().clone()
        _, _, terminated, truncated, _ = env.step(normalize_action(cmd))
        after = env._robot_pose().clone()
        valid &= ~(terminated | truncated)
        if tick >= command_steps - measure_steps:
            delta = after[:, :2] - before[:, :2]
            heading = before[:, 2]
            forward.append(
                (delta[:, 0] * torch.cos(heading)
                 + delta[:, 1] * torch.sin(heading)) / DT)
            lateral.append(
                (-delta[:, 0] * torch.sin(heading)
                 + delta[:, 1] * torch.cos(heading)) / DT)
            planar.append(torch.linalg.vector_norm(delta, dim=1) / DT)
            dyaw = torch.atan2(torch.sin(after[:, 2] - before[:, 2]),
                               torch.cos(after[:, 2] - before[:, 2]))
            yaw_rate.append(dyaw / DT)

    assert valid.any(), (
        f"every environment terminated during drive probe {command}; use more "
        "bring-up environments or inspect spawn clearance")
    forward_speed = torch.stack(forward, dim=1)[valid].median().item()
    lateral_speed = torch.stack(lateral, dim=1)[valid].median().item()
    planar_speed = torch.stack(planar, dim=1)[valid].median().item()
    achieved_yaw = torch.stack(yaw_rate, dim=1)[valid].median().item()
    print(f"drive response      cmd v={command[0]:.2f} w={command[1]:.2f}  "
          f"achieved v={forward_speed:.3f} lateral={lateral_speed:.3f} "
          f"w={achieved_yaw:.3f}  "
          f"({int(valid.sum())}/{env.num_envs} envs valid)")
    if hasattr(env, "robot"):
        wheel = env._wheel_ids
        suspension = env._suspension_ids
        wheel_target = env.robot.data.joint_vel_target[:, wheel].median(dim=0).values
        wheel_velocity = env.robot.data.joint_vel[:, wheel].median(dim=0).values
        suspension_target = env.robot.data.joint_pos_target[:, suspension].median(dim=0).values
        suspension_position = env.robot.data.joint_pos[:, suspension].median(dim=0).values
        print(f"                    wheel target/actual L "
              f"{wheel_target[0].item():.2f}/{wheel_velocity[0].item():.2f} "
              f"R {wheel_target[1].item():.2f}/{wheel_velocity[1].item():.2f} rad/s; "
              f"suspension target/actual L "
              f"{suspension_target[0].item():.3f}/{suspension_position[0].item():.3f} "
              f"R {suspension_target[1].item():.3f}/{suspension_position[1].item():.3f} m")
    return {"forward_speed": forward_speed, "lateral_speed": lateral_speed,
            "planar_speed": planar_speed, "yaw_rate": achieved_yaw}


def _assert_tracks(name: str, achieved: float, commanded: float,
                   *, tolerance: float = 0.20) -> None:
    """Assert signed steady-state response is within a relative tolerance."""
    error = abs(achieved - commanded)
    allowed = tolerance * abs(commanded)
    assert error <= allowed, (
        f"{name} response {achieved:.3f} does not track command "
        f"{commanded:.3f} within {tolerance:.0%}; check joint signs, "
        "suspension, drive gains and wheel contact")


if __name__ == "__main__":
    main()
