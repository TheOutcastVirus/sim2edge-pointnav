"""Collect behaviour-cloning demonstrations in Isaac Lab.

    ~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 256 --episodes 2000

Bring it up in the order in ``docs/RUNBOOK.md`` -- each step isolates a
different failure. ``--num_envs 4 --gui --check`` is the first one.

What gets written
-----------------
Raw measurements only: ranges as the sensor reported them, beam bearings in
base_link, true poses, and the oracle's **commanded** ``(v, omega)``. Noise and
binning happen in the loader, so retuning either does not invalidate a run that
took hours.

Two label conventions that look like details and are not:

**The label is the command, never the achieved velocity.** Reading back what
physics produced would teach the policy to emit a value the actuator lags a
second time -- a systematic underactuation that is invisible in sim.

**The label is the *unperturbed* oracle command; only execution is perturbed.**
That is what puts the robot in a tube of states around the expert path while
still labelling each one with what the expert would do there. Without it every
state in the dataset is one the oracle put itself in, and the policy has no
recovery behaviour at all.
"""

from __future__ import annotations

import argparse
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
                    help="successful episodes to collect, not episodes attempted")
    ap.add_argument("--num_layouts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--shard_steps", type=int, default=200_000,
                    help="steps per shard; ~610 B each, so 200k is ~120 MB")
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="execution noise as a fraction of each axis limit. "
                         "0 collects pure on-policy expert data, which is the "
                         "distribution BC fails on -- see the module docstring")
    ap.add_argument("--keep_failures", action="store_true",
                    help="keep collided and timed-out episodes")
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
    from sim2edge.spec import LIDAR_BINS

    cfg = NavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.num_layouts = args.num_layouts
    cfg.layout_seed = args.seed

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

    collector = Collector(args.out, args.shard_steps, args.num_envs,
                          max_steps=int(env.max_episode_length),
                          n_beams=int(env._beam_angles.shape[0]),
                          device=device)
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

        label = env.oracle_command()                 # unperturbed: this is the label
        executed = perturb_action(label, args.sigma, generator=gen)

        _t = time.perf_counter()
        collector.record(ranges, pose, env._goal_w - env.scene.env_origins[:, :2],
                         label, env._action)
        t_rec += time.perf_counter() - _t

        # Collision verdict BEFORE stepping. DirectRLEnv.step() resets finished
        # environments internally, which reassigns env._layout -- pairing the
        # new layout with this pose would classify against the wrong room and
        # silently drop good episodes.
        #
        # An episode is worth keeping only if it ended at the goal without the
        # body ever overlapping an obstacle: imitating a trajectory that ends in
        # a wall teaches ending in walls.
        collector.mark_collisions(~env.pool.is_free(env._layout, pose[:, :2]))

        _, _, terminated, truncated, _ = env.step(normalize_action(executed))

        done = terminated | truncated
        if done.any():
            collector.flush(done, terminated, angles,
                            keep_failures=args.keep_failures)

        step += 1
        if step % 50 == 0:
            wall = time.perf_counter() - t0
            t_sim = wall - t_rec
            print(f"step {step:6d}  kept {collector.episodes_kept:5d} "
                  f"(full {collector.kept_complete:5d} "
                  f"trunc {collector.kept_truncated:5d} "
                  f"part {collector.kept_partial:5d})  "
                  f"dropped {collector.episodes_dropped:5d}  "
                  f"| {wall / step * 1000:6.1f} ms/step "
                  f"(sim {t_sim / step * 1000:6.1f}  rec {t_rec / step * 1000:5.1f})  "
                  f"| {args.num_envs * step / wall:7.0f} env-steps/s")

    collector.close(angles)
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
    try:
        simulation_app.close()
    except Exception as exc:                     # pragma: no cover
        print(f"[collect] ignoring error during shutdown: {exc}")
        sys.stdout.flush()
    os._exit(0)


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
                 max_steps: int, n_beams: int, device):
        self.out, self.shard_steps, self.device = out, shard_steps, device
        self.num_envs, self.max_steps = num_envs, max_steps

        self.b_ranges = np.zeros((num_envs, max_steps, n_beams), np.float16)
        self.b_pose = np.zeros((num_envs, max_steps, 3), np.float32)
        self.b_goal = np.zeros((num_envs, max_steps, 2), np.float32)
        self.b_action = np.zeros((num_envs, max_steps, 2), np.float32)
        self.b_prev = np.zeros((num_envs, max_steps, 2), np.float32)
        nbytes = sum(a.nbytes for a in (self.b_ranges, self.b_pose, self.b_goal,
                                        self.b_action, self.b_prev))
        print(f"[collect] recorder buffers: {nbytes / 1e9:.2f} GB "
              f"({num_envs} envs x {max_steps} steps x {n_beams} beams), fixed",
              flush=True)

        self.t = np.zeros(num_envs, np.int32)          # steps buffered per env
        self.collide_at = np.full(num_envs, -1, np.int32)   # first collision step
        self.env_idx = np.arange(num_envs)

        self.episode_id = 0
        self.total_steps = 0
        self.kept_complete = self.kept_truncated = self.kept_partial = 0
        self.dropped_short = self.dropped_no_progress = 0
        self.pending: list[dict] = []
        self.pending_steps = 0

        # Resume rather than clobber: a fresh run used to restart numbering at
        # bc_000 and overwrite an earlier run's shards in place.
        out.mkdir(parents=True, exist_ok=True)
        existing = sorted(out.glob("bc_*.npz"))
        self.shards_written = (
            1 + max(int(f.stem.split("_")[1]) for f in existing) if existing else 0)
        if existing:
            print(f"[collect] {len(existing)} shard(s) already in {out}; "
                  f"appending from bc_{self.shards_written:03d}", flush=True)

    # ── per-step, O(1) Python ───────────────────────────────────────────────

    def record(self, ranges, pose, goal, label, prev_action):
        """One host transfer per field, then one vectorized scatter."""
        t = self.t
        room = t < self.max_steps                    # ignore overflow, if any
        i, j = self.env_idx[room], t[room]

        # .half() is a Tensor method, so this needs no module-level torch
        # (which main() imports only after AppLauncher). Casting on device
        # also halves the host transfer.
        self.b_ranges[i, j] = ranges.half().cpu().numpy()[room]
        self.b_pose[i, j] = pose.cpu().numpy()[room]
        self.b_goal[i, j] = goal.cpu().numpy()[room]
        self.b_action[i, j] = label.cpu().numpy()[room]
        self.b_prev[i, j] = prev_action.cpu().numpy()[room]
        self.t[room] += 1

    def mark_collisions(self, hit):
        """Remember *when* an environment first collided, not just whether.

        The step index is what makes truncation possible: everything before a
        collision is clean expert data, and discarding it threw away most of the
        dataset -- including precisely the off-path recovery states that action
        perturbation exists to create.
        """
        h = hit.cpu().numpy()
        first = h & (self.collide_at < 0)
        self.collide_at[first] = np.maximum(self.t[first] - 1, 0)

    # ── episode end ─────────────────────────────────────────────────────────

    def flush(self, done, terminated, angles, *, keep_failures: bool):
        reached = terminated.cpu().numpy()
        for i in np.flatnonzero(done.cpu().numpy()):
            n, c = int(self.t[i]), int(self.collide_at[i])
            # Truncate before the collision instead of discarding the episode.
            upto = c if c >= 0 else n

            if upto >= self.MIN_STEPS:
                d = np.linalg.norm(self.b_goal[i, :upto] - self.b_pose[i, :upto, :2],
                                   axis=1)
                progress = float(d[0] - d[-1])
                if progress >= self.MIN_PROGRESS_M or keep_failures:
                    self._emit(i, upto)
                    if c >= 0:
                        self.kept_truncated += 1
                    elif reached[i]:
                        self.kept_complete += 1
                    else:
                        # Timed out, but every label is the oracle's action at
                        # the state it was in, so the supervision is still
                        # correct -- only the outcome is missing.
                        self.kept_partial += 1
                else:
                    self.dropped_no_progress += 1
            else:
                self.dropped_short += 1

            self.t[i] = 0
            self.collide_at[i] = -1

        if self.pending_steps >= self.shard_steps:
            self._write(angles)

    def _emit(self, i: int, upto: int) -> None:
        self.pending.append({
            "ranges": self.b_ranges[i, :upto].copy(),
            "pose": self.b_pose[i, :upto].copy(),
            "goal": self.b_goal[i, :upto].copy(),
            "action": self.b_action[i, :upto].copy(),
            "prev_action": self.b_prev[i, :upto].copy(),
            "episode": np.full(upto, self.episode_id, np.int32),
        })
        self.pending_steps += upto
        self.episode_id += 1

    @property
    def episodes_kept(self) -> int:
        return self.kept_complete + self.kept_truncated + self.kept_partial

    @property
    def episodes_dropped(self) -> int:
        return self.dropped_short + self.dropped_no_progress

    def _write(self, angles):
        from sim2edge.dataset import save_shard

        path = self.out / f"bc_{self.shards_written:03d}.npz"
        cat = lambda k: np.concatenate([e[k] for e in self.pending])
        save_shard(path, ranges=cat("ranges"), angles=angles.cpu().numpy(),
                   pose=cat("pose"), goal=cat("goal"), action=cat("action"),
                   prev_action=cat("prev_action"), episode=cat("episode"))
        self.total_steps += self.pending_steps
        self.shards_written += 1
        print(f"  wrote {path} ({self.pending_steps} steps)", flush=True)
        self.pending, self.pending_steps = [], 0

    def close(self, angles):
        if self.pending:
            self._write(angles)


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

    print("--- all checks passed ---\n")


if __name__ == "__main__":
    main()
