"""Check that a collected shard actually contains usable demonstrations.

    .venv/bin/python scripts/verify_shard.py data/bc_000.npz

No Isaac needed -- that is the point. Collection is the expensive, hard-to-debug
half; verification should be runnable anywhere, immediately, before anyone spends
GPU hours on a dataset that is subtly wrong.

The checks are ordered from cheap and structural to the ones that actually prove
the demonstrations are demonstrations *of the task*. The important one is
PROGRESS: if goal distance does not fall over an episode, the labels are not
goal-reaching behaviour no matter how well-formed the arrays are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from sim2edge.dataset import load_shard
from sim2edge.spec import (DT, INVALID_FILL, LIDAR_BINS, LIDAR_MAX_M,
                           LIDAR_MIN_M, OBS_DIM, OMEGA_MAX, V_MAX, fingerprint)

PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "", *, warn_only: bool = False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"  [{tag}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok and not warn_only:
        _failures.append(label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard", type=Path)
    ap.add_argument("--goal-tolerance", type=float, default=0.25)
    args = ap.parse_args()

    print(f"\n=== {args.shard} ===")
    shard = load_shard(args.shard)          # raises on fingerprint mismatch
    n = len(shard)
    ep = shard.episode.numpy()
    ids = np.unique(ep)

    print(f"\nspec fingerprint {fingerprint()}   steps {n}   episodes {len(ids)}")

    # ── structure ───────────────────────────────────────────────────────────
    print("\nstructure")
    check(n > 0, "shard is not empty")
    check(shard.ranges.ndim == 2, "ranges are (T, beams)",
          f"{tuple(shard.ranges.shape)}")
    check(shard.angles.shape[0] == shard.ranges.shape[1],
          "one bearing per beam")
    check(shard.pose.shape == (n, 3) and shard.goal.shape == (n, 2),
          "pose and goal shapes")
    check(shard.action.shape == (n, 2) and shard.prev_action.shape == (n, 2),
          "action shapes")

    # ── the shard holds RAW data, not observations ──────────────────────────
    print("\nraw storage (so a spec change does not invalidate the run)")
    finite = torch.isfinite(shard.ranges.float())
    r = shard.ranges.float()[finite]
    check(r.numel() > 0, "some finite ranges exist")
    check(float(r.max()) > 1.5, "ranges are metres, not normalized",
          f"max {float(r.max()):.2f} m")
    check(shard.ranges.shape[1] != LIDAR_BINS,
          "ranges are not pre-binned", f"{shard.ranges.shape[1]} beams "
          f"vs {LIDAR_BINS} bins", warn_only=True)
    drop = 1.0 - finite.float().mean().item()
    check(drop < 0.5, "not mostly dropped beams", f"{drop:.1%} non-finite")

    # ── labels are physical commands within the robot's envelope ────────────
    print("\nlabels")
    v, w = shard.action[:, 0], shard.action[:, 1]
    check(bool((v >= -1e-6).all()), "v is forward-only", f"min {float(v.min()):.3f}")
    check(float(v.max()) <= V_MAX + 1e-4, "v within V_MAX",
          f"max {float(v.max()):.3f} / {V_MAX}")
    check(float(w.abs().max()) <= OMEGA_MAX + 1e-4, "omega within OMEGA_MAX",
          f"max |w| {float(w.abs().max()):.3f} / {OMEGA_MAX}")
    check(float(v.max()) > 0.05, "the expert actually drove",
          f"mean v {float(v.mean()):.3f} m/s")

    # ── DART: execution was perturbed, labels were not ─────────────────────
    print("\nperturbation (recovery states)")
    delta = (shard.prev_action[1:] - shard.action[:-1]).abs()
    same_ep = torch.from_numpy(ep[1:] == ep[:-1])
    delta = delta[same_ep]
    spread = float(delta[:, 0].mean()) if delta.numel() else 0.0
    check(spread > 1e-4, "executed action differs from the label",
          f"mean |v_exec - v_label| = {spread:.4f} m/s",
          warn_only=True)
    if spread <= 1e-4:
        print("         --sigma was 0: every state is one the oracle put itself")
        print("         in, which is the distribution BC is known to fail on.")

    # ── PROGRESS: the labels are goal-reaching behaviour ───────────────────
    print("\nprogress toward the goal (the check that matters)")
    dist = torch.linalg.vector_norm(shard.goal - shard.pose[:, :2], dim=1).numpy()
    starts, ends, lengths, improved = [], [], [], 0
    for e in ids:
        m = ep == e
        d = dist[m]
        starts.append(d[0]); ends.append(d[-1]); lengths.append(int(m.sum()))
        improved += int(d[-1] < d[0])

    starts, ends, lengths = np.array(starts), np.array(ends), np.array(lengths)
    check(improved == len(ids), "every episode ends closer than it started",
          f"{improved}/{len(ids)}")
    # The last *recorded* sample is one control step short of the terminal
    # state. Collection stores (state, action) pairs and evaluates success after
    # stepping, so the pose that actually satisfies the goal condition is never
    # recorded -- there is no action to pair with it. The final recorded
    # distance is therefore bounded by the tolerance plus one step of travel,
    # not by the tolerance itself. Asserting the tighter bound fails on
    # perfectly good data, which is what it did.
    reach_bound = args.goal_tolerance + V_MAX * DT
    check(float(np.median(ends)) < reach_bound,
          "median episode ends within one step of the goal tolerance",
          f"median final distance {np.median(ends):.3f} m "
          f"(tol {args.goal_tolerance} + one step {V_MAX * DT:.3f} "
          f"= {reach_bound:.3f})")
    check(float(np.percentile(ends, 90)) < reach_bound + 0.1,
          "the tail also converges, not just the median",
          f"p90 final distance {np.percentile(ends, 90):.3f} m")
    check(float(starts.mean()) > 1.0, "episodes started a useful distance out",
          f"mean start {starts.mean():.2f} m")
    check(int(lengths.min()) > 3, "no degenerate one-step episodes",
          f"lengths min {lengths.min()} median {int(np.median(lengths))} "
          f"max {lengths.max()}")

    # ── the loader produces a well-formed observation ──────────────────────
    print("\nobservation assembly (noise + binning happen here, not in the shard)")
    obs, act = next(iter(shard.batches(min(256, n), shuffle=False)))
    check(obs.shape[1] == OBS_DIM, "observation width", f"{obs.shape[1]}")
    check(bool(torch.isfinite(obs).all()), "no NaN or inf in the observation")
    lidar = obs[:, :LIDAR_BINS]
    check(bool(((lidar >= 0) & (lidar <= 1)).all()), "lidar bins in [0, 1]")
    occ = (lidar < INVALID_FILL).float().mean().item()
    check(occ > 0.05, "bins carry returns, the scan is not empty",
          f"{occ:.1%} of bins below INVALID_FILL")
    goal_feat = obs[:, LIDAR_BINS:LIDAR_BINS + 3]
    check(bool(((goal_feat[:, 0] >= 0) & (goal_feat[:, 0] <= 1)).all()),
          "goal distance normalized to [0, 1]")
    unit = (goal_feat[:, 1] ** 2 + goal_feat[:, 2] ** 2)
    check(bool(torch.allclose(unit, torch.ones_like(unit), atol=1e-4)),
          "goal bearing is a unit sin/cos pair")
    last = obs[:, -2:]
    check(bool((last.abs() <= 1.0 + 1e-6).all()), "last action normalized")
    check(bool(torch.equal(act, shard.action[:act.shape[0]])),
          "labels stay in SI units, unlike the last-action input")

    # ── fresh noise every epoch ────────────────────────────────────────────
    print("\nnoise is applied at load time")
    a, _ = next(iter(shard.batches(min(256, n), shuffle=False)))
    b, _ = next(iter(shard.batches(min(256, n), shuffle=False)))
    check(not torch.allclose(a, b), "two passes give different corruptions")
    c, _ = next(iter(shard.batches(min(256, n), noise_scale=0.0, shuffle=False)))
    d, _ = next(iter(shard.batches(min(256, n), noise_scale=0.0, shuffle=False)))
    check(torch.equal(c, d), "noise_scale=0 is reproducible")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\nsummary: {len(ids)} episodes, {n} steps, "
          f"{n / max(1, len(ids)):.0f} steps/episode, "
          f"{args.shard.stat().st_size / 1e6:.1f} MB")
    if _failures:
        print(f"\n\033[31m{len(_failures)} FAILED:\033[0m " + "; ".join(_failures))
        sys.exit(1)
    print("\n\033[32mall checks passed -- this shard is usable for BC\033[0m\n")


if __name__ == "__main__":
    main()
