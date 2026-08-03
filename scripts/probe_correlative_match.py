"""Long-baseline correlative scan matching: attack the variance, not the bias.

Why this exists
===============
The per-tick consensus vote (``probe_scan_vote.py``) removed most of the *bias*
in command dead reckoning -- 0.96 cm/m of drift against 2.10 -- and gained
nothing, because its per-tick scatter is 5.5x the command's and the two effects
cancel over an episode. Belief error is a sum, and the sum of an estimator's
residuals obeys

    belief_error  ~  sigma_match * sqrt(S / L)

where ``S`` is total distance travelled, ``L`` is the baseline each measurement
spans, and ``sigma_match`` is the matcher's absolute accuracy. Per-tick voting
has ``L`` = one tick = 2.3 cm, which is why 0.566 cm of scatter becomes 8.4 cm
over 5 m of travel. Nothing about the *estimator* fixes that; only a longer
baseline does, and the gain goes as sqrt(L).

Lengthening the baseline was already tried and failed badly -- 0.96 cm/m of
drift at gap 1, 10.8 at gap 2, 81 at gap 4. That failure is specific to the
range-flow model the vote uses: it is first order in displacement and it
compares *equal bearings*, so once the robot moves far enough to shift scene
content between bearings, the correspondence it assumes stops existing.

Correlative scan matching has neither limitation. It works in Cartesian space,
so displacement is exact rather than linearized, and it never assumes a bearing
correspondence: it asks how well the transformed point cloud lands on a
likelihood field built from the other scan. Outliers score low under every
candidate and abstain, exactly as in the per-tick vote, so the robustness that
made voting work is retained.

This probe measures ``sigma_match`` against baseline length, and reports the
implied episode belief error so the comparison is against the number that
matters rather than against a per-tick residual.

    .venv/bin/python scripts/probe_correlative_match.py \
        --data data/belief_onpolicy_v15 --baseline 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from sim2edge.models.belief import compose_belief
from sim2edge.spec import DT, LIDAR_MAX_M, V_MAX
from sim2edge.training.data import ShardCorpus


def scan_points(metric: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor,
                                                                    torch.Tensor]:
    """Cartesian points for one scan, in its own body frame."""
    beams = metric.shape[-1]
    theta = -math.pi + (torch.arange(beams, device=metric.device,
                                     dtype=metric.dtype) + 0.5) * (
                                         2 * math.pi / beams)
    x = metric * torch.cos(theta)
    y = metric * torch.sin(theta)
    return torch.stack([x, y], dim=-1), valid


def likelihood_field(points: torch.Tensor, weight: torch.Tensor, extent: float,
                     resolution: float, blur_sigma: float) -> torch.Tensor:
    """Rasterize a point cloud into a blurred occupancy likelihood.

    The blur is what makes this an *agreement* score rather than a hard hit
    test: a point landing within a range-noise-width of a surface still scores,
    and a point landing nowhere near one scores ~0 under every candidate. That
    bounded response is the same property that lets the per-tick vote survive
    30% gross outliers.
    """
    n = points.shape[0]
    cells = int(round(2 * extent / resolution))
    index = ((points + extent) / resolution).round().long()
    inside = ((index >= 0) & (index < cells)).all(-1) & (weight > 0)
    flat = (index[..., 0].clamp(0, cells - 1) * cells
            + index[..., 1].clamp(0, cells - 1))
    field = points.new_zeros((n, cells * cells))
    field.scatter_add_(1, flat, inside.to(points.dtype))
    field = field.view(n, 1, cells, cells).clamp(max=1.0)

    radius = max(1, int(round(3.0 * blur_sigma / resolution)))
    offset = torch.arange(-radius, radius + 1, device=points.device,
                          dtype=points.dtype) * resolution
    kernel = torch.exp(-0.5 * (offset / blur_sigma) ** 2)
    kernel = kernel / kernel.max()
    field = F.conv2d(field, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    field = F.conv2d(field, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return field.clamp(max=1.0)


def score_candidates(field: torch.Tensor, points: torch.Tensor,
                     weight: torch.Tensor, candidates: torch.Tensor,
                     extent: float) -> torch.Tensor:
    """Sum the field under every candidate placement of ``points``."""
    n, beams, _ = points.shape
    c = candidates.shape[0]
    placed = points[:, None] + candidates[None, :, None]      # (N, C, B, 2)
    # grid_sample wants normalized (x, y) in [-1, 1], and indexes (row, col) as
    # (y, x); the field was built with row = points[..., 0].
    normalized = placed / extent
    grid = torch.stack([normalized[..., 1], normalized[..., 0]], dim=-1)
    sampled = F.grid_sample(field, grid.view(n, c, beams, 2),
                            mode="bilinear", align_corners=False,
                            padding_mode="zeros")
    return (sampled.view(n, c, beams) * weight[:, None]).sum(-1)


def refine(score: torch.Tensor, candidates: torch.Tensor, side: int,
           radius: int) -> torch.Tensor:
    """Hard peak, then a soft argmax inside a window around it."""
    n = score.shape[0]
    best = score.argmax(1)
    row, column = best // side, best % side
    index = torch.arange(side, device=score.device)
    near = (((index[None] - row[:, None]).abs() <= radius)[:, :, None]
            & ((index[None] - column[:, None]).abs()
               <= radius)[:, None, :]).reshape(n, -1)
    peak = score.amax(1, keepdim=True).clamp_min(1e-9)
    mass = torch.softmax(
        torch.where(near, score / peak * 40.0,
                    torch.full_like(score, -1.0e4)), dim=1)
    return mass @ candidates


def grid(extent: float, side: int, device, dtype) -> torch.Tensor:
    axis = torch.linspace(-extent, extent, side, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def match(old_metric, old_valid, new_metric, new_valid, delta_yaw,
          field_extent, resolution, blur, coarse_extent, coarse_side,
          fine_extent, fine_side, coarse_resolution=0.04, coarse_blur=0.08):
    """Displacement from the old pose to the new one, in the *old* body frame.

    With world point ``P``, ``p_new = R(-dtheta)(p_old - d)``, so
    ``p_old = R(dtheta) p_new + d``: the new scan's points are rotated by the
    measured yaw and then translated by the candidate. ``d`` therefore comes out
    in the old frame, which is the frame the odometry head answers in.

    **Two resolutions, and that is not an optimization.** The blur width is the
    distance over which a candidate still scores, so a search grid coarser than
    the blur steps clean over the peak and never sees it. With a 1 cm blur and a
    4 cm coarse grid the score is dominated by whatever accidental overlap
    exists, which is usually near zero displacement -- the estimator then
    confidently reports that the robot did not move. So the coarse pass runs on
    a heavily blurred field that bridges its own grid spacing, and only the fine
    pass, already within a few centimetres, uses a sharp one.
    """
    old_points, _ = scan_points(old_metric, old_valid)
    new_points, _ = scan_points(new_metric, new_valid)
    old_weight = old_valid.to(old_metric.dtype) * (
        old_metric < LIDAR_MAX_M - 1e-3).to(old_metric.dtype)
    new_weight = new_valid.to(new_metric.dtype) * (
        new_metric < LIDAR_MAX_M - 1e-3).to(new_metric.dtype)

    cos, sin = torch.cos(delta_yaw), torch.sin(delta_yaw)
    rotated = torch.stack([
        cos[:, None] * new_points[..., 0] - sin[:, None] * new_points[..., 1],
        sin[:, None] * new_points[..., 0] + cos[:, None] * new_points[..., 1],
    ], dim=-1)

    coarse_field = likelihood_field(old_points, old_weight, field_extent,
                                    coarse_resolution, coarse_blur)
    coarse = grid(coarse_extent, coarse_side, old_metric.device,
                  old_metric.dtype)
    score = score_candidates(coarse_field, rotated, new_weight, coarse,
                             field_extent)
    centre = refine(score, coarse, coarse_side, 1)

    fine_field = likelihood_field(old_points, old_weight, field_extent,
                                  resolution, blur)
    fine = grid(fine_extent, fine_side, old_metric.device, old_metric.dtype)
    offsets = centre[:, None] + fine[None]
    placed = rotated[:, None] + offsets[:, :, None]
    normalized = placed / field_extent
    sample_grid = torch.stack([normalized[..., 1], normalized[..., 0]], dim=-1)
    n, c, bb, _ = placed.shape
    sampled = F.grid_sample(fine_field, sample_grid.view(n, c, bb, 2),
                            mode="bilinear", align_corners=False,
                            padding_mode="zeros")
    fine_score = (sampled.view(n, c, bb) * new_weight[:, None]).sum(-1)
    return centre + refine(fine_score, fine, fine_side, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=int, default=20,
                        help="frames between the matched scans")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--field-extent", type=float, default=6.0)
    parser.add_argument("--resolution", type=float, default=0.005)
    parser.add_argument("--blur", type=float, default=0.01)
    parser.add_argument("--coarse-extent", type=float, default=0.6)
    parser.add_argument("--coarse-side", type=int, default=31)
    parser.add_argument("--coarse-resolution", type=float, default=0.04)
    parser.add_argument("--coarse-blur", type=float, default=0.08)
    parser.add_argument("--fine-extent", type=float, default=0.06)
    parser.add_argument("--fine-side", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--stride", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    corpus = ShardCorpus(args.data)
    generator = torch.Generator(device=device.type).manual_seed(0)

    truths, estimates, commands = [], [], []
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        for batch in shard.episode_batches(args.batch_size, args.noise_scale,
                                           shuffle=False, generator=generator):
            raw, valid = batch.raw_lidar, batch.raw_valid
            gyro_last = batch.gyro[:, :, -1]
            steps = raw.shape[0]
            history = raw.shape[2]
            if args.baseline >= history:
                raise ValueError(
                    f"baseline {args.baseline} needs more than {history} frames "
                    "of stored scan history")
            for t in range(args.baseline, steps, args.stride):
                # The whole window must be inside the episode, at both ends.
                # Padding repeats the terminal *sample*, which repeats the
                # terminal `translation` too -- so composing per-tick
                # translations across a padded tail accumulates motion that
                # never happened (measured: 40 cm of it), while the scans
                # correctly show none. That mismatch reads as a broken matcher.
                live = batch.valid_steps[t] & batch.valid_steps[t - args.baseline]
                if not bool(live.any()):
                    continue
                # Both scans come from the *same* stored window, so the baseline
                # is limited by MOTION_HISTORY_FRAMES rather than by the episode.
                new_metric, new_valid = raw[t, :, -1], valid[t, :, -1]
                old_metric = raw[t, :, -1 - args.baseline]
                old_valid = valid[t, :, -1 - args.baseline]
                turn = batch.gyro[t, :, -args.baseline:].sum(1)

                with torch.no_grad():
                    estimate = match(
                        old_metric, old_valid, new_metric, new_valid, turn,
                        args.field_extent, args.resolution, args.blur,
                        args.coarse_extent, args.coarse_side,
                        args.fine_extent, args.fine_side,
                        args.coarse_resolution, args.coarse_blur)

                # Truth from the *true belief*, not from composing per-tick
                # translations. belief is the true goal offset, so
                # belief[t] = R(-turn)(belief[t-L] - d) gives the window
                # displacement directly in the old frame, with no accumulation
                # to go wrong and no sensitivity to padding.
                zero = torch.zeros_like(estimate)
                truth = batch.belief[t - args.baseline].float() - compose_belief(
                    batch.belief[t].float(), zero, -turn)
                command = torch.zeros_like(estimate)
                for k in range(t - args.baseline + 1, t + 1):
                    speed = batch.previous_action[k][:, -1, 0] * V_MAX
                    command = compose_belief(
                        command,
                        torch.stack([speed * DT, torch.zeros_like(speed)], 1),
                        gyro_last[k])
                # ``match`` answers in the *old* frame; the command composes
                # forward into the new one, so it is rotated back by the
                # window's yaw to match.
                command = compose_belief(-command, zero, -turn)
                truths.append(truth[live].cpu())
                estimates.append(estimate[live].cpu())
                commands.append(command[live].cpu())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break

    truth = torch.cat(truths).double()
    estimate = torch.cat(estimates).double()
    command = torch.cat(commands).double()
    baseline_m = float(truth.norm(dim=1).mean())

    def report(value: torch.Tensor) -> dict:
        residual = value - truth
        error = residual.norm(dim=1)
        scatter = float((residual - residual.mean(0)).norm(dim=1).pow(2)
                        .mean().sqrt())
        bias = float(residual.mean(0).norm())
        # The number that decides everything: an estimator measuring a baseline
        # of L, applied over S metres of travel, contributes bias*S/L
        # systematically and scatter*sqrt(S/L) randomly.
        travel = 5.0
        windows = travel / max(baseline_m, 1e-9)
        systematic = bias * windows
        random = scatter * math.sqrt(windows)
        return {
            "bias_cm": bias * 100.0,
            "scatter_cm": scatter * 100.0,
            "error_mean_cm": float(error.mean()) * 100.0,
            "implied_belief_error_cm": math.hypot(systematic, random) * 100.0,
            "implied_systematic_cm": systematic * 100.0,
            "implied_random_cm": random * 100.0,
        }

    result = {
        "samples": int(truth.shape[0]),
        "baseline_frames": args.baseline,
        "baseline_travel_cm": baseline_m * 100.0,
        "noise_scale": args.noise_scale,
        "resolution_cm": args.resolution * 100.0,
        "blur_cm": args.blur * 100.0,
        "correlative_match": report(estimate),
        "command": report(command),
        "note": "implied_* assume 5 m of travel per episode",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
