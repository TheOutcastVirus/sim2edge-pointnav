"""Scan-to-map: measure *accumulated* command error, not per-tick displacement.

The bound this is trying to escape
==================================
Measured on ``belief_onpolicy_v15``, per control tick:

* the quantity worth estimating -- the slip residual ``truth - command`` -- is
  **0.078 cm**;
* the noise floor of frame-to-frame scan matching is **0.566 cm**.

The signal is seven times below the noise, and the consequence is not subtle:
the per-tick consensus vote recovers the slip residual with a correlation of
**0.03**. Its apparent 0.92 correlation with truth is entirely explained by
truth being nearly equal to the command already. Every encoder improvement in
this project has been improving how well the head reads a quantity it cannot
measure.

Pooling does not fix it. Estimating one constant from ``N`` measurements of
noise ``sigma`` and applying it over ``N`` ticks gives ``sigma * sqrt(N)`` --
exactly what using the raw per-tick estimates gives. The variance follows the
sum either way, which is why ``persistent_calibration`` was never the answer.

Only a longer baseline raises the signal, and frame-to-frame cannot have one:
by four ticks a static alignment already explains the newer scan better than
the true one does (``probe_correlative_match.py``).

Why a map escapes it
====================
A local map breaks the coupling between *baseline* and *overlap* that makes
frame-to-frame matching fail. Frames from the last second are placed into a
common grid using dead reckoning, and the newest scan is matched against the
accumulation. Then:

* the **overlap** is short-baseline. The map contains geometry observed a tick
  ago as well as a second ago, so the current scan always has recent, dense
  support to land on -- the occlusion and viewpoint changes that destroy a
  19-tick frame pair never arise;
* the **baseline** is long. Because every map frame was placed using the
  command, the offset that best aligns the newest scan is the command's
  *accumulated* error over the whole window: 19 ticks x 0.078 cm = 1.5 cm of
  signal against a matching noise nearer 0.5 cm.

That is an SNR of ~3 where frame-to-frame has 0.14. This probe measures whether
that argument survives contact with the corpus, and the number to read is
``recovers_accumulated_slip_corr`` -- not the alignment error, which the command
alone predicts well.

    .venv/bin/python scripts/probe_scan_to_map.py --data data/belief_onpolicy_v15
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

from probe_correlative_match import (grid, likelihood_field, refine,
                                     scan_points, score_candidates)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def dead_reckoned_map(raw, valid, gyro, previous_action, frames, newest):
    """Place history frames into the newest frame using command dead reckoning.

    Returns stacked points and weights. Frame ``f`` is carried forward to the
    newest frame by composing the commanded displacement and the measured gyro
    over the intervening ticks, which is exactly what the belief integrator
    does -- so the residual misalignment *is* the integrator's error.
    """
    points, weights = [], []
    for f in frames:
        # Rotation and commanded translation from frame f to the newest frame.
        offset = torch.zeros(raw.shape[0], 2, device=raw.device, dtype=raw.dtype)
        for k in range(f + 1, newest + 1):
            speed = previous_action[:, k, 0] * V_MAX
            step = torch.stack([speed * DT, torch.zeros_like(speed)], dim=1)
            offset = compose_belief(offset, step, gyro[:, k])
        turn = gyro[:, f + 1:newest + 1].sum(1)

        metric, mask = raw[:, f], valid[:, f]
        pts, _ = scan_points(metric, mask)
        cos, sin = torch.cos(turn), torch.sin(turn)
        # Express frame f's points in the newest frame: rotate by -turn, then
        # subtract the displacement travelled since (offset is that
        # displacement expressed in the newest frame).
        rotated = torch.stack([
            cos[:, None] * pts[..., 0] + sin[:, None] * pts[..., 1],
            -sin[:, None] * pts[..., 0] + cos[:, None] * pts[..., 1],
        ], dim=-1)
        # compose_belief integrated from zero yields the vector *from* the
        # newest pose back to frame f, i.e. the negated displacement, so this
        # adds. Subtracting places every frame at twice its true offset, which
        # smears the map without obviously breaking it -- distant frames go
        # wrong while recent ones stay nearly right.
        points.append(rotated + offset[:, None])
        weights.append(mask.to(metric.dtype)
                       * (metric < LIDAR_MAX_M - 1e-3).to(metric.dtype))
    return torch.cat(points, dim=1), torch.cat(weights, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--window", type=int, default=19,
                        help="history frames accumulated into the map")
    parser.add_argument("--map-stride", type=int, default=2,
                        help="use every Nth history frame, to bound cost")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--field-extent", type=float, default=6.0)
    parser.add_argument("--resolution", type=float, default=0.005)
    parser.add_argument("--blur", type=float, default=0.01)
    parser.add_argument("--coarse-resolution", type=float, default=0.02)
    parser.add_argument("--coarse-blur", type=float, default=0.04)
    parser.add_argument("--coarse-extent", type=float, default=0.12)
    parser.add_argument("--coarse-side", type=int, default=25)
    parser.add_argument("--fine-extent", type=float, default=0.02)
    parser.add_argument("--fine-side", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    corpus = ShardCorpus(args.data)
    generator = torch.Generator(device=device.type).manual_seed(0)

    offsets, accumulated_slip, commands = [], [], []
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        for batch in shard.episode_batches(args.batch_size, args.noise_scale,
                                           shuffle=False, generator=generator):
            raw, valid = batch.raw_lidar, batch.raw_valid
            steps, history = raw.shape[0], raw.shape[2]
            newest = history - 1
            frames = list(range(newest - args.window, newest, args.map_stride))
            for t in range(args.window, steps, args.stride):
                live = batch.valid_steps[t] & batch.valid_steps[t - args.window]
                if not bool(live.any()):
                    continue
                with torch.no_grad():
                    map_points, map_weight = dead_reckoned_map(
                        raw[t], valid[t], batch.gyro[t],
                        batch.previous_action[t], frames, newest)
                    current, _ = scan_points(raw[t, :, newest],
                                             valid[t, :, newest])
                    current_weight = (
                        valid[t, :, newest].to(raw.dtype)
                        * (raw[t, :, newest] < LIDAR_MAX_M - 1e-3).to(raw.dtype))

                    coarse_field = likelihood_field(
                        map_points, map_weight, args.field_extent,
                        args.coarse_resolution, args.coarse_blur)
                    coarse = grid(args.coarse_extent, args.coarse_side,
                                  device, raw.dtype)
                    score = score_candidates(coarse_field, current,
                                             current_weight, coarse,
                                             args.field_extent)
                    centre = refine(score, coarse, args.coarse_side, 1)

                    fine_field = likelihood_field(
                        map_points, map_weight, args.field_extent,
                        args.resolution, args.blur)
                    fine = grid(args.fine_extent, args.fine_side, device,
                                raw.dtype)
                    placed = (current[:, None]
                              + (centre[:, None] + fine[None])[:, :, None])
                    normalized = placed / args.field_extent
                    sample_grid = torch.stack(
                        [normalized[..., 1], normalized[..., 0]], dim=-1)
                    n, c, b, _ = placed.shape
                    sampled = F.grid_sample(
                        fine_field, sample_grid.view(n, c, b, 2),
                        mode="bilinear", align_corners=False,
                        padding_mode="zeros")
                    fine_score = (sampled.view(n, c, b)
                                  * current_weight[:, None]).sum(-1)
                    offset = centre + refine(fine_score, fine, args.fine_side, 2)

                # Truth and command over the same window, in the newest frame.
                turn = batch.gyro[t, :, -args.window:].sum(1)
                zero = torch.zeros_like(offset)
                truth = compose_belief(
                    batch.belief[t - args.window].float(), zero, turn
                ) - batch.belief[t].float()
                command = torch.zeros_like(offset)
                for k in range(t - args.window + 1, t + 1):
                    speed = batch.previous_action[k][:, -1, 0] * V_MAX
                    command = compose_belief(
                        command,
                        torch.stack([speed * DT, torch.zeros_like(speed)], 1),
                        batch.gyro[k, :, -1])
                command = -command

                offsets.append(offset[live].cpu())
                accumulated_slip.append((truth - command)[live].cpu())
                commands.append(command[live].cpu())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break

    offset = torch.cat(offsets).double()
    slip = torch.cat(accumulated_slip).double()
    command = torch.cat(commands).double()

    residual = offset - slip
    result = {
        "samples": int(offset.shape[0]),
        "window_ticks": args.window,
        "map_frames": len(list(range(0, args.window, args.map_stride))),
        "noise_scale": args.noise_scale,
        "command_travel_cm": float(command.norm(dim=1).mean()) * 100.0,
        "accumulated_slip_cm": float(slip.norm(dim=1).mean()) * 100.0,
        # The number that decides this. Frame-to-frame scores 0.03 here.
        "recovers_accumulated_slip_corr_x": correlation(offset[:, 0], slip[:, 0]),
        "recovers_accumulated_slip_corr_y": correlation(offset[:, 1], slip[:, 1]),
        "offset_mean_cm": float(offset.norm(dim=1).mean()) * 100.0,
        "residual_after_correction_cm": float(residual.norm(dim=1).mean()) * 100.0,
        "implied_snr": (float(slip.norm(dim=1).std())
                        / max(float(residual.norm(dim=1).std()), 1e-9)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
