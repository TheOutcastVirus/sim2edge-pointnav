"""Can a consensus vote recover ego-motion where least squares cannot?

The 2026-08-02 log established two facts that together point at a specific
answer. The motion *is* in the scans -- the median range change between
consecutive reconstructed frames tracks distance travelled at a ratio of about
1.0. And both geometric solvers tried against those scans, a point-to-line solve
and a full ICP, recovered nothing (ICP correlation +0.089), because roughly a
third of bearings are gross outliers at depth edges.

Both of those solvers minimize a sum of squared residuals. A squared residual is
unbounded, so a single outlier at a depth edge contributes more to the objective
than a hundred good bearings, and 30% of them do not merely add noise -- they
relocate the minimum. Least squares has a breakdown point of zero. No amount of
tuning fixes that; it is the wrong estimator for this data.

The classical fix is to stop *minimizing* and start *voting*. Score each
candidate translation by how many bearings agree with it:

    e(theta) = dr(theta) + dx cos theta + dy sin theta
    score(dx, dy) = sum_theta valid(theta) * exp(-e(theta)^2 / 2 sigma^2)

An outlier's residual is large under *every* candidate, so it contributes ~0 to
all of them and cannot move the peak -- it abstains rather than votes. Inliers
agree only near the true displacement, so they accumulate there. This is a
Hough/kernel-density consensus and its breakdown point is governed by whether
inliers form the largest coherent cluster, not by their fraction.

This probe measures whether that is true on the actual corpus, and prints the
least-squares estimate on the identical residuals as the control. If the vote
correlates with truth and least squares does not, the difference is the
estimator, not the data.

    .venv/bin/python scripts/probe_scan_vote.py --data data/belief_onpolicy_v15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sim2edge.models.belief import OdometryHead, compose_belief
from sim2edge.spec import DT, LIDAR_MAX_M, V_MAX
from sim2edge.training.data import ShardCorpus


def bearings(beams: int, device, dtype) -> torch.Tensor:
    """Bin-centre bearings matching ``preprocess.rasterize_lidar``.

    That function floors ``(wrap_angle(angle) + pi) / width`` into ``bins``, so
    bin ``b`` spans ``[-pi + b*width, -pi + (b+1)*width)`` and its centre is half
    a width in. Getting this offset wrong rotates every recovered translation by
    a constant, which reads as a null result rather than as a bug.
    """
    width = 2.0 * math.pi / beams
    index = torch.arange(beams, device=device, dtype=dtype)
    return -math.pi + (index + 0.5) * width


def true_displacement(translation: torch.Tensor, gyro_last: torch.Tensor,
                      start: int, stop: int) -> torch.Tensor:
    """Displacement from step ``start`` to ``stop``, in the body frame at ``stop``.

    ``translation[k]`` is the displacement over the tick *ending* at ``k``,
    expressed in the body frame at ``k-1`` -- the backward-looking convention
    ``update_belief`` integrates in. Composing it forward from a zero vector
    yields the vector *from* the robot at ``stop`` *to* its position at
    ``start``, which is the negation of what this returns.
    """
    carried = torch.zeros_like(translation[0])
    for k in range(start + 1, stop + 1):
        carried = compose_belief(carried, translation[k], gyro_last[k])
    return -carried


def derotate(scan: torch.Tensor, valid: torch.Tensor,
             delta_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample an older scan into the newer frame's bearing grid."""
    return OdometryHead._rotate_scan(scan, valid, delta_yaw)


def vote_translation(difference: torch.Tensor, weight: torch.Tensor,
                     theta: torch.Tensor, extent: float, steps: int,
                     sigma: float, radius: int = 3
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Consensus-vote the translation. Returns ``(estimate, sharpness)``.

    ``difference`` is ``r_new - r_old_derotated`` per bearing and ``weight`` is
    the both-frames-valid mask. The returned sharpness is the peak's mass
    fraction, which is an observability measure the learned gate never had: a
    corridor with no bearing-diverse structure produces a ridge rather than a
    peak, and says so.
    """
    axis = torch.linspace(-extent, extent, steps, device=theta.device,
                          dtype=theta.dtype)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    # (C, B) predicted range change for every candidate at every bearing.
    projection = (grid_x.reshape(-1, 1) * torch.cos(theta)[None]
                  + grid_y.reshape(-1, 1) * torch.sin(theta)[None])
    residual = difference[:, None, :] + projection[None]
    support = torch.exp(-0.5 * (residual / sigma) ** 2) * weight[:, None, :]
    score = support.sum(-1)                                   # (N, C)

    # A soft-argmax over the whole grid is biased toward the grid centre by
    # whatever pedestal the surface carries, and since the true displacement is
    # ~2 cm forward while the grid is centred on zero, that shrinkage appears as
    # a systematic *underestimate* of travel -- exactly the quantity that
    # integrates into drift. The peak is located hard and refined only within a
    # local window, so the pedestal cannot pull it.
    peak = score.argmax(1)
    candidates = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    row, column = peak // steps, peak % steps
    index = torch.arange(steps, device=theta.device)
    near = ((index[None, :] - row[:, None]).abs() <= radius)[:, :, None] & (
        (index[None, :] - column[:, None]).abs() <= radius)[:, None, :]
    near = near.reshape(score.shape[0], -1)
    peak_value = score.amax(1, keepdim=True).clamp_min(1e-9)
    mass = torch.softmax(
        torch.where(near, score / peak_value * 40.0,
                    torch.full_like(score, -1e4)), dim=1)
    estimate = mass @ candidates
    # Sharpness stays a *global* quantity: it must be able to report that a
    # corridor produced a ridge, which is only visible outside the window.
    global_mass = torch.softmax(score / peak_value * 40.0, dim=1)
    sharpness = global_mass.gather(1, peak[:, None]).squeeze(1)
    return estimate, sharpness


def least_squares_translation(difference: torch.Tensor, weight: torch.Tensor,
                              theta: torch.Tensor) -> torch.Tensor:
    """The control: the same residuals, solved the way ICP solves them."""
    direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    design = direction[None] * weight[:, :, None]              # (N, B, 2)
    normal = design.transpose(1, 2) @ design
    target = -(design.transpose(1, 2) @ (difference * weight)[:, :, None])
    normal = normal + torch.eye(2, device=theta.device,
                                dtype=theta.dtype)[None] * 1e-6
    return torch.linalg.solve(normal, target).squeeze(-1)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    denominator = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=1,
                        help="frames between the compared scans")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="consensus kernel width, metres")
    parser.add_argument("--steps", type=int, default=61)
    parser.add_argument("--radius", type=int, default=3,
                        help="local refinement window, in grid cells")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--stride", type=int, default=7,
                        help="timestep stride, to decorrelate samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    corpus = ShardCorpus(args.data)
    generator = torch.Generator(device=device.type).manual_seed(0)

    theta = None
    collected: dict[str, list[torch.Tensor]] = {
        "truth": [], "vote": [], "least_squares": [], "sharpness": [],
        "inlier": [], "command": [],
    }
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        for batch in shard.episode_batches(args.batch_size, args.noise_scale,
                                           shuffle=False, generator=generator):
            raw, valid = batch.raw_lidar, batch.raw_valid
            gyro_last = batch.gyro[:, :, -1]
            steps = raw.shape[0]
            if theta is None:
                theta = bearings(raw.shape[-1], device, raw.dtype)
            for t in range(args.gap, steps, args.stride):
                live = batch.valid_steps[t]
                if live.sum() == 0:
                    continue
                newest = raw[t, :, -1]
                newest_valid = valid[t, :, -1]
                older = raw[t, :, -1 - args.gap]
                older_valid = valid[t, :, -1 - args.gap]
                # Yaw accumulated over the gap, same sign convention the
                # encoder's _align_history uses.
                turn = gyro_last[t - args.gap + 1:t + 1].sum(0)
                aligned, aligned_valid = derotate(older, older_valid, turn)

                difference = newest - aligned
                weight = (newest_valid & aligned_valid).to(raw.dtype)
                # A max-range fill paired against a real return is not a range
                # change; it is a missing measurement, and including it is what
                # manufactures metre-scale "closures".
                weight = weight * (newest < LIDAR_MAX_M - 1e-3).to(raw.dtype)

                extent = max(args.gap * V_MAX * DT * 1.5, 0.02)
                vote, sharpness = vote_translation(
                    difference, weight, theta, extent, args.steps, args.sigma,
                    args.radius)
                lstsq = least_squares_translation(difference, weight, theta)
                truth = true_displacement(batch.translation, gyro_last,
                                          t - args.gap, t)

                # The estimator that has to be beaten. Not zero -- command dead
                # reckoning is already a good estimate of displacement, and the
                # whole quantity in dispute is the slip residual it leaves.
                command = torch.zeros_like(truth)
                for k in range(t - args.gap + 1, t + 1):
                    speed = batch.previous_action[k][:, -1, 0] * V_MAX
                    step = torch.stack(
                        [speed * DT, torch.zeros_like(speed)], dim=1)
                    command = compose_belief(command, step, gyro_last[k])
                command = -command

                keep = live
                collected["command"].append(command[keep].cpu())
                collected["truth"].append(truth[keep].cpu())
                collected["vote"].append(vote[keep].cpu())
                collected["least_squares"].append(lstsq[keep].cpu())
                collected["sharpness"].append(sharpness[keep].cpu())
                # Fraction of bearings consistent with the truth, which is the
                # outlier rate the log estimated at ~30% from the mean/median
                # ratio, measured directly here.
                agree = (difference + (truth[:, :1] * torch.cos(theta)[None]
                                       + truth[:, 1:] * torch.sin(theta)[None])
                         ).abs() < 3.0 * args.sigma
                collected["inlier"].append(
                    ((agree.to(raw.dtype) * weight).sum(1)
                     / weight.sum(1).clamp_min(1.0))[keep].cpu())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break

    truth = torch.cat(collected["truth"]).double()
    vote = torch.cat(collected["vote"]).double()
    lstsq = torch.cat(collected["least_squares"]).double()
    sharpness = torch.cat(collected["sharpness"]).double()
    inlier = torch.cat(collected["inlier"]).double()
    command = torch.cat(collected["command"]).double()

    travel = truth.norm(dim=1)
    metres = float(travel.sum())

    def report(estimate: torch.Tensor) -> dict:
        residual = estimate - truth
        error = residual.norm(dim=1)
        # Belief error is an *integral* of these residuals, so the number that
        # decides whether an estimator helps is the systematic component. A
        # zero-mean estimator with large per-tick scatter accumulates as a random
        # walk (sqrt(n)); a biased one accumulates linearly (n) and wins the
        # per-tick contest while losing the episode.
        drift = residual.sum(0) / max(metres, 1e-9)
        scatter = (residual - residual.mean(0)).norm(dim=1)
        return {
            "corr_x": correlation(estimate[:, 0], truth[:, 0]),
            "corr_y": correlation(estimate[:, 1], truth[:, 1]),
            "error_mean_cm": float(error.mean()) * 100.0,
            "error_median_cm": float(error.median()) * 100.0,
            "bias_cm": [float(v) * 100.0 for v in residual.mean(0)],
            "drift_cm_per_m": [float(v) * 100.0 for v in drift],
            "drift_magnitude_cm_per_m": float(drift.norm()) * 100.0,
            "scatter_rms_cm": float(scatter.pow(2).mean().sqrt()) * 100.0,
        }

    # The decisive quantity. The command is already a good displacement
    # estimate; the only thing an estimator can add is the *slip residual*
    # truth - command. If (estimate - command) does not correlate with it, the
    # estimator is reproducing the command and every improvement in its
    # apparent accuracy is circular -- which is a live risk here, because the
    # sweep reconstruction deskews shared beams using that same command.
    residual_truth = truth - command
    result = {
        "samples": int(truth.shape[0]),
        "slip_residual_cm": float(residual_truth.norm(dim=1).mean()) * 100.0,
        "vote_recovers_slip_corr_x": correlation(
            (vote - command)[:, 0], residual_truth[:, 0]),
        "vote_recovers_slip_corr_y": correlation(
            (vote - command)[:, 1], residual_truth[:, 1]),
        "vote_vs_command_corr_x": correlation(vote[:, 0], command[:, 0]),
        "vote_vs_truth_corr_x": correlation(vote[:, 0], truth[:, 0]),
        "gap": args.gap,
        "sigma_m": args.sigma,
        "noise_scale": args.noise_scale,
        "true_travel_median_cm": float(travel.median()) * 100.0,
        "inlier_fraction_mean": float(inlier.mean()),
        "command": report(command),
        "vote": report(vote),
        "least_squares": report(lstsq),
        "zero": report(torch.zeros_like(truth)),
        "sharpness_mean": float(sharpness.mean()),
        "sharpness_vs_error_corr": correlation(
            sharpness, (vote - truth).norm(dim=1)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
