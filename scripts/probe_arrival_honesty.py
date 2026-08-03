"""Is the declared uncertainty an honest description of the belief error?

On hardware the belief is the only arrival signal there is. The robot stops when
its own upper confidence bound on remaining distance drops inside
``arrival_radius_m``, so a covariance that under-states the error does not
produce a cautious robot that stops late -- it produces a confident one that
stops in the wrong place and reports success. The closed-loop evaluation
measured exactly that: when the policy concludes it has arrived it is truly
31-34 cm away, against a 25 cm tolerance and a 15 cm declared radius.

There is a specific structural reason to expect this. ``belief_losses``
calibrates the variance head against the *per-step* squared translation error,
and ``_propagate_covariance`` then adds those variances independently down the
episode. That is a random-walk model, and it is the wrong model: the dominant
error is a systematic wheel-scale bias, measured at ~2.5 cm per metre travelled.
A random walk grows as the square root of the number of steps and a scale error
grows linearly with distance, so the declared bound falls behind the true error
by roughly the square root of the episode length -- and does so silently.

This probe measures the gap rather than assuming it:

* ``calibration_ratio`` -- true belief error over declared 1-sigma. One is
  honest, and much above one is the failure above.
* ``coverage_1.96_sigma`` -- fraction of steps whose true error is inside the
  bound the arrival test uses. A calibrated bound covers 95%.
* ``error_vs_travel_slope`` / ``sigma_vs_travel_slope`` -- how each grows with
  distance travelled. If the first is linear and the second is not, the model
  form is wrong and no amount of retraining the variance head fixes it.

    .venv/bin/python scripts/probe_arrival_honesty.py \
        runs/belief_ppo_biasfix_v14/iter_1130.pt --data data/belief_onpolicy_v15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sim2edge.models.belief import compose_belief
from sim2edge.training.belief import load_belief_checkpoint
from sim2edge.training.data import ShardCorpus


def linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """Least-squares slope through the origin -- both quantities start at zero."""
    denominator = float((x * x).sum())
    return float((x * y).sum()) / denominator if denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--drift-scale", type=float,
                        help="override drift_scale_uncertainty, to score what "
                             "the systematic term would have declared")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = load_belief_checkpoint(args.checkpoint, device=device).eval()
    if args.drift_scale is not None:
        from dataclasses import replace
        policy.config = replace(policy.config,
                                drift_scale_uncertainty=args.drift_scale)
    corpus = ShardCorpus(args.data)
    generator = torch.Generator(device=device.type).manual_seed(0)

    errors, sigmas, travels = [], [], []
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        for batch in shard.episode_batches(args.batch_size, args.noise_scale,
                                           shuffle=False, generator=generator):
            steps = batch.lidar.shape[0]
            belief = batch.initial_belief
            covariance = torch.eye(2, device=device)[None].repeat(
                belief.shape[0], 1, 1) * 1.0e-8
            calibration = policy.initial_calibration(belief)
            travelled = policy.initial_travel(belief)
            with torch.no_grad():
                for t in range(1, steps):
                    translation, variance, _, calibration = (
                        policy.calibrated_motion_estimate(
                            calibration, batch.lidar[t], batch.gyro[t],
                            batch.previous_action[t], batch.raw_lidar[t],
                            batch.raw_valid[t]))
                    delta_yaw = batch.gyro[t, :, -1]
                    belief = compose_belief(belief, translation, delta_yaw)
                    step_length = torch.linalg.vector_norm(translation, dim=1)
                    drift = None
                    if policy.config.drift_scale_uncertainty > 0.0:
                        drift = policy._drift_variance(travelled, step_length)
                    covariance = policy._propagate_covariance(
                        covariance, variance, delta_yaw, drift)
                    travelled = travelled + step_length

                    live = batch.valid_steps[t]
                    if not bool(live.any()):
                        continue
                    error = torch.linalg.vector_norm(
                        belief - batch.belief[t].float(), dim=1)
                    sigma = policy._largest_covariance_eigenvalue(
                        covariance).clamp_min(0.0).sqrt()
                    errors.append(error[live].cpu())
                    sigmas.append(sigma[live].cpu())
                    travels.append(travelled[live].cpu())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break

    error = torch.cat(errors).double()
    sigma = torch.cat(sigmas).double()
    travel = torch.cat(travels).double()
    ratio = error / sigma.clamp_min(1e-9)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "steps_scored": int(error.numel()),
        "noise_scale": args.noise_scale,
        "drift_scale_uncertainty": policy.config.drift_scale_uncertainty,
        "belief_error_mean_cm": float(error.mean()) * 100.0,
        "declared_sigma_mean_cm": float(sigma.mean()) * 100.0,
        "calibration_ratio_mean": float(ratio.mean()),
        "calibration_ratio_median": float(ratio.median()),
        "coverage_1.96_sigma": float((error <= 1.96 * sigma).double().mean()),
        "error_vs_travel_slope_cm_per_m": linear_slope(travel, error) * 100.0,
        "sigma_vs_travel_slope_cm_per_m": linear_slope(travel, sigma) * 100.0,
        "travel_mean_m": float(travel.mean()),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
