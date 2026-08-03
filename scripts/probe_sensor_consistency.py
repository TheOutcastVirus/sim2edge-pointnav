"""Is the reconstructed sweep self-consistent enough to measure ego-motion?

This is the acceptance test for the sensor model, and it is deliberately not a
test of any estimator. Take two reconstructed frames, place the older one using
the **true** relative pose -- ground truth, not a command, not an estimate --
and measure how far the best alignment still sits from zero. A correct
reconstruction answers zero.

Measured before the sub-tick fix, on ``belief_onpolicy_v15``:

    baseline 1 tick:   0.665 cm at 0x sensor noise, 0.629 cm at 1x
    baseline 4 ticks:  7.81 cm  at 0x sensor noise, 7.13 cm  at 1x

Sensor noise is irrelevant -- the columns agree -- so this is the
reconstruction disagreeing with itself. The 4-tick figure is the sweep period:
at ``LIDAR_SPIN_HZ`` 5.5 a revolution is 3.6 ticks, and full speed covers 8.4 cm
in one, so frames more than a revolution apart are stitched from disjoint
sub-scan sets whose distortions do not cancel.

**The bar.** The slip residual any estimator here must measure is 0.078 cm per
tick. The reconstruction's own inconsistency has to sit below that, not merely
below the range noise -- the original 25 ms rounding was justified against range
noise, which is the wrong comparison for a sensor feeding an odometry estimator.

Run both arms; the environment variable is what ``dataset`` reads:

    .venv/bin/python scripts/probe_sensor_consistency.py --data data/belief_onpolicy_v15
    SIM2EDGE_SUBTICK_SWEEP=1 .venv/bin/python scripts/probe_sensor_consistency.py \\
        --data data/belief_onpolicy_v15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from probe_correlative_match import match

from sim2edge.models.belief import compose_belief
from sim2edge.spec import DT, V_MAX
from sim2edge.training.data import ShardCorpus

#: The quantity the estimator exists to measure, per control tick. Any
#: reconstruction inconsistency above this is the binding constraint.
SLIP_SIGNAL_CM = 0.078


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baselines", type=int, nargs="+", default=[1, 4, 9])
    parser.add_argument("--noise-scales", type=float, nargs="+",
                        default=[0.0, 1.0])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    subtick = os.environ.get("SIM2EDGE_SUBTICK_SWEEP", "0") == "1"
    report: dict = {
        "subtick_sweep": subtick,
        "slip_signal_cm_per_tick": SLIP_SIGNAL_CM,
        "arms": {},
    }

    for noise in args.noise_scales:
        corpus = ShardCorpus(args.data)
        generator = torch.Generator(device=device.type).manual_seed(0)
        collected: dict[int, list[torch.Tensor]] = {b: [] for b in args.baselines}
        batches = 0
        for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
            for batch in shard.episode_batches(args.batch_size, noise,
                                               shuffle=False,
                                               generator=generator):
                steps = batch.raw_lidar.shape[0]
                for span in args.baselines:
                    for t in range(span, steps, args.stride):
                        live = (batch.valid_steps[t]
                                & batch.valid_steps[t - span])
                        if not bool(live.any()):
                            continue
                        older = batch.raw_lidar[t, :, -1 - span]
                        older_valid = batch.raw_valid[t, :, -1 - span]
                        newest = batch.raw_lidar[t, :, -1]
                        newest_valid = batch.raw_valid[t, :, -1]
                        turn = batch.gyro[t, :, -span:].sum(1)
                        zero = torch.zeros(older.shape[0], 2, device=device)
                        truth = batch.belief[t - span].float() - compose_belief(
                            batch.belief[t].float(), zero, -turn)
                        # The search must bracket the displacement it is
                        # looking for. A fixed +/-10 cm extent silently caps the
                        # answer once the baseline exceeds ~4 ticks, and then
                        # reports the *travel* as if it were reconstruction
                        # error -- which is how a 9-tick "15.6 cm
                        # inconsistency" was manufactured out of a search
                        # failure.
                        reach = max(0.10, 2.0 * span * V_MAX * DT)
                        with torch.no_grad():
                            estimate = match(
                                older, older_valid, newest, newest_valid, turn,
                                6.0, 0.005, 0.01, reach,
                                int(2 * reach / 0.01) | 1, 0.02, 21,
                                0.02, 0.04)
                        collected[span].append(
                            (estimate - truth).norm(dim=1)[live].cpu())
                batches += 1
                if args.max_batches and batches >= args.max_batches:
                    break
            if args.max_batches and batches >= args.max_batches:
                break

        arm = {}
        for span, values in collected.items():
            if not values:
                continue
            error = torch.cat(values).double()
            arm[str(span)] = {
                "median_cm": float(error.median()) * 100.0,
                "mean_cm": float(error.mean()) * 100.0,
                "samples": int(error.numel()),
                "times_the_slip_signal": float(error.median()) * 100.0
                / SLIP_SIGNAL_CM,
            }
        report["arms"][f"noise_{noise:g}x"] = arm

    print(json.dumps(report, indent=2))
    header = "sub-tick" if subtick else "nearest-tick"
    for name, arm in report["arms"].items():
        for span, row in arm.items():
            print(f"[{header}] {name} baseline {span:>2} tick: "
                  f"median {row['median_cm']:.3f} cm "
                  f"({row['times_the_slip_signal']:.1f}x the slip signal)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
