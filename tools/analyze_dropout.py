#!/usr/bin/env python3
"""Is the A1's beam dropout what ``noise.DROPOUT_PROB`` says, and is it i.i.d.?

``sim2edge.noise.corrupt_lidar`` drops beams independently at 3%. Two things
about that need checking against hardware, and they are different questions:

  * **the rate.** Measured against a real sensor, not assumed.
  * **the structure.** Independent dropout scatters missing beams uniformly.
    Real no-returns come from dark, specular and glancing surfaces, which are
    contiguous in bearing. For convolutions that wrap around bearing, 20%
    scattered and 20% in a few arcs are not the same input at all.

The structure test is a run-length comparison: for the same measured rate, an
i.i.d. process has a known geometric run-length distribution, so the observed
one can be compared directly against it rather than against intuition.

One caveat this script cannot resolve on its own: an infinite range means
either "no return" (what DROPOUT_PROB models) or "nothing within range" (which
the simulator never produces, because rays always hit a wall). In a small
closed room the second is rare and the measured rate is a clean dropout rate.
In open space it is not. The scene label is recorded for that reason.
"""
import argparse
import sys

import numpy as np

SIM_DROPOUT_PROB = 0.03


def runs_of_true(mask: np.ndarray) -> np.ndarray:
    """Lengths of contiguous True runs along a 1-D boolean, wrapping the ends.

    Wrapping matters: bin 0 and bin 719 are adjacent on one revolution, so an
    arc straddling the seam is one run, not two.
    """
    if mask.all():
        return np.array([mask.size])
    if not mask.any():
        return np.array([], dtype=int)
    # Rotate so the array starts at a False, making the seam a non-issue.
    start = int(np.argmin(mask))
    m = np.roll(mask, -start)
    padded = np.concatenate(([False], m, [False]))
    edges = np.diff(padded.astype(np.int8))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+", help="npz files from capture_scans.py")
    args = ap.parse_args()

    for path in args.captures:
        d = np.load(path, allow_pickle=True)
        ranges = d["ranges"]
        label = str(d["label"]) if "label" in d else ""
        rmax = float(d["range_max"]) if "range_max" in d else float("inf")
        n_scans, n_beams = ranges.shape

        invalid = ~np.isfinite(ranges)
        rate = invalid.mean()

        print(f"\n=== {path} ===")
        print(f"  label                 {label!r}")
        print(f"  {n_scans} scans x {n_beams} beams, range_max {rmax:.1f} m")
        print(f"  invalid fraction      {rate:.4f}   "
              f"(sim DROPOUT_PROB {SIM_DROPOUT_PROB}) -> {rate/SIM_DROPOUT_PROB:.1f}x")
        per_scan = invalid.mean(axis=1)
        print(f"  per-scan spread       min {per_scan.min():.4f}  "
              f"median {np.median(per_scan):.4f}  max {per_scan.max():.4f}")

        # ---- structure: is a beam more likely to be invalid if its neighbour is?
        left = invalid[:, :-1]
        right = invalid[:, 1:]
        p_inv_given_inv = right[left].mean() if left.any() else float("nan")
        print(f"\n  P(invalid)                      {rate:.4f}")
        print(f"  P(invalid | neighbour invalid)  {p_inv_given_inv:.4f}")
        lift = p_inv_given_inv / rate if rate > 0 else float("nan")
        print(f"  lift                            {lift:.1f}x")
        if lift > 1.5:
            print("  -> NOT independent. Dropout arrives in contiguous arcs,")
            print("     which i.i.d. sampling cannot reproduce.")
        else:
            print("  -> consistent with independent dropout.")

        # ---- run lengths, observed vs the i.i.d. process at the same rate
        observed = np.concatenate([runs_of_true(invalid[i]) for i in
                                   range(n_scans)]) if invalid.any() else np.array([])
        rng = np.random.default_rng(0)
        synthetic = rng.random(ranges.shape) < rate
        expected = np.concatenate([runs_of_true(synthetic[i]) for i in
                                   range(n_scans)])

        print(f"\n  invalid-run lengths (beams), observed vs i.i.d. at the same rate")
        print(f"  {'':>10}{'observed':>12}{'i.i.d.':>12}")
        for name, fn in (("count", len), ("mean", np.mean),
                         ("median", np.median), ("p95",
                          lambda a: np.quantile(a, 0.95)), ("max", np.max)):
            o = fn(observed) if len(observed) else float("nan")
            e = fn(expected) if len(expected) else float("nan")
            print(f"  {name:>10}{o:>12.1f}{e:>12.1f}")

        if len(observed):
            long_o = (observed >= 5).sum() / len(observed)
            long_e = (expected >= 5).sum() / len(expected)
            print(f"\n  runs of >=5 beams   observed {long_o:.3%}   "
                  f"i.i.d. {long_e:.3%}")
            frac_in_long = observed[observed >= 5].sum() / observed.sum()
            print(f"  share of all invalid beams sitting in runs >=5: "
                  f"{frac_in_long:.1%}")

        # ---- where in bearing, which says whether it is geometry or the sensor
        by_bearing = invalid.mean(axis=0)
        worst = np.argsort(by_bearing)[-5:][::-1]
        print(f"\n  invalid rate by bearing: min {by_bearing.min():.3f} "
              f"median {np.median(by_bearing):.3f} max {by_bearing.max():.3f}")
        print(f"  worst bins {worst.tolist()} at "
              f"{[round(float(by_bearing[b]), 3) for b in worst]}")
        persistent = (by_bearing > 0.9).sum()
        print(f"  bins invalid >90% of the time: {persistent} of {n_beams}"
              f"  ({persistent/n_beams:.1%})")
        if persistent > 0.02 * n_beams:
            print("  -> a fixed set of bearings never returns. That is geometry or")
            print("     occlusion, not random dropout, and it should be modelled")
            print("     as such rather than folded into DROPOUT_PROB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
