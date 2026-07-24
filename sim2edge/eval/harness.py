"""Simulation evaluation harness.

Implements the protocol in §5 of the implementation plan: a fixed set of
held-out episodes, reported as Success Rate, SPL, collision rate, time-to-goal
and smoothness, at three noise tiers (clean / deployment / 2x deployment).

Two properties matter more than the metric list:

* **Held-out episodes are held out by seed range**, not by resampling. Every
  checkpoint is evaluated on the *same* layouts, so a 3-point SR difference
  between two checkpoints is a difference between policies and not between
  two draws of the room generator.
* **Interventions are a first-class metric.** A run that only succeeds because
  the safety layer kept stopping it is a finding, not a success. Evaluating with
  ``apply_safety=True`` and reporting the count is how that stays visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..envs.reference_env import EnvConfig, NavEnv
from ..spec import DEFAULT_SPEC, NavSpec

__all__ = ["EvalMetrics", "evaluate", "evaluate_noise_tiers", "oracle_policy",
           "HELD_OUT_SEED_BASE"]

#: Evaluation seeds live in a disjoint range from training seeds. Training uses
#: [0, 1e6); evaluation starts here. Overlapping the two is the single easiest
#: way to report a success rate that does not survive contact with new layouts.
HELD_OUT_SEED_BASE = 10_000_000

#: A policy maps an observation dict to a physical ``(v, omega)`` command.
PolicyFn = Callable[[Dict[str, np.ndarray]], Sequence[float]]


@dataclass
class EvalMetrics:
    episodes: int = 0
    success_rate: float = 0.0
    spl: float = 0.0
    collision_rate: float = 0.0
    timeout_rate: float = 0.0
    #: Seconds to reach the goal, successful episodes only.
    mean_time_to_goal: float = 0.0
    #: Mean |da/dt| over the episode in normalized action units per second.
    mean_jerk: float = 0.0
    mean_return: float = 0.0
    #: Safety-layer interventions per episode; 0 when the layer is disabled.
    mean_interventions: float = 0.0
    #: Fraction of episodes in which the safety layer acted at all.
    intervention_episode_rate: float = 0.0
    noise_scale: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        return (f"SR {self.success_rate:6.1%}  SPL {self.spl:5.3f}  "
                f"coll {self.collision_rate:5.1%}  timeout {self.timeout_rate:5.1%}  "
                f"t_goal {self.mean_time_to_goal:5.1f}s  jerk {self.mean_jerk:5.2f}  "
                f"interv {self.mean_interventions:5.2f}")


@dataclass
class EpisodeRecord:
    """Per-episode detail, kept so failures can be inspected individually."""

    seed: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    spl: float
    ret: float
    jerk: float
    interventions: int


def evaluate(
    policy: PolicyFn,
    n_episodes: int = 500,
    config: Optional[EnvConfig] = None,
    seed_base: int = HELD_OUT_SEED_BASE,
    spec: NavSpec = DEFAULT_SPEC,
    progress: bool = False,
    records: Optional[List[EpisodeRecord]] = None,
    env: Optional[NavEnv] = None,
) -> EvalMetrics:
    """Run ``n_episodes`` held-out episodes and aggregate the metrics.

    ``env`` may be supplied so that a policy needing privileged access to the
    environment — :func:`oracle_policy` reads ground-truth pose from it — drives
    the same instance the harness steps. Passing an env overrides ``config``.
    """
    cfg = config or EnvConfig(spec=spec)
    if env is None:
        env = NavEnv(cfg, seed=seed_base)
    else:
        cfg = env.config
    dt = spec.action.dt
    max_steps = cfg.max_episode_steps or spec.reward.max_episode_steps

    succ = coll = tout = 0
    spls: List[float] = []
    times: List[float] = []
    jerks: List[float] = []
    rets: List[float] = []
    intervs: List[int] = []
    interv_eps = 0

    for i in range(n_episodes):
        ep_seed = seed_base + i
        obs = env.reset(seed=ep_seed)
        prev_norm = np.zeros(2, dtype=np.float32)
        total_jerk = 0.0
        ret = 0.0
        result = None

        for _ in range(max_steps):
            action = policy(obs)
            result = env.step(action)
            obs = result.obs
            ret += result.reward

            norm = np.array(spec.action.normalize(float(action[0]),
                                                  float(action[1])),
                            dtype=np.float32)
            total_jerk += float(np.linalg.norm(norm - prev_norm)) / dt
            prev_norm = norm

            if result.terminated or result.truncated:
                break

        assert result is not None
        info = result.info
        steps = max(1, info["steps"])
        jerk = total_jerk / steps
        is_success = bool(info["success"])
        is_coll = bool(info["collision"])
        n_interv = int(info["interventions"])

        succ += is_success
        coll += is_coll
        tout += (not is_success and not is_coll)
        if is_success:
            spls.append(float(info.get("spl", 0.0)))
            times.append(steps * dt)
        else:
            spls.append(0.0)  # SPL is 0 for a failed episode by definition
        jerks.append(jerk)
        rets.append(ret)
        intervs.append(n_interv)
        interv_eps += int(n_interv > 0)

        if records is not None:
            records.append(EpisodeRecord(
                ep_seed, is_success, is_coll, not (is_success or is_coll),
                steps, spls[-1], ret, jerk, n_interv))
        if progress and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_episodes}] SR so far {succ / (i + 1):.1%}")

    n = max(1, n_episodes)
    return EvalMetrics(
        episodes=n_episodes,
        success_rate=succ / n,
        spl=float(np.mean(spls)) if spls else 0.0,
        collision_rate=coll / n,
        timeout_rate=tout / n,
        mean_time_to_goal=float(np.mean(times)) if times else float("nan"),
        mean_jerk=float(np.mean(jerks)) if jerks else 0.0,
        mean_return=float(np.mean(rets)) if rets else 0.0,
        mean_interventions=float(np.mean(intervs)) if intervs else 0.0,
        intervention_episode_rate=interv_eps / n,
        noise_scale=(cfg.randomization.scale if cfg.randomization is not None
                     else spec.randomization.scale),
    )


def evaluate_noise_tiers(
    policy: PolicyFn,
    n_episodes: int = 200,
    tiers: Sequence[float] = (0.0, 1.0, 2.0),
    base_config: Optional[EnvConfig] = None,
    spec: NavSpec = DEFAULT_SPEC,
    seed_base: int = HELD_OUT_SEED_BASE,
) -> Dict[str, EvalMetrics]:
    """Evaluate at clean / deployment / 2x-deployment noise.

    The 2x tier is a robustness margin, not a target. A policy that scores well
    at 1.0 but falls apart at 2.0 has fit the randomization ranges rather than
    become robust, and the ranges were only ever estimates from a few rosbags.
    """
    base = base_config or EnvConfig(spec=spec)
    out: Dict[str, EvalMetrics] = {}
    for scale in tiers:
        cfg = EnvConfig(
            spec=base.spec,
            use_depth=base.use_depth,
            randomization=spec.randomization.at_scale(scale),
            apply_safety=base.apply_safety,
            max_episode_steps=base.max_episode_steps,
            terminate_on_collision=base.terminate_on_collision,
        )
        # Same seeds at every tier: the layouts are identical and only the
        # noise differs, so the comparison isolates robustness.
        out[f"noise_{scale:g}x"] = evaluate(policy, n_episodes, cfg,
                                            seed_base, spec)
    return out


def oracle_policy(env: NavEnv, spec: NavSpec = DEFAULT_SPEC) -> PolicyFn:
    """A policy that consults the expert. The upper bound for the benchmark.

    Note it reads ground-truth pose from ``env`` rather than from the
    observation — it is an oracle, not a competitor, and it exists to establish
    the ceiling and to sanity-check that a layout set is solvable at all.
    """
    from ..oracle.expert import PurePursuitExpert

    state: Dict[str, object] = {"room": None, "expert": None}

    def act(_obs: Dict[str, np.ndarray]) -> Sequence[float]:
        if state["room"] is not env.room:
            state["room"] = env.room
            state["expert"] = PurePursuitExpert(env.room, spec)
        cmd = state["expert"].act(*env.pose)  # type: ignore[union-attr]
        return (cmd.v, cmd.omega)

    return act
