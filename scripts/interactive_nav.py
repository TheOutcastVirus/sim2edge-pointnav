"""Drive a trained policy by hand: click a point on the floor, watch it go.

    ~/isaac-venv/bin/python scripts/interactive_nav.py

One environment, rendered, with the layout the policy has never seen. Left-click
anywhere on the floor to send the robot there; the panel on the left resets it,
moves it somewhere else in the same room, or draws a new room.

Two drivers, and the choice matters more than it looks:

``pose``       the actor's three goal features come from simulator truth. This
               is the actor's ceiling -- 96.5% success at 1x noise on held-out
               layouts -- and it answers "is the *policy* any good".
``localized``  the deployable stack: Create3-style body-frame odometry as the
               motion prior, the CPU exact-bearing scan-to-submap tracker as the
               correction, and arrival declared by the robot's own uncertainty.
               No pose reaches the actor. This answers "would the *robot* do
               this", and the orange disc shows where it believes the goal is.

The clicked goal is fed to the policy exactly the way a goal is fed anywhere
else in this project: as a relative bearing and distance in the robot's frame.
Nothing about a click is privileged -- which is precisely why an arbitrary point
works at all, and why the layout's precomputed geodesic field is neither needed
nor recomputed. There is no planner here. What you are watching is a 291k
parameter reactive policy with a 180-bin lidar view and four frames of history.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

DEFAULT_CHECKPOINT = Path("runs/body_v2_actor_prod/best.pt")

#: Consecutive uncertainty-aware confirmations before the localized driver
#: declares arrival. Matches the production evaluator; a single in-tolerance
#: sample is noise, and on hardware nothing announces arrival.
ARRIVAL_CONFIRM_TICKS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--driver", choices=("pose", "localized"),
                        default="pose")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="initial sensor and odometry corruption tier; the "
                             "panel's selector owns it thereafter")
    parser.add_argument("--num-layouts", type=int, default=32,
                        help="rooms in the pool the buttons draw from")
    parser.add_argument("--layout-seed", type=int, default=20_000_000,
                        help="held out from every collection and training seed")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="wall-clock multiple of real time; 0 runs free")
    parser.add_argument("--robot-usd", type=Path,
                        default=Path("assets/turtlebot4_lite_visual.usd"),
                        help="robot asset; defaults to the one carrying visual "
                             "meshes, which only a viewer needs")
    # Localizer settings, defaulting to the configuration the production
    # evaluation reported. They are exposed because the tracker is the part of
    # the stack still being tuned, not because a viewer should retune it.
    parser.add_argument("--prior-source",
                        choices=("create3-ekf", "command-gyro"),
                        default="create3-ekf",
                        help="'create3-ekf' synthesizes the Create3's odometry "
                             "increment from achieved motion plus encoder "
                             "noise, which is the only quantity in the "
                             "localized driver derived from simulator truth; "
                             "'command-gyro' uses nothing but the command and "
                             "the gyro, and is the configuration that missed "
                             "the localization gate")
    parser.add_argument("--correction-gain", type=float, default=0.30)
    parser.add_argument("--yaw-correction-gain", type=float, default=0.50)
    parser.add_argument("--arrival-radius-m", type=float, default=0.15)
    parser.add_argument("--max-submap-scans", type=int, default=128)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true",
                        help="load an actor from a superseded sensor contract")
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def pacing_delay(elapsed_s: float, control_period_s: float,
                 speed: float) -> float:
    """Seconds to sleep so one control step occupies its real duration.

    A single environment on a 4090 simulates far faster than real time, and a
    robot crossing a room in two seconds is unreadable -- worse, it looks like a
    different, twitchier policy than the one that will run at 20 Hz on the
    hardware. ``speed <= 0`` means "as fast as it will go", for which the answer
    is no delay rather than an infinite one.
    """
    if speed <= 0.0:
        return 0.0
    return max(0.0, control_period_s / speed - elapsed_s)


def body_offset(delta_xy, yaw: float) -> np.ndarray:
    """A world-frame offset rotated into a frame with the given heading."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([cos * delta_xy[0] + sin * delta_xy[1],
                     -sin * delta_xy[0] + cos * delta_xy[1]])


def world_offset(body_xy, yaw: float) -> np.ndarray:
    """Inverse of :func:`body_offset`: a body-frame offset back into world."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([cos * body_xy[0] - sin * body_xy[1],
                     sin * body_xy[0] + cos * body_xy[1]])


# ─────────────────────────────────────────────────────────────────────────────
# Interface state
#
# Every UI callback fires inside ``app.update()``, which this process reaches
# from inside ``env.step()`` -- so a callback that touched the environment would
# be re-entering a half-stepped simulator. They queue; the loop applies.
# ─────────────────────────────────────────────────────────────────────────────

class Controls:
    """What the person asked for since the last control tick."""

    def __init__(self, noise_scale: float):
        self.clicked_world: np.ndarray | None = None
        self.reset = False
        self.randomize = False
        self.new_layout = False
        self.paused = False
        self.noise_scale = noise_scale

    def take_click(self) -> np.ndarray | None:
        point, self.clicked_world = self.clicked_world, None
        return point


class ViewportPicker:
    """Left-click in the viewport into a world-space point on the floor.

    An invisible ``sc.Screen`` covering the viewport reports the click in the
    viewport's normalized coordinates; the render query turns that pixel into
    the world position of whatever surface it landed on. Reading the floor's
    actual hit position rather than intersecting a nominal ground plane means a
    click on top of an obstacle reports the obstacle, which is a goal the caller
    can then refuse instead of a floor point hidden underneath it.
    """

    def __init__(self, on_point):
        from omni.kit.viewport.utility import get_active_viewport_window
        from omni.kit.viewport.window.raycast import perform_raycast_query
        from omni.ui import scene as sc

        window = get_active_viewport_window()
        if window is None:                           # pragma: no cover
            raise RuntimeError("no viewport window; is this running headless?")
        self._raycast = perform_raycast_query
        self._viewport = window.viewport_api
        self._on_point = on_point
        # Held on the instance because omni.ui.scene objects are destroyed with
        # their last Python reference, and a dropped Screen stops reporting
        # clicks with no error anywhere.
        with window.get_frame("sim2edge.interactive"):
            self._view = sc.SceneView()
            with self._view.scene:
                self._screen = sc.Screen(
                    gesture=sc.ClickGesture(self._clicked, mouse_button=0))

    def _clicked(self, *args) -> None:
        payload = None
        for candidate in args:
            payload = getattr(candidate, "gesture_payload", None)
            if payload is not None:
                break
        if payload is None:
            payload = self._screen.gesture_payload
        mouse = payload.mouse
        pixel, viewport = self._viewport.map_ndc_to_texture_pixel(mouse)
        if viewport is None:                         # click fell outside
            return
        self._raycast(self._viewport, mouse, pixel, self._resolved,
                      query_name="sim2edge.interactive.goal")

    def _resolved(self, prim_path, world_position, *args) -> None:
        # A miss reports an empty path and, on the RTX path, the origin -- which
        # is a *legal-looking* arena corner. Testing the position would accept
        # it; testing the path is the only thing that distinguishes a click on
        # the floor from a click on the sky.
        if not prim_path or world_position is None:
            return
        self._on_point(np.asarray(world_position, dtype=np.float64),
                       str(prim_path))


class Panel:
    """The button and readout window. Writes flags, reads nothing."""

    def __init__(self, controls: Controls, driver_name: str):
        import omni.ui as ui

        self._controls = controls
        self._window = ui.Window("Sim2Edge — interactive navigation",
                                 width=380, height=430)
        self._labels: dict[str, object] = {}
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Left-click the floor to set a goal.",
                         style={"font_size": 16})
                ui.Label(f"driver: {driver_name}", style={"color": 0xFF9FD8A0})
                ui.Spacer(height=4)
                with ui.HStack(spacing=4, height=28):
                    ui.Button("Reset", clicked_fn=self._on_reset,
                              tooltip="Back to this episode's start pose")
                    ui.Button("Randomize robot", clicked_fn=self._on_randomize,
                              tooltip="New start pose in the same room")
                with ui.HStack(spacing=4, height=28):
                    ui.Button("New layout", clicked_fn=self._on_new_layout,
                              tooltip="Draw a different room from the pool")
                    self._pause = ui.Button("Pause", clicked_fn=self._on_pause)
                ui.Spacer(height=4)
                with ui.HStack(height=24):
                    ui.Label("sensor noise", width=110)
                    self._noise = ui.ComboBox(
                        _noise_index(controls.noise_scale),
                        "0x  clean", "1x  deployment", "2x  stress")
                ui.Spacer(height=6)
                for key, caption in (("state", "state"),
                                     ("goal", "goal"),
                                     ("distance", "distance"),
                                     ("belief", "believed"),
                                     ("command", "command"),
                                     ("layout", "layout"),
                                     ("rate", "rate")):
                    with ui.HStack(height=20):
                        ui.Label(caption, width=110,
                                 style={"color": 0xFF909090})
                        self._labels[key] = ui.Label("—")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _on_reset(self):
        self._controls.reset = True

    def _on_randomize(self):
        self._controls.randomize = True

    def _on_new_layout(self):
        self._controls.new_layout = True

    def _on_pause(self):
        self._controls.paused = not self._controls.paused
        self._pause.text = "Resume" if self._controls.paused else "Pause"

    # ── per-tick refresh ─────────────────────────────────────────────────────

    def selected_noise_scale(self) -> float:
        return float(self._noise.model.get_item_value_model().as_int)

    def refresh(self, readout: dict) -> None:
        for key, label in self._labels.items():
            label.text = str(readout.get(key, "—"))


def _noise_index(scale: float) -> int:
    return min(2, max(0, int(round(scale))))


# ─────────────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────────────

class PoseDriver:
    """Goal features from simulator truth: the actor, and only the actor.

    Sensors are still the real noisy swept and deskewed path -- the lidar half
    of the observation is exactly what any other consumer sees. Only the three
    goal numbers are privileged, which is the same isolation
    ``scripts/eval_bc.py`` measures the actor's ceiling with.
    """

    name = "pose  (goal from simulator truth)"
    shows_belief = False

    def __init__(self, env, actor, torch):
        from sim2edge.preprocess import goal_features
        from sim2edge.spec import GOAL_TOLERANCE_M
        self._env, self._actor, self._torch = env, actor, torch
        self._goal_features = goal_features
        self._tolerance = GOAL_TOLERANCE_M

    def respawn(self) -> None:
        pass

    def retarget(self) -> None:
        pass

    def step(self) -> dict:
        env, torch = self._env, self._torch
        pose = env._robot_pose()
        goal = env._goal_w - env.scene.env_origins[:, :2]
        observation = env._observation_history.observation(
            self._goal_features(pose, goal))
        with torch.inference_mode():
            command = self._actor(observation)
        distance = float(torch.linalg.vector_norm(goal[0] - pose[0, :2]))
        return {"command": command, "distance": distance,
                "believed_distance": distance, "sigma": 0.0,
                "believed_goal_local": None,
                "arrived": distance < self._tolerance}


class LocalizedDriver:
    """The deployable stack. No pose reaches the actor or the localizer.

    The goal is stored where the robot can keep it: as a point in the frame the
    tracker started in. Every tick the tracker's relative pose turns that back
    into a body-frame offset, and drift in the tracker moves the goal the robot
    is driving at -- which is the failure this viewer exists to make visible,
    and which a pose-fed run cannot show.

    A click reads true pose exactly once, to turn the clicked world point into
    an offset in the robot's frame, and then stores it in the tracker's frame
    and never consults truth again. That is the operator-with-a-map act, and it
    is the same status the goal has in ``eval_localized.py``, where
    ``initial_goal`` is likewise a body-frame offset taken from true pose at
    episode start. Nothing drives on it; drift after the click moves the goal
    the robot is steering at, which is the point.

    Under ``--prior-source create3-ekf`` exactly one *driving* quantity is
    derived from
    simulator truth: the odometry increment, built from achieved motion plus
    encoder noise because the simulator has no wheel-encoder model. Say what
    that costs, because it is not conservative. The simulator deliberately makes
    the base slip relative to its command, and an increment measured from
    achieved displacement *sees through* that slip -- where a real wheel encoder
    counts rotations and would report the commanded travel, missing it. So this
    prior carries additive noise but none of the drivetrain scale error that the
    tracker's ``motion_scale`` exists to estimate.

    ``--prior-source command-gyro`` is the truth-free configuration, and its
    three inputs are worth naming exactly: the lidar sweep, the gyro's yaw
    increment, and **the robot's own commanded forward speed**. The last is not
    a measurement -- ``LocalPoseTracker`` integrates
    ``commanded_speed * dt * motion_scale`` -- so it is blind to wheel slip in
    precisely the way a wheel encoder is, and ``motion_scale`` is the only thing
    that can recover it. That estimate comes from the scan match, so in this
    mode the lidar is carrying the entire translation correction alone. It is
    not the default because it is the arm that missed the localization gate,
    which is why the evidence-gated fallback to Create3 odometry was exercised.
    """

    shows_belief = True

    def __init__(self, env, actor, torch, args, generator):
        self.name = f"localized  ({args.prior_source} + scan matching)"
        from sim2edge.localization import LocalPoseTracker
        from sim2edge.noise import ODOM_XY_WALK_M
        from sim2edge.preprocess import build_observation, relative_goal_features
        from sim2edge.sensors import PolarSweep
        from sim2edge.spec import DT

        self._env, self._actor, self._torch, self._args = env, actor, torch, args
        self._build_observation = build_observation
        self._relative_goal_features = relative_goal_features
        self._sweep_type = PolarSweep
        self._odom_walk = ODOM_XY_WALK_M
        self._dt = DT
        self._generator = generator
        self._tracker = LocalPoseTracker(
            max_submap_scans=args.max_submap_scans,
            correction_gain=args.correction_gain,
            yaw_correction_gain=args.yaw_correction_gain,
            max_correction_m=0.04,
            max_correction_yaw_rad=math.radians(2.0),
            motion_scale_update_rate=0.10,
            motion_prior_source=args.prior_source)
        self._goal_in_start_frame = np.zeros(2)
        self._previous_truth = env.pose_local()
        self._travelled = 0.0
        self._confirmations = 0
        self._first = True

    def respawn(self) -> None:
        """A respawn is a new robot: the tracker's frame is gone with it."""
        self._tracker.reset()
        self._previous_truth = self._env.pose_local()
        self._travelled = 0.0
        self._first = True
        self.retarget()

    def retarget(self) -> None:
        env = self._env
        truth = env.pose_local()
        relative = body_offset(env.goal_local() - truth[:2], float(truth[2]))
        estimate = self._tracker.pose
        self._goal_in_start_frame = (
            estimate[:2] + world_offset(relative, float(estimate[2])))
        self._confirmations = 0

    def step(self) -> dict:
        env, torch, args = self._env, self._torch, self._args
        sweep = env.canonical_sweep().to("cpu")
        gyro = float(env.gyro_increment()[0, -1])
        speed = float(env._action[0, 0])
        truth = env.pose_local()

        prior = None
        if not self._first:
            # Path length feeds the systematic term in the arrival bound, so it
            # is accumulated whatever the prior is -- the bound must not quietly
            # tighten just because the prior changed.
            self._travelled += abs(speed) * self._dt
            if args.prior_source == "create3-ekf":
                world_delta = truth[:2] - self._previous_truth[:2]
                encoder = self._generator.normal(
                    0.0,
                    self._odom_walk * float(env.cfg.observation_noise_scale),
                    size=2)
                body = body_offset(world_delta, float(self._previous_truth[2]))
                prior = np.array([body[0] + encoder[0],
                                  body[1] + encoder[1], gyro])
        self._previous_truth = truth

        one = self._sweep_type(sweep.ranges[0], sweep.bearings[0],
                               sweep.ages[0], sweep.valid[0])
        estimate = self._tracker.update(
            one, 0.0 if self._first else gyro,
            0.0 if self._first else speed, prior_delta_body=prior)
        self._first = False

        pose = estimate.relative_pose
        goal_body = body_offset(self._goal_in_start_frame - pose[:2],
                                float(pose[2]))
        observation = self._build_observation(
            env._observation_history.lidar,
            self._relative_goal_features(
                torch.from_numpy(goal_body.astype(np.float32))
                     .to(env.device).unsqueeze(0)),
            env._observation_history.action)
        with torch.inference_mode():
            command = self._actor(observation)

        # The declared-arrival test the evaluator uses: the *bound* on the
        # remaining distance has to fit inside the arrival radius, and it has to
        # do so several ticks running. The systematic scale term is carried
        # separately from the matcher's covariance because it grows with
        # distance travelled rather than as a random walk.
        largest = float(np.linalg.eigvalsh(estimate.covariance[:2, :2])[-1])
        sigma = math.sqrt(max(0.0, largest) + (0.01 * self._travelled) ** 2)
        believed = float(np.linalg.norm(goal_body))
        if believed + 1.96 * sigma <= args.arrival_radius_m:
            self._confirmations += 1
        else:
            self._confirmations = 0

        truth_goal = env.goal_local()
        return {
            "command": command,
            "distance": float(np.linalg.norm(truth_goal - truth[:2])),
            "believed_distance": believed,
            "sigma": sigma,
            # Where the robot thinks the goal is, drawn in the world we can see.
            "believed_goal_local": truth[:2] + world_offset(goal_body,
                                                            float(truth[2])),
            "arrived": self._confirmations >= ARRIVAL_CONFIRM_TICKS,
        }


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    # Isaac's native modules become importable only after AppLauncher starts.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=False).app

    import torch

    from isaaclab_env.interactive import InteractiveNavEnv, frame_camera, snap_goal
    from isaaclab_env.nav_task import NavEnvCfg, normalize_action
    from sim2edge.spec import DT
    from sim2edge.training import load_policy_checkpoint

    sys.stdout.reconfigure(line_buffering=True)

    config = NavEnvCfg()
    config.scene.num_envs = 1
    config.num_layouts = args.num_layouts
    config.layout_seed = args.layout_seed
    config.seed = args.layout_seed
    config.collection_mode = False
    config.observation_noise_scale = args.noise_scale
    # Nothing about a viewer is episodic: the person decides when the run is
    # over. `_get_dones` withholds termination and this withholds truncation, so
    # the only thing that moves the robot is a button or a click.
    config.terminate_on_goal = False
    config.episode_length_s = 24.0 * 3600.0

    # The task's default asset predates the `package://` fix in
    # `tools/convert_tb4.py`, so its only visuals are the URDF's primitive
    # wheels -- the body, bumper, lidar and camera meshes are absent. Every
    # other consumer is indifferent (collision is primitives, and the ray-cast
    # lidar sees obstacles, never the robot), which is exactly why it went
    # unnoticed; a viewer is the one consumer that cares. Prefer the asset with
    # meshes and say so, rather than silently rendering a pair of wheels.
    if args.robot_usd.exists():
        config.robot_cfg.spawn.usd_path = str(args.robot_usd)
    else:
        print(f"[interactive] {args.robot_usd} is missing; falling back to "
              f"{config.robot_cfg.spawn.usd_path}, whose body has no visual "
              f"meshes. Rebuild it with tools/convert_tb4.py.")

    print(f"[interactive] building 1 environment, {args.num_layouts} layouts "
          f"from held-out seed {args.layout_seed} ...")
    env = InteractiveNavEnv(config)
    device = torch.device(env.device)
    actor = load_policy_checkpoint(
        args.checkpoint, device=device,
        allow_legacy_sensor_model=args.allow_legacy_checkpoint)
    print(f"[interactive] actor: {sum(p.numel() for p in actor.parameters()):,} "
          f"parameters from {args.checkpoint}")

    env.reset()
    env._gyro.scale = args.noise_scale
    env._odometry_drift.scale = args.noise_scale
    spawn = (int(env._layout[0]), int(env._start_id[0]),
             float(env.pose_local()[2]))

    generator = np.random.default_rng(args.layout_seed + 71)
    driver = (PoseDriver(env, actor, torch) if args.driver == "pose"
              else LocalizedDriver(env, actor, torch, args, generator))
    driver.respawn()

    controls = Controls(args.noise_scale)

    def on_click(world_position, prim_path):
        controls.clicked_world = world_position

    # The viewport window exists only once the app has drawn a frame or two.
    for _ in range(4):
        simulation_app.update()
    picker = ViewportPicker(on_click)
    panel = Panel(controls, driver.name)
    _aim_camera(env, frame_camera)

    print("[interactive] left-click the floor to set a goal; "
          "the panel resets and randomizes.")

    state = "driving"
    message = ""
    steps = 0
    zero = torch.zeros(1, 2, device=device)
    measured_rate = 0.0

    while simulation_app.is_running():
        tick_started = time.perf_counter()

        # ── what the person asked for ────────────────────────────────────────
        scale = panel.selected_noise_scale()
        if scale != controls.noise_scale:
            controls.noise_scale = scale
            env.cfg.observation_noise_scale = scale
            env._gyro.scale = scale
            env._odometry_drift.scale = scale

        respawned = False
        if controls.new_layout:
            controls.new_layout = False
            env.respawn()
            spawn = (int(env._layout[0]), int(env._start_id[0]),
                     float(env.pose_local()[2]))
            respawned = True
        elif controls.randomize:
            controls.randomize = False
            env.respawn(layout=spawn[0])
            spawn = (spawn[0], int(env._start_id[0]),
                     float(env.pose_local()[2]))
            respawned = True
        elif controls.reset:
            controls.reset = False
            env.respawn(layout=spawn[0], start_id=spawn[1], yaw=spawn[2])
            respawned = True

        if respawned:
            _aim_camera(env, frame_camera)
            driver.respawn()
            state, message, steps = "driving", "", 0

        clicked = controls.take_click()
        if clicked is not None:
            room = env.room
            local = clicked[:2] - env.env_origin()
            goal = snap_goal(local, room.free, room.res)
            if goal is None:
                message = "that point is not reachable"
            else:
                env.set_goal(goal)
                driver.retarget()
                state, message, steps = "driving", "", 0

        # ── one control step ─────────────────────────────────────────────────
        driving = state == "driving" and not controls.paused
        readout = driver.step()
        command = readout["command"] if driving else zero
        env.step(normalize_action(command))
        steps += int(driving)

        collided = bool(env._collided[0])
        if driving and collided:
            state, message = "collided", "reset or click a new goal"
        elif driving and readout["arrived"]:
            state = "arrived"
            message = (f"{readout['distance'] * 100:.0f} cm from the goal"
                       if args.driver == "localized" else "")

        # ── draw ─────────────────────────────────────────────────────────────
        believed = readout["believed_goal_local"] if driver.shows_belief else None
        env.show_markers(env.goal_local() + env.env_origin(),
                         None if believed is None
                         else believed + env.env_origin())

        physical = command[0].detach().cpu().numpy()
        room = env.room
        panel.refresh({
            "state": f"{state}  {message}".strip() + (
                "  (paused)" if controls.paused else ""),
            "goal": "{:.2f}, {:.2f} m".format(*env.goal_local()),
            "distance": f"{readout['distance']:.2f} m true",
            "belief": (f"{readout['believed_distance']:.2f} m "
                       f"± {readout['sigma'] * 100:.0f} cm"
                       if driver.shows_belief else "same as true"),
            # Kit's UI font has no omega, and a missing glyph draws as "?".
            "command": f"v {physical[0]:.2f} m/s   yaw {physical[1]:+.2f} rad/s",
            "layout": (f"#{int(env._layout[0])}  "
                       f"{room.size_xy[0]:.1f} × {room.size_xy[1]:.1f} m"),
            "rate": f"{measured_rate:.1f} Hz   {steps} steps",
        })

        elapsed = time.perf_counter() - tick_started
        delay = pacing_delay(elapsed, DT, args.speed)
        if delay > 0.0:
            time.sleep(delay)
        measured_rate = 1.0 / max(time.perf_counter() - tick_started, 1e-6)

    simulation_app.close()


def _aim_camera(env, frame_camera) -> None:
    """Point the viewport at the middle of the arena the robot is actually in."""
    origin = env.env_origin()
    eye, lookat = frame_camera(env.room.size_xy)
    env.sim.set_camera_view(
        (eye[0] + origin[0], eye[1] + origin[1], eye[2]),
        (lookat[0] + origin[0], lookat[1] + origin[1], lookat[2]))


if __name__ == "__main__":
    main()
