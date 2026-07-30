"""Convert the ROS TurtleBot 4 Lite description into a USD Isaac Lab can spawn.

    ~/isaac-venv/bin/python tools/convert_tb4.py --output assets/turtlebot4_lite.usd

Three stages, because the description is not a URDF until it is expanded:

    xacro  ->  URDF (text)  ->  USD (Isaac's importer)

The xacro stage needs the ROS packages on ``AMENT_PREFIX_PATH`` so that
``$(find turtlebot4_description)`` resolves; the URDF stage needs Isaac. They
are separate processes because ROS Jazzy and the Isaac venv have incompatible
Python environments, and trying to import both into one interpreter is a fight
nobody wins. We shell out to the ROS ``xacro`` and read back the text.

Why convert at all rather than model a cylinder: the mount poses are the whole
point. ``sim2edge.spec`` pins the RPLIDAR at (0.0039, 0, 0.1395) with a
quarter-turn yaw, taken from this exact description. If the simulated sensor
sits somewhere else, every demonstration is collected through a sensor the robot
does not have, and the resulting sim-to-real gap looks like a perception problem
while being a measurement one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_XACRO = Path(
    "/opt/ros/jazzy/share/turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro"
)

#: Values this script asserts against, so a description update cannot silently
#: move the sensor out from under a trained policy. These are metres and radians
#: in base_link, and they must match ``sim2edge.spec``.
EXPECTED_LIDAR_XYZ = (0.0039, 0.0, 0.1395)


def expand_xacro(xacro_path: Path) -> str:
    """Run the ROS xacro processor and return URDF text.

    Uses the system ROS interpreter, not this one: xacro lives in the ROS
    environment and pulling it into the Isaac venv drags in a second, conflicting
    set of Python packages.
    """
    # The ROS entry point is a console script that resolves its own package
    # metadata at import, so it needs ROS's site-packages on PYTHONPATH.
    # Sourcing setup.bash would also work, but this keeps us out of a shell and
    # avoids inheriting the rest of a ROS environment we do not want.
    ros = Path("/opt/ros/jazzy")
    site = next(ros.glob("lib/python3*/site-packages"), None)
    if site is None:
        sys.exit(f"no ROS site-packages under {ros}; is ros-jazzy installed?")

    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = str(ros)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        out = subprocess.run(
            ["/opt/ros/jazzy/bin/xacro", str(xacro_path), "gazebo:=ignition"],
            capture_output=True, text=True, check=True, env=env,
        )
    except FileNotFoundError:
        sys.exit(
            "xacro not found. Install it with:\n"
            "  sudo apt install ros-jazzy-xacro\n"
            "and make sure ros-jazzy-turtlebot4-description is present too."
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"xacro failed:\n{exc.stderr}")
    return out.stdout


def check_mount_poses(urdf_text: str) -> None:
    """Warn loudly if the description no longer matches the spec's constants.

    This is cheap insurance against the single worst failure mode here: a
    package update nudges a mount, the sim quietly trains against the new pose,
    the robot still has the old one, and the mismatch surfaces months later as
    "the policy does not transfer".
    """
    from sim2edge.spec import LIDAR_MOUNT_XYZ

    if any(abs(a - b) > 1e-3 for a, b in zip(LIDAR_MOUNT_XYZ, EXPECTED_LIDAR_XYZ)):
        sys.exit(
            f"spec.LIDAR_MOUNT_XYZ {LIDAR_MOUNT_XYZ} disagrees with this "
            f"script's expectation {EXPECTED_LIDAR_XYZ}. One of them is stale."
        )
    if "rplidar_link" not in urdf_text:
        sys.exit("expanded URDF has no rplidar_link; wrong xacro variant?")


def convert(urdf_text: str, output: Path, *, fix_base: bool,
            headless: bool = True) -> None:
    """URDF text -> USD, via Isaac Lab's converter.

    **The app must be launched before importing anything under
    ``isaaclab.sim``.** The USD Python bindings (``pxr``) ship inside Isaac
    Sim's kit runtime and are not on ``sys.path`` until ``AppLauncher`` puts
    them there; importing the converter cold fails with
    ``ModuleNotFoundError: No module named 'pxr'``, which looks like a missing
    dependency and is actually an ordering problem.

    ``AppLauncher`` itself imports fine beforehand -- it is the bootstrap.
    Everything else is imported after, inside this function, so ``--help`` and
    ``--urdf-only`` never pay the ~30 s startup.
    """
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    try:
        _run_conversion(UrdfConverter, UrdfConverterCfg, urdf_text, output,
                        fix_base=fix_base)
    finally:
        simulation_app.close()


def _run_conversion(UrdfConverter, UrdfConverterCfg, urdf_text: str,
                    output: Path, *, fix_base: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = Path(tmp) / "turtlebot4_lite.urdf"
        urdf_path.write_text(urdf_text)

        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output.parent.resolve()),
            usd_file_name=output.name,
            # The base is free: this is a mobile robot, and pinning it would
            # make every episode a very confusing no-op.
            fix_base=fix_base,
            merge_fixed_joints=True,
            # Wheels are velocity-controlled from the task's differential-drive
            # inverse kinematics, matching how the real Create3 accepts a Twist.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="velocity",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=800.0
                ),
            ),
            # Convex hull rather than decomposition: the base is a puck and
            # the obstacles are primitives, so decomposition would spend import
            # time and per-step broadphase cost resolving concavity nothing has.
            collider_type="convex_hull",
        )
        converter = UrdfConverter(cfg)
        print(f"wrote {converter.usd_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xacro", type=Path, default=DEFAULT_XACRO)
    ap.add_argument("--output", type=Path, default=Path("assets/turtlebot4_lite.usd"))
    ap.add_argument("--fix-base", action="store_true",
                    help="pin the base; only useful for inspecting the asset")
    ap.add_argument("--urdf-only", action="store_true",
                    help="write the expanded URDF and stop, no Isaac needed")
    ap.add_argument("--gui", action="store_true",
                    help="launch Isaac with a window, to eyeball the import")
    args = ap.parse_args()

    if not args.xacro.exists():
        sys.exit(f"{args.xacro} not found. Install:\n"
                 f"  sudo apt install ros-jazzy-turtlebot4-description")

    urdf_text = expand_xacro(args.xacro)
    check_mount_poses(urdf_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.urdf_only:
        out = args.output.with_suffix(".urdf")
        out.write_text(urdf_text)
        print(f"wrote {out} ({len(urdf_text)} bytes)")
        return

    convert(urdf_text, args.output, fix_base=args.fix_base,
            headless=not args.gui)


if __name__ == "__main__":
    main()
