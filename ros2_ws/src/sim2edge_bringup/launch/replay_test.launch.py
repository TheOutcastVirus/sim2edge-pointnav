"""The on-board test: drive the real node graph from a recorded corpus.

No lidar, no OAK-D, no Create 3. The replay node publishes ``/scan``, ``/imu``
and ``/odom`` exactly as the drivers would, the navigator and safety layer come
up unmodified, and what is being tested is the thing that will run on the robot
rather than a harness that resembles it.

    ros2 launch sim2edge_bringup replay_test.launch.py \\
        replay:=$HOME/sim2edge-ventuno/ros2_ws/replays/validation_12.s2e \\
        model:=$HOME/sim2edge-ventuno/ros2_ws/models/actor_qnn.pte

What this answers: does the graph come up, do the topics and QoS match, does
the 20 Hz timer hold under real DDS load, does the NPU deliver a command inside
the period, and does the safety layer see the scan it thinks it sees.

What it does **not** answer: whether the numbers agree with Python -- that is
``sim2edge_parity``, which compares against the recorded reference -- and
whether the robot navigates, which nothing offline can answer. The corpus was
driven by the oracle, so the recorded pose does not respond to what the policy
commands. This is an input/output test, and calling it anything else would be
claiming a sim-to-real result that has not been earned.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    replay = LaunchConfiguration("replay")
    model = LaunchConfiguration("model")
    backend = LaunchConfiguration("backend")
    qnn_lib_dir = LaunchConfiguration("qnn_lib_dir")
    rate_scale = LaunchConfiguration("rate_scale")
    loop = LaunchConfiguration("loop")
    use_safety = LaunchConfiguration("use_safety")
    nav_cpus = LaunchConfiguration("nav_cpus")

    navigator_config = PathJoinSubstitution(
        [FindPackageShare("sim2edge_nodes"), "config", "navigator.yaml"])
    safety_config = PathJoinSubstitution(
        [FindPackageShare("sim2edge_nodes"), "config", "safety.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("replay",
            description="path to a .s2e replay from tools/export_replay.py"),
        DeclareLaunchArgument("model", default_value="",
            description="actor .pte; empty runs perception and localization "
                        "only, which is the preprocessing bring-up check"),
        DeclareLaunchArgument("backend", default_value="qnn",
            description="qnn | cpu"),
        DeclareLaunchArgument("qnn_lib_dir", default_value="",
            description="directory holding libQnnHtp.so. Must be the "
                        "aarch64-oe-linux-gcc11.2 variant: the gcc9.4 one "
                        "ships no V75 stub and rejects this SoC"),
        DeclareLaunchArgument("rate_scale", default_value="1.0",
            description="replay speed. Above 1.0 the loop is deliberately "
                        "over-driven, which is how you find out whether it "
                        "holds; latency figures from such a run are not "
                        "representative of the robot"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("nav_cpus", default_value="2,3",
            description="CPU affinity for the navigator, as a taskset list. "
                        "This is NOT tuning -- it decides whether the control "
                        "loop closes. The Ventuno is 4x Cortex-A78C (cpu0-3) "
                        "plus 4x Cortex-A55 (cpu4-7), and the localizer costs "
                        "6.98 ms pinned to cpu2 against 53.6 ms pinned to "
                        "cpu6: 7.7x, against a clock ratio of only 1.2x. The "
                        "scan-to-submap search is memory-bound "
                        "nearest-neighbour work and the in-order A55 is its "
                        "worst case. Left to the scheduler, the loop misses "
                        "its 50 ms period whenever it migrates. cpu2-3 are the "
                        "fastest pair at 2.36 GHz. Empty disables pinning"),
        DeclareLaunchArgument("use_safety", default_value="true",
            description="also run the safety layer, so /cmd_vel is exercised. "
                        "Nothing is connected to it here"),

        LogInfo(msg=["replay test: ", replay, "  backend=", backend]),

        Node(
            package="sim2edge_nodes",
            executable="navigator_node",
            name="sim2edge_navigator",
            output="screen",
            emulate_tty=True,
            prefix=["taskset -c ", nav_cpus, " "],
            parameters=[navigator_config, {
                "model_path": model,
                "backend": backend,
                "qnn_lib_dir": qnn_lib_dir,
                # The replay publishes odometry, and requiring it here means a
                # gap in the replay shows up as a hold rather than being
                # papered over.
                "require_odom": True,
            }],
        ),

        Node(
            package="sim2edge_nodes",
            executable="safety_node",
            name="sim2edge_safety",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(use_safety),
            parameters=[safety_config],
        ),

        Node(
            package="sim2edge_nodes",
            executable="replay_node",
            name="sim2edge_replay",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "replay_path": replay,
                "rate_scale": rate_scale,
                "loop": loop,
            }],
        ),
    ])
