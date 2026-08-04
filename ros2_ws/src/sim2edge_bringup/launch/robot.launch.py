"""The whole robot: lidar, IMU, navigator, safety.

    ros2 launch sim2edge_bringup robot.launch.py \\
        model:=$HOME/sim2edge-ventuno/ros2_ws/models/actor_qnn.pte

Three inputs reach the policy and no others: the **lidar** sweep, the **gyro**
from the OAK-D, and the base's **velocity** from the Create 3. There is no map,
no external pose, and no camera in this path. Somebody points at a goal once --
``/goal_pose``, as an offset in the robot's body frame -- and after that the
robot drives on its own estimate, with the gap between where it believes the
goal is and where the goal actually is being the localization error.

The Create 3 link is a USB gadget: the board is the USB *device* and the base
is the host, over the board's USB-C port. Its DDS discovery has to be pinned to
``usb0`` or Fast DDS joins multicast on the Wi-Fi interface and never hears the
base -- see ``config/fastdds_usb0.xml`` and the reference project's
``ventuno-setup`` skill. This launch does **not** start ``create3_republisher``:
this base runs Iron at the root namespace and subscribes to ``/cmd_vel``
directly.

The lidar and the OAK-D have both been run on this board -- see
``docs/daily/2026_08_04_hardware_log.md`` for what that changed. The A1 measures
**7.3 Hz**, not the 5.5 Hz nominal in ``spec.py``, and delivers 1080 bins of
which ~855 carry a return. Deskew reads ``time_increment`` from the message and
is unaffected; the constant matters to training data, not to this launch.

The gyro's yaw axis is **y** on this mount, not the ``z`` the code read for its
whole life before hardware — measured by turning the robot, 100.9 deg on y
against 3.3 on z. It is ``gyro_yaw_axis`` in ``navigator.yaml``; re-mount the
OAK-D and re-measure it with ``tools/probe_yaw_axis.py``.

Still untested against hardware: the Create 3. No base has ever been connected,
so ``/odom`` and ``/cmd_vel`` are unverified, and without one the navigator
holds on ``no odometry yet`` by design.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    model = LaunchConfiguration("model")
    backend = LaunchConfiguration("backend")
    qnn_lib_dir = LaunchConfiguration("qnn_lib_dir")
    serial_port = LaunchConfiguration("serial_port")
    use_lidar = LaunchConfiguration("use_lidar")
    use_imu = LaunchConfiguration("use_imu")
    use_odometry_prior = LaunchConfiguration("use_odometry_prior")
    nav_cpus = LaunchConfiguration("nav_cpus")

    navigator_config = PathJoinSubstitution(
        [FindPackageShare("sim2edge_nodes"), "config", "navigator.yaml"])
    safety_config = PathJoinSubstitution(
        [FindPackageShare("sim2edge_nodes"), "config", "safety.yaml"])
    imu_config = PathJoinSubstitution(
        [FindPackageShare("oak_imu"), "config", "oak_imu.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("model", default_value="",
            description="actor .pte, lowered for this SoC"),
        DeclareLaunchArgument("backend", default_value="qnn"),
        DeclareLaunchArgument("qnn_lib_dir", default_value="",
            description="must be the aarch64-oe-linux-gcc11.2 QNN libraries; "
                        "the gcc9.4 set ships no V75 stub for this silicon"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0",
            description="the RPLIDAR A1's serial device"),
        DeclareLaunchArgument("use_lidar", default_value="true"),
        DeclareLaunchArgument("use_imu", default_value="true"),
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
        DeclareLaunchArgument("use_odometry_prior", default_value="false",
            description="false keeps the motion prior to gyro + velocity, "
                        "which is the self-contained arm. true uses the "
                        "Create 3's own odometry increment -- stronger, but "
                        "the run is then no longer lidar/IMU-only evidence"),

        LogInfo(msg="sim2edge robot bringup: lidar + gyro + base velocity"),

        # ── Sensors ───────────────────────────────────────────────────────────
        GroupAction(condition=IfCondition(use_lidar), actions=[
            Node(
                package="sllidar_ros2",
                executable="sllidar_node",
                name="sllidar",
                output="screen",
                parameters=[{
                    "serial_port": serial_port,
                    "serial_baudrate": 115200,      # A1
                    "frame_id": "laser_frame",
                    "inverted": False,
                    "angle_compensate": True,
                    # Sensitivity, measured on this unit: 8 kHz sampling, 1080
                    # bins per revolution at 0.334 deg, ~79% of them carrying a
                    # return, at 7.32 Hz. The driver *reports* "scan frequency:
                    # 10.0 Hz" for the mode and that figure is not what the
                    # motor does -- 8000 samples/s over 1080 bins is 7.4 rev/s,
                    # which is what arrives.
                    #
                    # It reports time_increment as 121 us (nonzero), so deskew
                    # uses real per-beam timing and never reaches its
                    # LIDAR_SPIN_HZ fallback. That is why this whole path is
                    # indifferent to the 5.5 Hz constant being wrong.
                    "scan_mode": "Sensitivity",
                }],
            ),
        ]),

        GroupAction(condition=IfCondition(use_imu), actions=[
            Node(
                package="oak_imu",
                executable="oak_imu_node",
                name="oak_imu",
                output="screen",
                parameters=[imu_config],
            ),
        ]),

        # ── The stack ─────────────────────────────────────────────────────────
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
                "use_odometry_prior": use_odometry_prior,
            }],
            # /odom and /cmd_vel are at the root namespace because that is
            # where this Create 3's Iron firmware publishes and subscribes.
            remappings=[("odom", "/odom")],
        ),

        # A separate process on purpose. It owns /cmd_vel, and it keeps
        # stopping the robot if the navigator crashes -- which it cannot do if
        # the thing that crashed was also the only publisher.
        Node(
            package="sim2edge_nodes",
            executable="safety_node",
            name="sim2edge_safety",
            output="screen",
            emulate_tty=True,
            parameters=[safety_config],
            remappings=[("cmd_vel", "/cmd_vel")],
        ),
    ])
