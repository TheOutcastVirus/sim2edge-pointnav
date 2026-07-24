"""Launch the safety layer alone.

Useful for bench-testing the watchdog and the bubble against a recorded or
hand-published command stream, without the policy or the NPU involved.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('nav_safety')
    return LaunchDescription([
        DeclareLaunchArgument('publish_cmd_vel', default_value='true'),
        Node(
            package='nav_safety',
            executable='nav_safety_node',
            name='nav_safety',
            output='screen',
            parameters=[
                os.path.join(share, 'config', 'nav_safety.yaml'),
                {'publish_cmd_vel': LaunchConfiguration('publish_cmd_vel')},
            ],
        ),
    ])
