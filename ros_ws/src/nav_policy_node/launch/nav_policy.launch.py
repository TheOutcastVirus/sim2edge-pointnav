"""Launch the policy node and the safety layer together.

They are always launched as a pair. The policy node does not publish to
/cmd_vel; nav_safety does. Launching the policy alone moves nothing, which is
deliberate -- there is no configuration in which the learned policy drives the
motors unsupervised.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy_share = get_package_share_directory('nav_policy_node')
    safety_share = get_package_share_directory('nav_safety')

    args = [
        DeclareLaunchArgument('backend', default_value='npu',
                              description="'npu' (QNN/HTP) or 'cpu' (XNNPACK)"),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('goal_x', default_value='3.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument(
            'publish_cmd_vel', default_value='true',
            description='false = dry run; computes commands but does not move'),
    ]

    policy = Node(
        package='nav_policy_node',
        executable='nav_policy_node',
        name='nav_policy',
        output='screen',
        parameters=[
            os.path.join(policy_share, 'config', 'nav_policy.yaml'),
            {
                'backend': LaunchConfiguration('backend'),
                'model_path': LaunchConfiguration('model_path'),
                'goal_x': LaunchConfiguration('goal_x'),
                'goal_y': LaunchConfiguration('goal_y'),
            },
        ],
    )

    safety = Node(
        package='nav_safety',
        executable='nav_safety_node',
        name='nav_safety',
        output='screen',
        parameters=[
            os.path.join(safety_share, 'config', 'nav_safety.yaml'),
            {'publish_cmd_vel': LaunchConfiguration('publish_cmd_vel')},
        ],
    )

    return LaunchDescription(args + [policy, safety])
