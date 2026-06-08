from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('ros2_package'),
        'config',
        'turtle_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            output='screen'
        ),
        Node(
            package='ros2_package',
            executable='turtle_circle',
            name='turtle_circle',
            output='screen',
            parameters=[config_file]
        )
    ])
