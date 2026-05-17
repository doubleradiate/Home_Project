from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('Semanti_Map'),
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='Semanti_Map',
            executable='semantic_map_node',
            name='semantic_map_node',
            output='screen',
            parameters=[config],
        ),
    ])
