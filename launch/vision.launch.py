from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    enable_ram = LaunchConfiguration("enable_ram")
    enable_grounding_dino = LaunchConfiguration("enable_grounding_dino")
    enable_bbox_center_3d = LaunchConfiguration("enable_bbox_center_3d")
    enable_bbox_all_3d_marker = LaunchConfiguration("enable_bbox_all_3d_marker")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time"
        ),

        DeclareLaunchArgument(
            "enable_ram",
            default_value="true",
            description="Launch RAM node"
        ),

        DeclareLaunchArgument(
            "enable_grounding_dino",
            default_value="true",
            description="Launch GroundingDINO node"
        ),

        DeclareLaunchArgument(
            "enable_bbox_center_3d",
            default_value="true",
            description="Launch bbox center 3D node"
        ),

        DeclareLaunchArgument(
            "enable_bbox_all_3d_marker",
            default_value="true",
            description="Launch bbox all 3D marker node"
        ),

        Node(
            package="vision_package",
            executable="ram_node",
            name="ram_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time}
            ],
            condition=IfCondition(enable_ram),
        ),

        Node(
            package="vision_package",
            executable="grounding_dino_node",
            name="grounding_dino_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time}
            ],
            condition=IfCondition(enable_grounding_dino),
        ),

        Node(
            package="vision_package",
            executable="bbox_center_3d_node",
            name="bbox_center_3d_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time}
            ],
            condition=IfCondition(enable_bbox_center_3d),
        ),

        Node(
            package="vision_package",
            executable="bbox_all_3d_marker_node",
            name="bbox_all_3d_marker_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time}
            ],
            condition=IfCondition(enable_bbox_all_3d_marker),
        ),
    ])