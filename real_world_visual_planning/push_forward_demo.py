"""
Push Forward Demo: FrankaPanda Push Action

This script demonstrates a simple push forward action:
1. Move to pre-push pose
2. Push forward along z-axis

Before running:
1. Make sure the perception pipeline is running:
   python -m frankapanda.perception.perception_pipeline --continuous

2. Make sure the robot is connected and powered on

Usage:
    python push_forward_demo.py
"""

import numpy as np
import torch
from robo_utils.conversion_utils import move_pose_along_local_z, pose_to_transformation
from robo_utils.visualization.plotting import plot_pcd, make_gripper_visualization

# Import from frankapanda package
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner


def visualize_poses_in_pointcloud(pcd, poses, rgb=None, colors=None):
    """Visualize multiple gripper poses in a point cloud."""
    if colors is None:
        colors = [(1, 0, 0)] * len(poses)

    combined_pcd = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else None

    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        gripper_transform = pose_to_transformation(pose, format='wxyz')
        gripper_points, gripper_colors = make_gripper_visualization(
            rotation=gripper_transform[:3, :3],
            translation=gripper_transform[:3, 3],
            length=0.05,
            density=50,
            color=colors[i % len(colors)]
        )
        combined_pcd = np.vstack([combined_pcd, gripper_points])
        if combined_rgb is not None:
            combined_rgb = np.vstack([combined_rgb, gripper_colors])

    if combined_rgb is not None:
        plot_pcd(combined_pcd, combined_rgb, base_frame=True)
    else:
        plot_pcd(combined_pcd, base_frame=True)


def main():

    # Initialize robot controller
    controller = FrankaPandaController()

    # Initialize perception pipeline client
    perception = PerceptionPipeline(publish_port=1235, timeout_ms=10000)

    # Capture point cloud
    print("Capturing point cloud from dual cameras...")
    try:
        pcd, rgb = perception.get_point_cloud()
    except TimeoutError as e:
        print("Make sure the perception pipeline is running!")
        return

    controller.close_gripper(num_steps=80)
    controller.move_to_joints(controller.home_joints, controller.close_gripper_action)

    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device="cuda:0")

    # Initialize motion planner
    motion_planner = MotionPlanner(pcd)

    # Pre-push and push poses
    pre_push_pose = torch.tensor(
        [0.519, -0.25, 0.35, -0.5, 0.5, 0.5, 0.5],
        dtype=torch.float32,
        device="cuda:0"
    )
    push_pose = move_pose_along_local_z(pre_push_pose, 0.33)
    push_pose = torch.tensor(push_pose, dtype=torch.float32, device="cuda:0")

    # Visualize poses: red=pre_push, green=push
    print("Visualizing poses: red=pre_push, green=push")
    visualize_poses_in_pointcloud(
        pcd,
        [pre_push_pose, push_pose],
        rgb=rgb,
        colors=[(1, 0, 0), (0, 1, 0)]
    )

    # Plan to pre-push pose
    pre_push_trajectories, pre_push_success = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=pre_push_pose.unsqueeze(0),
    )
    print(f"Pre-push planning success: {pre_push_success.item()}")

    # Plan push motion
    push_trajectories, push_success = motion_planner.plan_to_goal_poses(
        current_joints=pre_push_trajectories[0, -1].unsqueeze(0),
        goal_poses=push_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[:],
        plan_config=motion_planner.along_z_axis_plan_config
    )
    print(f"Push planning success: {push_success.item()}")

    success = pre_push_success.item() & push_success.item()

    if success:
        controller.move_along_trajectory(pre_push_trajectories[0].cpu().numpy(), controller.close_gripper_action)
        controller.move_along_trajectory(push_trajectories[0].cpu().numpy(), controller.close_gripper_action)

    controller.move_to_joints(controller.home_joints, controller.close_gripper_action)

    perception.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
