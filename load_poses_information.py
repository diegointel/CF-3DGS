import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def load_saved_files(result_path, epoch):
    # Load the pose file
    pose_file_path = f"{result_path}/pose/ep{epoch:02d}_init.pth"
    pose_dict = torch.load(pose_file_path)

    # Load the checkpoint file
    checkpoint_file_path = f"{result_path}/chkpnt/ep{epoch:02d}_init.pth"
    gaussian_state = torch.load(checkpoint_file_path)

    return pose_dict, gaussian_state

def extract_pose_information(pose_dict):
    poses_pred = pose_dict["poses_pred"]
    poses_gt = pose_dict["poses_gt"]
    match_results = pose_dict["match_results"]

    return poses_pred, poses_gt, match_results

def visualize_camera_poses(poses_pred, poses_gt, image_names, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted poses
    for i, pose in enumerate(poses_pred):
        t = pose[:3, 3].numpy()
        ax.scatter(t[0], t[1], t[2], c='r', marker='o', s=50)  # Increase point size
        R = pose[:3, :3].numpy()
        camera_direction = R @ np.array([0, 0, 1])
        ax.quiver(t[0], t[1], t[2], camera_direction[0], camera_direction[1], camera_direction[2], length=0.1, color='b', linewidth=2)  # Increase arrow size
        # ax.text(t[0], t[1], t[2], f"{image_names[i]}", color='red')  # Add label

    # Plot ground truth poses
    for i, pose in enumerate(poses_gt):
        t = pose[:3, 3].numpy()
        ax.scatter(t[0], t[1], t[2], c='g', marker='^', s=50)  # Increase point size
        R = pose[:3, :3].numpy()
        camera_direction = R @ np.array([0, 0, 1])
        ax.quiver(t[0], t[1], t[2], camera_direction[0], camera_direction[1], camera_direction[2], length=0.1, color='y', linewidth=2)  # Increase arrow size
        # ax.text(t[0], t[1], t[2], f"{image_names[i]}", color='green')  # Add label

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def generate_3d_images(result_path, epoch):
    # Load the saved files
    pose_dict, gaussian_state = load_saved_files(result_path, epoch)

    # Extract pose information
    poses_pred, poses_gt, match_results = extract_pose_information(pose_dict)
    image_names = [f"image_{i}.jpg" for i in range(len(pose_dict["poses_pred"]))]  # Replace with actual image names if available
    # Visualize camera poses
    save_path = f"{result_path}/camera_poses_epoch_{epoch:02d}.png"
    visualize_camera_poses(poses_pred, poses_gt, image_names, save_path=save_path)


result_path = "output/progressive/data_tt360"
epoch = 0  # Specify the epoch number

generate_3d_images(result_path, epoch)