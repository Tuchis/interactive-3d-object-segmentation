import os
import json
import shutil
import numpy as np
from PIL import Image
import cv2

SOURCE_DIR = "/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/data/data"
TARGET_DIR = "/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/dataset"

# def create_pose_and_intrinsics(source_dir, target_dir):
#     """
#     Creates folders for intrinsics and poses, and stores corresponding values.
    
#     Args:
#         source_dir: Path to the source directory containing ScanNet++ data
#         target_dir: Path to the target directory where the dataset will be created
#     """
#     # Create directories
#     pose_dir = os.path.join(target_dir, 'pose')
#     intrinsics_dir = os.path.join(target_dir, 'intrinsics')
#     os.makedirs(pose_dir, exist_ok=True)
#     os.makedirs(intrinsics_dir, exist_ok=True)
    
#     # Find camera transforms file (could be in nerfstudio format or colmap format)
#     transforms_path = None
#     cameras_path = None
    
#     # Check if iphone has pose_intrinsic_imu.json
#     iphone_dir = os.path.join(source_dir, 'iphone')
#     if os.path.exists(iphone_dir):
#         for file in os.listdir(iphone_dir):
#             if file.endswith('.json') and 'pose_intrinsic_imu' in file:
#                 transforms_path = os.path.join(iphone_dir, file)
#                 break

#     # Read the transforms_path
#     with open(transforms_path, 'r') as f:
#         transforms = json.load(f)

#     for frame_id, transform in transforms.items():
#         frame_id = int(frame_id[-6:])
#         pose = transform['pose']
#         intrinsics = transform['intrinsic']

        # "pose": [
        #     [
        #         0.63281906,
        #         0.5936474,
        #         -0.4971145,
        #         0.39389375
        #     ],
        #     [
        #         0.03214595,
        #         -0.6616073,
        #         -0.7491611,
        #         0.06876414
        #     ],
        #     [
        #         -0.7736321,
        #         0.45810315,
        #         -0.43776107,
        #         0.6076599
        #     ],
        #     [
        #         0.0,
        #         0.0,
        #         0.0,
        #         1.0
        #     ]
        # ],
        # "intrinsic": [
        #     [
        #         1429.0793,
        #         0.0,
        #         954.5517
        #     ],
        #     [
        #         0.0,
        #         1429.0793,
        #         722.9511
        #     ],
        #     [
        #         0.0,
        #         0.0,
        #         1.0
        #     ]
        # ],
        

        # Pose file has to be in the next format:
        # 0.115846 0.139738 -0.983389 5.164835
        # 0.992027 -0.065733 0.107523 3.944635
        # -0.049616 -0.988004 -0.146239 1.639819
        # 0.000000 0.000000 0.000000 1.000000

        # Intrinsics file has to be in the next format:
        # 5.778706049999999550e+02 0.000000000000000000e+00 3.195000000000000000e+02 0.000000000000000000e+00
        # 0.000000000000000000e+00 5.778706049999999550e+02 2.395000000000000000e+02 0.000000000000000000e+00
        # 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00 0.000000000000000000e+00
        # 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00

def create_pose_and_intrinsics(source_dir, target_dir, frame_interval=10):
    """
    Creates pose and intrinsics files from ScanNet++ data.
    
    Args:
        source_dir: Path to the source directory containing ScanNet++ data
        target_dir: Path to the target directory where the dataset will be created
    """
    # Create directories
    pose_dir = os.path.join(target_dir, 'pose')
    os.makedirs(pose_dir, exist_ok=True)
    
    # Find pose and intrinsic data
    pose_intrinsic_file = None
    for device in ['iphone', 'dslr']:
        potential_file = os.path.join(source_dir, device, 'pose_intrinsic_imu.json')
        if os.path.exists(potential_file):
            pose_intrinsic_file = potential_file
            break
    
    if not pose_intrinsic_file:
        print("Warning: Could not find pose_intrinsic_imu.json file")
        return
    
    # Load pose and intrinsic data
    with open(pose_intrinsic_file, 'r') as f:
        pose_data = json.load(f)
    
    # Create intrinsics directory
    intrinsics_dir = os.path.join(target_dir, 'intrinsics')
    os.makedirs(intrinsics_dir, exist_ok=True)
    
    # Also create a global intrinsics.txt file for backward compatibility
    intrinsics_path = os.path.join(target_dir, 'intrinsics.txt')
    
    # Process intrinsics for each frame
    for frame_key, frame_data in pose_data.items():
        # Extract frame index
        frame_idx = int(frame_key.split('_')[1])
        if frame_idx % frame_interval != 0:
            continue
        frame_idx = frame_idx // frame_interval
        
        # Get intrinsic matrix
        intrinsic = np.array(frame_data['intrinsic'])
        
        # Convert to 4x4 matrix format
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = intrinsic
        
        # Save per-frame intrinsics
        frame_intrinsics_path = os.path.join(intrinsics_dir, f'{frame_idx}.txt')
        np.savetxt(frame_intrinsics_path, intrinsic_4x4, fmt='%.15e')
    
    # Use the first frame's intrinsic as the global intrinsic for backward compatibility
    first_frame_key = list(pose_data.keys())[0]
    intrinsic = np.array(pose_data[first_frame_key]['intrinsic'])
    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = intrinsic
    np.savetxt(intrinsics_path, intrinsic_4x4, fmt='%.15e')
    
    # Create pose files
    for frame_key, frame_data in pose_data.items():
        # Extract frame index
        frame_idx = int(frame_key.split('_')[1])
        if frame_idx % frame_interval != 0:
            continue
        frame_idx = frame_idx // frame_interval
        
        # Get pose matrix
        if 'aligned_pose' in frame_data:
            pose_matrix = np.array(frame_data['aligned_pose'])
        else:
            pose_matrix = np.array(frame_data['pose'])
        
        # Ensure it's a 4x4 matrix
        if pose_matrix.shape != (4, 4):
            # If it's not complete, try to fill in the missing parts
            if pose_matrix.shape[0] == 3:
                # Add the last row [0, 0, 0, 1]
                pose_matrix = np.vstack([pose_matrix, [0, 0, 0, 1]])
        
        # Save pose
        pose_path = os.path.join(pose_dir, f'{frame_idx}.txt')
        np.savetxt(pose_path, pose_matrix, fmt='%.6f')
            

def create_color(source_dir, target_dir, frame_interval=10):
    """
    Creates a folder for color images and stores RGB frames.
    
    Args:
        source_dir: Path to the source directory containing ScanNet++ data
        target_dir: Path to the target directory where the dataset will be created
    """
    # Create directory
    color_dir = os.path.join(target_dir, 'color')
    os.makedirs(color_dir, exist_ok=True)
    
    # Get color images from iphone/rgb
    rgb_dir = os.path.join(source_dir, 'iphone', 'rgb')
    for img_file in os.listdir(rgb_dir):
        if img_file.endswith(('.jpg', '.png', '.JPG', '.PNG')):
            src_path = os.path.join(rgb_dir, img_file)
            # Get rid of frame_ 
            frame_idx = str(int(img_file.split('frame_')[1].split('.')[0]))
            if int(frame_idx) % frame_interval != 0:
                continue
            frame_idx = str(int(frame_idx) // frame_interval)
            dst_path = os.path.join(color_dir, frame_idx + '.jpg')
            with open(src_path, 'rb') as src_file:
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            

def create_depth(source_dir, target_dir, frame_interval=10):
    """
    Creates a folder for depth images and stores depth frames.
    
    Args:
        source_dir: Path to the source directory containing ScanNet++ data
        target_dir: Path to the target directory where the dataset will be created
    """
    # Create directory
    depth_dir = os.path.join(target_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)
    
    # Get depth images from iphone/depth
    source_depth_dir = os.path.join(source_dir, 'iphone', 'depth')
    for img_file in os.listdir(source_depth_dir):
        if img_file.endswith(('.png', '.exr', '.pfm')):
            src_path = os.path.join(source_depth_dir, img_file)
            # Take only part after frame_, make int of it, and make it back to string
            frame_idx = str(int(img_file.split('frame_')[1].split('.')[0]))
            if int(frame_idx) % frame_interval != 0:
                continue
            frame_idx = str(int(frame_idx) // frame_interval)
            dst_path = os.path.join(depth_dir, frame_idx + '.png')
            with open(src_path, 'rb') as src_file:
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())

def create_ply(source_dir, target_dir):
    """
    Copies PLY files to the target folder.
    
    Args:
        source_dir: Path to the source directory containing ScanNet++ data
        target_dir: Path to the target directory where the dataset will be created
    """
    # Create directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Find PLY files
    ply_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.ply') and 'semantic' not in file:
                ply_files.append(os.path.join(root, file))
    
    # Copy PLY files
    for idx, ply_file in enumerate(ply_files):
        filename = os.path.basename(target_dir) + "_vh_clean_2.ply"
        dst_path = os.path.join(target_dir, filename)
        shutil.copy2(ply_file, dst_path)
        print(f"Copied {filename} to {dst_path}")

def convert_scannetpp_to_dataset(source_dir, target_dir, scene_name, frame_interval=10):
    """
    Main function to convert ScanNet++ data to the target dataset format.
    
    Args:
        source_dir: Path to the source directory containing ScanNet++ data
        target_dir: Path to the target directory where the dataset will be created
    """
    source_dir = os.path.join(source_dir, scene_name)
    target_dir = os.path.join(target_dir, scene_name)

    os.makedirs(target_dir, exist_ok=True)
    
    print("Creating pose and intrinsics...")
    create_pose_and_intrinsics(source_dir, target_dir, frame_interval)
    
    print("Creating color images...")
    create_color(source_dir, target_dir, frame_interval)
    
    print("Creating depth images...")
    create_depth(source_dir, target_dir, frame_interval)
    
    print("Copying PLY files...")
    create_ply(source_dir, target_dir)
    
    print(f"Conversion complete. Dataset saved to {target_dir}")

# Example usage:
# convert_scannetpp_to_dataset('/path/to/scannetpp/data', '/path/to/output/dataset')

if __name__ == "__main__":
    # Get all scene names from source directory
    scene_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"Found {len(scene_names)} scenes in {SOURCE_DIR}")
    
    # Convert each scene to dataset
    for scene_name in scene_names:
        convert_scannetpp_to_dataset(SOURCE_DIR, TARGET_DIR, scene_name, frame_interval=10)
