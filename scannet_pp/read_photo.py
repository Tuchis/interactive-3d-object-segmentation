import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Define the path to the depth image
# depth_image_path = '/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/data/data/56a0ec536c/iphone/depth/frame_000012.png'

depth_image_path = "/home/rpartsey/code/eai/SAMPro3D-fork/data/scannet/scans/56a0ec536c/depth/1.png"
# depth_image_path = "/home/rpartsey/code/eai/SAMPro3D-fork/data/scannet/scans/scene0000_02/depth/1.png"

# Check if the file exists
if not os.path.exists(depth_image_path):
    print(f"Error: File not found at {depth_image_path}")
else:
    # Read the depth image using PIL and convert to numpy array
    depth_img = np.array(Image.open(depth_image_path))
    
    # Print information about the depth image
    print(f"Depth image shape: {depth_img.shape}")
    print(f"Depth image data type: {depth_img.dtype}")
    print(f"Min depth value: {np.min(depth_img)}")
    print(f"Max depth value: {np.max(depth_img)}")
    print(f"Mean depth value: {np.mean(depth_img)}")
    
    # # Display the depth image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(depth_img, cmap='viridis')
    # plt.colorbar(label='Depth')
    # plt.title('Depth Image: frame_000012.png')
    # plt.show()

    # Save the depth image with colormap
    depth_img_path = "/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/1.png"
    # Apply colormap to the depth image
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_img, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(depth_img_path, dpi=300, bbox_inches='tight')
    plt.close()
