import open3d as o3d
import numpy as np
import json
import os
from constants import PLY_SAVE_PATH, VIDEO_POINTS_SAVE_PATH

def save_points(input_points):
    """
    Save the picked points as a point cloud (with colors) to 'output.ply'
    using Open3D.
    """
    # Parse input points
    points = []
    point_types = []
    for point_data in input_points:
        points.append(point_data["position"])
        point_types.append(point_data["type"])

    if not points:
        print("No points to save.")
        if os.path.exists(PLY_SAVE_PATH):
            os.remove(PLY_SAVE_PATH)
        return
        
    
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    points_array = np.array(points)
    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    # Set colors: green ([0,1,0]) for '+' and red ([1,0,0]) for '-'
    colors = np.array([[0, 1, 0] if t == "+" else [1, 0, 0] for t in point_types])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    output_file = PLY_SAVE_PATH
    o3d.io.write_point_cloud(output_file, pcd)

def save_video_points(input_points):
    """
    Save the picked points as a point cloud (with colors) to 'output.ply'
    using Open3D.
    """
    if not input_points:
        print("No points to save.")
        if os.path.exists(VIDEO_POINTS_SAVE_PATH):
            os.remove(VIDEO_POINTS_SAVE_PATH)
        return
    
    with open(VIDEO_POINTS_SAVE_PATH, "w") as f:
        json.dump(input_points, f)
