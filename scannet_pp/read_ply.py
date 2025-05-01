# Script, that reads a ply file and returns a numpy array of the points, and their labels

import numpy as np
import open3d as o3d

def read_ply(file_path):
    print(o3d.io.read_point_cloud(file_path))
    print(o3d.io.read_point_cloud(file_path).labels)
    return np.asarray(o3d.io.read_point_cloud(file_path).points), np.asarray(o3d.io.read_point_cloud(file_path).colors)


print(read_ply("/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/data/ScanNet/scans/scene0003_02.ply"))