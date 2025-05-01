# Script, that reads a npy file and returns a numpy array of the points, and their labels

import numpy as np

def read_npy(file_path):
    return np.load(file_path)

print(read_npy("/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/data/ScanNetPP/single/object_ids.npy"))