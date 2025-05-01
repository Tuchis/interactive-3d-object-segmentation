import numpy as np

object_classes_list_file = "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/data/ScanNetPP/single/object_classes.txt"

print(np.loadtxt(object_classes_list_file, dtype=str))

