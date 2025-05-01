# Script, that converts a pointcloud to a dataset, that is user for evaluation of AGILE3D

# Output format:
# scans/scene_id.ply
# single/object_classes.txt
# single/object_ids.npy

# scene_id.ply is a ply file with next header:
# ply
# format binary_little_endian 1.0
# element vertex 81369
# property float32 x
# property float32 y
# property float32 z
# property uint8 R
# property uint8 G
# property uint8 B
# property float64 label

# object_classes.txt is a text file with next content:
# object_class_1
# object_class_2
# ...
# object_class_n

# object_ids.npy is a (x, 2) numpy array of the scene and object ids
# scene_id object_id

# On input we have the next files:
# data/data/scene_id/scans/mesh_aligned_0.05.ply
# data/data/scene_id/scans/mesh_aligned_0.05_semantic.ply
# data/data/scene_id/scans/segments_anno.json

# mesh_aligned_0.05.ply is a ply file with next header:
# ply
# format binary_little_endian 1.0
# element vertex 1646677
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# element face 3269344
# property list uchar int vertex_indices

# mesh_aligned_0.05_semantic.ply is a ply file with next header:
# ply
# format binary_little_endian 1.0
# element vertex 1646677
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property int label
# element face 3269344
# property list uchar int vertex_indices

# segments_anno.json is a json file with next content:
# {"sceneId": "scannetpp.2022-11-25_19-24", "appId": "stk.v1", "annId": "6673", "segGroups": [{"id": 1, "objectId": 1, "label": "fake ceiling", "segments": [45644, 45687, ..., 1321908], "obb": {"centroid": [6.650374448475603, 1.900442331056328, 3.092642188072204], "axesLengths": [3.63003116534402, 0.2991878986358648, 1.2790911125582163], "normalizedAxes": [-0.0011438008336857619, -0.9999993458596126, -2.2204445967668742e-16, 0, -2.220446049250313e-16, 1, -0.9999993458596126, 0.0011438008336857619, 2.5397480422867643e-19], "min": [6.008753294212459, 0.08469642291886226, 2.943048238754271], "max": [7.291995602738746, 3.716188239193794, 3.242236137390137]}, "dominantNormal": [0, -2.220446049250313e-16, 1], "partId": 1, "index": 0}}

import json
import numpy as np
from ply import read_ply, write_ply
import os
from numpy.lib import recfunctions as rfn
import shutil
import random

def read_and_parse_input_files(scene_id):
    # Define file paths
    mesh_ply_path = f"data/data/{scene_id}/scans/mesh_aligned_0.05.ply"
    semantic_ply_path = f"data/data/{scene_id}/scans/mesh_aligned_0.05_semantic.ply"
    segments_json_path = f"data/data/{scene_id}/scans/segments_anno.json"

    # Read PLY files
    mesh_data = read_ply(mesh_ply_path, triangular_mesh=True)
    semantic_data = None

    # Parse JSON file
    with open(segments_json_path, 'r') as f:
        segments_data = json.load(f)

    return mesh_data, semantic_data, segments_data

def generate_output_files(scene_id, mesh_data, semantic_data, segments_data, save_folder, percentage_of_objects_to_save):
    # Define output file paths
    scene_ply_path = f"{save_folder}/scans/{scene_id}.ply"
    object_classes_path = f"{save_folder}/single/object_classes.txt"
    object_ids_path = f"{save_folder}/single/object_ids.npy"

    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(f"{save_folder}/scans", exist_ok=True)
    os.makedirs(f"{save_folder}/single", exist_ok=True)

    # Extract labels from segments_data
    labels = np.zeros(mesh_data[0].shape[0], dtype=np.int32)
    for group in segments_data['segGroups']:
        for segment in group['segments']:
            labels[segment] = group['objectId']

    # Add labels to mesh_data
    mesh_with_labels = rfn.append_fields(mesh_data[0], 'label', labels, usemask=False)

    print(f"{type(mesh_with_labels[0])}")

    # Convert that to a list of 1D numpy arrays
    mesh_with_labels = [mesh_with_labels[key] for key in mesh_with_labels.dtype.names]

    # Write scene_id.ply with labels
    # write_ply(scene_ply_path, np.random.rand(10, 7), ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply(scene_ply_path, mesh_with_labels, ['x', 'y', 'z', 'R', 'G', 'B', 'label'])

    # Get random percentage of objects to save
    number_of_objects_to_save = int(len(segments_data['segGroups']) * percentage_of_objects_to_save)
    segments_data['segGroups'] = random.sample(segments_data['segGroups'], number_of_objects_to_save)

    # Extract and write object classes
    object_classes=[group['label'] for group in segments_data['segGroups']]
    with open(object_classes_path, 'a') as f:
        for obj_class in sorted(object_classes):
            f.write(f"{obj_class.replace(' ', '_')}\n")
    # Extract and write object IDs
    # Check if object_ids.npy already exists and load it if it does
    existing_object_ids = []
    if os.path.exists(object_ids_path):
        try:
            existing_object_ids = np.load(object_ids_path, allow_pickle=True).tolist()
            print(f"Loaded existing object_ids with {len(existing_object_ids)} entries")
        except Exception as e:
            print(f"Error loading existing object_ids: {e}")
            existing_object_ids = []
    
    # Create new object_ids for current scene
    new_object_ids = [(scene_id, group['objectId']) for group in segments_data['segGroups']]
    
    # Combine existing and new object_ids
    combined_object_ids = existing_object_ids + new_object_ids
    object_ids = np.array(combined_object_ids)
    print(f"object_ids: {object_ids}")
    np.save(object_ids_path, object_ids)

# Example usage
# scene_id = "56a0ec536c"
save_folder = "scannet_pp_dataset"
# read all scenes from data/data/
# Clear save_folder if it exists
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)

# Set percentage of objects to save
percentage_of_objects_to_save = 0.4

for scene_id in os.listdir("data/data/"):
    print(f"Processing scene: {scene_id}")
    mesh_data, semantic_data, segments_data = read_and_parse_input_files(scene_id)
    generate_output_files(scene_id, mesh_data, semantic_data, segments_data, save_folder, percentage_of_objects_to_save)

# Save all labels from file object_classes.txt to next format:
# 'scannetpp': {
#             'wall',
#             'floor',
#             'cabinet'
#             }

with open(save_folder + "/single/object_classes.txt", "r") as f:
    object_classes = f.readlines()

object_classes = [obj_class.replace(' ', '_') for obj_class in object_classes]

print(object_classes)

# Create folder labels if it doesn't exist
os.makedirs(save_folder + "/labels", exist_ok=True)

# Save object classes to file
object_classes = set([obj_class.strip() for obj_class in object_classes])
with open(save_folder + "/labels/object_classes.txt", "w") as f:
    for obj_class in object_classes:
        f.write(f"'{obj_class}',\n")
