# {
    # "ebff4de90b": "office",
    # "39f36da05b": "office",
    # "5a269ba6fe": "conference room",
    # "a24858e51e": "conference room",
    # "21d970d8de": "conference room",
    # "dc263dfbf0": "classroom",
    # "08bbbdcc3d": "classroom",
    # "4bc04e0cde": "conference room",
    # "fb564c935d": "conference room",
    # "a897272241": "classroom",
    # "bde1e479ad": "classroom",
    # "ef18cf0708": "office",
    # "56a0ec536c": "office",
    # "a1d9da703c": "office",

# We want to sample 1 random scenes from each category

import json
import random

# Load the scene types
with open('/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/data/metadata/scene_types.json', 'r') as f:
    scene_types = json.load(f)

# Get a list of unique scene types
unique_scene_types = list(set(scene_types.values()))

# Sample 1 random scene from each category

for scene_type in unique_scene_types:
    # Get all scenes of this type
    scenes_of_type = [scene for scene, scene_type in scene_types.items() if scene_type == scene_type]
    
    # Sample 1 random scene from this category
    random_scene = random.choice(scenes_of_type)

    # Print the scene ID
    print(random_scene, scene_type) 
