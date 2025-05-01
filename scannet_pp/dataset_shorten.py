# Take object_classes.txt file and object_ids.npy file, and take random 30% of the data

import numpy as np
import random
import os

object_classes = open('/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/data/ScanNetPP_short_old/single/object_classes.txt', 'r').readlines()
object_classes = [line.strip() for line in object_classes]
object_classes = np.array(object_classes)
object_ids = np.load('/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/data/ScanNetPP_short_old/single/object_ids.npy')

# Take random 30% of the data
num_samples = len(object_classes)
num_samples_to_take = int(num_samples * 0.3)

random_indices = random.sample(range(num_samples), num_samples_to_take)

random_indices.sort()

# Create folder for short dataset
folder = 'short_dataset'
os.makedirs(folder, exist_ok=True)

# Save the random indices
# np.save(os.path.join(folder, 'random_indices.txt'), random_indices)

print(object_classes[random_indices])

with open(os.path.join(folder, 'random_indices.txt'), 'w') as f:
    for i in random_indices:
        f.write(str(i) + '\n')

with open(os.path.join(folder, 'object_classes_short.txt'), 'w') as f:
    for i in random_indices:
        f.write(object_classes[i] + '\n')

# Save the object classes and ids
# np.savetxt(os.path.join(folder, 'object_classes_short.txt'), object_classes[random_indices])
np.save(os.path.join(folder, 'object_ids_short.npy'), object_ids[random_indices])
