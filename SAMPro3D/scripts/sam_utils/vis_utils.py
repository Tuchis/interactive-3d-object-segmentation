import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
functions for visualization
"""

def show_mask(mask, ax, random_color=False):
    if random_color:
        rgb = np.random.random(3)
        color = np.concatenate([rgb, np.array([0.65])], axis=0)
    else:
        rgb = None
        color = np.array([30/255, 144/255, 255/255, 0.65])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return rgb

def show_mask_ins(mask, ax, color, random_color=False):
    if random_color:
        rgb = np.random.random(3)
        color = np.concatenate([rgb, np.array([0.65])], axis=0)
    else:
        color = np.concatenate([color, np.array([0.65])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=20, linewidth=1.25)   
    
def show_points_color(coords, labels, ax, rgb, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=rgb, marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=20, linewidth=1.25) 
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        rgb = np.random.random(3)
        color_mask = np.concatenate([rgb, [0.65]])
        img[m] = color_mask
        # also show the corresponding point coords on each instance:
        coords = ann['point_coords']
        color_point = np.concatenate([rgb, [0.95]])  # point color is deeper than mask
        show_points_color(np.array(coords), np.array([1]), ax, rgb=color_point)
    ax.imshow(img)
    
def show_anns_sem(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns.shape[0], anns.shape[1], 4))
    img[:,:,3] = 0
    for i in range(0, 21):
        m = np.where(anns==i)
        color_mask = np.concatenate([np.random.random(3), [0.65]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_anns_ins(anns, num):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns.shape[0], anns.shape[1], 4))
    img[:,:,3] = 0
    for i in range(0, num+1):
        m = np.where(anns==i)
        color_mask = np.concatenate([np.random.random(3), [0.45]])
        img[m] = color_mask
    ax.imshow(img)
    
def cal_iou(pred, gt):
    assert pred.shape == gt.shape  # H * W
    I = np.sum(np.logical_and(pred == 1, gt == 1))
    U = np.sum(np.logical_or(pred == 1, gt == 1))
    iou = I / float(U)
    return iou

def show_iou(data_path_ins, scene_name, frame_id, ins_id, pred, image):
    # instance seg:
    data_path_ins = "sample_data/scannet_2d_allframe_label_ins/"
    label_ins = cv2.imread(os.path.join(data_path_ins, scene_name, 'label', str(frame_id) + '.png'), # TODO: try 'label' instead of 'label_filt'
                   cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

    if image.shape[0] != label_ins.shape[0] or image.shape[1] != label_ins.shape[1]:
            raise (RuntimeError("Image & label shape mismatch!"))
        
    mask_coord = np.where(label_ins == ins_id)
    other_coord = np.where(label_ins != ins_id)
    label_ins[mask_coord] = 1
    label_ins[other_coord] = 0
    
    iou = cal_iou(pred, label_ins)
    
    plt.imshow(image)
    show_mask(label_ins, plt.gca())
    plt.title(f"Instance Annotation, the IoU is: {iou:.5f}", fontsize=10)
    plt.axis('off')
    
    return iou

def rand_cmap(nlabels, type='bright', first_color_black=False, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib ranging from 0~1
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            #print(list(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])))
            randRGBcolors.append(list(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        if nlabels == 0:
            return randRGBcolors

        print(f"nlabels: {nlabels}")

        if nlabels == 1:
            randRGBcolors[0] = [0, 0, 1]
    
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

#     return random_colormap
    return randRGBcolors


def rand_cmap_sns(nlabels):

    import seaborn as sns
    
    # palette = "husl"
    palette = "bright"
    randRGBcolors = sns.color_palette(palette, nlabels)

    return randRGBcolors


def write_ply_color_rgb(points, labels, rgb, out_filename, n_classes):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]

    np.random.seed(1)
    colors = rand_cmap(n_classes, type='bright', first_color_black=False, last_color_black=False, verbose=False)

    ignore_idx_list = np.where(labels==-1)[0] # list of ignore_idx

    for i in range(N):
        if i in ignore_idx_list:  # if ignore_idx, using original rgb value
            c = rgb[i]  
        else:  # else, using the given label rgb
            c = colors[labels[i]]
            c = [int(x * 255) for x in c]  # change rgb value from 0-1 to 0-255
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def get_color_rgb(points, labels, rgb, cmap, make_gray=False):
    labels = labels.astype(int)
    N = points.shape[0]
    
    # Create output array directly
    c_all = np.zeros((N, 1, 3))
    
    # Find ignore indices
    ignore_mask = labels == -1
    
    # Handle ignored points (use original RGB values)
    c_all[ignore_mask, 0, :] = rgb[ignore_mask] / 255.0
    
    # Handle labeled points (use colors from colormap)
    valid_mask = ~ignore_mask
    valid_labels = labels[valid_mask]
    
    if cmap:
        # Vectorized lookup in the colormap
        c_all[valid_mask, 0, :] = np.array([cmap[label] for label in valid_labels])

    if make_gray:
        invalid_labels = labels[ignore_mask]
        c_all[ignore_mask, 0, :] = np.array([[211/255, 211/255, 211/255] for label in invalid_labels])
    
    # Split into list of arrays for compatibility with existing code
    return c_all

def show_mask_ins_video(mask, frame, color):
    # Create a color mask where the mask is True
    color_mask = frame.copy()
    color_mask[mask > 0] = color

    # Blend the color mask with the frame
    alpha = 0.5  # Transparency factor
    frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

    return frame

def show_points_color_video(points, frame, color, marker_size=5):
    for coord in points:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(frame, (x, y), marker_size, color, -1)  # Filled circle
    return frame

def create_visualization_video_with_opencv(start_frame_idx, end_frame_idx, output_video_path, fps=30, frame_size=(1920, 1080), colors=None,
                                           dataset_dir=None, scene_name=None, experiments_dir=None, object_names=None, model="sam2"):
    # Define the codec and initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for i in tqdm(range(start_frame_idx, end_frame_idx + 1)):
        frame_path = f"{dataset_dir}/{scene_name}/color/{i}.jpg"
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Frame {i} not found at {frame_path}")
            continue

        original_height, original_width = frame.shape[:2]

        # Resize frame to desired size if necessary
        if (original_width, original_height) != frame_size:
            frame = cv2.resize(frame, frame_size)

        for j, object_name in enumerate(object_names):
            sam_output_path = f"{experiments_dir}/{scene_name}/{object_name}/sam_output"
            mask_path = f"{sam_output_path}/masks_npy/{i}.npy"
            points_path = f"{sam_output_path}/points_npy/{i}.npy"

            # Overlay mask if it exists
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

                # print(f"Mask shape: {mask.shape}")
                
                # if model == "sam2":
                #     mask = mask.reshape(-1, 480, 640)

                if mask is None or mask.size == 0:
                    print(f"Mask at {mask_path} is empty or invalid.")
                    continue

                # **Squeeze the mask to remove singleton dimensions**
                mask = np.squeeze(mask)

                # # Ensure mask is 2D
                # if mask.ndim != 2:
                #     print(f"Mask at {mask_path} is not 2D after squeezing.")
                #     continue

                # # Convert mask to uint8 if necessary
                # if mask.dtype != np.uint8:
                #     mask = mask.astype(np.uint8)

                # Resize mask to match frame size if necessary
                if mask.shape != (frame_size[1], frame_size[0]):
                    # print(f"Mask shape: {mask.shape}")
                    # print(f"Frame size: {frame_size}")
                    if frame_size[0] > 0 and frame_size[1] > 0:
                        mask = cv2.resize(mask, (frame_size[0], frame_size[1]), interpolation=cv2.INTER_NEAREST)
                    else:
                        print(f"Invalid frame size: {frame_size}")
                        continue

                frame = show_mask_ins_video(mask, frame, colors[j])

            # Draw points if they exist
            if os.path.exists(points_path):
                points = np.load(points_path)

                # Adjust points if frame size changed
                if (original_width, original_height) != frame_size:
                    scale_x = frame_size[0] / original_width
                    scale_y = frame_size[1] / original_height
                    points[:, 0] = points[:, 0] * scale_x
                    points[:, 1] = points[:, 1] * scale_y

                frame = show_points_color_video(points, frame, colors[j], marker_size=5)

        # Write the processed frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")