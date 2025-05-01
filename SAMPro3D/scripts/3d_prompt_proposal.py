"""
Script for the stage of 3D Prompt Proposal in the paper

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn) and Xingyilang Yin
"""

import cProfile
import pstats
import io
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import cv2
import argparse
import torch
import numpy as np
import open3d as o3d
import pointops
import sys
# Add the parent directory to the path to properly import utils as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sam_utils.main_utils import *
from sam_utils.sam_utils import *
from tqdm import trange
from typing import Tuple, Dict, Any, Optional, List
import json
sys.path.append("/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/dataset_preprocess")
from preprocess_util import adjust_intrinsic

# Global cache for predictors
_predictor_cache = {}

def get_predictor_cache_key(args: argparse.Namespace) -> str:
    """
    Create a unique key for the predictor cache based on the arguments.
    """
    if args.sam2:
        key = f"sam2_{args.sam2_checkpoint}_{args.sam2_config}"
        if args.sam2_video:
            key += "_video"
    else:
        key = f"sam_{args.model_type}_{args.sam_checkpoint}"
    
    key += f"_{args.device}"
    return key

def create_predictor(args: argparse.Namespace):
    """
    Create a SAM or SAM2 predictor based on the arguments.
    If a predictor with the same configuration exists in cache, return it.
    """
    cache_key = get_predictor_cache_key(args)
    
    # Check if predictor is already in cache
    if cache_key in _predictor_cache:
        print(f"Using cached predictor with key: {cache_key}")
        return _predictor_cache[cache_key]
    
    # Initialize SAM:
    device = torch.device(args.device)
    print(f"Creating new predictor with key: {cache_key}")

    if args.sam2:
        # Import SAM2
        checkpoint = args.sam2_checkpoint
        model_cfg = args.sam2_config
        if args.sam2_video:
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            predictor = build_sam2_video_predictor(model_cfg, checkpoint).to(device=device)
            print(f"{type(predictor)=}")
            # predictor = SAM2VideoPredictor(build_sam2_video_predictor(model_cfg, checkpoint))
        else:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint).to(device=device))
    else:
        # Import SAM
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[args.model_type](
            checkpoint=args.sam_checkpoint
        ).to(device=device)
        predictor = SamPredictor(sam)
    
    # Store predictor in cache
    _predictor_cache[cache_key] = predictor
    return predictor


def load_prompt(args, device):
    # prompt_xyz, prompt_rgb = None, None
    prompt_xyz = np.array([])
    prompt_rgb = np.array([])

    if args.prompt_path != "init_prompt":
        if os.path.exists(args.prompt_path):
            prompt_xyz, prompt_rgb = load_ply(args.prompt_path)
    else:
        if os.path.exists(f"{args.data_path}/{args.scene_name}/{args.scene_name}_{args.prompt_name}.ply"):
            prompt_xyz, prompt_rgb = load_ply(f"{args.data_path}/{args.scene_name}/{args.scene_name}_{args.prompt_name}.ply")
    prompt_xyz = torch.from_numpy(prompt_xyz).to(device=device)
    return prompt_xyz, prompt_rgb

def load_video_points(args):
    if not os.path.exists(args.video_points_path):
        return {}
    with open(args.video_points_path, "r") as f:
        video_points = json.load(f)
    return video_points

def create_output_folders(args):
    # Create folder to save SAM outputs:
    create_folder(args.sam_output_path)
    # Create subfolder for saving different output types:
    create_folder(f"{args.sam_output_path}/points_npy")
    create_folder(f"{args.sam_output_path}/iou_preds_npy")
    create_folder(f"{args.sam_output_path}/masks_npy")
    create_folder(f"{args.sam_output_path}/corre_3d_ins_npy")


def prompt_init(xyz, rgb, voxel_size, device):
    # Here we only use voxelization to decide the number of fps-sampled points, \
    # since voxel_size is more controllable. We use fps later for prompt initialization
    idx_sort, num_pt = voxelize(xyz, voxel_size, mode=1)
    print("the number of initial 3D prompts:", len(num_pt))
    xyz = torch.from_numpy(xyz).cuda().contiguous()
    o, n_o = len(xyz), len(num_pt)
    o, n_o = torch.cuda.IntTensor([o]), torch.cuda.IntTensor([n_o])
    idx = pointops.farthest_point_sampling(xyz, o, n_o)
    fps_points = xyz[idx.long(), :]
    fps_points = torch.from_numpy(fps_points.cpu().numpy()).to(device=device)
    rgb = rgb / 256.
    rgb = torch.from_numpy(rgb).cuda().contiguous()
    fps_colors = rgb[idx.long(), :]
    
    return fps_points, fps_colors
    

def save_init_prompt(xyz, rgb, args):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    prompt_ply_file = f"{args.prompt_path}/{args.scene_name}.ply"
    o3d.io.write_point_cloud(prompt_ply_file, point_cloud)
    
    
def process_batch_sam(
    predictor,
    points: torch.Tensor,
    point_labels: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    transformed_points = predictor.transform.apply_coords_torch(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.from_numpy(point_labels).to(device=predictor.device)

    masks, iou_preds, _ = predictor.predict_torch(
        in_points[None, :, :],
        in_labels[None, :],
        multimask_output=False,
        return_logits=True,
    )

    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points,
        corre_3d_ins=ins_idxs
    )

    return data_original


def process_batch_sam2(
    predictor,
    points: torch.Tensor,
    point_labels: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    transformed_points = points
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.from_numpy(point_labels).to(device=predictor.device)

    masks, iou_preds, _ = predictor.predict(
        in_points[None, :, :],
        in_labels[None, :],
        multimask_output=False,
        return_logits=True,
    )

    data_original = MaskData(
        masks=masks.reshape(-1),
        iou_preds=iou_preds.reshape(-1),
        points=points,
        corre_3d_ins=ins_idxs
    )

    return data_original

def process_batch_sam2_batched(
    predictor,
    points_batch: List[torch.Tensor],
    point_labels_batch: List[np.ndarray],
    ins_idxs_batch: List[torch.Tensor],
) -> List[MaskData]:
    """
    Process a batch of points with SAM2 batch processing.
    
    Args:
        predictor: The SAM2 predictor
        points_batch: List of point coordinates for each image
        point_labels_batch: List of point labels for each image
        ins_idxs_batch: List of instance indices for each image
        
    Returns:
        List of MaskData objects
    """
    # Convert point_labels to list of numpy arrays
    point_labels_np_batch = [labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels for labels in point_labels_batch]
    
    # Call predict_batch
    masks_batch, iou_preds_batch, _ = predictor.predict_batch(
        point_coords_batch=points_batch,
        point_labels_batch=point_labels_np_batch,
        multimask_output=False,
        return_logits=True,
    )
    
    # Create MaskData objects for each item in the batch
    results = []
    for i in range(len(points_batch)):
        data = MaskData(
            masks=masks_batch[i],
            iou_preds=iou_preds_batch[i],
            points=points_batch[i],
            corre_3d_ins=ins_idxs_batch[i]
        )
        results.append(data)
    
    return results

def process_batch_sam2_video(
    inference_state,
    predictor,
    points: torch.Tensor,
    point_labels: torch.Tensor,
    ann_obj_id: int,
    ins_idxs: torch.Tensor,
    frame_id: int,
):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_id,
        obj_id=ann_obj_id,
        points=points,
        labels=point_labels,
    )
    

def sam_seg(predictor, frame_id_init, frame_id_end, init_prompt, point_labels, video_points, args):
    device = torch.device(args.device)
    
    if args.sam2 and args.sam2_video:
        first_frame = -100
        set_frame = -100
        inference_state = predictor.init_state(os.path.join(args.data_path, args.scene_name, 'color'),
                                                offload_video_to_cpu=False,
                                                offload_state_to_cpu=False,
                                                async_loading_frames=False)
    
    # For SAM2 batch processing
    if args.sam2 and not args.sam2_video:
        batch_size = args.batch_size
        batch_frames = []
        batch_input_points = []
        batch_point_labels = []
        batch_ins_idxs = []
        batch_frame_ids = []
    
    for frame_id in trange(frame_id_init, frame_id_end):

        video_points_torch = []
        video_point_labels = []
        if str(frame_id) in video_points:
            video_points_frame = video_points[str(frame_id)]
            for point in video_points_frame:
                video_points_torch.append(point['position'])
                video_point_labels.append(np.array(point['type'] == '+'))
        video_points_torch = torch.tensor(video_points_torch).to(device=predictor.device)
        video_point_labels = np.array(video_point_labels)


        depth_intrinsic_path = os.path.join(args.data_path, args.scene_name, 'intrinsics', f'{frame_id}.txt')
        if os.path.exists(depth_intrinsic_path):
            # Load the intrinsic
            rgb_intrinsic = torch.tensor(
                np.loadtxt(depth_intrinsic_path),
                dtype=torch.float64
            ).to(device=predictor.device)
            depth_image_dim = (256, 192)
            rgb_image_dim = (1920, 1440)
            # print(f"RGB intrinsic: {rgb_intrinsic=}")
            depth_intrinsic = adjust_intrinsic(rgb_intrinsic, rgb_image_dim, depth_image_dim)
            # print(f"Depth intrinsic: {depth_intrinsic=}")
        else:
            # Load the intrinsic
            depth_intrinsic = torch.tensor(
                np.loadtxt(f"{args.data_path}/intrinsics.txt"),
                dtype=torch.float64
            ).to(device=predictor.device)

        # Load the depth, and pose
        # Construct paths
        depth_png_path = os.path.join(args.data_path, args.scene_name, 'depth', f'{frame_id}.png')
        depth_npy_path = os.path.join(args.data_path, args.scene_name, 'depth', f'{frame_id}.npy')

        # Check if .npy cache exists
        if os.path.exists(depth_npy_path):
            depth = np.load(depth_npy_path)
        else:
            depth = cv2.imread(depth_png_path, -1).astype(np.float64)
            np.save(depth_npy_path, depth)

        depth = torch.from_numpy(depth).to(device=predictor.device)

        pose = torch.tensor(
            np.loadtxt(f"{args.data_path}/{args.scene_name}/pose/{frame_id}.txt"),
            dtype=torch.float64
        ).to(device=predictor.device)
        
        if str(pose[0, 0].item()) == '-inf': # skip frame with '-inf' pose
            continue

        # 3D-2D projection
        input_point_pos, corre_ins_idx = transform_pt_depth_scannet_torch(
            points=init_prompt, 
            depth_intrinsic=depth_intrinsic, 
            depth=depth, 
            pose=pose, 
            device=predictor.device
        )  # [valid, 2], [valid]
        if (corre_ins_idx.shape[0] == 0 or point_labels[corre_ins_idx.cpu().numpy()].sum() == 0) and video_points_torch.shape[0] == 0:
            continue

        if os.path.exists(depth_intrinsic_path):
            scale_x = 1920 / 256
            scale_y = 1440 / 192
            # Scale input_point_pos to the original image size
            if input_point_pos.shape[0] > 0:
                input_point_pos = input_point_pos * torch.tensor([scale_x, scale_y]).to(device=predictor.device)
            input_point_pos = input_point_pos.to(device=predictor.device)

            if video_points_torch.shape[0] > 0:  
                video_points_torch = video_points_torch * torch.tensor([1920, 1440]).to(device=predictor.device)
            video_points_torch = video_points_torch.to(device=predictor.device)

        # print(f"input_point_pos: {input_point_pos}")
        # print(f"video_points_torch: {video_points_torch}")

        # print(f"point_labels: {point_labels}")
        # print(f"video_point_labels: {video_point_labels}")

        # print(f"point_labels.type: {type(point_labels)}")
        # print(f"video_point_labels.type: {type(video_point_labels)}")
        
        ins_idxs = torch.arange(1).to(device) # override the ins_idxs to 0 for all points

        input_point_pos = torch.cat([input_point_pos, video_points_torch], dim=0)
        point_labels = np.concatenate([point_labels, video_point_labels], axis=0)

        # corre_ins_idx = corre_ins_idx.to(device=predictor.device)

        corre_ins_idx = torch.cat([corre_ins_idx, torch.arange(start=len(input_point_pos) - len(video_points_torch), end=len(input_point_pos), dtype=corre_ins_idx.dtype).to(device=corre_ins_idx.device)]).to(dtype=torch.int64)

        # raise Exception("Stop here")
        
        if args.sam2 and args.sam2_video and frame_id - set_frame >= args.frame_interval:
            set_frame = frame_id
            if first_frame == -100:
                first_frame = frame_id
                # continue
            process_batch_sam2_video(
                inference_state,
                predictor,
                input_point_pos,
                (point_labels[corre_ins_idx.cpu().numpy()] == 1).astype(np.int32),
                1,
                ins_idxs,
                frame_id,
            )
        elif args.sam2 and not args.sam2_video:
            # For batch processing with SAM2
            if len(batch_frames) < batch_size:
                # Construct paths
                image_jpg_path = os.path.join(args.data_path, args.scene_name, 'color', f'{frame_id}.jpg')
                image_npy_path = os.path.join(args.data_path, args.scene_name, 'color', f'{frame_id}.npy')

                # Check if .npy cache exists
                if os.path.exists(image_npy_path):
                    image = np.load(image_npy_path)
                else:
                    image = cv2.imread(image_jpg_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    np.save(image_npy_path, image)
                
                # Add to batch
                batch_frames.append(image)
                batch_input_points.append(input_point_pos.cpu().numpy())
                batch_point_labels.append((point_labels[corre_ins_idx.cpu().numpy()] == 1).astype(np.int32))
                batch_ins_idxs.append(ins_idxs)
                batch_frame_ids.append(frame_id)
                
                # If batch is full or this is the last frame, process the batch
                if len(batch_frames) == batch_size or frame_id == frame_id_end - 1:
                    # Set batch of images
                    predictor.set_image_batch(batch_frames)
                    
                    # Process batch
                    batch_results = process_batch_sam2_batched(
                        predictor,
                        batch_input_points,
                        batch_point_labels,
                        batch_ins_idxs
                    )
                    
                    # Save results for each frame in the batch
                    for i, frame_id in enumerate(batch_frame_ids):
                        data_original = batch_results[i]
                        data_original.to_numpy()
                        
                        save_file_name = str(frame_id) + ".npy"
                        np.save(f"{args.sam_output_path}/points_npy/{save_file_name}", data_original["points"])
                        np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", data_original["masks"])  
                        np.save(f"{args.sam_output_path}/iou_preds_npy/{save_file_name}", data_original["iou_preds"])  
                        np.save(f"{args.sam_output_path}/corre_3d_ins_npy/{save_file_name}", data_original["corre_3d_ins"])
                    
                    # Clear batch
                    batch_frames = []
                    batch_input_points = []
                    batch_point_labels = []
                    batch_ins_idxs = []
                    batch_frame_ids = []
            
        elif not args.sam2:
            if os.path.exists(f"{args.data_path}/{args.scene_name}/features"):
                # if features are pre-computed, load them
                features = np.load(os.path.join(args.data_path, args.scene_name, 'features', str(frame_id) + '.npy'))
                features = torch.from_numpy(features).to(device=predictor.device)
                predictor.features = features
                predictor.original_size = (480, 640)
                predictor.input_size = (768, 1024)
                predictor.is_image_set = True
            else:
                # Construct paths
                image_jpg_path = os.path.join(args.data_path, args.scene_name, 'color', f'{frame_id}.jpg')
                image_npy_path = os.path.join(args.data_path, args.scene_name, 'color', f'{frame_id}.npy')

                # Check if .npy cache exists
                if os.path.exists(image_npy_path):
                    image = np.load(image_npy_path)
                else:
                    image = cv2.imread(image_jpg_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    np.save(image_npy_path, image)
                predictor.set_image(image)
                
            data_original = process_batch_sam(
                predictor, 
                input_point_pos, 
                (point_labels[corre_ins_idx.cpu().numpy()] == 1).astype(np.int32), 
                ins_idxs, 
                predictor.original_size
            )
            predictor.reset_image()
            
            data_original.to_numpy()

            save_file_name = str(frame_id) + ".npy"
            np.save(f"{args.sam_output_path}/points_npy/{save_file_name}", data_original["points"])
            np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", data_original["masks"])  
            np.save(f"{args.sam_output_path}/iou_preds_npy/{save_file_name}", data_original["iou_preds"])  
            np.save(f"{args.sam_output_path}/corre_3d_ins_npy/{save_file_name}", data_original["corre_3d_ins"])

        if args.sam2_video:
            save_file_name = str(frame_id) + ".npy"
            np.save(f"{args.sam_output_path}/points_npy/{save_file_name}", input_point_pos.cpu().numpy())
            np.save(f"{args.sam_output_path}/iou_preds_npy/{save_file_name}", np.ones(len(input_point_pos)))
            np.save(f"{args.sam_output_path}/corre_3d_ins_npy/{save_file_name}", ins_idxs.cpu().numpy())
        
    if args.sam2 and args.sam2_video:
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track=500, start_frame_idx=first_frame):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            save_file_name = str(out_frame_idx) + ".npy"
            np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", out_mask_logits.cpu().numpy())
        if args.sam2_reverse:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True, max_frame_num_to_track=500, start_frame_idx=first_frame):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                save_file_name = str(out_frame_idx) + ".npy"
                np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", out_mask_logits.cpu().numpy())
    
    # Process any remaining frames in the batch
    elif args.sam2 and not args.sam2_video and len(batch_frames) > 0:
        # Set batch of images
        predictor.set_image_batch(batch_frames)
        
        # Process batch
        batch_results = process_batch_sam2_batched(
            predictor,
            batch_input_points,
            batch_point_labels,
            batch_ins_idxs
        )
        
        # Save results for each frame in the batch
        for i, frame_id in enumerate(batch_frame_ids):
            data_original = batch_results[i]
            data_original.to_numpy()
            
            save_file_name = str(frame_id) + ".npy"
            np.save(f"{args.sam_output_path}/points_npy/{save_file_name}", data_original["points"])
            np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", data_original["masks"])  
            np.save(f"{args.sam_output_path}/iou_preds_npy/{save_file_name}", data_original["iou_preds"])  
            np.save(f"{args.sam_output_path}/corre_3d_ins_npy/{save_file_name}", data_original["corre_3d_ins"])


class Args:
    """A class to hold arguments for 3D prompt proposal."""
    def __init__(
        self,
        data_path="dataset/scannet",
        scene_name="scene0030_00",
        prompt_name=None,
        prompt_path="init_prompt",
        video_points_path="video_points.json",
        experiments_path="experiments",
        model_type="vit_h",
        sam_checkpoint="sam_vit_h_4b8939.pth",
        device="cuda:0",
        sam2=False,
        sam2_checkpoint="sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_video=False,
        sam2_reverse=False,
        frame_interval=100,
        batch_size=8
    ):
        self.data_path = data_path
        self.scene_name = scene_name
        self.prompt_name = prompt_name
        self.prompt_path = prompt_path
        self.video_points_path = video_points_path
        self.experiments_path = experiments_path
        self.model_type = model_type
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.sam2 = sam2
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.sam2_video = sam2_video
        self.sam2_reverse = sam2_reverse
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        
        # Set the output path
        if prompt_name:
            self.sam_output_path = f"{self.experiments_path}/{self.scene_name}/{self.prompt_name}/sam_output"


def run_3d_prompt_proposal(
    data_path: str,
    scene_name: str,
    prompt_name: str,
    prompt_path: str = "init_prompt",
    video_points_path: str = "video_points.json",
    experiments_path: str = "experiments",
    model_type: str = "vit_h",
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    device: str = "cuda:0",
    sam2: bool = False,
    sam2_checkpoint: str = "sam2.1_hiera_large.pt",
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_video: bool = False,
    sam2_reverse: bool = False,
    frame_interval: int = 100,
    batch_size: int = 32,
    profiling: bool = True
) -> None:
    """
    Run the 3D prompt proposal process.
    
    Args:
        data_path: Path to the dataset containing ScanNet 2d frames and 3d .ply files.
        scene_name: The scene names in ScanNet.
        prompt_name: The name of the prompt.
        prompt_path: Path to save the sampled 3D initial prompts.
        video_points_path: Path to the video points.
        experiments_path: Path to the experiments folder.
        model_type: The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b'].
        sam_checkpoint: The path to the SAM checkpoint to use for mask generation.
        device: The device to run generation on.
        sam2: Use SAM2 or not.
        sam2_checkpoint: The path to the SAM2 checkpoint to use for mask generation.
        sam2_config: The path to the SAM2 config file.
        sam2_video: Make SAM2 work on video frames.
        sam2_reverse: Make SAM2 work not only forward but also backward.
        frame_interval: Interval between frames for SAM2 video processing.
        batch_size: Batch size for SAM2 batch processing.
        profiling: Whether to enable profiling.
    """
    if profiling:
        pr = cProfile.Profile()
        pr.enable()
    
    # Create args object
    args = Args(
        data_path=data_path,
        scene_name=scene_name,
        prompt_name=prompt_name,
        prompt_path=prompt_path,
        video_points_path=video_points_path,
        experiments_path=experiments_path,
        model_type=model_type,
        sam_checkpoint=sam_checkpoint,
        device=device,
        sam2=sam2,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_video=sam2_video,
        sam2_reverse=sam2_reverse,
        frame_interval=frame_interval,
        batch_size=batch_size
    )
    
    print("Arguments:")
    print(args.__dict__)

    # Get or create predictor (cached)
    predictor = create_predictor(args)
    
    device = torch.device(args.device)
    
    # Load prompt
    prompt_xyz, prompt_rgb = load_prompt(args, device)
    # print(f"prompt_xyz.shape {prompt_xyz.shape}")

    # Load video points
    video_points = load_video_points(args)

    # Create output folders
    create_output_folders(args)
    
    # Perform SAM on each 2D RGB frame
    frame_id_init = 0
    frame_id_end = len([f for f in os.listdir(os.path.join(args.data_path, args.scene_name, 'depth')) if not f.endswith('.npy')])
    print(f"Start performing SAM segmentations on {frame_id_end} 2D frames...")
    
    sam_seg(
        predictor=predictor, 
        frame_id_init=frame_id_init, 
        frame_id_end=frame_id_end, 
        init_prompt=prompt_xyz, 
        point_labels=np.isclose(np.linalg.norm(prompt_rgb - np.array([0., 255., 0.]), axis=1), 0) if prompt_rgb.shape[0] > 0 else np.array([]),
        video_points=video_points,
        args=args
    )
    print("Finished performing SAM segmentations!")

    if profiling:
        pr.disable()
        # Save stats to a file for snakeviz visualization
        profile_output = "profile_results.prof"
        ps = pstats.Stats(pr)
        ps.sort_stats('cumulative')
        ps.dump_stats(profile_output)
        print(f"Profile data saved to {profile_output}")
        print(f"View visualization with: snakeviz {profile_output}")
        
        # Also print summary to console
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(25)  # Print top 25 entries
        print(s.getvalue())


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3d prompt proposal on ScanNet.")
    # path arguments:
    parser.add_argument('--data_path', default="dataset/scannet", type=str, help='Path to the dataset containing ScanNet 2d frames and 3d .ply files.')
    parser.add_argument('--scene_name', default="scene0030_00", type=str, help='The scene names in ScanNet.')
    parser.add_argument('--prompt_name', required=True, type=str, help='The name of the prompt.')
    parser.add_argument('--prompt_path', default="init_prompt", type=str, help='Path to the save the sampled 3D initial prompts.')
    parser.add_argument('--experiments_path', default="experiments", type=str, help='Path to the experiments folder.')
    # sam arguments:
    parser.add_argument('--model_type', default="vit_h", type=str, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument('--sam_checkpoint', default="sam_vit_h_4b8939.pth", type=str, help='The path to the SAM checkpoint to use for mask generation.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")

    # sam or sam2 arguments:
    parser.add_argument("--sam2", action='store_true', help="Use SAM2 or not.")
    parser.add_argument("--sam2_checkpoint", default="sam2.1_hiera_large.pt", type=str, help="The path to the SAM2 checkpoint to use for mask generation.")
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str, help="The path to the SAM2 config file.")

    # sam2 video arguments:
    parser.add_argument("--sam2_video", action='store_true', help="Make SAM2 work on video frames.")
    parser.add_argument("--sam2_reverse", action='store_true', help="Make SAM2 work not only forward but also backward.")
    parser.add_argument("--frame_interval", default=100, type=int, help="Interval between frames for SAM2 video processing.")
    
    # Batch processing arguments
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for SAM2 batch processing.")
    
    args = parser.parse_args()

    # set the output path
    args.sam_output_path = f"{args.experiments_path}/{args.scene_name}/{args.prompt_name}/sam_output"

    return args
    

if __name__ == "__main__":
    args = get_args()
    
    # Convert args to kwargs for run_3d_prompt_proposal
    kwargs = vars(args)
    run_3d_prompt_proposal(**kwargs, profiling=True)