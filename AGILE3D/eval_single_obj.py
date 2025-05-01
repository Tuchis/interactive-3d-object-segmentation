# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import sys
import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model, build_criterion
import MinkowskiEngine as ME
from utils.seg import mean_iou_scene, extend_clicks, get_simulated_clicks
import utils.misc as utils

sys.path.append('../SAMPro3D')

from scripts.sam_pro_3d_custom import SAMPro3DCustom

from evaluation.evaluator_SO import EvaluatorSO
import wandb
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)

    # dataset
    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--dataset_mode', default='single_obj')
    parser.add_argument('--scan_folder', default='data/ScanNet/scans', type=str)
    parser.add_argument('--crop', default=False, action='store_true', help='whether evaluate on whole scan or object crops')
    parser.add_argument('--val_list', default='data/ScanNet/single/object_ids.npy', type=str)
    parser.add_argument('--val_list_classes', default='data/ScanNet/single/object_classes.txt', type=str)
    parser.add_argument('--train_list', default='', type=str)
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool)

    # evaluation
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint1099.pth', help='resume from checkpoint')

    parser.add_argument('--max_num_clicks', default=5, help='maximum number of clicks per object on average', type=int)

    # SAMPro3D
    parser.add_argument('--use_sampro3d', help='use SAMPro3D model for evaluation', action='store_true')
    # Neg background
    parser.add_argument('--neg_bg', help='use negative background for SAMPro3D model', action='store_true')
    # Use SAM2
    parser.add_argument('--use_sam2', help='use SAM2 model for evaluation', action='store_true')
    # Propagate video
    parser.add_argument('--propagate_video', help='propagate video for SAMPro3D model', action='store_true')
    # Segmentation threshold
    parser.add_argument('--segmentation_threshold', default=0.5, type=float, help='segmentation threshold for SAMPro3D model')
    # Reverse propagation
    parser.add_argument('--reverse_propagation', help='reverse propagation for SAMPro3D model', action='store_true')
    # Frame interval
    parser.add_argument('--frame_interval', default=1, type=int, help='frame interval for SAMPro3D model')

    return parser



def Evaluate(model, data_loader, args, device):
    if not args.use_sampro3d:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    instance_counter = 0
    results_file_name = 'val_results_single_'
    if not args.use_sampro3d:
        results_file_name += "AGILE3D"
    else:
        results_file_name += "SAMPro3D"
        if args.use_sam2:
            results_file_name += "_SAM2"
        results_file_name += f"_seg_thresh_{args.segmentation_threshold}"
        if args.neg_bg:
            results_file_name += "_neg_bg"
        if args.use_sam2:
            if args.propagate_video:
                results_file_name += "_propagate_video"
                if args.reverse_propagation:
                    results_file_name += "_reverse_propagation"
                if args.frame_interval:
                    results_file_name += f"_frame_interval_{args.frame_interval}"
    
    results_file_name += ".csv"
    results_file = os.path.join(args.output_dir, results_file_name)
    if os.path.exists(results_file):
        # Read results_file
        with open(results_file, 'r') as f:
            results = f.readlines()
        
        # Get last object instance id
        last_object_instance_id = results[-1].split(' ')[0]
        # Convert to int
        last_object_instance_id = int(last_object_instance_id)
    else:
        last_object_instance_id = -1
    
    f = open(results_file, 'a')

    print(f"{results_file=}")

    sampro3d = args.use_sampro3d

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, object_id = batched_inputs

        # Skip if object was already evaluated
        if instance_counter <= last_object_instance_id:
            instance_counter += len(object_id)
            continue

        print(f"{scene_name=}")
        print(f"{labels=}")
        print(f"{labels_full=}")
        print(f"{object_id=}")

        if sampro3d:
            scene_exists = model.set_scene(scene_name[0])
            model.reset_clicks()
            if not scene_exists:
                continue

        coords = coords.to(device)
        raw_coords = raw_coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_full = [l.to(device) for l in labels_full]

        data = ME.SparseTensor(
                                coordinates=coords,
                                features=feats,
                                device=device
                                )

        ###### interactive evaluation ######
        batch_idx = coords[:,0]
        batch_size = batch_idx.max()+1

        # click ids set null
        click_idx = [{'0':[],'1':[]} for b in range(batch_size)]

        click_time_idx = copy.deepcopy(click_idx)

        current_num_clicks = 0

        # pre-compute backbone features only once
        if not sampro3d:
            pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        max_num_clicks = args.max_num_clicks

        last_click_id = None

        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(device) for l in labels]
            else:

                # Main model forward pass, can be swaped to our models.
                # Input is:
                # 1. pcd_features: backbone features
                # 2. aux: auxiliary features
                # 3. coordinates: coordinates of the input point cloud
                # 4. pos_encodings_pcd: positional encodings of the input point cloud
                # 5. click_idx: click ids (that is the main thing that can be used for our approach)
                # 6. click_time_idx: click time ids
                if sampro3d:
                    try:
                        # Get the last click from 0 or 1
                        if click_idx[-1]['1']:
                            if last_click_id == click_idx[-1]['1'][-1]:
                                click_id = click_idx[-1]['0'][-1]
                            else:
                                click_id = click_idx[-1]['1'][-1]
                            last_click_id = click_id
                        else:
                            click_id = click_idx[-1]['0'][-1]

                        model.add_click(raw_coords[click_id])
                    except IndexError:
                        pass
                    model.run_prompt_proposal()
                    segmentation = model.run_segmentation(visualize=False, coords_qv=raw_coords.cpu().numpy())

                    unique, counts = np.unique(segmentation, return_counts=True)

                    # Change -1 to 0 and 0 to 1
                    segmentation[segmentation == 0] = 1
                    segmentation[segmentation == -1] = 0

                    # Make not numpy but tensor and to cuda
                    pred = [torch.from_numpy(segmentation).to(device)]
                else:
                    try:
                        outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                                    click_idx=click_idx, click_time_idx=click_time_idx)
                        pred_logits = outputs['pred_masks']
                        pred = [p.argmax(-1) for p in pred_logits]
                    except IndexError:
                        pass


            # Count values in pred
            unique, counts = np.unique(pred[0].cpu().numpy(), return_counts=True)


            updated_pred = []

            for idx in range(batch_idx.max()+1):
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]

                # sample_feats = feats[sample_mask]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_map[idx]]

                sample_labels_full = labels_full[idx]
                try:
                    sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)
                except ZeroDivisionError:
                    sample_iou = torch.tensor(0.0)
                except torch.OutOfMemoryError:
                    print(f"Out of memory error for instance {instance_counter+idx}")
                    sample_iou = torch.tensor(0.0)

                line = str(instance_counter+idx) + ' ' + scene_name[idx].replace('scene','') + ' '  + object_id[idx] + ' ' + str(current_num_clicks) +  ' ' + str(
                sample_iou.cpu().numpy()) + '\n'
                f.write(line)
                f.flush()

                print(scene_name[idx], 'Object: ', object_id[idx], 'num clicks: ', current_num_clicks, 'IOU: ', sample_iou.item())
    
                # Can be used to simulate clicks for evaluation even for our approach
                new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, current_num_clicks, training=False)

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)


            current_num_clicks += 1

        instance_counter += len(object_id)

    f.close()
    evaluator = EvaluatorSO(args.dataset, args.val_list, args.val_list_classes, results_file, [0.5,0.65,0.8,0.85,0.9])
    results_dict = evaluator.eval_results()

def main(args):
    sampro3d = args.use_sampro3d

    print(f"{args=}")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    if not sampro3d:
        model = build_model(args)
        model.to(device)
    else:
        print("AGILE3D not available, using SAMPro3D")
        model = SAMPro3DCustom()
        model.set_model('sam2' if args.use_sam2 else 'sam')
        model.set_neg_bg(args.neg_bg)
        model.set_video(args.propagate_video)
        model.set_reverse(args.reverse_propagation)
        model.set_frame_interval(args.frame_interval)
        model.set_segmentation_threshold(args.segmentation_threshold)

    # build dataset and dataloader
    dataset_val, collation_fn_val = build_dataset(split='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collation_fn_val, num_workers=args.num_workers,
                                 pin_memory=True)

    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not sampro3d:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
      
    Evaluate(model, data_loader_val, args, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script on interactive multi-object segmentation ', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)