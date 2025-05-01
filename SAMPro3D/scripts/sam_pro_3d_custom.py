import os
import shutil
import numpy as np
import open3d as o3d
import torch

output_ply_file = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments/temp/output.ply"
video_points_path = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments/temp/video_points.json"
sam_pro_3d_custom_path = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/scripts"

# Import the 3d_prompt_proposal module
import sys
sys.path.append(sam_pro_3d_custom_path)
# Import using string to avoid syntax error with module name starting with number
from importlib import import_module
prompt_proposal_module = import_module("3d_prompt_proposal")
run_3d_prompt_proposal = prompt_proposal_module.run_3d_prompt_proposal

main_module = import_module("main_sam")

# Load required functions from main module
perform_3dsegmentation = main_module.perform_3dsegmentation
prompt_consolidation = main_module.prompt_consolidation
num_to_natural = main_module.num_to_natural
create_folder = main_module.create_folder
get_color_rgb = main_module.get_color_rgb
rand_cmap = main_module.rand_cmap
load_ply = main_module.load_ply
mean_iou_single = main_module.mean_iou_single
get_labels = main_module.get_labels
merge_floor = main_module.merge_floor
ransac_plane_seg = main_module.ransac_plane_seg


class SAMPro3DCustom:
    def __init__(self):
        self.data_path = "/home/rpartsey/code/eai/SAMPro3D-fork/data/scannet/scans"
        self.scene_name = None
        self.prompt_path = output_ply_file
        self.video_points_path = video_points_path
        self.experiments_path = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments"
        self.experiment_name = 'sampro3d'
        self.experiments_dir = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments"

        # self.sam2_checkpoint = "sam2.1_hiera_large.pt"
        # self.sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # self.sam2_checkpoint = "sam2.1_hiera_tiny.pt"
        self.sam_checkpoint = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/scripts/sam_vit_h_4b8939.pth"
        self.sam2_checkpoint = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/scripts/sam2.1_hiera_tiny.pt"
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"

        self.device = "cuda:0"

        self.sam2 = False

        self.env_name = "sampro3d" if not self.sam2 else "sam2pro3d"

        self.sam2_video = False
        self.sam2_reverse = False
        self.frame_interval = False

        self.neg_bg = False
        self.segmentation_threshold = 0.0

        model = 'sam' if not self.sam2 else 'sam2'
        self.model = model

        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

        self.clicks = []
        self.point_types = []
    def set_model(self, model):
        if model not in ['sam', 'sam2']:
            raise ValueError(f"Invalid model: {model}")
        if model == 'sam2':
            self.sam2 = True
            self.env_name = "sam2pro3d"
            self.model = 'sam2'
            self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"
        else:
            self.sam2 = False
            self.env_name = "sampro3d"
            self.model = 'sam'
            self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

    def set_video(self, video=False):
        self.sam2_video = video
        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"


    def set_neg_bg(self, neg_bg=False):
        self.neg_bg = neg_bg
        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

    def set_segmentation_threshold(self, segmentation_threshold=0.0):
        self.segmentation_threshold = segmentation_threshold
        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

    def set_reverse(self, reverse=False):
        self.sam2_reverse = reverse
        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

    def set_frame_interval(self, frame_interval=False):
        self.frame_interval = frame_interval
        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

    def save_clicks(self):
        """
        Save the picked points as a point cloud (with colors) to 'output.ply'
        using Open3D.
        """
        points = self.clicks
        if not points:
            print("No points to save.")
            return
        print("Saving points to 'output.ply'...")
        print(f"Points: {points}")

        point_types = self.point_types

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points_array = np.array([point.tolist() for point in points])
        pcd.points = o3d.utility.Vector3dVector(points_array)
        # Set colors: green ([0,1,0]) for '+' and red ([1,0,0]) for '-'
        colors = np.array([[0, 1, 0] if t == "+" else [1, 0, 0] for t in point_types])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_ply_file, pcd)
        print(f"Saved {len(points)} point(s) to '{output_ply_file}'.")

    def add_click(self, click, type="+"):
        self.clicks.append(click)
        self.point_types.append(type)
        print(f"{self.clicks=}")
        self.save_clicks()


    def reset_clicks(self):
        self.clicks = []
        self.point_types = []
        self.save_clicks()


    def set_scene(self, scene_name):
        scene_path = os.path.join(self.data_path, scene_name)
        if not os.path.exists(scene_path):
            return False
        self.scene_name = scene_name

        model = 'sam' if not self.sam2 else 'sam2'

        self.output_folder = f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}_{model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"

        return True

    def set_prompt_path(self, prompt_path):
        self.prompt_path = prompt_path

    def run_prompt_proposal(self):
        # Import the 3d_prompt_proposal module
        import sys
        sys.path.append(sam_pro_3d_custom_path)
        # Import using string to avoid syntax error with module name starting with number
        from importlib import import_module
        prompt_proposal_module = import_module("3d_prompt_proposal")
        run_3d_prompt_proposal = prompt_proposal_module.run_3d_prompt_proposal
        
        print(f"video_points_path: {self.video_points_path}")
        # Run the 3D prompt proposal directly
        run_3d_prompt_proposal(
            data_path=self.data_path,
            scene_name=self.scene_name,
            prompt_name=self.experiment_name,
            prompt_path=self.prompt_path,
            video_points_path=self.video_points_path,
            experiments_path=self.experiments_path,
            device=self.device,
            sam2=self.sam2,
            sam_checkpoint=self.sam_checkpoint,
            sam2_checkpoint=self.sam2_checkpoint,
            sam2_config=self.sam2_config,
            sam2_video=self.sam2_video,
            sam2_reverse=self.sam2_reverse,
            frame_interval=self.frame_interval
        )

        # If folder exists, remove it
        if os.path.exists(f"{self.output_folder}"):
            shutil.rmtree(f"{self.output_folder}")
        shutil.move(f"{self.experiments_dir}/{self.scene_name}/{self.experiment_name}", f"{self.output_folder}")

    def run_segmentation(self, return_vis_path=False, visualize=True, coords_qv=None):
        import open3d as o3d
        # Create a mock args object to match the expected interface
        class Args:
            pass
        
        args = Args()
        args.data_path = self.data_path
        args.scene_name = self.scene_name
        args.prompt_path = self.prompt_path
        args.sam_output_path = f"{self.output_folder}/sam_output"
        args.pred_path = f"{self.output_folder}/predictions"
        args.output_vis_path = f"{self.output_folder}/visualizations"
        args.device = self.device
        args.sam2 = self.sam2
        args.sam2_video = self.sam2_video
        args.neg_bg = self.neg_bg
        args.segmentation_threshold = self.segmentation_threshold
        # Default values for other parameters
        args.pred_iou_thres = 0.7
        args.stability_score_thres = 0.6
        args.box_nms_thres = 0.8
        args.keep_thres = 0.4
        args.post_floor = False
        
        # Load the initial 3D input prompts
        init_prompt, _ = load_ply(self.prompt_path)
        print("the number of initial prompts", init_prompt.shape[0])

        if coords_qv is not None:
            xyz = coords_qv
            rgb = None
        else:
            # Load all 3D points of the input scene
            scene_plypath = os.path.join(self.data_path, self.scene_name, self.scene_name + '_vh_clean_2.ply')
            xyz, rgb = load_ply(scene_plypath)
        
        # Load SAM segmentations
        scene_output_path = args.sam_output_path
        create_folder(scene_output_path)
        create_folder(os.path.join(scene_output_path, "points_npy"))
        create_folder(os.path.join(scene_output_path, "iou_preds_npy"))
        create_folder(os.path.join(scene_output_path, "masks_npy"))
        create_folder(os.path.join(scene_output_path, "corre_3d_ins_npy"))
        create_folder(args.pred_path)
        create_folder(args.output_vis_path)
        
        # Get the list of npy files
        points_npy_dir = os.path.join(scene_output_path, "points_npy")
        if os.path.exists(points_npy_dir) and os.listdir(points_npy_dir):
            points_npy_path = sorted([f for f in os.listdir(points_npy_dir) if f.endswith('.npy')])
        else:
            print("No points_npy files found. This may be expected if this is the first run.")
            points_npy_path = []
        
        # Create a tensor with all prompt indices
        device = torch.device(self.device)
        keep_idx = torch.arange(0, init_prompt.shape[0]).to(device)

        # Perform 3D segmentation
        pt_score_abs, pt_pred_abs, pt_score_mean = perform_3dsegmentation(xyz, keep_idx, scene_output_path, points_npy_path, args)
        
        # Prompt Consolidation
        pt_pred = prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean)
        
        # Finalize prediction
        pt_pred = num_to_natural(pt_pred)

        if coords_qv is None:
            labels = get_labels(self.scene_name)

            iou_pred = pt_pred.copy()
            # Change -1 to 0 and 0 to 1
            iou_pred[iou_pred == 0] = 1
            iou_pred[iou_pred == -1] = 0

            iou_pred = torch.from_numpy(iou_pred)
            labels = torch.from_numpy(labels)

            # We have to find the label, that has the biggest intersection with pt_pred
            res = iou_pred*labels
            # Count the number of each label and find the one with the biggest count, except negative labels
            unique, counts = np.unique(res, return_counts=True)
            counts = counts[unique != 0]
            unique = unique[unique != 0]
            most_common_label = unique[np.argmax(counts)]
            print(f"most_common_label: {most_common_label}")

            # We now compute what labels are the most predicted for 0, and compute IoU for that label
            iou_with_most_common_label = mean_iou_single(iou_pred==1, labels==most_common_label)
            print(f"iou_with_most_common_label: {iou_with_most_common_label}")

            # If results.txt file does not exist, create it
            if not os.path.exists("results.txt"):
                with open("results.txt", "w") as f:
                    f.write("scene_name label iou num_clicks model neg_bg segmentation_threshold video reverse frame_interval\n")

            # Open results.txt file
            with open(f"results.txt", "a") as f:
                f.write(f"{self.scene_name} {most_common_label} {iou_with_most_common_label} {len(self.clicks)} {self.model} {self.neg_bg} {self.segmentation_threshold} {self.sam2_video} {self.sam2_reverse} {self.frame_interval}\n")
        
        # Save the prediction result
        pred_file = os.path.join(args.pred_path, self.scene_name + '_seg.npy')
        np.save(pred_file, pt_pred)

        if coords_qv is None:
            args.post_floor = False
            # Post process for perfect floor segmentation:
            if args.post_floor:
                print("Start post-processing the floor ...")
                # Generate floor instance proposal (min height of the current scene + scene_height_threshold)
                floor_proposal_masks = xyz[:, 2] < min(xyz[:, 2]) + args.scene_ht_thres  # define an initial area of the floor according to the height
                xyz_id = np.arange(len(xyz))
                floor_proposal_ids = xyz_id[floor_proposal_masks]
                floor_id = int(pt_pred.max()) + 1
                # Merge instances that have large overlap with the floor_proposal
                pt_pred = merge_floor(pt_pred, floor_proposal_ids, floor_id, args.scene_inter_thres)
                # Run RANSAC to finally refine the previous plane segmentation if there are still some actual floor areas does not segmented as floor (this can usually be skipped)
                pt_pred = ransac_plane_seg(scene_plypath, pt_pred, floor_id, args.scene_dist_thres)
                print("Finished post-processing the floor!")
                print("********************************************************")
                # save the prediction result:
                pred_file = os.path.join(args.pred_path, args.scene_name + '_seg_floor.npy')
                np.save(pred_file, pt_pred)
            
            print("Creating the visualization result ...")
            create_folder(args.output_vis_path)
            mesh_ori = o3d.io.read_triangle_mesh(scene_plypath)
            pred_ins_num = int(pt_pred.max())+1
            cmap = rand_cmap(pred_ins_num, type='bright', first_color_black=False, last_color_black=False, verbose=False)
            c_all = get_color_rgb(xyz, pt_pred, rgb, cmap, make_gray=True)
            # Reshape from (N, 1, 3) to (N, 3)
            c = c_all.reshape(-1, 3)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = mesh_ori.vertices
            mesh.triangles = mesh_ori.triangles
            mesh.vertex_colors = o3d.utility.Vector3dVector(c)
            output_vis_file = os.path.join(args.output_vis_path, args.scene_name + '_seg.ply')
            o3d.io.write_triangle_mesh(output_vis_file, mesh)
            print("Successfully save the visualization result of final segmentation!")
            
        # Create visualization
        if visualize:
            # Create video visualization if enabled
            sys.path.append('/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/scripts/sam_utils')
            from vis_utils import create_visualization_video_with_opencv
            
            # Generate random colors for each object
            colors = [[255, 0, 0]]  # BGR format
            object_names = [name for name in os.listdir(f"{self.experiments_dir}/{self.scene_name}") 
                        if f"_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}_" in name]
            
            object_names = [f"{self.experiment_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}{'_' + str(self.frame_interval) if self.frame_interval else ''}_{self.segmentation_threshold}{'_neg' if self.neg_bg else ''}"]
            
            # Get number of frames in the scene
            color_files = [f for f in os.listdir(f"{self.data_path}/{self.scene_name}/color") if not (f.endswith('.png') or f.endswith('.jpg'))]
            num_frames = len(color_files)
            
            create_visualization_video_with_opencv(
                start_frame_idx=0,
                end_frame_idx=num_frames,
                output_video_path=f"{self.scene_name}_{self.model}{'video' if self.sam2_video else ''}{'reverse' if self.sam2_reverse else ''}.mp4",
                frame_size= (1920, 1440),
                colors=colors,
                dataset_dir=self.data_path, 
                scene_name=self.scene_name, 
                experiments_dir=self.experiments_dir,
                object_names=object_names,
                model=self.model,
            )
        
        # Return visualization path if requested
        if return_vis_path:
            return f"{self.output_folder}/visualizations/{self.scene_name}_seg.ply"
        
        # Return predictions
        return pt_pred
        