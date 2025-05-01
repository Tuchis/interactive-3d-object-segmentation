"""
Constants and configuration values for the backend application.
"""
import os

# Server configuration
HOST = "0.0.0.0"
PORT = 5100

# CORS configuration
ALLOWED_ORIGINS = ["http://localhost:5173"]
ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
ALLOWED_HEADERS = ["Content-Type", "Authorization"]

# File storage configuration
UPLOAD_FOLDER_NAME = "files"
SAMPLE_PLY_NAME = "sample.ply"

# Specific PLY file configuration
SPECIFIC_PLY_FILENAME = "scene0008_00_seg_sam.ply"
SPECIFIC_PLY_PATH = os.path.join("/Users/vhumenn/Downloads", SPECIFIC_PLY_FILENAME)

# API paths
API_PREFIX = ""  # Set to "/api" if you want all endpoints to be prefixed with /api
PROCESS_MARKERS_ENDPOINT = "/process_markers"
AVAILABLE_PLY_FILES_ENDPOINT = "/available_ply_files"
DOWNLOAD_PLY_ENDPOINT = "/download_ply/{filename}" 

# PLY file save path
PLY_SAVE_PATH = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments/temp/output.ply"
VIDEO_POINTS_SAVE_PATH = "/home/kuzhum/sam/promptable-3d-object-segmentation/SAMPro3D/experiments/temp/video_points.json"