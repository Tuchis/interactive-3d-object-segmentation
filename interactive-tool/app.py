from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import tempfile
import shutil
from typing import Optional, List
import uvicorn
import sys
from utils import save_points, save_video_points

# Import SAMPro3D
sys.path.append("..")
from SAMPro3D.scripts.sam_pro_3d_custom import SAMPro3DCustom

# Import constants
from constants import (
    HOST, PORT, 
    ALLOWED_ORIGINS, ALLOWED_METHODS, ALLOWED_HEADERS,
    UPLOAD_FOLDER_NAME, SAMPLE_PLY_NAME, SPECIFIC_PLY_PATH,
    API_PREFIX, PROCESS_MARKERS_ENDPOINT, AVAILABLE_PLY_FILES_ENDPOINT, DOWNLOAD_PLY_ENDPOINT
)

app = FastAPI(title="PLY Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)

# Create a directory to store PLY files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample PLY file for testing
SAMPLE_PLY = os.path.join(UPLOAD_FOLDER, SAMPLE_PLY_NAME)

@app.post(API_PREFIX + PROCESS_MARKERS_ENDPOINT)
async def process_markers(
    point_cloud_markers: str = Form(...),
    video_markers: str = Form(...),
    settings: str = Form(...),
    ply_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None)
):
    """
    Process both 3D point cloud markers and 2D video markers and return a modified PLY file
    
    Args:
        point_cloud_markers: JSON string of 3D marker data
        video_markers: JSON string of 2D video marker data
        settings: JSON string of processing settings
        ply_file: Optional PLY file to process
        video_file: Optional video file to process
    
    Returns:
        The processed PLY file
    """
    try:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()

        # Parse settings
        try:
            settings_data = json.loads(settings)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid settings JSON format"}
            )

        # Get video filename if video file was uploaded
        video_filename = None
        scene_name = None
        if video_file:
            video_filename = video_file.filename
            scene_name = video_filename.split('.')[0]
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "Video file is required"}
            )
        
        # Parse markers from JSON
        try:
            point_cloud_data = json.loads(point_cloud_markers)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid point cloud markers JSON format"}
            )

        # Save point cloud data to a ply file
        save_points(point_cloud_data)
        print("Point cloud data saved to 'output.ply'")

        video_points_data = json.loads(video_markers)
        print(video_points_data)
        save_video_points(video_points_data)


        # Run SAMPro3D
        model = SAMPro3DCustom()
        scene_exists = model.set_scene(scene_name)
        
        if not scene_exists:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Scene '{scene_name}' not found"}
            )
            
        model.set_neg_bg(settings_data.get('negativeBackground', False))
        model.set_model('sam2' if settings_data.get('useSAM2', False) else 'sam')
        model.set_video(settings_data.get('propagateVideo', False))
        model.set_segmentation_threshold(settings_data.get('segmentationThreshold', 0.5))
        model.set_reverse(settings_data.get('reversePropagation', False))
        model.set_frame_interval(settings_data.get('frameInterval', 1))

        model.run_prompt_proposal()
        ply_path = model.run_segmentation(return_vis_path=True, visualize=True)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        return FileResponse(
            ply_path, 
            media_type="application/octet-stream",
            filename="processed.ply"
        )

    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error in process_markers: {str(e)}")
        print(traceback.format_exc())
        
        # Return appropriate error response
        if isinstance(e, HTTPException):
            raise e
        
        # Determine if this is a client error (400) or server error (500)
        if any(keyword in str(e).lower() for keyword in ["invalid", "not found", "missing", "required"]):
            return JSONResponse(
                status_code=400,
                content={"detail": str(e)}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )

@app.get(API_PREFIX + AVAILABLE_PLY_FILES_ENDPOINT, response_model=List[dict])
async def available_ply_files():
    """
    Return a list of available PLY and video files
    
    Returns:
        List of file objects with name, type, and size
    """
    try:
        # Get all PLY files
        ply_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.ply')]
        
        # Get all video files
        video_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp4', '.mov'))]
        
        # Add the specific PLY file if it exists
        if os.path.exists(SPECIFIC_PLY_PATH):
            specific_filename = os.path.basename(SPECIFIC_PLY_PATH)
            if specific_filename not in ply_files:
                ply_files.append(specific_filename)
        
        # Create a list of file objects with name, type, and size
        files = []
        
        # Add PLY files
        for filename in ply_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(file_path):
                file_path = SPECIFIC_PLY_PATH if filename == os.path.basename(SPECIFIC_PLY_PATH) else None
            
            if file_path and os.path.exists(file_path):
                files.append({
                    'name': filename,
                    'type': 'ply',
                    'size': os.path.getsize(file_path)
                })
        
        # Add video files
        for filename in video_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                files.append({
                    'name': filename,
                    'type': 'video',
                    'size': os.path.getsize(file_path)
                })
        
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(API_PREFIX + DOWNLOAD_PLY_ENDPOINT)
async def download_ply(filename: str):
    """
    Download a specific PLY file
    
    Args:
        filename: Name of the PLY file to download
    
    Returns:
        The requested PLY file
    """
    try:
        # Check if the requested file is the specific PLY file
        specific_filename = os.path.basename(SPECIFIC_PLY_PATH)
        if filename == specific_filename and os.path.exists(SPECIFIC_PLY_PATH):
            return FileResponse(
                SPECIFIC_PLY_PATH,
                media_type="application/octet-stream",
                filename=filename
            )
        
        # Ensure the filename is safe
        if ".." in filename or filename.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path, 
            media_type="application/octet-stream",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"API documentation available at http://{HOST}:{PORT}/docs")
    
    # Check if the specific PLY file exists
    if os.path.exists(SPECIFIC_PLY_PATH):
        print(f"Specific PLY file found: {SPECIFIC_PLY_PATH}")
    else:
        print(f"Warning: Specific PLY file not found at {SPECIFIC_PLY_PATH}")
        print("The server will fall back to using sample PLY files.")
    
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True) 