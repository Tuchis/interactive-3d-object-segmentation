import numpy as np
import plotly.graph_objects as go
import pandas as pd

def read_colmap_images(file_path):
    """
    Read the COLMAP images.txt file and extract camera positions.
    
    Args:
        file_path: Path to the images.txt file
    
    Returns:
        DataFrame with camera positions and image names
    """
    # Initialize lists to store data
    image_ids = []
    qws = []
    qxs = []
    qys = []
    qzs = []
    txs = []
    tys = []
    tzs = []
    camera_ids = []
    names = []
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first 4 lines (header)
    lines = lines[4:]
    
    # Process every other line (skipping the POINTS2D lines)
    for i in range(0, len(lines), 2):
        if i < len(lines):
            line = lines[i].strip()
            if line:
                parts = line.split()
                if len(parts) >= 10:  # Ensure we have enough parts
                    image_ids.append(int(parts[0]))
                    qws.append(float(parts[1]))
                    qxs.append(float(parts[2]))
                    qys.append(float(parts[3]))
                    qzs.append(float(parts[4]))
                    txs.append(float(parts[5]))
                    tys.append(float(parts[6]))
                    tzs.append(float(parts[7]))
                    camera_ids.append(int(parts[8]))
                    names.append(parts[9])
    
    # Create a DataFrame
    data = {
        'image_id': image_ids,
        'qw': qws,
        'qx': qxs,
        'qy': qys,
        'qz': qzs,
        'tx': txs,
        'ty': tys,
        'tz': tzs,
        'camera_id': camera_ids,
        'name': names
    }
    
    return pd.DataFrame(data)

def plot_camera_trajectory(file_path):
    """
    Plot the 3D trajectory of camera positions from COLMAP images.txt file.
    
    Args:
        file_path: Path to the images.txt file
    """
    # Read the camera positions
    df = read_colmap_images(file_path)
    
    # Create a 3D plot for camera positions
    camera_trace = go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['tz'],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=np.arange(len(df)),
            colorscale='Viridis',
            opacity=0.8
        ),
        line=dict(
            color='darkblue',
            width=2
        ),
        text=df['name'],
        hoverinfo='text',
        name='Camera Positions'
    )
    
    # Create traces for quaternion components
    qw_trace = go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['qw'],
        mode='lines',
        line=dict(color='red', width=2),
        name='QW'
    )
    
    qx_trace = go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['qx'],
        mode='lines',
        line=dict(color='green', width=2),
        name='QX'
    )
    
    qy_trace = go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['qy'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='QY'
    )
    
    qz_trace = go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['qz'],
        mode='lines',
        line=dict(color='purple', width=2),
        name='QZ'
    )
    
    # Create figure with all traces
    fig = go.Figure(data=[camera_trace, qw_trace, qx_trace, qy_trace, qz_trace])
    
    # Update layout
    fig.update_layout(
        title='Camera Trajectory and Quaternions from COLMAP',
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position / Quaternion Value',
            aspectmode='data'
        ),
        width=900,
        height=700,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )
    
    return fig

if __name__ == "__main__":
    # Path to the images.txt file
    file_path = "/home/kuzhum/sam/promptable-3d-object-segmentation/scannet_pp_dataset/data/data/56a0ec536c/iphone/colmap/images.txt"
    
    # Plot the trajectory
    fig = plot_camera_trajectory(file_path)
    
    # Show the plot
    fig.show()
