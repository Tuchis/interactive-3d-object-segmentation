#!/usr/bin/env python3
import os
import numpy as np
import pyvista as pv
import open3d as o3d

# Global lists for storing picked point coordinates, their types, and the actor objects
points = []
point_types = []
point_actors = []

# Current point type ("+" = green, "-" = red)
current_point_type = "+"

def load_mesh(filepath):
    """
    Load a .ply mesh. If the file does not exist, create a default sphere mesh.
    """
    if not os.path.exists(filepath):
        print(f"File '{filepath}' not found. Creating a default sphere mesh.")
        sphere = pv.Sphere(radius=1.0)
        sphere.save(filepath)
    mesh = pv.read(filepath)
    return mesh


def add_point_callback(all_points_data, event=None):
    """
    Callback for point picking. Called when the user left-clicks on the mesh.
    """
    global points, point_types, point_actors, current_point_type, plotter, mesh
    if all_points_data is None:
        return
    # Save the picked point and its type
    picked_point = all_points_data.points[event]
    points.append(picked_point)
    point_types.append(current_point_type)
    # Choose a color based on the current point type
    color = "green" if current_point_type == "+" else "red"
    # Create a small sphere (glyph) to mark the point
    sphere = pv.Sphere(radius=.01, center=picked_point)
    actor = plotter.add_mesh(sphere, color=color, name=f"point_{len(points)}", pickable=False)
    point_actors.append(actor)
    print(f"Added point at {all_points_data} with type '{current_point_type}'")

def remove_last_point():
    """
    Remove the most recently added point from both the scene and our records.
    """
    global points, point_types, point_actors, plotter
    if points:
        removed_point = points.pop()
        removed_type = point_types.pop()
        actor = point_actors.pop()
        plotter.remove_actor(actor)
        print(f"Removed point at {removed_point} with type '{removed_type}'")
    else:
        print("No points to remove.")

def save_points():
    """
    Save the picked points as a point cloud (with colors) to 'output.ply'
    using Open3D.
    """
    global points, point_types
    if not points:
        print("No points to save.")
        return
    print("Saving points to 'output.ply'...")
    print(f"Points: {points}")
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    points_array = np.array(points)
    pcd.points = o3d.utility.Vector3dVector(points_array)
    # Set colors: green ([0,1,0]) for '+' and red ([1,0,0]) for '-'
    colors = np.array([[0, 1, 0] if t == "+" else [1, 0, 0] for t in point_types])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output_file = "output.ply"
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved {len(points)} point(s) to '{output_file}'.")

def toggle_point_type():
    """
    Toggle the current point type between '+' (green) and '-' (red).
    Also update the on-screen text.
    """
    global current_point_type, plotter
    current_point_type = "-" if current_point_type == "+" else "+"
    # Update the text actor (using the same name "pt_type" to replace the old text)
    plotter.add_text(f"Current Point Type: {current_point_type}", position="upper_left",
                     name="pt_type", font_size=10, color="white")
    print(f"Current point type toggled to '{current_point_type}'")

if __name__ == "__main__":
    # Create a PyVista plotter window
    plotter = pv.Plotter(window_size=(1920, 1080))
    
    # Load the mesh from "input.ply" (or create a default sphere if not found)
    mesh = load_mesh("input.ply")
    plotter.add_mesh(mesh, opacity=1, name="mesh", rgb=True)
    
    # Display the current point type on the upper left corner
    plotter.add_text(f"Current Point Type: {current_point_type}", position="upper_left",
                     name="pt_type", font_size=10, color="white")
    
    # Enable point picking on left mouse clicks (click on the mesh to add a point)
    plotter.enable_point_picking(callback=add_point_callback, use_mesh=True,
                                 show_message=True, left_clicking=False, 
                                 color="green" if current_point_type == "+" else "red", point_size=100)
    
    # Add key event callbacks:
    # Press 'r' to remove the last point.
    plotter.add_key_event('r', remove_last_point)
    # Press 's' to save points to output.ply.
    plotter.add_key_event('s', save_points)
    # Press 't' to toggle the current point type.
    plotter.add_key_event('t', toggle_point_type)
    
    # Print instructions to the console
    print("Instructions:")
    print(" • Left-click on the mesh to add a point (a small sphere will be drawn).")
    print(" • Press 't' to toggle the current point type between '+' (green) and '-' (red).")
    print(" • Press 'r' to remove the last point.")
    print(" • Press 's' to save points to 'output.ply'.")
    
    # Start the interactive visualization window
    plotter.show()
