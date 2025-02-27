import open3d as o3d
import pymeshlab
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

def load_off_mesh(file_path):
    """Load an OFF mesh file using Open3D"""
    if not file_path.lower().endswith('.off'):
        raise ValueError("This function is designed for OFF files only")
    
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_stats(mesh):
    """Get key statistics about a mesh"""
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    return f"Vertices: {n_vertices}, Faces: {n_faces}"

def save_simplified_mesh(mesh, object_path, Initial_Name, show_details = False):
    """Save the simplified mesh to disk"""
    file_name = object_path.split("/")[-1]
    object_folder = object_path.removesuffix(object_path.split("/")[-1])
    simplified_dir = 'Data\Simplified_' + str(Initial_Name) + '/' + object_folder
    simplified_filename = simplified_dir + '/' + file_name
    if show_details:
        print("Filename: " + file_name + "\n" + "Simplified Filename: " + simplified_filename + "\n" + "Simplified Dir: " + simplified_dir + "\n" + "Object Path: " + object_path + "\n")         

    os.makedirs(simplified_dir, exist_ok=True)

    o3d.io.write_triangle_mesh(simplified_filename, mesh)
    
    return simplified_dir


def simplify_mesh_vertex_clustering(input_mesh, voxel_size):
    """Simplify mesh using Vertex Clustering via Open3D"""
    return input_mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

def simplify_mesh_quadric_decimation(input_mesh, target_reduction):
    """Further simplify mesh using Quadric Decimation."""
    num_faces = len(input_mesh.triangles)
    target_faces = int(num_faces * target_reduction)  # Reduce by percentage
    return input_mesh.simplify_quadric_decimation(target_faces)


def visualize_comparison(original, simplified, method_name, original_stats, simplified_stats):
    """Visualize original and simplified meshes side by side"""
    # Create visualization window
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=f"Original Mesh - {original_stats}", width=640, height=480, left=0, top=0)
    vis1.add_geometry(original)
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=f"Simplified Mesh ({method_name}) - {simplified_stats}", width=640, height=480, left=640, top=0)
    vis2.add_geometry(simplified)
    
    # Set visualization options
    opt1 = vis1.get_render_option()
    opt1.mesh_show_wireframe = True
    opt1.point_size = 2.0
    
    opt2 = vis2.get_render_option()
    opt2.mesh_show_wireframe = True
    opt2.point_size = 2.0
    
    # Update visualization and capture screen
    vis1.poll_events()
    vis1.update_renderer()
    img1 = vis1.capture_screen_float_buffer(True)
    
    vis2.poll_events()
    vis2.update_renderer()
    img2 = vis2.capture_screen_float_buffer(True)
    
    # Close visualization windows
    vis1.destroy_window()
    vis2.destroy_window()
    
    # Convert to numpy arrays
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(img1_np)
    axes[0].set_title(f"Original Mesh\n{original_stats}")
    axes[0].axis('off')
    
    axes[1].imshow(img2_np)
    axes[1].set_title(f"Simplified Mesh ({method_name})\n{simplified_stats}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_modelnet10_df(df, num_samples=None, visualize=False, voxel_size_ratio=20, target_face_reduction=0.5, show_details = True):
    """
    Process ModelNet10 dataset by applying vertex clustering and quadric decimation.
    """
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df))).reset_index(drop=True)
    
    all_results = []
    summary_data = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing meshes"):
        mesh_path = row['path']
        object_id = row['object_id']
        object_class = row['class']
        split = row['split']
        object_path = row['object_path']
        if show_details:
            print(f"\nProcessing {object_id} ({object_class}, {split})")
            print(f"File: {mesh_path}")
        
        try:
            original_mesh = load_off_mesh(mesh_path)
            original_stats = get_mesh_stats(original_mesh)
            
            # Compute voxel size for clustering
            bbox = original_mesh.get_axis_aligned_bounding_box()
            max_extent = np.max(bbox.get_extent())
            voxel_size = max_extent / voxel_size_ratio  
            if show_details:
                print(f"Applying Vertex Clustering (Voxel Size: {voxel_size:.6f})...")
            start_time = time.time()
            simplified_mesh = simplify_mesh_vertex_clustering(original_mesh, voxel_size)
            if show_details:
                print(f"Applying Quadric Decimation (Target Reduction: {target_face_reduction * 100:.0f}%)...")
            simplified_mesh = simplify_mesh_quadric_decimation(simplified_mesh, target_face_reduction)
            elapsed_time = time.time() - start_time
            
            simplified_stats = get_mesh_stats(simplified_mesh)
            output_path = save_simplified_mesh(simplified_mesh, object_path, str(voxel_size_ratio) + '_' + str(target_face_reduction), show_details)
            if show_details:
                print(f"Completed in {elapsed_time:.2f} seconds")
            
            summary_data.append({
                'object_id': object_id,
                'class': object_class,
                'split': split,
                'original_path': mesh_path,
                'original_vertices': int(original_stats.split(',')[0].split(':')[1]),
                'original_faces': int(original_stats.split(',')[1].split(':')[1]),
                'simplified_vertices': int(simplified_stats.split(',')[0].split(':')[1]),
                'simplified_faces': int(simplified_stats.split(',')[1].split(':')[1]),
                'processing_time': elapsed_time,
                'output_path': output_path
            })
            
            all_results.append({
                'object_id': object_id,
                'original_stats': original_stats,
                'simplified_stats': simplified_stats,
                'processing_time': elapsed_time,
                'output_path': output_path
            })

            if visualize:
                visualize_comparison(
                    original_mesh,
                    simplified_mesh,
                    f'Vertex Clustering (voxel_size={voxel_size:.6f})',
                    original_stats,
                    simplified_stats
                )
            
        except Exception as e:
            print(f"Error processing {mesh_path}: {str(e)}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = 'simplification_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to: {summary_csv_path}")
    
    return summary_df, all_results


