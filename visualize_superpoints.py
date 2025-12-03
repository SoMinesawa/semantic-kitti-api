#!/usr/bin/env python3
"""
SemanticKITTI Superpoint Visualization Tool

This script visualizes superpoint/segment data from NPY or PLY files combined with 3D point clouds from PLY files.
Each superpoint is colored differently for easy identification.
"""

import argparse
import numpy as np
import sys
import os
from pathlib import Path

try:
    import vispy
    from vispy import app, scene
    from vispy.scene import visuals
    from vispy.color import get_colormap
except ImportError:
    print("Error: VisPy not installed. Please run: pip install vispy")
    sys.exit(1)

try:
    from plyfile import PlyData
except ImportError:
    print("Error: plyfile not installed. Please run: pip install plyfile")
    sys.exit(1)


class SuperpointVisualizer:
    def __init__(self, ply_path, npy_path, black_outlier=False):
        """
        Initialize the visualizer with PLY file paths.
        
        Args:
            ply_path (str): Path to PLY file containing 3D points
            npy_path (str): Path to NPY or PLY file containing superpoint/segment IDs
            black_outlier (bool): If True, display cluster with ID (max_id - 1) in black
        """
        self.ply_path = Path(ply_path)
        self.npy_path = Path(npy_path)
        self.black_outlier = black_outlier
        
        # Check if files exist
        if not self.ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {self.ply_path}")
        if not self.npy_path.exists():
            raise FileNotFoundError(f"NPY file not found: {self.npy_path}")
        
        self.points = None
        self.superpoint_ids = None
        self.colors = None
        
    def load_data(self):
        """Load data from PLY and NPY/PLY files."""
        print("Loading data...")
        
        # Load superpoint IDs from NPY or PLY file
        if self.npy_path.suffix.lower() == '.npy':
            # Load from NPY file
            self.superpoint_ids = np.load(self.npy_path, allow_pickle=True)
            print(f"Loaded superpoint IDs from NPY file")
        elif self.npy_path.suffix.lower() == '.ply':
            # Load from PLY file
            print(f"Loading superpoint IDs from PLY file: {self.npy_path}")
            segment_plydata = PlyData.read(self.npy_path)
            
            # Debug: print PLY file structure
            print(f"PLY elements: {[element.name for element in segment_plydata.elements]}")
            
            segment_vertex = segment_plydata['vertex']
            
            # Debug: print vertex info
            print(f"Vertex type: {type(segment_vertex)}")
            print(f"Vertex data type: {type(segment_vertex.data)}")
            
            # Get the actual data array
            vertex_data = segment_vertex.data
            print(f"Vertex data dtype: {vertex_data.dtype}")
            print(f"Available fields: {vertex_data.dtype.names}")
            
            # Try to find segment/superpoint ID field
            possible_names = ['segment', 'superpoint', 'label', 'cluster', 'id', 'class']
            segment_field = None
            
            for name in possible_names:
                if name in vertex_data.dtype.names:
                    segment_field = name
                    break
            
            if segment_field is None:
                print(f"Available fields in segment PLY: {vertex_data.dtype.names}")
                raise ValueError("Could not find segment/superpoint field in PLY file. Available fields: " + 
                               str(vertex_data.dtype.names))
            
            self.superpoint_ids = vertex_data[segment_field].astype(np.int32)
            print(f"Using field '{segment_field}' as superpoint IDs")
        else:
            raise ValueError(f"Unsupported file format for superpoint data: {self.npy_path}")
        
        print(f"Loaded {len(self.superpoint_ids)} superpoint IDs")
        print(f"Unique superpoints: {len(np.unique(self.superpoint_ids))}")
        
        # Load PLY file for coordinates
        plydata = PlyData.read(self.ply_path)
        vertex = plydata['vertex']
        
        # Extract coordinates
        self.points = np.column_stack([
            vertex['x'].astype(np.float32),
            vertex['y'].astype(np.float32),
            vertex['z'].astype(np.float32)
        ])
        
        print(f"Loaded {len(self.points)} 3D points")
        
        # Verify data consistency
        if len(self.points) != len(self.superpoint_ids):
            raise ValueError(f"Mismatch: {len(self.points)} points vs {len(self.superpoint_ids)} superpoint IDs")
        
        # Generate colors based on superpoint IDs
        self._generate_colors()
        
    def _generate_colors(self):
        """Generate colors for each superpoint."""
        unique_ids = np.unique(self.superpoint_ids)
        n_superpoints = len(unique_ids)
        max_id = unique_ids.max()
        outlier_id = max_id - 1
        
        print(f"Generating colors for {n_superpoints} superpoints...")
        print(f"Max cluster ID: {max_id}")
        
        if self.black_outlier:
            print(f"Black outlier mode: Cluster ID {outlier_id} will be displayed in black")
        
        # Generate random colors for each unique superpoint ID
        np.random.seed(42)  # For reproducible colors
        
        # Map each unique ID to a random RGB color
        id_to_color = {}
        for unique_id in unique_ids:
            if self.black_outlier and unique_id == outlier_id:
                # Set outlier cluster to black
                color = np.array([0.0, 0.0, 0.0])  # Black
                print(f"Assigned black color to cluster ID {unique_id}")
            else:
                # Generate random RGB color (values between 0 and 1)
                color = np.random.rand(3)
            id_to_color[unique_id] = color
        
        # Assign colors to all points
        self.colors = np.array([id_to_color[sp_id] for sp_id in self.superpoint_ids])
        
        print(f"Generated colors shape: {self.colors.shape}")
        print(f"Color range: [{self.colors.min():.3f}, {self.colors.max():.3f}]")
        
        if self.black_outlier:
            outlier_count = np.sum(self.superpoint_ids == outlier_id)
            print(f"Number of points in black outlier cluster: {outlier_count:,}")
        
    def visualize(self, point_size=2, background_color='black'):
        """
        Create and show the 3D visualization.
        
        Args:
            point_size (float): Size of each point
            background_color (str): Background color
        """
        if self.points is None or self.superpoint_ids is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        # Create canvas
        canvas = scene.SceneCanvas(keys='interactive', 
                                   size=(1200, 800),
                                   title='SemanticKITTI Superpoint Visualization')
        canvas.measure_fps()
        
        # Set up view
        view = canvas.central_widget.add_view()
        view.bgcolor = background_color
        view.camera = 'turntable'
        
        # Create scatter plot
        scatter = visuals.Markers()
        scatter.set_data(self.points, 
                        face_color=self.colors,
                        size=point_size,
                        edge_width=0)
        
        view.add(scatter)
        
        # Add axis
        axis = visuals.XYZAxis(parent=view.scene)
        
        # Auto-scale view to fit all points
        view.camera.set_range()
        
        # Add key bindings
        self._setup_key_bindings(canvas, scatter)
        
        # Display info
        self._print_info()
        
        # Show canvas
        canvas.show()
        
        if sys.flags.interactive != 1:
            app.run()
    
    def _setup_key_bindings(self, canvas, scatter):
        """Setup keyboard shortcuts."""
        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.text == 'r':
                # Reset camera view
                canvas.central_widget.children[0].camera.set_range()
            elif event.text == 'q':
                # Quit
                app.quit()
            elif event.text == 'h':
                # Show help
                print("\n=== Keyboard Shortcuts ===")
                print("r: Reset camera view")
                print("q: Quit")
                print("h: Show this help")
                print("Mouse: Left click + drag to rotate")
                print("Mouse: Right click + drag to zoom")
                print("Mouse: Middle click + drag to pan")
                print("========================\n")
    
    def _print_info(self):
        """Print information about the loaded data."""
        print("\n=== Visualization Info ===")
        print(f"PLY file: {self.ply_path.name}")
        print(f"NPY file: {self.npy_path.name}")
        print(f"Total points: {len(self.points):,}")
        print(f"Unique superpoints: {len(np.unique(self.superpoint_ids)):,}")
        
        # Point cloud bounds
        min_coords = self.points.min(axis=0)
        max_coords = self.points.max(axis=0)
        print(f"Point cloud bounds:")
        print(f"  X: [{min_coords[0]:.2f}, {max_coords[0]:.2f}]")
        print(f"  Y: [{min_coords[1]:.2f}, {max_coords[1]:.2f}]")
        print(f"  Z: [{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
        
        # Superpoint statistics
        unique_ids, counts = np.unique(self.superpoint_ids, return_counts=True)
        print(f"Superpoint statistics:")
        print(f"  Average points per superpoint: {counts.mean():.1f}")
        print(f"  Min points per superpoint: {counts.min()}")
        print(f"  Max points per superpoint: {counts.max()}")
        
        if self.black_outlier:
            max_id = unique_ids.max()
            outlier_id = max_id - 1
            outlier_count = np.sum(self.superpoint_ids == outlier_id)
            print(f"Black outlier mode:")
            print(f"  Outlier cluster ID: {outlier_id}")
            print(f"  Outlier points: {outlier_count:,}")
        
        print("=========================")
        print("\nPress 'h' in the visualization window for keyboard shortcuts.")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Visualize SemanticKITTI superpoint data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize with NPY segment file
  python visualize_superpoints.py data/users/minesawa/semantickitti/growsp/00/000000.ply data/users/minesawa/semantickitti/growsp_sp/00/000000_superpoint.npy
  
  # Visualize with PLY segment file
  python visualize_superpoints.py coordinates.ply segments.ply --background white
  
  # With black outlier cluster (max_id - 1)
  python visualize_superpoints.py coordinates.ply segments.ply --black-outlier --background white
  
  # With custom point size and black outlier
  python visualize_superpoints.py --point-size 3 --black-outlier points.ply segments.ply
        """
    )
    
    parser.add_argument('ply_file', help='Path to PLY file containing 3D points')
    parser.add_argument('segment_file', help='Path to NPY or PLY file containing superpoint/segment IDs')
    parser.add_argument('--point-size', type=float, default=2.0,
                        help='Size of each point (default: 2.0)')
    parser.add_argument('--background', choices=['black', 'white', 'gray'], 
                        default='black', help='Background color (default: black)')
    parser.add_argument('--black-outlier', action='store_true',
                        help='Display cluster with ID (max_id - 1) in black color')
    
    args = parser.parse_args()
    
            # Create visualizer
    viz = SuperpointVisualizer(args.ply_file, args.segment_file, 
                                black_outlier=args.black_outlier)
    
    # Load data
    viz.load_data()
    
    # Start visualization
    viz.visualize(point_size=args.point_size, 
                    background_color=args.background)



if __name__ == '__main__':
    main() 