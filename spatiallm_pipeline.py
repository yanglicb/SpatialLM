import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd
from inference import preprocess_point_cloud, generate_layout
import sys
import pyvista as pv

# Add path to preprocessing scripts
# Assuming directory structure:
# root/
#   SpatialLM/spatiallm_pipeline.py
#   Interior3D-server/SpatialLLM/Scripts/PreProcess_and_visualization/utils_processing.py
preprocessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Interior3D-server/SpatialLLM/Scripts/PreProcess_and_visualization"))
if preprocessing_dir not in sys.path:
    sys.path.append(preprocessing_dir)

try:
    import utils_processing
    import utils_alignment
except ImportError as e:
    print(f"Warning: Could not import preprocessing modules: {e}")
    utils_processing = None
    utils_alignment = None

class SpatialLMPipeline:
    """
    Pipeline for SpatialLM inference.
    """
    def __init__(
        self, 
        model_path: str = "manycore-research/SpatialLM1.1-Qwen-0.5B",
        device: str = "cuda",
        inference_dtype: str = "float32"
    ):
        self.model_path = model_path
        self.device = device
        self.inference_dtype = inference_dtype
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SpatialLM model and tokenizer."""
        print(f"Loading SpatialLM model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=getattr(torch, self.inference_dtype)
        )
        self.model.to(self.device)
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()
        print("SpatialLM model loaded.")

    def generate_layout_from_pcd(
        self,
        pcd_path: str,
        code_template_file: str = "code_template.txt",
        detect_type: str = "all",
        categories: list = [],
        top_k: int = 10,
        top_p: float = 0.95,
        temperature: float = 0.6,
        num_beams: int = 1,
        seed: int = -1,
        no_cleanup: bool = False,
    ) -> str:
        """
        Generate layout from a point cloud file.

        Args:
            pcd_path: Path to the input point cloud file (.ply).
            code_template_file: Path to the code template file.
            detect_type: Type of indoor elements to detect.
            categories: List of categories to detect.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            temperature: Sampling temperature.
            num_beams: Number of beams for beam search.
            seed: Random seed.
            no_cleanup: Whether to skip point cloud cleanup.

        Returns:
            Generated layout string.
        """
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")

        if utils_processing is not None:
             print("Applying preprocessing to point cloud...")
             try:
                 # Load with PyVista
                 mesh = pv.read(pcd_path)
                 
                 # 1. Uniform Voxel Sampling (0.005)
                 mesh = utils_processing.sample_mesh_uniform_voxel(mesh=mesh, voxel_size=0.005)
                 
                 # 1. Uniform Voxel Sampling (0.005)
                 mesh = utils_processing.sample_mesh_uniform_voxel(mesh=mesh, voxel_size=0.005)
                 
                 # 2. XY Alignment (Rotate around Z ONLY)
                 # Note: align_mesh_vertical_pca calls align_scene_to_z_up with fit_ground=False 
                 # This ensures we DO NOT change the Z-axis (gravity), only rotate walls to align with XY.
                 print("Aligning walls to XY (rotating around Z-axis only)...")
                 mesh = utils_processing.align_mesh_vertical_pca(mesh=mesh)
                 
                 # 3. Scale to Ceiling Height (2.7m default in script)
                 mesh, _ = utils_processing.scale_pyvista_to_ceiling_height(mesh=mesh, target_height=2.7)
                 
                 # Save preprocessed PLY
                 pcd_path_preprocessed = pcd_path.replace(".ply", "_preprocessed.ply")
                 
                 # Use utils_processing.save_ply_file to ensure colors are preserved
                 # Detect correct color key
                 color_key = "RGB"
                 if "RGBA" in mesh.point_data:
                     color_key = "RGBA"
                 elif "COLOR_0" in mesh.point_data:
                     color_key = "COLOR_0"
                     
                 utils_processing.save_ply_file(mesh, pcd_path_preprocessed, color_variable=color_key)
                 print(f"Saved preprocessed PLY to {pcd_path_preprocessed}")
                 
                 # Update pcd_path to use the preprocessed file
                 pcd_path = pcd_path_preprocessed
                 
             except Exception as e:
                 print(f"Error during preprocessing: {e}. Proceeding with original PLY.")
                 import traceback
                 traceback.print_exc()

        # Load point cloud (potentially preprocessed)
        point_cloud = load_o3d_pcd(pcd_path)
        num_bins = self.model.config.point_config["num_bins"]
        grid_size = Layout.get_grid_size(num_bins)

        if not no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # Preprocess
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
        input_pcd = input_pcd.to(self.device)

        # Generate layout
        # Note: generate_layout in inference.py takes model, point_cloud, tokenizer, etc.
        # We need to make sure we pass the correct arguments.
        # Also, generate_layout prints to stdout, we might want to capture that or just let it print.
        
        # We need to ensure code_template_file path is correct. 
        # If it's relative, it might be relative to where the server runs.
        # We should probably resolve it relative to the SpatialLM directory if not absolute.
        if not os.path.isabs(code_template_file):
             # Assuming code_template.txt is in the SpatialLM directory
             spatiallm_dir = os.path.dirname(os.path.abspath(__file__))
             # But this file is in SpatialLM/spatiallm_pipeline.py (if I put it there)
             # Wait, I am putting it in SpatialLM/spatiallm_pipeline.py
             # So __file__ is SpatialLM/spatiallm_pipeline.py
             # code_template.txt is likely in SpatialLM/code_template.txt
             code_template_file = os.path.join(spatiallm_dir, code_template_file)

        layout = generate_layout(
            model=self.model,
            point_cloud=input_pcd,
            tokenizer=self.tokenizer,
            code_template_file=code_template_file,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            seed=seed,
            detect_type=detect_type,
            categories=categories,
        )
        
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()
        
        return pred_language_string

    def generate_mesh_from_layout(self, layout_string: str, output_path: str):
        """
        Generate a 3D mesh from the layout string.
        """
        import utils_roomplan
        import open3d as o3d
        import tempfile
        import json

        # Create a temporary txt file for the layout string because utils_roomplan expects a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(layout_string)
            tmp_path = tmp.name

        try:
             # Load objects using utils_roomplan
            data_json = utils_roomplan.load_objects(txt_file=tmp_path)
            
            # Generate meshes
            meshes = utils_roomplan.make3DMeshRoom(data_json=data_json)
            
            if not meshes:
                print("No meshes generated.")
                return None
                
            # Combine all meshes
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes:
                if mesh: # check if None
                    combined_mesh += mesh
            
            # Save mesh
            o3d.io.write_triangle_mesh(output_path, combined_mesh)
            print(f"Saved mesh to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating mesh: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
