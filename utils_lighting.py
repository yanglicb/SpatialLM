import open3d as o3d
import numpy as np

# def visualize_with_lighting(meshes):
#     """
#     Windows-friendly Open3D 0.12 visualization.
#     Uses classic Visualizer + basic lighting.
#     Shadows and advanced PBR are not available in 0.12.
#     """
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="3D Room Viewer (Open3D 0.12)", width=1280, height=800)

#     # Add meshes
#     for mesh in meshes:
#         # Ensure mesh has vertex colors if you want per-object color
#         if not mesh.has_vertex_colors():
#             mesh.paint_uniform_color([0.8, 0.8, 0.8])
#         vis.add_geometry(mesh)

#     # Add coordinate frame
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#     vis.add_geometry(axis)

#     # Lighting: only basic headlight is supported
#     render_opt = vis.get_render_option()
#     render_opt.light_on = True  # turn on simple headlight
#     render_opt.background_color = np.array([0.95, 0.95, 0.95])

#     # Camera: optional preset
#     ctr = vis.get_view_control()
#     ctr.set_up([0, 0, 1])
#     ctr.set_front([0, -1, -1])
#     ctr.set_lookat([0, 0, 0])
#     ctr.set_zoom(0.5)

#     print("🟢 3D viewer running — close the window to continue.")
#     vis.run()
#     vis.destroy_window()

import open3d as o3d
import numpy as np


def visualize_with_lighting(meshes):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # GUI window
    window = o3d.visualization.O3DVisualizer("3D Room Viewer (Directional Light)", 1280, 800)
    window.show_settings = True

    # Add meshes with basic material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    if hasattr(mat, "base_metallic"):
        mat.base_metallic = 0.0
    if hasattr(mat, "base_roughness"):
        mat.base_roughness = 0.5

    for i, mesh in enumerate(meshes):
        window.add_geometry(f"mesh_{i}", mesh, mat)

    # Add coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    window.add_geometry("axis", axis, mat)

    # Access low-level Scene
    scene = window.scene.scene

    # Add directional light with shadows
    light_name = "sun"
    direction = np.array([0.5, -1.0, -0.5], dtype=np.float32)
    color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    intensity = 50000.0
    cast_shadows = True
    scene.add_directional_light(light_name, color, direction, intensity, cast_shadows)
    scene.enable_light_shadow(light_name, True)

    # Optional indirect lighting
    scene.enable_indirect_light(True)

    # Background
    # scene.set_background([0.95, 0.95, 0.95, 1.0])

    # Camera
    window.reset_camera_to_default()

    print("🟢 3D viewer running — close the window to continue.")
    app.add_window(window)
    app.run()
