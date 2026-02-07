import trimesh as tr
import numpy as np
import copy
import open3d as o3d



def make_detailed_mesh(scene_objects_list):
    for scene_object in scene_objects_list:
        if scene_object.category == "dining_table":
            detailed_mesh = make_detailed_table(scene_object)
            scene_object.set_detailed_mesh(detailed_mesh)


    return scene_objects_list




def make_detailed_table(scene_object):

    import copy

    # -------------------------------------------------------
    # 1. GET OBB & POSE
    # -------------------------------------------------------
    bbox = scene_object.mesh.get_oriented_bounding_box()
    center_world = bbox.center
    extent = np.array(bbox.extent)
    R_obb = bbox.R    # columns = local axes in world coordinates
    
    print("OBB extent:", extent)
    print("OBB axes (columns):\n", R_obb)

    # -------------------------------------------------------
    # 2. FIND VERTICAL AXIS (OBB AXIS CLOSEST TO WORLD Z)
    # -------------------------------------------------------
    world_z = np.array([0,0,1])
    alignment = np.abs(R_obb.T @ world_z)
    vertical_axis = np.argmax(alignment)

    # Get vertical axis direction in world space
    vertical_dir_world = R_obb[:, vertical_axis]

    # -------------------------------------------------------
    # 2b. ENSURE THE VERTICAL AXIS POINTS UP (+Z)
    # -------------------------------------------------------
    # Make a writable copy
    R_obb_copy = R_obb.copy()

    # Flip if pointing down
    if np.dot(R_obb_copy[:, vertical_axis], world_z) < 0:
        print("Vertical axis was upside down → flipping it.")
        R_obb_copy[:, vertical_axis] *= -1

    vertical_dir_world = R_obb_copy[:, vertical_axis]

    # vertical axis length
    height = extent[vertical_axis]

    # horizontal axes
    axes = [0,1,2]
    axes.remove(vertical_axis)
    length_axis, width_axis = axes

    length = extent[length_axis]
    width  = extent[width_axis]

    print(f"Vertical axis index = {vertical_axis}")
    print(f"Table size (L,W,H) = {length}, {width}, {height}")

    # -------------------------------------------------------
    # 3. MOVE ORIGINAL MESH INTO LOCAL OBB COORDINATES
    # -------------------------------------------------------
    mesh_local = copy.deepcopy(scene_object.mesh)

    T_to_local = np.eye(4)
    T_to_local[:3,:3] = R_obb.T
    T_to_local[:3, 3] = -R_obb.T @ center_world
    mesh_local.transform(T_to_local)

    # -------------------------------------------------------
    # 4. LOCAL COORDINATE SYSTEM:
    #    +Y = up (table height direction)
    #    +X, +Z = horizontal table axes
    # -------------------------------------------------------
    # NOTE: We reassign axes: (local_x, local_y, local_z)
    #       where local_y is the vertical axis.
    #
    # Our generated geometry WILL FOLLOW THIS FRAME.

    top_thickness = 0.05
    leg_thickness = 0.05
    leg_height = height - top_thickness

    # Table top at top of legs
    table_top = o3d.geometry.TriangleMesh.create_box(
        width=length,
        height=top_thickness,
        depth=width
    )
    table_top.translate([-length/2, leg_height, -width/2])

    # Legs (start at Y=0)
    def make_leg(x, z):
        leg = o3d.geometry.TriangleMesh.create_box(
            width=leg_thickness,
            height=leg_height,
            depth=leg_thickness
        )
        leg.translate([x, 0, z])
        return leg

    leg_positions = [
        (-length/2, -width/2),
        ( length/2 - leg_thickness, -width/2),
        (-length/2,  width/2 - leg_thickness),
        ( length/2 - leg_thickness,  width/2 - leg_thickness)
    ]

    legs = [make_leg(x, z) for (x, z) in leg_positions]

    detailed_local = table_top
    for leg in legs:
        detailed_local += leg

    # -------------------------------------------------------
    # 5. TRANSFORM DETAILED TABLE BACK TO WORLD SPACE
    # -------------------------------------------------------
    T_to_world = np.eye(4)
    T_to_world[:3,:3] = R_obb_copy
    T_to_world[:3, 3] = center_world

    detailed_world = copy.deepcopy(detailed_local)
    detailed_world.transform(T_to_world)

    # -------------------------------------------------------
    # 6. ENSURE TABLE TOP IS UP-FACING IN WORLD SPACE
    # -------------------------------------------------------
    # Check the table top normal (local +Y becomes world `R_obb[:, vertical_axis]`)
    # Make a writable copy
    R_obb_copy = R_obb.copy()

    # Flip if pointing down
    if np.dot(R_obb_copy[:, vertical_axis], world_z) < 0:
        print("Vertical axis was upside down → flipping it.")
        R_obb_copy[:, vertical_axis] *= -1

    vertical_dir_world = R_obb_copy[:, vertical_axis]


    # -------------------------------------------------------
    # 7. GROUND CORRECTION – LEGS, NOT TABLE TOP, TOUCH THE FLOOR
    # -------------------------------------------------------
    detailed_vertices = np.asarray(detailed_world.vertices)
    min_z = detailed_vertices[:, 2].min()

    orig_vertices = np.asarray(scene_object.mesh.vertices)
    orig_min_z = orig_vertices[:, 2].min()

    dz = orig_min_z - min_z

    detailed_world.translate([0, 0, dz])

    print("Ground correction applied:", dz)

    return detailed_world

def make_detailed_chair(scene_object):

    import copy
    import numpy as np
    import open3d as o3d

    # -------------------------------------------------------
    # 1. GET OBB AND BASIC INFO
    # -------------------------------------------------------
    bbox = scene_object.mesh.get_oriented_bounding_box()
    center_world = bbox.center
    R_obb = bbox.R                      # 3x3, columns = axes in world space
    extent = np.array(bbox.extent)

    world_up = np.array([0, 0, 1])

    # -------------------------------------------------------
    # 2. DETECT VERTICAL AXIS
    # -------------------------------------------------------
    alignment = np.abs(R_obb.T @ world_up)
    vertical_axis = np.argmax(alignment)

    R = R_obb.copy()
    if np.dot(R[:, vertical_axis], world_up) < 0:
        R[:, vertical_axis] *= -1

    vertical_dir_world = R[:, vertical_axis]

    # determine horizontal
    axes = [0,1,2]
    axes.remove(vertical_axis)
    hor1, hor2 = axes
    L = extent[hor1]
    W = extent[hor2]
    H = extent[vertical_axis]

    # -------------------------------------------------------
    # 3. LOCAL OBB-ALIGNED FRAME
    # -------------------------------------------------------
    # local +Z = vertical
    # local +X = first horizontal
    # local +Y = second horizontal
    R_local = np.zeros((3,3))
    R_local[:,2] = R[:, vertical_axis]
    R_local[:,0] = R[:, hor1]
    R_local[:,1] = R[:, hor2]

    # move to local
    T_to_local = np.eye(4)
    T_to_local[:3,:3] = R_local.T
    T_to_local[:3,3] = -R_local.T @ center_world

    mesh_local = copy.deepcopy(scene_object.mesh)
    mesh_local.transform(T_to_local)

    # -------------------------------------------------------
    # 4. BUILD CHAIR GEOMETRY (LOCAL)
    # -------------------------------------------------------
    leg_thickness = 0.05
    seat_thickness = 0.05
    backrest_thickness = 0.04
    backrest_height = H * 0.45        # 45% of OBB height

    leg_height = H - seat_thickness

    # SEAT (centered in X/Y, sits at Z = leg_height)
    seat = o3d.geometry.TriangleMesh.create_box(
        width=L,
        height=W,
        depth=seat_thickness
    )
    seat.translate([-L/2, -W/2, leg_height])


    # LEGS (follow local +Z = world vertical)
    def make_leg(x,y):
        leg = o3d.geometry.TriangleMesh.create_box(
            width=leg_thickness,
            height=leg_thickness,
            depth=leg_height
        )
        leg.translate([x,y,0])
        return leg

    leg_positions = [
        (-L/2, -W/2),
        ( L/2 - leg_thickness, -W/2),
        (-L/2,  W/2 - leg_thickness),
        ( L/2 - leg_thickness, W/2 - leg_thickness),
    ]
    legs = [make_leg(x,y) for x,y in leg_positions]


    # -------------------------------------------------------
    # 5. BACKREST (aligned with OBB horizontal axes)
    # -------------------------------------------------------
    # Backrest is placed at the "back" side, meaning
    # along the +Y direction of the OBB horizontal axis #2.

    backrest = o3d.geometry.TriangleMesh.create_box(
        width=L,
        height=backrest_thickness,
        depth=backrest_height
    )

    # bottom of backrest at seat top
    backrest.translate([
        -L/2,
        W/2 - backrest_thickness,   # attached to back edge
        leg_height + seat_thickness
    ])


    # -------------------------------------------------------
    # MERGE LOCAL GEOMETRY
    # -------------------------------------------------------
    detailed_local = seat
    for lg in legs:
        detailed_local += lg
    detailed_local += backrest

    # -------------------------------------------------------
    # 6. TRANSFORM BACK TO WORLD
    # -------------------------------------------------------
    T_to_world = np.eye(4)
    T_to_world[:3,:3] = R_local
    T_to_world[:3,3] = center_world

    detailed_world = copy.deepcopy(detailed_local)
    detailed_world.transform(T_to_world)

    # -------------------------------------------------------
    # 7. FLOOR ALIGNMENT
    # -------------------------------------------------------
    v = np.asarray(detailed_world.vertices)
    min_z_detailed = v[:, 2].min()

    orig_v = np.asarray(scene_object.mesh.vertices)
    orig_min_z = orig_v[:, 2].min()

    dz = orig_min_z - min_z_detailed
    detailed_world.translate([0,0,dz])

    return detailed_world


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find rotation matrix that rotates vec1 to vec2 """
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        # 180-degree rotation around any perpendicular axis
        perp = np.array([1,0,0])
        if np.allclose(a, perp):
            perp = np.array([0,1,0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = -np.eye(3) + 2 * K @ K
        return R
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


def safe_rotation_matrix_from_vectors(vec1, vec2):
    """Return rotation matrix that rotates vec1 -> vec2, safely handling parallel vectors."""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):  # already aligned
        return np.eye(3)
    if np.isclose(c, -1.0):  # opposite
        # Rotate 180 degrees around any perpendicular axis
        perp = np.array([1,0,0])
        if np.allclose(a, perp):
            perp = np.array([0,1,0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = -np.eye(3) + 2 * K @ K
        return R

    # general case
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R
