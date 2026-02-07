import re
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, unary_union, polygonize
import plotly.graph_objects as go
import shapely.geometry as sg
import trimesh

import utils_lighting
import utils_detailed

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Wall:
    a_x: float
    a_y: float
    a_z: float
    b_x: float
    b_y: float
    b_z: float
    height: float
    rotation: float = 0.0

@dataclass
class Door:
    id: str
    object_id: str
    position_x: float
    position_y: float
    position_z: float
    width: float
    height: float

@dataclass
class Window:
    id: str
    object_id: str
    position_x: float
    position_y: float
    position_z: float
    width: float
    height: float

@dataclass
class Bbox:
    id: str
    object_id: str
    position_x: float
    position_y: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float

@dataclass
class SceneObject:
    def __init__(self, mesh, category):
        self.mesh = mesh
        self.category = category
        self.detailed_mesh = None

    def set_detailed_mesh(self, detailed_mesh):
        self.detailed_mesh = detailed_mesh

# -----------------------------
# Regex
# -----------------------------
LINE_RE = re.compile(r'^(?P<name>\w+)\s*=\s*(?P<class>\w+)\s*\((?P<args>.*)\)\s*$')
FLOAT_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def safe_float(s: str) -> float:
    m = FLOAT_RE.search(s)
    if not m:
        raise ValueError(f"Cannot parse float from '{s}'")
    return float(m.group(0))

# -----------------------------
# Parsing
# -----------------------------
def parse_line(line: str, lineno: int):
    m = LINE_RE.match(line.strip())
    if not m:
        raise ValueError(f"Line {lineno}: invalid format -> {line}")

    name, cls, args_raw = m.group("name"), m.group("class"), m.group("args").strip()

    if cls == "Wall":
        nums = FLOAT_RE.findall(args_raw)
        vals = [float(x) for x in nums]
        if len(vals) == 7:
            w = Wall(*vals)
        elif len(vals) >= 8:
            w = Wall(*vals[:8])
        else:
            raise ValueError(f"Line {lineno}: Wall needs at least 7 numbers")
        return "wall", name, asdict(w)

    elif cls in ("Door", "Window"):
        parts = [p.strip() for p in args_raw.split(",")]
        object_id = parts[0]
        nums = [safe_float(p) for p in parts[1:]]
        if len(nums) < 5:
            raise ValueError(f"Line {lineno}: {cls} needs 5 numbers, got {len(nums)}")
        px, py, pz, width, height = nums[:5]
        if cls == "Door":
            return "door", name, asdict(Door(name, object_id, px, py, pz, width, height))
        else:
            return "window", name, asdict(Window(name, object_id, px, py, pz, width, height))
        
    elif cls == "Bbox":
        parts = [p.strip() for p in args_raw.split(",")]
        object_id = parts[0]
        nums = [safe_float(p) for p in parts[1:]]
        if len(nums) < 7:
            raise ValueError(f"Line {lineno}: {cls} needs 7 numbers, got {len(nums)}")
        px, py, pz, angle_z, scale_x, scale_y, scale_z = nums[:7]
        return "bbox", name, asdict(Bbox(name, object_id, px, py, pz, angle_z, scale_x, scale_y, scale_z))
    else:
        raise ValueError(f"Line {lineno}: Unknown class '{cls}'")

def load_objects(txt_file):
    data = {"walls": {}, "doors": [], "windows": [], "bboxes": []}
    with open(txt_file, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            typ, name, obj = parse_line(line, i)
            if typ == "wall":
                data["walls"][name] = obj
            elif typ == "door":
                data["doors"].append(obj)
            elif typ == "window":
                data["windows"].append(obj)
            elif typ == "bbox":
                data["bboxes"].append(obj)
    return data

# -----------------------------
# Plotting
# -----------------------------
def plot_layout(data, out_img="layout.png"):
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, wall in data["walls"].items():
        ax.plot([wall["a_x"], wall["b_x"]], [wall["a_y"], wall["b_y"]], "k-", linewidth=2)
        ax.text((wall["a_x"] + wall["b_x"]) / 2,
                (wall["a_y"] + wall["b_y"]) / 2,
                name, fontsize=8, color="blue")

    for door in data["doors"]:
        ax.plot(door["position_x"], door["position_y"], "ro", markersize=6)
        ax.text(door["position_x"], door["position_y"], "Door", fontsize=8, color="red")

    for window in data["windows"]:
        ax.plot(window["position_x"], window["position_y"], "gs", markersize=6)
        ax.text(window["position_x"], window["position_y"], "Window", fontsize=8, color="green")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Layout from SpatialLM Predictions")

    plt.savefig(out_img, dpi=300)
    plt.show()

# -----------------------------
# 3D Mesh generation
# -----------------------------
# def make_wall_mesh(wall, color, thickness=0.1):
#     ax, ay, az = wall["a_x"], wall["a_y"], wall["a_z"]
#     bx, by, bz = wall["b_x"], wall["b_y"], wall["b_z"]
#     height = wall["height"]

#     vec = np.array([bx - ax, by - ay])
#     length = np.linalg.norm(vec)
#     if length < 1e-6:
#         print("⚠️ Degenerate wall skipped (too short).")
#         return None

#     unit_vec = vec / length
#     angle = np.arctan2(unit_vec[1], unit_vec[0])

#     mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=thickness, depth=height)
#     mesh.compute_vertex_normals()
#     mesh.paint_uniform_color(color)

#     R = mesh.get_rotation_matrix_from_xyz([0, 0, angle])
#     mesh.rotate(R, center=(0, 0, 0))
#     mesh.translate([ax, ay, az])
#     return mesh


def make_wall_from_layout(dict_data_layout, data, color=[0.8, 0.8, 0.8]):
    """Given layout data with 'inner_ring' and 'outer_ring', create 3D wall meshes.

    Args:
        dict_data_layout (dict): output from makeLayout()
        height (float): wall height

    Returns:
        list of open3d.geometry.TriangleMesh
    """
    walls_list = [wall for wall in data.get('walls', {}).values()]

    # Get height of the first wall
    wall_height = walls_list[0]["height"]
    a_z = walls_list[0]["a_z"]

    inner_ring = dict_data_layout.get('inner_ring', None)
    outer_ring = dict_data_layout.get('outer_ring', None)

    if outer_ring is None:
        raise ValueError("❌ 'outer_ring' is required to define the wall boundary.")

    if not isinstance(outer_ring, sg.LineString):
        raise TypeError("❌ 'outer_ring' must be a shapely LineString.")

    # --- Create the wall footprint polygon ---
    # Outer boundary = outer_ring, inner hole = inner_ring (if provided)
    if inner_ring and isinstance(inner_ring, sg.LineString):
        wall_polygon = sg.Polygon(outer_ring.coords, [inner_ring.coords])
    else:
        wall_polygon = sg.Polygon(outer_ring.coords)

    if not wall_polygon.is_valid:
        wall_polygon = wall_polygon.buffer(0)  # fix slight geometry issues

    if wall_polygon.is_empty:
        raise ValueError("❌ Wall polygon is empty after construction.")

    # --- Extrude the 2D wall footprint into 3D using trimesh ---
    mesh_tri = trimesh.creation.extrude_polygon(wall_polygon, wall_height)

    # --- Convert the trimesh mesh to an Open3D TriangleMesh ---
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_tri.vertices),
        o3d.utility.Vector3iVector(mesh_tri.faces)
    )
    mesh_o3d.compute_vertex_normals()

    mesh_o3d.translate([0, 0, 0])
    mesh_o3d.paint_uniform_color(color)

    return mesh_o3d, -a_z


def make_floor(dict_data_layout, color=[0.9, 0.9, 0.9], thickness=0.2):
    """Create a 3D floor mesh from the inner ring of the layout.

    Args:
        dict_data_layout (dict): Output from makeLayout(), must contain 'inner_ring'
        height (float): Extrusion height (floor thickness)

    Returns:
        list of open3d.geometry.TriangleMesh
    """

    outer_ring = dict_data_layout.get('outer_ring', None)

    if outer_ring is None:
        raise ValueError("❌ 'outer_ring' is required to create the floor.")

    if not isinstance(outer_ring, sg.LineString):
        raise TypeError("❌ 'outer_ring' must be a shapely LineString.")

    # --- Create the floor footprint polygon ---
    floor_polygon = sg.Polygon(outer_ring.coords)
    if not floor_polygon.is_valid:
        floor_polygon = floor_polygon.buffer(0)

    if floor_polygon.is_empty:
        raise ValueError("❌ Floor polygon is empty or invalid.")

    # --- Extrude the polygon using trimesh ---
    mesh_tri = trimesh.creation.extrude_polygon(floor_polygon, thickness)

    # --- Convert to Open3D TriangleMesh ---
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_tri.vertices),
        o3d.utility.Vector3iVector(mesh_tri.faces)
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color(color)

    return mesh_o3d

def make_door_mesh(door, wall_angle=0.0, color=[0.85,0.7,0.5], thickness=0.05, a_z=0.0):
    px, py, pz = door["position_x"], door["position_y"], door["position_z"]
    width, height = door["width"], door["height"]
    pz += a_z


    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=thickness, depth=height)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    mesh.translate([-width/2, -thickness/2, -height/2])
    R = mesh.get_rotation_matrix_from_xyz([0, 0, wall_angle])
    mesh.rotate(R, center=(0,0,0))
    mesh.translate([px, py, pz])
    return mesh

def make_window_mesh(window, wall_angle=0.0, color=[0.6,0.8,1.0], thickness=0.03, a_z=0.0):
    px, py, pz = window["position_x"], window["position_y"], window["position_z"]
    width, height = window["width"], window["height"]
    pz += a_z

    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=thickness, depth=height)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    mesh.translate([-width/2, -thickness/2, -height/2])
    R = mesh.get_rotation_matrix_from_xyz([0, 0, wall_angle])
    mesh.rotate(R, center=(0,0,0))
    mesh.translate([px, py, pz])
    return mesh

def make_bbox_mesh(bbox: Bbox, color, a_z=0.0):
    px, py, pz = bbox.position_x, bbox.position_y, bbox.position_z
    sx, sy, sz = bbox.scale_x, bbox.scale_y, bbox.scale_z
    angle = bbox.angle_z

    pz += a_z


    mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate([-sx/2, -sy/2, -sz/2])
    R = mesh.get_rotation_matrix_from_xyz([0,0,angle])
    mesh.rotate(R, center=(0,0,0))
    mesh.translate([px, py, pz])
    return mesh

# -----------------------------
# Layout
# -----------------------------

def make_layout(data, out_img="layout.html", wall_thickness=0.01, center_color="black", outer_color="#ff7f0e", snap_tol=1e-8):
    """
    Simpler, robust approach requested by user:
      - Detect inner boundary (attempt polygonize of merged centerlines; fallback to convex hull)
      - Show detected outward normals for each wall (plotted as short lines)
      - Build outer boundary by buffering the inner polygon using Shapely.buffer
        with square caps and mitre joins (to create sharp corners).

    Returns dict with 'center_lines' and 'outer_ring' (LineString or None).
    """

    # Accept either a filename or dict
    if isinstance(data, str):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    walls = data.get('walls', {})

    # collect centerline segments and per-segment midpoints
    center_lines = []
    seg_infos = []
    for name, w in walls.items():
        try:
            a = (float(w['a_x']), float(w['a_y']))
            b = (float(w['b_x']), float(w['b_y']))
        except Exception:
            continue
        ln = LineString([a, b])
        center_lines.append(ln)
        seg_infos.append({'name': name, 'a': np.array(a, dtype=float), 'b': np.array(b, dtype=float), 'line': ln})

    if not seg_infos:
        print('No walls to plot in makeLayout().')
        return {'center_lines': [], 'outer_ring': None}

    # Try to reconstruct an inner polygon by polygonizing merged centerlines
    merged = linemerge(unary_union(center_lines))
    polys = list(polygonize(merged))

    # compute an approximate scene extent for numeric epsilons
    try:
        all_pts = np.vstack([np.array(s['a']) for s in seg_infos] + [np.array(s['b']) for s in seg_infos])
        span = np.max(all_pts, axis=0) - np.min(all_pts, axis=0)
        extent = max(1.0, np.linalg.norm(span))
    except Exception:
        extent = 1.0

    if polys:
        inner_poly = max(polys, key=lambda p: p.area)
    else:
        # fallback 1: try a very small buffer around centerlines to build a region
        small = extent * 1e-3
        try:
            buffered_lines = [ln.buffer(small, resolution=8) for ln in center_lines]
            unioned = unary_union(buffered_lines)
            if unioned.geom_type == 'Polygon':
                inner_poly = unioned
            else:
                geoms = list(unioned.geoms) if hasattr(unioned, 'geoms') else []
                if geoms:
                    inner_poly = max(geoms, key=lambda p: p.area)
                else:
                    pts = []
                    for s in seg_infos:
                        pts.append(tuple(s['a']))
                        pts.append(tuple(s['b']))
                    inner_poly = Polygon(pts).convex_hull
        except Exception:
            pts = []
            for s in seg_infos:
                pts.append(tuple(s['a']))
                pts.append(tuple(s['b']))
            inner_poly = Polygon(pts).convex_hull

    # compute a reliable interior test point
    try:
        interior_pt = np.array(inner_poly.representative_point().coords[0])
    except Exception:
        interior_pt = np.array([0.0, 0.0])

    half = float(wall_thickness) / 2.0

    # compute and collect outward normals robustly by testing small offsets
    normals = []
    eps = max(extent * 1e-4, half * 0.1, 1e-6)
    for s in seg_infos:
        a = s['a']
        b = s['b']
        v = b - a
        L = np.linalg.norm(v)
        mid = (a + b) / 2.0
        if L < 1e-8:
            normals.append({'mid': mid, 'n': np.array([0.0, 0.0])})
            continue
        u = v / L
        n = np.array([-u[1], u[0]])

        # test both candidate normals by probing a small offset point
        cand_pos = tuple((mid + n * eps).tolist())
        cand_neg = tuple((mid - n * eps).tolist())
        inside_pos = False
        inside_neg = False
        try:
            inside_pos = inner_poly.contains(Point(cand_pos))
        except Exception:
            inside_pos = False
        try:
            inside_neg = inner_poly.contains(Point(cand_neg))
        except Exception:
            inside_neg = False

        # outward should be the direction that is NOT inside the inner polygon
        if inside_pos and not inside_neg:
            chosen = -n
        elif inside_neg and not inside_pos:
            chosen = n
        elif (not inside_pos) and (not inside_neg):
            # both outside: pick the one further from interior point
            dpos = np.linalg.norm(np.array(cand_pos) - interior_pt)
            dneg = np.linalg.norm(np.array(cand_neg) - interior_pt)
            chosen = n if dneg < dpos else -n
        else:
            # both inside (degenerate): fallback to centroid heuristic
            chosen = n if np.dot(n, mid - interior_pt) > 0 else -n

        normals.append({'mid': mid, 'n': chosen})

    # Build outer polygon by buffering the inner polygon outward by `half`
    # Use square caps and mitre joins to get sharp corners
    try:
        # shapely buffer cap_style/join_style: 1=round, 2=mitre/flat, 3=square (varies by version)
        # We'll use cap_style=3 (square) and join_style=2 (mitre) which are common mappings.
        buffered = inner_poly.buffer(distance=half, resolution=16, cap_style=3, join_style=2, mitre_limit=5.0)
    except TypeError:
        # older/newer shapely signatures might accept cap_style/join_style as strings
        try:
            buffered = inner_poly.buffer(distance=half, resolution=16, cap_style='square', join_style='mitre', mitre_limit=5.0)
        except Exception:
            # final fallback: simple buffer with defaults
            buffered = inner_poly.buffer(distance=half)

    # If buffering produced a MultiPolygon, take the largest polygon by area
    if buffered is None:
        outer_ring = None
    else:
        if buffered.geom_type == 'Polygon':
            outer_poly = buffered
        else:
            # MultiPolygon
            geoms = list(buffered.geoms)
            if not geoms:
                outer_poly = None
            else:
                outer_poly = max(geoms, key=lambda p: p.area)
        outer_ring = LineString(list(outer_poly.exterior.coords)) if outer_poly is not None else None

    # Always produce an interactive Plotly HTML output (user requested Plotly-only)
    if isinstance(out_img, str) and out_img.lower().endswith('.html'):
        out_html = out_img
    else:
        base = os.path.splitext(str(out_img))[0]
        out_html = base + '.html'

    # arrow length for normals — scale relative to wall thickness
    arrow_len = max(float(wall_thickness) * 2.0, 0.05)


    # center_ring = LineString([pt for ln in center_lines for pt in ln.xy])
    merged_line = linemerge(center_lines)  # could be LineString or MultiLineString

    center_pts = []

    if merged_line.geom_type == 'LineString':
        xs, ys = merged_line.xy
        center_pts = list(zip(xs, ys))
    elif merged_line.geom_type == 'MultiLineString':
        for ln in merged_line.geoms:
            xs, ys = ln.xy
            center_pts.extend(list(zip(xs, ys)))
    else:
        raise ValueError(f"Unexpected geometry type: {merged_line.geom_type}")

    center_ring = merged_line

    # # Remove consecutive duplicate points that occur at shared endpoints
    # if center_pts:
    #     clean_pts = [center_pts[0]]
    #     for p in center_pts[1:]:
    #         if p != clean_pts[-1]:
    #             clean_pts.append(p)
        
    # else:
    #     center_ring = None

    # plot_layout_interactive(center_ring=center_ring, normals=normals, outer_ring=outer_ring,
    #                         out_html=out_html, center_color=center_color, outer_color=outer_color,
    #                         title='Interactive 2D Layout: centerlines, normals and buffered outer boundary',
    #                         arrow_len=arrow_len)
    

    return {'inner_ring': center_ring, 'outer_ring': outer_ring}


# -----------------------------
# Layout helpers
# -----------------------------

def plot_layout_interactive(center_ring, normals, outer_ring, out_html="layout.html", center_color='black', outer_color='#ff7f0e', title='Interactive 2D Layout', arrow_len=0.5):
    """Create an interactive Plotly HTML showing centerlines, normals (as arrows) and an outer ring.

    Args:
        center_lines (list[LineString])
        normals (list[dict]) : dicts with keys 'mid' (np.array) and 'n' (np.array)
        outer_ring (LineString | None)
        out_html (str): output HTML filename
    """
    fig = go.Figure()

    if center_ring is not None:
        xs_arr, ys_arr = center_ring.xy
        xs, ys = list(xs_arr), list(ys_arr)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=center_color, width=2),
                                 hovertemplate='%{x:.3f}, %{y:.3f}<extra>centerline</extra>'))

    # normals as annotations
    annotations = []
    for ninfo in normals:
        mid = ninfo['mid']
        n = ninfo['n']
        if np.linalg.norm(n) < 1e-12:
            continue
        sx, sy = float(mid[0]), float(mid[1])
        tx, ty = float(mid[0] + n[0] * arrow_len), float(mid[1] + n[1] * arrow_len)
        annotations.append(dict(x=tx, y=ty, ax=sx, ay=sy, xref='x', yref='y', axref='x', ayref='y',
                                showarrow=True, arrowhead=4, arrowsize=2, arrowwidth=4, arrowcolor='cyan'))

    if outer_ring is not None:
        xs_arr, ys_arr = outer_ring.xy
        xs, ys = list(xs_arr), list(ys_arr)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=outer_color, width=3),
                                 hovertemplate='%{x:.3f}, %{y:.3f}<extra>outer_ring</extra>'))

    fig.update_layout(annotations=annotations)
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False, scaleanchor='x', scaleratio=1)
    fig.update_layout(title=title)

    plotly_plot(fig, filename=out_html, auto_open=True)
    print(f"Saved interactive plot to {out_html}")

def plotly_plot(fig, filename="plot.html", auto_open=True):
    """Save a Plotly figure to an HTML file and optionally open it in a web browser."""
    import plotly.io as pio
    pio.write_html(fig, file=filename, auto_open=auto_open)
    print(f"Saved interactive plot to {filename}")

# -----------------------------
# Wall helpers for door/window rotation
# -----------------------------
def closest_wall_angle(x, y, walls):
    min_dist = float('inf')
    angle = 0.0
    for wall in walls.values():
        ax, ay = wall["a_x"], wall["a_y"]
        bx, by = wall["b_x"], wall["b_y"]
        t = max(0, min(1, ((x-ax)*(bx-ax) + (y-ay)*(by-ay)) / ((bx-ax)**2 + (by-ay)**2)))
        proj_x = ax + t*(bx-ax)
        proj_y = ay + t*(by-ay)
        dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        if dist < min_dist:
            min_dist = dist
            angle = np.arctan2(by- ay, bx - ax)
    return angle

# -----------------------------
# Utilities
# -----------------------------
def round_point(pt, decimals=3):
    return tuple(np.round(pt, decimals))

def create_grids(size=10, step=1, color=[0.8,0.8,0.8]):
    def make_grid_xy(size, step):
        lines, points = [], []
        half = size/2
        for i in np.arange(-half, half+step, step):
            points.append([i,-half,0]); points.append([i,half,0])
            lines.append([len(points)-2, len(points)-1])
            points.append([-half,i,0]); points.append([half,i,0])
            lines.append([len(points)-2, len(points)-1])
        grid = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
        grid.paint_uniform_color(color)
        return grid
    def make_grid_xz(size, step):
        lines, points = [], []
        half = size/2
        for i in np.arange(-half, half+step, step):
            points.append([i,0,-half]); points.append([i,0,half])
            lines.append([len(points)-2, len(points)-1])
            points.append([-half,0,i]); points.append([half,0,i])
            lines.append([len(points)-2, len(points)-1])
        grid = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
        grid.paint_uniform_color(color)
        return grid
    return make_grid_xz(size, step), make_grid_xy(size, step)

def project_outer_edges_and_extrude(meshes, extrusion_height=2.5, color=[0.85,0.85,0.85], show_debug=True):
    edges_xy = []
    all_xy_points = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        for tri in faces:
            tri_pts = vertices[tri][:,:2]
            for i in range(3):
                a = tri_pts[i]; b = tri_pts[(i+1)%3]
                edges_xy.append((tuple(a), tuple(b)))
            all_xy_points.extend(tri_pts)
    all_xy_points = np.array(all_xy_points)
    outer_edges = [(round_point(a), round_point(b)) for a,b in edges_xy]

    adjacency = {}
    for a,b in outer_edges:
        adjacency.setdefault(a,[]).append(b)
        adjacency.setdefault(b,[]).append(a)

    start = outer_edges[0][0]
    contour = [start]
    current = start; prev = None

    while True:
        neighbors = adjacency[current]
        next_v = None
        for n in neighbors:
            if n != prev:
                next_v = n
                break
        if next_v is None or next_v == start:
            break
        contour.append(next_v)
        prev, current = current, next_v
        if len(contour) > len(adjacency)*2:
            break

    contour = np.array(contour)
    contour_3d = np.column_stack((contour, np.zeros(len(contour))))
    triangles = [[0,i,i+1] for i in range(1,len(contour_3d)-1)]

    base = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(contour_3d),
                                     triangles=o3d.utility.Vector3iVector(np.array(triangles)))
    base.compute_vertex_normals()
    top = base.translate([0,0,extrusion_height], relative=False)
    extruded = base + top
    extruded.paint_uniform_color(color)
    return extruded

# -----------------------------
# Main 3D room builder
# -----------------------------
def make3DMeshRoom(data_json):
    if isinstance(data_json, str):
        with open(data_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = data_json

    meshes = []
    scene_objects = []
    color_map = {
        "wall": [0.9, 0.9, 0.9],
        "floor": [0.8, 0.8, 0.8],
        "door": [0.85, 0.7, 0.5],
        "window": [0.6, 0.8, 1.0],
        "bbox": [0.2, 0.2, 0.2]
    }

    bbox_color_map = {
        "wall_decoration": [1.0, 0.5, 0.5],
        "dining_chair": [0.8, 0.6, 0.4],
        "dining_table": [0.6, 0.4, 0.2],
        "carpet": [0.9, 0.8, 0.4],
        "curtain": [0.4, 0.6, 1.0],
        "coffee_table": [0.5, 0.4, 0.3],
        "sofa": [0.6, 0.2, 0.2],
        "wine_cabinet": [0.4, 0.3, 0.2],
        "painting": [0.8, 0.2, 0.5],
        "bed": [0.7, 0.5, 0.3],
        "nightstand": [0.5, 0.4, 0.3],
        "desk": [0.6, 0.4, 0.3],
        "tv": [0.1, 0.1, 0.1],
        "hand_sink": [0.7, 0.7, 0.9],
        "toilet": [0.9, 0.9, 0.9],
        "shower": [0.8, 0.8, 1.0],
        "sink": [0.7, 0.7, 0.9],
    }

    # Make Layout
    layout_dict = make_layout(data, out_img="layout.html", wall_thickness=0.1, center_color="black", outer_color="#ff7f0e", snap_tol=1e-8)
    
    # Walls
    mesh, a_z = make_wall_from_layout(layout_dict, data=data, color=color_map["wall"])
    meshes.append(mesh)

    # wall_meshes = []
    # for name, wall in data.get("walls", {}).items(): 
    #     mesh_wall = make_wall_mesh(wall, color=color_map["wall"], thickness=0.01)
    #     if mesh_wall:
    #         meshes.append(mesh_wall)
    #         wall_meshes.append(mesh_wall)

 


    # Floor
    # Floor
    meshes.append(make_floor(dict_data_layout=layout_dict, color=color_map.get("floor", [0.8, 0.8, 0.8]), thickness=-0.1))

    # Doors
    for door in data.get("doors", []):
        angle = closest_wall_angle(door["position_x"], door["position_y"], data["walls"])
        meshes.append(make_door_mesh(door, wall_angle=angle, color=color_map["door"], a_z=a_z))

    # Windows
    for win in data.get("windows", []):
        angle = closest_wall_angle(win["position_x"], win["position_y"], data["walls"])
        meshes.append(make_window_mesh(win, wall_angle=angle, color=color_map["window"], a_z=a_z))

    # Bboxes
    for item in data.get("bboxes", []):
        bbox = Bbox(**item)
        color = bbox_color_map.get(bbox.object_id, [0.2,0.2,0.2])
        mesh = make_bbox_mesh(bbox, color=color, a_z=a_z)
        scene_object = SceneObject(mesh=mesh, category=bbox.object_id)
        if scene_object.category in ["dining_table","desk"]:
            scene_object.detailed_mesh = utils_detailed.make_detailed_table(scene_object=scene_object)
            # meshes.append(scene_object.mesh)
        elif scene_object.category in ["chair", "dining_chair"]:
            scene_object.detailed_mesh = utils_detailed.make_detailed_chair(scene_object=scene_object)
            # meshes.append(scene_object.mesh)
        else:
            scene_object.detailed_mesh = mesh
            
        # Paint the detailed mesh (if it was newly created by utils_detailed, it lacks color)
        if scene_object.detailed_mesh:
             scene_object.detailed_mesh.paint_uniform_color(color)
        
        meshes.append(scene_object.detailed_mesh)
        scene_objects.append(scene_object)

    # scene_objects = utils_detailed.make_detailed_mesh(scene_objects)
    # detailed_meshes = [obj.detailed_mesh for obj in scene_objects if obj.detailed_mesh is not None]
    for obj in scene_objects:
        print(f"Object ID: {obj.category}, Detailed Mesh: {'Yes' if obj.detailed_mesh is not None else 'No'}")



    # Visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    print(f"Axis directions: X-red, Y-green, Z-blue, are:", axis)
    
    return meshes
    utils_lighting.visualize_with_lighting(meshes=meshes)

    print(f"Created {len(meshes)} meshes (Walls={len(data.get('walls', {}))}, Doors={len(data.get('doors', []))}, Windows={len(data.get('windows', []))}, Bboxes={len(data.get('bboxes', []))})")
    return meshes
