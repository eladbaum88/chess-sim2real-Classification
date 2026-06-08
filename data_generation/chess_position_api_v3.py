"""
Chess synthetic rendering — v3 (simplified, single chess set).

Reverted to the v1 piece source (chess-set.blend). Per-render randomization
adds variety via HDRI rotation, sun direction/strength, camera pitch/yaw/
distance jitter — without ever swapping the underlying piece geometry.

One image per Blender invocation. The outer driver picks all randomization
values, passes them as CLI args, this script renders.

Invocation (typically from build_dataset_v2.py):
    blender chess-set.blend --background --python chess_position_api_v3.py -- \
        --fen "<fen>" \
        --hdri hdris/studio_small_03.exr \
        --hdri-rotation 137 --hdri-strength 1.0 \
        --sun-energy 3.0 --sun-temp-k 5000 \
        --cam-height-mul 1.0 --cam-angle-deg 40 --cam-yaw-deg 2 \
        --cam-lens 30 --cam-view white --cam-roll-deg 0 \
        --output dataset_v2/images/test.png
"""

import bpy
import math
import os
import sys
import argparse
from mathutils import Vector, Matrix

# Blender's bundled Python doesn't import pip --user packages by default.
import site
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from bpy_extras.object_utils import world_to_camera_view
import numpy as np
from PIL import Image


# ======================================================================
# CONFIG
# ======================================================================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2.0

# chess-set.blend mesh names
BASE_BOARD_PLANE = "Black & white"
BASE_BOARD_FRAME = "Outer frame"


# ======================================================================
# BOARD GEOMETRY
# ======================================================================
def fix_base_board_rotation():
    """The 90° rotation of 'Black & white' that aligns checker with files/ranks."""
    plane = bpy.data.objects.get(BASE_BOARD_PLANE)
    frame = bpy.data.objects.get(BASE_BOARD_FRAME)
    if plane is None or frame is None:
        return
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    f_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    f_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (f_min + f_max) / 2
    original_pos = plane.location.copy()
    offset = original_pos - center
    plane.rotation_euler.z = math.radians(90)
    rot = Matrix.Rotation(math.radians(90), 3, "Z")
    plane.location = center + (rot @ offset)


def get_board_info():
    plane = bpy.data.objects.get(BASE_BOARD_PLANE)
    frame = bpy.data.objects.get(BASE_BOARD_FRAME)
    if plane is None:
        raise RuntimeError(f"Base scene missing '{BASE_BOARD_PLANE}' mesh.")
    if frame is None:
        frame = plane
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8.0

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)
    scale_factor = board_size / REAL_BOARD_SIZE

    return {
        "square_size": square_size,
        "plane_min": plane_min,
        "plane_max": plane_max,
        "center": center,
        "scale_factor": scale_factor,
    }


def square_to_world_xy(square, board_info):
    s = board_info["square_size"]
    pmin = board_info["plane_min"]; pmax = board_info["plane_max"]
    file_idx = ord(square[0]) - ord("a")
    rank_idx = int(square[1]) - 1
    return (pmin.x + (7 - file_idx + 0.5) * s,
            pmax.y - (rank_idx + 0.5) * s)


def parse_fen(fen):
    """FEN board section -> {square: piece_char}. Strict validation."""
    if not fen or not isinstance(fen, str):
        raise ValueError(f"Empty/non-string FEN: {fen!r}")
    board = fen.split()[0]
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN has {len(ranks)} ranks, expected 8: {fen!r}")
    pos = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        rnum = 8 - rank_idx
        for c in rank:
            if c.isdigit():
                file_idx += int(c)
            elif c in "PNBRQKpnbrqk":
                if file_idx >= 8:
                    raise ValueError(f"FEN rank {rank!r} overflows file h: {fen!r}")
                pos[f"{chr(ord('a') + file_idx)}{rnum}"] = c
                file_idx += 1
            else:
                raise ValueError(f"FEN has unexpected char {c!r}: {fen!r}")
        if file_idx != 8:
            raise ValueError(f"FEN rank {rank!r} doesn't sum to 8 (got {file_idx}): {fen!r}")
    return pos


# ======================================================================
# PIECES (chess-set.blend native names)
# ======================================================================
def detect_base_pieces():
    """Identify chess-set.blend's pieces via the v1 name conventions."""
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        n = obj.name
        if n in [BASE_BOARD_PLANE, BASE_BOARD_FRAME]:
            continue
        t = None
        if n in ["B", "C", "D", "E", "F", "G", "H", "A(texture)"]:
            t = "P"
        elif n in ["B.001", "C.001", "D.001", "E.001", "F.001", "G.001", "H.001", "A(textures)"]:
            t = "p"
        elif "rook" in n.lower():
            t = "R" if "white" in n.lower() else "r"
        elif "knight" in n.lower():
            t = "N" if "white" in n.lower() else "n"
        elif "bitshop" in n.lower() or "bishop" in n.lower():
            t = "B" if "white" in n.lower() else "b"
        elif "queen" in n.lower():
            t = "Q" if "white" in n.lower() else "q"
        elif "king" in n.lower():
            t = "K" if "white" in n.lower() else "k"
        if t:
            pieces[n] = {"piece_type": t}
    return pieces


def make_piece_material(name, rgb, roughness=0.5):
    """Solid-opaque Principled BSDF. Explicitly zeros Subsurface +
    Transmission + forces Alpha=1.0 (defaults can leak translucency in
    Eevee Next on some Blender versions)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    def _set(input_name, value):
        if input_name in bsdf.inputs:
            bsdf.inputs[input_name].default_value = value

    _set("Base Color", (*rgb, 1.0))
    _set("Roughness", float(roughness))
    _set("Metallic", 0.0)
    _set("Alpha", 1.0)
    _set("Subsurface Weight", 0.0); _set("Subsurface", 0.0)
    _set("Transmission Weight", 0.0); _set("Transmission", 0.0)
    _set("Sheen Weight", 0.0); _set("Coat Weight", 0.0)
    _set("Emission Strength", 0.0)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    for attr, value in [("surface_render_method", "DITHERED"),
                        ("blend_method", "OPAQUE")]:
        try:
            setattr(mat, attr, value)
        except (AttributeError, TypeError):
            pass
    return mat


def override_board_material(light_rgb, dark_rgb, roughness=0.6):
    """Replace the playable area's material with a procedural 8x8 checker
    matching the reference image's warm cream/brown wood tones.

    Uses Generated texture coords so the pattern aligns with the mesh's
    bounding box (no UV setup required).
    """
    plane = bpy.data.objects.get(BASE_BOARD_PLANE)
    if plane is None:
        return
    mat = bpy.data.materials.new(name="BoardCheckerWarm")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (*light_rgb, 1.0)
    checker.inputs["Color2"].default_value = (*dark_rgb, 1.0)
    checker.inputs["Scale"].default_value = 8.0
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tc.outputs["Generated"], checker.inputs["Vector"])
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for attr, value in [("surface_render_method", "DITHERED"),
                        ("blend_method", "OPAQUE")]:
        try:
            setattr(mat, attr, value)
        except (AttributeError, TypeError):
            pass
    # Per-instance mesh copy so we don't pollute chess-set.blend's data block
    plane.data = plane.data.copy()
    if plane.data.materials:
        for i in range(len(plane.data.materials)):
            plane.data.materials[i] = mat
    else:
        plane.data.materials.append(mat)
    for poly in plane.data.polygons:
        poly.material_index = 0
    print(f"  BOARD: warm checker  light={tuple(round(v,2) for v in light_rgb)}  "
          f"dark={tuple(round(v,2) for v in dark_rgb)}  rough={roughness:.2f}")


def apply_fen_to_pieces(fen, base_pieces, board_info):
    """Place chess-set.blend pieces per FEN. Overrides each piece's material
    with a solid opaque PBR (white cream / near-black) so the .blend's
    native black material — which renders waxy/translucent in Eevee — is
    replaced. Uses per-piece mesh data copy to keep material override from
    propagating across instances that share mesh data.
    """
    target = parse_fen(fen)
    # Wood-tone pieces matching the reference image:
    # white = warm beige/tan, black = dark brown wood, both with rough=0.6.
    mat_white = make_piece_material("BaseWhite", (0.78, 0.68, 0.50), roughness=0.6)
    mat_black = make_piece_material("BaseBlack", (0.15, 0.08, 0.05), roughness=0.6)

    pool = {}
    for name, info in base_pieces.items():
        pool.setdefault(info["piece_type"], []).append(name)
    used = set()
    unfulfilled = []
    white_count = 0; black_count = 0
    for sq, ptype in target.items():
        if not pool.get(ptype):
            unfulfilled.append((sq, ptype)); continue
        name = pool[ptype].pop()
        obj = bpy.data.objects.get(name)
        if obj is None:
            unfulfilled.append((sq, ptype)); continue
        x, y = square_to_world_xy(sq, board_info)
        obj.location.x = x
        obj.location.y = y
        obj.hide_render = False
        obj.hide_viewport = False

        # Per-piece mesh copy so material assignment doesn't propagate
        obj.data = obj.data.copy()
        mat = mat_white if ptype.isupper() else mat_black
        if obj.data.materials:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
        else:
            obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0

        used.add(name)
        if ptype.isupper(): white_count += 1
        else: black_count += 1
    # Hide unused base pieces
    for name in base_pieces:
        if name in used:
            continue
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True; obj.hide_viewport = True
    print(f"  apply_fen: placed {len(used)}/{len(target)} "
          f"(white={white_count}, black={black_count}), unfulfilled={unfulfilled}")
    return len(unfulfilled) == 0


# ======================================================================
# LIGHTING
# ======================================================================
def setup_hdri_world(hdri_path, rotation_deg, strength):
    """HDRI as world environment, layered on top of the sun light below."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value[2] = math.radians(float(rotation_deg))
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    print(f"  HDRI: {os.path.basename(hdri_path)} rot={rotation_deg:.0f}° strength={strength:.2f}")


def setup_sun_light(board_info, energy, temp_k, az_jit, el_jit):
    """Place a sun light above the board with jittered direction + temperature."""
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    c = board_info["center"]
    sf = board_info["scale_factor"]
    h = c.z + 2.0 * DESIRED_CAMERA_HEIGHT * sf
    bpy.ops.object.light_add(type="SUN", location=(c.x, c.y, h))
    sun = bpy.context.active_object
    sun.data.energy = float(energy)
    sun.data.use_nodes = True
    nt = sun.data.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bb = nt.nodes.new("ShaderNodeBlackbody")
    bb.inputs["Temperature"].default_value = float(temp_k)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputLight")
    nt.links.new(bb.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    sun.rotation_euler = (math.radians(float(el_jit)), 0.0, math.radians(float(az_jit)))
    print(f"  SUN: energy={energy:.2f} temp={temp_k:.0f}K az={az_jit:+.1f}° el={el_jit:+.1f}°")


# ======================================================================
# CAMERA
# ======================================================================
def setup_camera(board_info, h_mul, angle_deg, lens, view, yaw_deg, roll_deg):
    """Place camera. Three views, matching v1 conventions:
      view='east'     -> camera at +X offset (board on west side from cam)
      view='west'     -> camera at -X offset
      view='overhead' -> camera directly above center, pitch forced to 0
                         (= directly top-down), yaw rotates the camera view.

    yaw_deg jitter rotates the camera POSITION around the board's vertical
    axis for east/west views (so the camera isn't perfectly aligned with
    the X axis). For overhead, yaw rotates the camera's roll-around-Z so
    different overhead frames show the board at different rotations.
    """
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    c = board_info["center"]
    sf = board_info["scale_factor"]
    cam_h = DESIRED_CAMERA_HEIGHT * sf * h_mul

    if view == "overhead":
        # Directly above the board, looking straight down. pitch forced
        # to effective 0; angle_deg parameter is ignored for overhead.
        cx, cy = c.x, c.y
        cz = c.z + cam_h
        bpy.ops.object.camera_add(location=(cx, cy, cz))
        cam = bpy.context.active_object
        # Point straight down (-Z)
        cam.rotation_euler = (0.0, 0.0, math.radians(float(yaw_deg)))
        eff_pitch = 0.0
    else:
        # east/west: camera offset in X
        h_off = cam_h * math.tan(math.radians(float(angle_deg)))
        sign = +1.0 if view == "east" else -1.0
        base_dx = sign * h_off
        base_dy = 0.0
        # Yaw rotation around board's Z axis through center
        yaw_rad = math.radians(float(yaw_deg))
        dx = base_dx * math.cos(yaw_rad) - base_dy * math.sin(yaw_rad)
        dy = base_dx * math.sin(yaw_rad) + base_dy * math.cos(yaw_rad)
        cx = c.x + dx
        cy = c.y + dy
        cz = c.z + cam_h
        bpy.ops.object.camera_add(location=(cx, cy, cz))
        cam = bpy.context.active_object
        direction = c - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        eff_pitch = angle_deg

    cam.rotation_euler.rotate_axis("Z", math.radians(float(roll_deg)))
    cam.data.lens = float(lens)
    cam.data.clip_start = 0.001
    cam.data.clip_end = max(1000.0, cam_h * 10.0)
    bpy.context.scene.camera = cam
    print(f"  CAMERA: h_mul={h_mul:.2f} pitch={eff_pitch:.1f}° yaw={yaw_deg:+.1f}° "
          f"lens={lens:.1f}mm view={view} roll={roll_deg:+.1f}°")
    return cam


# ======================================================================
# RECTIFICATION
# ======================================================================
def get_board_corners_3d(outer_padding=0.0):
    plane = bpy.data.objects.get(BASE_BOARD_PLANE)
    pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    top_z = max(p.z for p in pts)
    top_pts = [p for p in pts if abs(p.z - top_z) < 1e-4]
    seen = set(); uniq = []
    for p in top_pts:
        key = (round(p.x, 6), round(p.y, 6))
        if key not in seen:
            seen.add(key); uniq.append(p)
    if outer_padding > 0 and len(uniq) == 4:
        cx = sum(p.x for p in uniq) / 4.0
        cy = sum(p.y for p in uniq) / 4.0
        uniq = [Vector((cx + (p.x - cx) * (1 + outer_padding),
                        cy + (p.y - cy) * (1 + outer_padding),
                        p.z)) for p in uniq]
    return uniq


def get_max_piece_z():
    exclude = {BASE_BOARD_PLANE, BASE_BOARD_FRAME}
    mz = None
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render or obj.name in exclude:
            continue
        pts = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        z = max(p.z for p in pts)
        if mz is None or z > mz:
            mz = z
    return mz


def project_to_image(pts_3d, cam, scene):
    W = scene.render.resolution_x; H = scene.render.resolution_y
    res = []
    for p in pts_3d:
        co = world_to_camera_view(scene, cam, p)
        res.append((co.x * W, (1.0 - co.y) * H))
    return res


def piece_aware_source_corners(corners_3d, cam, scene, margin):
    board_z = max(p.z for p in corners_3d)
    top = get_max_piece_z()
    if top is None or top <= board_z:
        return project_to_image(corners_3d, cam, scene)
    lift = board_z + margin * (top - board_z)
    distances = [(cam.location - p).length for p in corners_3d]
    far = set(sorted(range(len(corners_3d)), key=lambda i: distances[i])[-2:])
    mixed = []
    for i, p in enumerate(corners_3d):
        if i in far:
            mixed.append(Vector((p.x, p.y, lift)))
        else:
            mixed.append(p)
    return project_to_image(mixed, cam, scene)


def order_tl_tr_br_bl(pts):
    """Order image points by their position in the IMAGE (legacy)."""
    s = sorted(pts, key=lambda p: p[1])
    top = sorted(s[:2], key=lambda p: p[0])
    bot = sorted(s[2:], key=lambda p: p[0], reverse=True)
    return [top[0], top[1], bot[0], bot[1]]


def order_by_world_xy(world_pts, image_pts):
    """Order image points so the rectified output has a CONSISTENT
    chess orientation across all camera views:

      world HIGH Y  (= rank 1, WHITE pieces)  ->  image BOTTOM
      world LOW Y   (= rank 8, BLACK pieces)  ->  image TOP

    This decouples the rectified frame from camera position. White
    always lands at the bottom of every rendered frame, regardless of
    whether the camera was east, west, or overhead.

    Returns image points in the order TL, TR, BR, BL (matching the
    dst quad we map to: (0,0), (W,0), (W,H), (0,H)).
    """
    paired = list(zip(world_pts, image_pts))
    paired.sort(key=lambda p: p[0].y)  # low world Y first (image top)
    top = sorted(paired[:2], key=lambda p: p[0].x)            # ascending world X
    bot = sorted(paired[2:], key=lambda p: p[0].x, reverse=True)
    return [top[0][1], top[1][1], bot[0][1], bot[1][1]]


def pil_coeffs(src, dst):
    M = []
    for (x, y), (X, Y) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        M.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(M, dtype=np.float64)
    B = np.array([v for p in src for v in p], dtype=np.float64)
    return tuple(np.linalg.solve(A, B))


def rectify(raw_path, corners_3d, cam, scene, out_size, out_path, piece_margin):
    if piece_margin > 0:
        image_pts = piece_aware_source_corners(corners_3d, cam, scene, piece_margin)
    else:
        image_pts = project_to_image(corners_3d, cam, scene)
    # Order corners by WORLD position so the rectified output always has
    # white pieces at the bottom and black at the top, regardless of
    # which view the camera was at (east, west, overhead).
    src = order_by_world_xy(corners_3d, image_pts)
    dst = [(0, 0), (out_size, 0), (out_size, out_size), (0, out_size)]
    coeffs = pil_coeffs(src, dst)
    img = Image.open(raw_path)
    img.transform((out_size, out_size), Image.PERSPECTIVE, coeffs, Image.BICUBIC).save(out_path)
    return src


# ======================================================================
# MAIN
# ======================================================================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--fen", type=str, required=True)
    p.add_argument("--output", type=str, required=True)

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--rectified-size", type=int, default=512)
    p.add_argument("--outer-padding", type=float, default=0.02)
    p.add_argument("--piece-margin", type=float, default=1.0)

    p.add_argument("--hdri", type=str, required=True)
    p.add_argument("--hdri-rotation", type=float, default=0.0)
    p.add_argument("--hdri-strength", type=float, default=1.0)

    p.add_argument("--sun-energy", type=float, default=3.0)
    p.add_argument("--sun-temp-k", type=float, default=5000.0)
    p.add_argument("--sun-azimuth-jitter", type=float, default=0.0)
    p.add_argument("--sun-elevation-jitter", type=float, default=0.0)

    p.add_argument("--cam-height-mul", type=float, default=1.0)
    p.add_argument("--cam-angle-deg", type=float, default=40.0)
    p.add_argument("--cam-yaw-deg", type=float, default=0.0)
    p.add_argument("--cam-lens", type=float, default=30.0)
    p.add_argument("--cam-view", type=str,
                   choices=["east", "west", "overhead"], default="east")
    p.add_argument("--cam-roll-deg", type=float, default=0.0)

    return p.parse_args(argv)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parse_args(argv)

    print(f"\n{'=' * 70}\nv3 RENDER  ->  {os.path.basename(args.output)}\n{'=' * 70}")
    print(f"FEN: {args.fen}")

    fix_base_board_rotation()
    board_info = get_board_info()
    # Warm cream/brown checker board overrides chess-set.blend's native
    # white/black material so the scene matches the reference image tone.
    override_board_material(
        light_rgb=(0.85, 0.75, 0.45),
        dark_rgb=(0.45, 0.30, 0.18),
        roughness=0.6,
    )
    base_pieces = detect_base_pieces()
    ok = apply_fen_to_pieces(args.fen, base_pieces, board_info)
    if not ok:
        print("FATAL: apply_fen unfulfilled", file=sys.stderr)
        sys.exit(2)

    # Eevee Next, opaque pieces, no Cycles needed.
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = int(args.samples)
    scene.render.resolution_x = int(args.resolution)
    scene.render.resolution_y = int(args.resolution)
    scene.render.image_settings.file_format = "PNG"
    # Color management: Blender's default Filmic desaturates colors and
    # was washing the cream piece tint into near-white. Standard preserves
    # the material's actual base color.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except (AttributeError, TypeError) as e:
        print(f"  WARN view_transform setting: {e}")
    print(f"  Render engine: {scene.render.engine}, "
          f"TAA samples: {scene.eevee.taa_render_samples}, "
          f"resolution: {args.resolution}, "
          f"view_transform: {scene.view_settings.view_transform}, "
          f"look: {scene.view_settings.look}")

    setup_hdri_world(args.hdri, args.hdri_rotation, args.hdri_strength)
    setup_sun_light(board_info, args.sun_energy, args.sun_temp_k,
                    args.sun_azimuth_jitter, args.sun_elevation_jitter)
    cam = setup_camera(board_info, args.cam_height_mul, args.cam_angle_deg,
                       args.cam_lens, args.cam_view, args.cam_yaw_deg, args.cam_roll_deg)

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, os.path.basename(args.output) + ".raw.png")
    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    corners = get_board_corners_3d(outer_padding=args.outer_padding)
    src = rectify(raw_path, corners, cam, scene, args.rectified_size, args.output, args.piece_margin)
    print(f"  ✓ Saved rectified: {args.output} ({args.rectified_size}x{args.rectified_size})")
    print(f"    board corners: {[(round(p[0], 1), round(p[1], 1)) for p in src]}")
    try:
        os.remove(raw_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
