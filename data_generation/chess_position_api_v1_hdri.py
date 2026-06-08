"""
v1-style render with optional HDRI world environment.

Continuation of chess_position_api_v2.py used to render the first 1500-image
dataset. Identical camera setup (25° tilt, lens=26, three views from black:
overhead/west/east) and identical rectified output (800 raw -> 512 rectified).
The ONLY addition is that a per-render HDRI is loaded into the world shader
so lighting/background colour varies across the dataset.

Usage:
    blender chess-set.blend --background --python chess_position_api_v1_hdri.py -- \
        --fen "<fen>" --view black \
        --resolution 800 --samples 128 --rectified-size 512 \
        --piece-margin 0.1 \
        --hdri hdris/studio_small_03.exr --hdri-rotation 137 --hdri-strength 1.0
"""

import bpy
import math
import os
from mathutils import Vector, Matrix
import sys
import argparse

import site
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

from bpy_extras.object_utils import world_to_camera_view
import numpy as np
from PIL import Image

# ==========================
# CONFIG (matches v1 exactly)
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
RECTIFIED_SIZE = 512
PIECE_MARGIN = 0.6
OUTER_PADDING = 0.0
FORCE_CPU = False
OUT_DIR = "//renders"

HDRI_PATH = None
HDRI_ROTATION_DEG = 0.0
HDRI_STRENGTH = 1.0


def get_board_info():
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)

    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'scale_factor': board_size / REAL_BOARD_SIZE,
    }


def position_to_square(pos, board_info):
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    return f"{chr(ord('a') + file_idx)}{rank_idx + 1}"


def square_to_world_xy(square, board_info):
    s = board_info['square_size']
    pmin = board_info['plane_min']
    pmax = board_info['plane_max']
    file_idx = ord(square[0]) - ord('a')
    rank_idx = int(square[1]) - 1
    return pmin.x + (7 - file_idx + 0.5) * s, pmax.y - (rank_idx + 0.5) * s


def detect_starting_positions(board_info):
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        name = obj.name
        piece_type = None
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'
        elif 'rook' in name.lower():
            piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower():
            piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower():
            piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower():
            piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower():
            piece_type = 'K' if 'white' in name.lower() else 'k'
        if piece_type:
            pieces[name] = {
                'square': position_to_square(obj.location, board_info),
                'piece_type': piece_type,
                'start_pos': obj.location.copy(),
            }
    return pieces


def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    position = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                position[f"{chr(ord('a') + file_idx)}{board_rank}"] = char
                file_idx += 1
    return position


def apply_fen(fen, starting_pieces, board_info):
    target_position = parse_fen(fen)
    pool = {}
    for piece_name, info in starting_pieces.items():
        pool.setdefault(info['piece_type'], []).append(piece_name)

    used_names = set()
    unfulfilled = []
    for target_square, piece_type in target_position.items():
        if not pool.get(piece_type):
            unfulfilled.append((target_square, piece_type))
            continue
        piece_name = pool[piece_type].pop()
        obj = bpy.data.objects.get(piece_name)
        if obj is None:
            unfulfilled.append((target_square, piece_type))
            continue
        tx, ty = square_to_world_xy(target_square, board_info)
        obj.location.x = tx
        obj.location.y = ty
        obj.hide_render = False
        obj.hide_viewport = False
        used_names.add(piece_name)

    for piece_name in starting_pieces.keys():
        if piece_name in used_names:
            continue
        obj = bpy.data.objects.get(piece_name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_viewport = True

    success = (len(unfulfilled) == 0)
    return {
        'success': success,
        'pieces_placed': len(used_names),
        'pieces_requested': len(target_position),
        'unfulfilled': unfulfilled,
    }


def get_board_corners_3d(outer_padding=0.0):
    plane = bpy.data.objects.get("Black & white")
    pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    top_z = max(p.z for p in pts)
    top_pts = [p for p in pts if abs(p.z - top_z) < 1e-4]
    seen = set(); unique = []
    for p in top_pts:
        key = (round(p.x, 6), round(p.y, 6))
        if key not in seen:
            seen.add(key); unique.append(p)
    # Expand the 4 corners outward by `outer_padding` (fraction of half-board)
    # so the rectified output includes a border of the surrounding scene
    # (frame, table, HDRI background) instead of cropping flush to the board.
    if outer_padding > 0 and len(unique) == 4:
        cx = sum(p.x for p in unique) / 4.0
        cy = sum(p.y for p in unique) / 4.0
        unique = [Vector((cx + (p.x - cx) * (1 + outer_padding),
                          cy + (p.y - cy) * (1 + outer_padding),
                          p.z)) for p in unique]
    return unique


def get_max_piece_z():
    exclude = {"Black & white", "Outer frame"}
    max_z = None
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.hide_render or obj.name in exclude:
            continue
        pts = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        z = max(p.z for p in pts)
        if max_z is None or z > max_z:
            max_z = z
    return max_z


def get_piece_aware_source_corners(board_corners_3d, cam, scene, margin=1.2):
    board_z = max(p.z for p in board_corners_3d)
    piece_top_z = get_max_piece_z()
    if piece_top_z is None or piece_top_z <= board_z:
        return project_corners_to_image(board_corners_3d, cam, scene)
    lift_z = board_z + margin * (piece_top_z - board_z)
    cam_pos = cam.location
    distances = [(cam_pos - p).length for p in board_corners_3d]
    sorted_by_dist = sorted(range(len(board_corners_3d)), key=lambda i: distances[i])
    far_indices = set(sorted_by_dist[-2:])
    mixed_3d = []
    for i, p in enumerate(board_corners_3d):
        if i in far_indices:
            mixed_3d.append(Vector((p.x, p.y, lift_z)))
        else:
            mixed_3d.append(p)
    return project_corners_to_image(mixed_3d, cam, scene)


def project_corners_to_image(corners_3d, cam, scene):
    W = scene.render.resolution_x; H = scene.render.resolution_y
    pts_2d = []
    for pt in corners_3d:
        co = world_to_camera_view(scene, cam, pt)
        pts_2d.append((co.x * W, (1.0 - co.y) * H))
    return pts_2d


def order_corners_tl_tr_br_bl(pts):
    pts_sorted_y = sorted(pts, key=lambda p: p[1])
    top = sorted(pts_sorted_y[:2], key=lambda p: p[0])
    bottom = sorted(pts_sorted_y[2:], key=lambda p: p[0], reverse=True)
    return [top[0], top[1], bottom[0], bottom[1]]


def compute_pil_coeffs(src_corners, dst_corners):
    matrix = []
    for (x, y), (X, Y) in zip(dst_corners, src_corners):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([v for pair in src_corners for v in pair], dtype=np.float64)
    return tuple(np.linalg.solve(A, B))


def rectify_image(image_path, corners_3d, cam, scene, out_size, out_path, piece_margin=1.2):
    if piece_margin > 0:
        pts_2d = get_piece_aware_source_corners(corners_3d, cam, scene, margin=piece_margin)
    else:
        pts_2d = project_corners_to_image(corners_3d, cam, scene)
    src = order_corners_tl_tr_br_bl(pts_2d)
    dst = [(0, 0), (out_size, 0), (out_size, out_size), (0, out_size)]
    coeffs = compute_pil_coeffs(src, dst)
    img = Image.open(image_path)
    warped = img.transform((out_size, out_size), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    warped.save(out_path)
    return src


def setup_hdri_world(hdri_path, rotation_deg, strength):
    """HDRI as world environment, layered on top of the sun (kept for fill)."""
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


def render_all_views(board_info, view='black'):
    center = board_info['center']
    scale_factor = board_info['scale_factor']
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)

    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 3.0

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.use_denoising = True

    # ---- Cycles compute device ----
    if FORCE_CPU:
        scene.cycles.device = 'CPU'
        device_msg = "CPU (forced)"
    else:
        device_msg = "CPU (no GPU backend usable)"
        try:
            cprefs = bpy.context.preferences.addons['cycles'].preferences
            for dtype in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'):
                try:
                    cprefs.compute_device_type = dtype
                except TypeError:
                    continue
                cprefs.get_devices()
                gpu_devs = [d for d in cprefs.devices if d.type != 'CPU']
                if not gpu_devs:
                    continue
                for d in cprefs.devices:
                    d.use = (d.type != 'CPU')
                scene.cycles.device = 'GPU'
                device_msg = f"GPU [{dtype}] - {', '.join(d.name for d in gpu_devs)}"
                break
            else:
                scene.cycles.device = 'CPU'
        except Exception as e:
            scene.cycles.device = 'CPU'
            device_msg = f"CPU (probe error: {e})"
    print(f"  Cycles device: {device_msg}")

    if HDRI_PATH:
        setup_hdri_world(HDRI_PATH, HDRI_ROTATION_DEG, HDRI_STRENGTH)

    board_corners_3d = get_board_corners_3d(outer_padding=OUTER_PADDING)
    camera_z = center.z + camera_height

    if view == 'white':
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east", False),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west", False),
        ]
        z_rotation_offset = math.radians(180)
    else:
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x - horizontal_offset, center.y, camera_z), "2_west", False),
            ((center.x + horizontal_offset, center.y, camera_z), "3_east", False),
        ]
        z_rotation_offset = 0

    for location, name, point_at_center in views:
        print(f"\nRendering: {name}")
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object
        if point_at_center:
            direction = center - cam.location
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        else:
            cam.rotation_euler = (0, 0, 0)
        cam.rotation_euler.z += z_rotation_offset
        cam.data.lens = LENS
        bpy.context.scene.camera = cam

        raw_path = f"{OUT_DIR}/{name}_raw_tmp.png"
        bpy.context.scene.render.filepath = raw_path
        bpy.ops.render.render(write_still=True)

        try:
            rectified_path = f"{OUT_DIR}/{name}_rectified.png"
            rectify_image(
                raw_path, board_corners_3d, cam, scene, RECTIFIED_SIZE, rectified_path,
                piece_margin=PIECE_MARGIN,
            )
            print(f"  ✓ Saved rectified: {name}_rectified.png ({RECTIFIED_SIZE}x{RECTIFIED_SIZE})")
            try:
                os.remove(raw_path)
            except OSError:
                pass
        except Exception as e:
            print(f"  ✗ Rectification failed for {name}: {e}")

        bpy.data.objects.remove(cam, do_unlink=True)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    parser.add_argument('--resolution', type=int, default=800)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--rectified-size', type=int, default=512)
    parser.add_argument('--piece-margin', type=float, default=0.1)
    parser.add_argument('--outer-padding', type=float, default=0.0,
                        help='Expand rectification quad outward by this '
                             'fraction of half-board size, so the output '
                             'shows a border of scene around the board.')
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--hdri', type=str, default=None,
                        help='Path to HDRI .exr/.hdr (relative to .blend dir or absolute).')
    parser.add_argument('--hdri-rotation', type=float, default=0.0)
    parser.add_argument('--hdri-strength', type=float, default=1.0)
    args = parser.parse_args(argv)

    global RES, SAMPLES, RECTIFIED_SIZE, PIECE_MARGIN, OUTER_PADDING, FORCE_CPU, OUT_DIR
    global HDRI_PATH, HDRI_ROTATION_DEG, HDRI_STRENGTH
    RES = args.resolution
    SAMPLES = args.samples
    RECTIFIED_SIZE = args.rectified_size
    PIECE_MARGIN = args.piece_margin
    OUTER_PADDING = args.outer_padding
    FORCE_CPU = args.cpu
    HDRI_ROTATION_DEG = args.hdri_rotation
    HDRI_STRENGTH = args.hdri_strength

    blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
    OUT_DIR = os.path.join(blend_dir, "renders")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    if args.hdri:
        HDRI_PATH = args.hdri if os.path.isabs(args.hdri) else os.path.join(blend_dir, args.hdri)
        if not os.path.exists(HDRI_PATH):
            print(f"ERROR: HDRI not found: {HDRI_PATH}", file=sys.stderr)
            sys.exit(3)

    board_info = get_board_info()

    # Fix board orientation (same as v1)
    plane = bpy.data.objects.get("Black & white")
    if plane:
        frame = bpy.data.objects.get("Outer frame")
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        f_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        f_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (f_min + f_max) / 2
        original_pos = plane.location.copy()
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        plane.location = center + (rot_matrix @ offset)

    starting_pieces = detect_starting_positions(board_info)
    result = apply_fen(args.fen, starting_pieces, board_info)
    if not result['success']:
        print(f"FATAL: cannot satisfy FEN — aborting render.", file=sys.stderr)
        sys.exit(2)

    render_all_views(board_info, view=args.view)


if __name__ == "__main__":
    main()
