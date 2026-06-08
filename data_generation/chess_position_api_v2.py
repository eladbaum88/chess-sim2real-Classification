"""
Chess FEN Parser - Auto-detect Starting Positions

Strategy:
1. Analyze current piece positions to determine which piece is on which square
2. Parse target FEN
3. Move pieces from their detected starting square to target square

Usage:
    blender chess-set.blend --background --python chess_position_api_v2.py -- --fen "r4rk1/1p1bqppp/n1p1pn2/p2pN3/2PP4/P1N3P1/1P1QPPBP/R4RK1" --view black
"""

import bpy
import math
import os
from mathutils import Vector
import sys
import argparse
# Rotate the offset vector
from mathutils import Matrix

# Blender's embedded Python disables user site-packages by default
# (site.ENABLE_USER_SITE is False). We install third-party packages
# (Pillow, chess, python-chess, zstandard) with `pip install --user`
# to avoid needing admin rights for Program Files. Add the user
# site-packages to sys.path so those imports work under blender.exe.
import site
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

from bpy_extras.object_utils import world_to_camera_view
import numpy as np
from PIL import Image
# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
RECTIFIED_SIZE = 512
PIECE_MARGIN = 0.6
FORCE_CPU = False
OUT_DIR = "//renders"

def get_board_info():
    """Get board dimensions"""
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
    
    scale_factor = board_size / REAL_BOARD_SIZE
    
    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'scale_factor': scale_factor,
    }

def position_to_square(pos, board_info):
    """Convert 3D position to chess square (e.g., 'e2')"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    
    # File (a-h) from X coordinate - scene is flipped
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    # Rank (1-8) from Y coordinate
    # Higher Y = lower rank (reversed)
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1
    
    return f"{file_letter}{rank_number}"

def square_to_world_xy(square, board_info):
    """Inverse of `position_to_square`: square name 'e2' → (world_x, world_y)
    of that square's center. Z is intentionally not returned — pieces keep
    whatever Z they currently have (board surface).
    """
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    file_idx = ord(square[0]) - ord('a')      # 'a'=0 .. 'h'=7
    rank_idx = int(square[1]) - 1             # '1'=0 .. '8'=7

    # Forward did:  file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    # so:           pos.x  = plane_min.x + (7 - file_idx + 0.5) * square_size
    pos_x = plane_min.x + (7 - file_idx + 0.5) * square_size
    # Forward did:  rank_idx = int((plane_max.y - pos.y) / square_size)
    # so:           pos.y  = plane_max.y - (rank_idx + 0.5) * square_size
    pos_y = plane_max.y - (rank_idx + 0.5) * square_size
    return pos_x, pos_y


def detect_starting_positions(board_info):
    """
    Detect which piece is on which square currently
    Returns: {piece_name: {'square': 'e2', 'piece_type': 'P'}}
    """
    print("\n" + "="*70)
    print("DETECTING STARTING POSITIONS")
    print("="*70)
    
    pieces = {}
    
    # Get all chess piece objects
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        
        name = obj.name
        
        # Determine piece type from name
        piece_type = None
        
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'  # White pawn
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'  # Black pawn
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
            square = position_to_square(obj.location, board_info)
            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'start_pos': obj.location.copy()
            }
            print(f"  {name:20s} → {square:4s} ({piece_type})")
    
    print(f"\n✓ Detected {len(pieces)} pieces")
    return pieces

def parse_fen(fen):
    """Parse FEN into dict {square: piece_char}"""
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
                file_letter = chr(ord('a') + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1
    
    return position

def apply_fen(fen, starting_pieces, board_info):
    """Apply a FEN by placing pieces at absolute world positions.

    Idempotent: depends only on `starting_pieces[name]['piece_type']`, NOT
    on the pieces' current locations or visibility. Safe to call repeatedly
    in a loop without any "reset scene state" step in between.

    Returns a dict:
        {
          'success': bool,
          'pieces_placed': int,
          'pieces_requested': int,
          'unfulfilled': [(square, piece_type), ...],   # what we couldn't place
        }
    """
    print("\n" + "=" * 70)
    print("APPLYING FEN")
    print("=" * 70)
    print(f"FEN: {fen}\n")

    target_position = parse_fen(fen)

    # Pool of available pieces grouped by type, e.g. {'P': ['A(texture)', 'B', ...]}.
    # Pieces are popped as they get assigned — no need to track a 'used' set.
    pool = {}
    for piece_name, info in starting_pieces.items():
        pool.setdefault(info['piece_type'], []).append(piece_name)

    used_names = set()
    unfulfilled = []

    for target_square, piece_type in target_position.items():
        if not pool.get(piece_type):
            print(f"  ✗ No '{piece_type}' available for {target_square}")
            unfulfilled.append((target_square, piece_type))
            continue

        piece_name = pool[piece_type].pop()
        obj = bpy.data.objects.get(piece_name)
        if obj is None:
            print(f"  ✗ Piece object '{piece_name}' missing in scene")
            unfulfilled.append((target_square, piece_type))
            continue

        # Place at the absolute world center of the target square.
        # Z is preserved — pieces sit on the board surface and we never
        # change that height.
        target_x, target_y = square_to_world_xy(target_square, board_info)
        obj.location.x = target_x
        obj.location.y = target_y

        obj.hide_render = False
        obj.hide_viewport = False
        used_names.add(piece_name)
        print(f"  Placed {piece_name:20s} at {target_square}  ({piece_type})")

    # Hide every piece NOT used in this FEN (captured / off-board).
    for piece_name in starting_pieces.keys():
        if piece_name in used_names:
            continue
        obj = bpy.data.objects.get(piece_name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_viewport = True

    success = (len(unfulfilled) == 0)
    requested = len(target_position)
    placed = len(used_names)
    if success:
        print(f"\n✓ Position set ({placed}/{requested} pieces placed, "
              f"{len(starting_pieces) - placed} hidden)")
    else:
        print(f"\n✗ FEN partially applied: {placed}/{requested} placed, "
              f"{len(unfulfilled)} unfulfilled: {unfulfilled}")

    return {
        'success': success,
        'pieces_placed': placed,
        'pieces_requested': requested,
        'unfulfilled': unfulfilled,
    }

def get_board_corners_3d():
    """Get the 4 top-surface corners of the playable board area in world coordinates.

    Uses the 'Black & white' plane (the 8x8 checkerboard), not the outer frame,
    so the rectified image contains only the playable squares.
    """
    plane = bpy.data.objects.get("Black & white")
    pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    top_z = max(p.z for p in pts)
    # Keep the 4 corners on the top surface (tolerate tiny float error)
    top_pts = [p for p in pts if abs(p.z - top_z) < 1e-4]
    # Deduplicate on (x, y) in case bound_box has 8 points with identical Z per corner
    seen = set()
    unique = []
    for p in top_pts:
        key = (round(p.x, 6), round(p.y, 6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def get_max_piece_z():
    """Find max Z (world) among visible chess pieces. Excludes board & frame."""
    exclude = {"Black & white", "Outer frame"}
    max_z = None
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        if obj.name in exclude:
            continue
        pts = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        obj_max_z = max(p.z for p in pts)
        if max_z is None or obj_max_z > max_z:
            max_z = obj_max_z
    return max_z


def get_piece_aware_source_corners(board_corners_3d, cam, scene, margin=1.2):
    """Return 4 image-space corners that include piece volume above the board.

    Strategy: only the 2 corners *furthest from the camera* get lifted to
    piece-top Z. The 2 near-camera corners stay at board level. This keeps
    the board tight on the near side (no frame leakage) while leaving room
    for piece heads on the far side (where pieces otherwise get cropped).
    """
    board_z = max(p.z for p in board_corners_3d)
    piece_top_z = get_max_piece_z()
    if piece_top_z is None or piece_top_z <= board_z:
        return project_corners_to_image(board_corners_3d, cam, scene)

    lift_z = board_z + margin * (piece_top_z - board_z)

    # Classify corners as near/far by distance from the camera
    cam_pos = cam.location
    distances = [(cam_pos - p).length for p in board_corners_3d]
    # Indices of the 2 farthest corners
    sorted_by_dist = sorted(range(len(board_corners_3d)), key=lambda i: distances[i])
    far_indices = set(sorted_by_dist[-2:])

    # Build mixed 3D quad: near corners at board Z, far corners at lift_z
    mixed_3d = []
    for i, p in enumerate(board_corners_3d):
        if i in far_indices:
            mixed_3d.append(Vector((p.x, p.y, lift_z)))
        else:
            mixed_3d.append(p)

    return project_corners_to_image(mixed_3d, cam, scene)


def project_corners_to_image(corners_3d, cam, scene):
    """Project 3D world points to 2D pixel coordinates (origin top-left)."""
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    pts_2d = []
    for pt in corners_3d:
        co = world_to_camera_view(scene, cam, pt)
        # co.x, co.y are normalized [0..1]; co.z is camera-space depth
        u = co.x * W
        v = (1.0 - co.y) * H  # flip y: image y increases downward
        pts_2d.append((u, v))
    return pts_2d


def order_corners_tl_tr_br_bl(pts):
    """Order 4 image-space points as [TL, TR, BR, BL]."""
    pts_sorted_y = sorted(pts, key=lambda p: p[1])
    top = sorted(pts_sorted_y[:2], key=lambda p: p[0])          # ascending x
    bottom = sorted(pts_sorted_y[2:], key=lambda p: p[0], reverse=True)  # descending x
    return [top[0], top[1], bottom[0], bottom[1]]  # TL, TR, BR, BL


def compute_pil_coeffs(src_corners, dst_corners):
    """Compute PIL PERSPECTIVE coefficients (8 values) mapping dst -> src.

    PIL resamples by inverse mapping, so it needs dst->src.
    Solves the 8-parameter homography for (TL, TR, BR, BL) correspondences.
    """
    matrix = []
    for (x, y), (X, Y) in zip(dst_corners, src_corners):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([v for pair in src_corners for v in pair], dtype=np.float64)
    return tuple(np.linalg.solve(A, B))


def rectify_image(image_path, corners_3d, cam, scene, out_size, out_path, piece_margin=1.2):
    """Warp `image_path` so the board fills the full output image.

    The output is `out_size` x `out_size` pixels. `piece_margin` lifts the
    source quad upward (in 3D) to include piece volume — prevents pawn heads
    from being cropped on the far edges. Set to 0.0 to use flat board corners.
    """
    if piece_margin > 0:
        pts_2d_unordered = get_piece_aware_source_corners(
            corners_3d, cam, scene, margin=piece_margin
        )
    else:
        pts_2d_unordered = project_corners_to_image(corners_3d, cam, scene)
    src = order_corners_tl_tr_br_bl(pts_2d_unordered)
    dst = [(0, 0), (out_size, 0), (out_size, out_size), (0, out_size)]
    coeffs = compute_pil_coeffs(src, dst)
    img = Image.open(image_path)
    warped = img.transform(
        (out_size, out_size), Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )
    warped.save(out_path)
    return src  # for debugging/logging


def render_all_views(board_info, view='black'):
    """Render views from white or black perspective"""
    print("\n" + "="*70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("="*70)
    
    center = board_info['center']
    scale_factor = board_info['scale_factor']
    
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    
    # Clean cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Setup lighting
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 3.0
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.use_denoising = True
    
    # ---- Cycles compute device: probe for a real GPU; clearly report what we got ----
    # Without this, `scene.cycles.device = 'GPU'` silently falls back to CPU when no
    # GPU compute backend is configured in Blender preferences. We try each backend
    # in order and only commit to GPU if at least one non-CPU device is discovered.
    if FORCE_CPU:
        scene.cycles.device = 'CPU'
        cycles_device_msg = "CPU (forced via --cpu)"
    else:
        cycles_device_msg = "CPU (no GPU backend usable)"
        try:
            cprefs = bpy.context.preferences.addons['cycles'].preferences
            # Order matters. CUDA before OPTIX: OPTIX only helps on RTX (RT cores)
            # and requires recent NVIDIA drivers, otherwise it silently errors at
            # render time even though it appears available at probe time.
            for dtype in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'):
                try:
                    cprefs.compute_device_type = dtype
                except TypeError:
                    continue  # backend not compiled into this Blender build
                cprefs.get_devices()
                gpu_devs = [d for d in cprefs.devices if d.type != 'CPU']
                if not gpu_devs:
                    continue
                for d in cprefs.devices:
                    d.use = (d.type != 'CPU')   # enable GPU(s), disable CPU device
                scene.cycles.device = 'GPU'
                names = ', '.join(d.name for d in gpu_devs)
                cycles_device_msg = f"GPU [{dtype}] - {names}"
                break
            else:
                scene.cycles.device = 'CPU'
        except Exception as e:
            scene.cycles.device = 'CPU'
            cycles_device_msg = f"CPU (probe error: {e})"
    print(f"  Cycles device: {cycles_device_msg}")
    
    # Board corners in 3D (computed once, used to rectify every view)
    board_corners_3d = get_board_corners_3d()

    # Camera positions
    camera_z = center.z + camera_height
    
    # Flip camera positions for white view (180 degree rotation)
    if view == 'white':
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east", False),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west", False),
        ]
        z_rotation_offset = math.radians(180)
    else:  # black view (default)
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
        
        # Apply rotation for white/black view
        cam.rotation_euler.z += z_rotation_offset
        
        cam.data.lens = LENS

        bpy.context.scene.camera = cam
        # Raw render goes to a temp file; we only keep the rectified output.
        raw_path = f"{OUT_DIR}/{name}_raw_tmp.png"
        bpy.context.scene.render.filepath = raw_path
        bpy.ops.render.render(write_still=True)

        # Rectify: warp so the board fills a RECTIFIED_SIZE x RECTIFIED_SIZE image.
        # Must happen BEFORE we remove the camera, since projection needs it.
        try:
            rectified_path = f"{OUT_DIR}/{name}_rectified.png"
            src_corners = rectify_image(
                raw_path, board_corners_3d, cam, scene, RECTIFIED_SIZE, rectified_path,
                piece_margin=PIECE_MARGIN,
            )
            print(f"  ✓ Saved rectified:  {name}_rectified.png  ({RECTIFIED_SIZE}x{RECTIFIED_SIZE})")
            print(f"    board corners (TL,TR,BR,BL): "
                  f"{[(round(p[0],1), round(p[1],1)) for p in src_corners]}")
            # Rectified saved OK — delete the temp raw render.
            try:
                os.remove(raw_path)
            except OSError:
                pass
        except Exception as e:
            # Keep the raw file for debugging if rectification fails.
            print(f"  ✗ Rectification failed for {name}: {e}")
            print(f"    (raw render kept at {raw_path} for debugging)")

        bpy.data.objects.remove(cam, do_unlink=True)
    
    print("\n✓ Rendering complete")

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    parser.add_argument('--resolution', type=int, default=800,
                        help='Render resolution in pixels (square). Higher = more detail but slower.')
    parser.add_argument('--samples', type=int, default=128,
                        help='Cycles render quality (higher = less noise but slower).')
    parser.add_argument('--rectified-size', type=int, default=512,
                        help='Output size of the rectified (board-fills-frame) image.')
    parser.add_argument('--piece-margin', type=float, default=0.1,
                        help='Lift factor for source quad to include piece heights '
                             '(1.0 = exact piece top, 1.2 = 20%% extra headroom, 0 = off).')
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Render from white or black perspective')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU rendering (skip GPU probe). Use this '
                             'when CUDA/OPTIX kernel load fails due to old drivers.')

    args = parser.parse_args(argv)

    global RES, SAMPLES, RECTIFIED_SIZE, PIECE_MARGIN, FORCE_CPU, OUT_DIR
    RES = args.resolution
    SAMPLES = args.samples
    RECTIFIED_SIZE = args.rectified_size
    PIECE_MARGIN = args.piece_margin
    FORCE_CPU = args.cpu

    # Anchor OUT_DIR to the .blend file's directory so Blender and Python
    # agree on the path. (Blender resolves "./" inconsistently — e.g. it
    # ended up writing to C:\renders\ instead of <project>\renders\.)
    blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
    OUT_DIR = os.path.join(blend_dir, "renders")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    
    # Get board info
    board_info = get_board_info()
    # Fix inverted board - rotate checkerboard 90 degrees around board center
    plane = bpy.data.objects.get("Black & white")
    if plane:
        # Get board center first (before rotating)
        frame = bpy.data.objects.get("Outer frame")
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (frame_min + frame_max) / 2
        
        # Store original position
        original_pos = plane.location.copy()
        
        # Move to center, rotate, move back
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        rotated_offset = rot_matrix @ offset
        
        plane.location = center + rotated_offset
    # Detect starting positions
    starting_pieces = detect_starting_positions(board_info)
    
    # Apply FEN
    result = apply_fen(args.fen, starting_pieces, board_info)
    if not result['success']:
        # Don't render a corrupted board — the outer driver
        # (build_dataset.py) will see the non-zero exit and skip this FEN.
        print(f"\nFATAL: cannot satisfy FEN — aborting render.", file=sys.stderr)
        sys.exit(2)

    # Render
    render_all_views(board_info, view=args.view)

if __name__ == "__main__":
    main()
