"""Registry — v1 chess set + HDRI list + per-render randomization ranges.

We use ONLY the v1 chess set (chess-set.blend) as the piece source.
The other downloaded sets had material issues in Eevee Next that weren't
worth debugging individually. The .blend files for those sets stay in
chess-sets/ for reference but aren't loaded.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

# Single chess set: v1's chess-set.blend at the project root. The pipeline
# uses this file as the base scene (board + pieces + camera + lights),
# overriding piece materials at render time for clean opaque rendering.
BASE_BLEND = str(PROJECT_DIR / "chess-set.blend")

HDRIS = [
    str(PROJECT_DIR / "hdris" / "studio_small_03.exr"),
    str(PROJECT_DIR / "hdris" / "brown_photostudio_02.exr"),
    str(PROJECT_DIR / "hdris" / "lebombo.exr"),
    str(PROJECT_DIR / "hdris" / "entrance_hall.exr"),
]

# Per-render randomization ranges. Per user spec:
#   - Camera pitch ±10° around a base oblique of 40° from vertical
#   - Camera yaw ±5°
#   - Camera distance (height_mul) ±15% of base
#   - Sun light strength ±30% of base
#   - HDRI rotation 0-360° + strength multiplier 0.7-1.4
RANGES = {
    "hdri_rotation_deg":   (0.0, 360.0),
    # Hard cap at 0.3 per reviewer so the HDRI doesn't overwhelm the
    # warm cream/brown board tones.
    "hdri_strength":       (0.15, 0.30),
    # Slight brightness bump per user: 0.5-1.0 (was 0.3-0.7). Still under
    # the clipping threshold for the warm tones.
    "sun_energy":           (0.5, 1.0),
    "sun_temp_k":           (3500, 6500),
    "sun_azimuth_jitter":   (-20.0, 20.0),
    "sun_elevation_jitter": (-15.0, 15.0),
    # For east/west, cam_angle_deg controls the camera's horizontal offset
    # via offset = cam_h * tan(angle). Range tuned to put the offset at
    # 30-50% of the board width — gives the visible piece "lean" toward
    # the camera-side edge that matches the reference photos.
    # For overhead, this value is ignored (camera placed directly above).
    "cam_angle_deg":        (5.0, 8.0),
    "cam_yaw_deg":          (-5.0, 5.0),
    "cam_height_mul":       (0.85, 1.15),  # ±15%
    "cam_lens":             (26.0, 35.0),
    "cam_roll_deg":         (-3.0, 3.0),
}
