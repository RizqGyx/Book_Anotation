"""Photo-realistic augmentation with construction-guaranteed label correctness.

Labels are carried as per-instance binary masks. Every geometric transform is
applied with the identical operator to the image and to each mask, and polygons
are re-derived from the transformed masks afterwards, so image and label cannot
drift apart. See docs/DESIGN_NOTES.md section 2.
"""

from .config import AugConfig, LIGHTING_NAMES, LIGHTING_PROFILES, PRESETS
from .geometric import (apply_geometric, apply_page_curve, apply_perspective,
                        apply_rotation, masks_area, surface_fill)
from .photometric import (apply_lighting, apply_photometric, apply_reflection,
                          apply_sensor, apply_shadow)
from .polygons import mask_to_polygon, polygon_bbox_xyxy, polygon_is_valid

__all__ = [
    "AugConfig", "LIGHTING_NAMES", "LIGHTING_PROFILES", "PRESETS",
    "apply_geometric", "apply_page_curve", "apply_perspective", "apply_rotation",
    "masks_area", "surface_fill",
    "apply_lighting", "apply_photometric", "apply_reflection", "apply_sensor",
    "apply_shadow",
    "mask_to_polygon", "polygon_bbox_xyxy", "polygon_is_valid",
]
