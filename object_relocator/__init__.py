"""Vision-based object relocation system for SO-101.

Detects objects on the desk via background subtraction and uses calibrated
joint-space waypoints to push objects to random new positions.
"""

from .relocator import ObjectRelocator
from .detection import detect_object

__all__ = ["ObjectRelocator", "detect_object"]
