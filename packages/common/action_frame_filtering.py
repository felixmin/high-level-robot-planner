"""Action-frame filtering API.

This module provides expressive naming for low-motion / low-action anchor
filtering while reusing the current core implementation.
"""

from __future__ import annotations

from common.anchor_filtering import AnchorFilterResult
from common.anchor_filtering import build_anchor_filter
from common.anchor_filtering import infer_motion_frame_gap
from common.anchor_filtering import normalize_filtering_config


def build_action_frame_filter(*args, **kwargs):
    return build_anchor_filter(*args, **kwargs)

__all__ = [
    "AnchorFilterResult",
    "build_action_frame_filter",
    "build_anchor_filter",
    "infer_motion_frame_gap",
    "normalize_filtering_config",
]
