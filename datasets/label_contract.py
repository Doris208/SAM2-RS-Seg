"""Explicit label contracts; no automatic remapping or image IO."""

from numbers import Integral

IGNORE_INDEX = 255
ISPRS_CLASSES = (
    "impervious_surface", "building", "low_vegetation", "tree", "car", "clutter"
)
ISPRS_PALETTE = (
    (255, 255, 255), (0, 0, 255), (0, 255, 255),
    (0, 255, 0), (255, 255, 0), (255, 0, 0),
)
LOVEDA_CLASSES = (
    "background", "building", "road", "water", "barren", "forest", "agricultural"
)


def validate_ids(values, num_classes, ignore_index=IGNORE_INDEX):
    """Reject invalid IDs, including floats, rather than silently truncating."""
    if num_classes <= 0 or ignore_index in range(num_classes):
        raise ValueError("Invalid class count or overlapping ignore index")
    for value in values:
        if (not isinstance(value, Integral) or isinstance(value, bool)
                or (value != ignore_index and not 0 <= value < num_classes)):
            raise ValueError(f"Invalid class ID: {value!r}")


def isprs_rgb_to_id(rgb):
    """Map one exact official RGB label; unknown colors are an error."""
    try:
        return ISPRS_PALETTE.index(tuple(rgb))
    except ValueError as exc:
        raise ValueError(f"Unknown ISPRS color: {rgb!r}") from exc


def loveda2021_raw_to_train_id(raw_id):
    """Explicit raw 0 -> ignore, raw 1..7 -> train 0..6. Apply exactly once."""
    if not isinstance(raw_id, Integral) or isinstance(raw_id, bool):
        raise ValueError(f"Invalid LoveDA raw ID: {raw_id!r}")
    if raw_id == 0:
        return IGNORE_INDEX
    if 1 <= raw_id <= 7:
        return raw_id - 1
    raise ValueError(f"Invalid LoveDA raw ID: {raw_id!r}")
