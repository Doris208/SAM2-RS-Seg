"""Small explicit loss configuration contract, without importing PyTorch."""

import math


def validate_loss_config(config, ignore_index):
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise ValueError("Loss ignore_index must be an integer")
    if (not isinstance(config, dict) or not isinstance(config.get("ignore_index"), int)
            or isinstance(config.get("ignore_index"), bool) or config.get("ignore_index") != ignore_index):
        raise ValueError("Loss ignore_index must match the dataset ignore_index")
    name = config.get("name")
    allowed = {
        "CrossEntropyLoss": {"name", "ignore_index"},
        "DiceLoss": {"name", "ignore_index", "smooth", "eps", "absent_class"},
    }
    if name not in allowed:
        raise ValueError(f"Unknown loss {name!r}; choose CrossEntropyLoss or DiceLoss")
    if set(config) - allowed[name]:
        raise ValueError(f"Unknown {name} options: {sorted(set(config) - allowed[name])}")
    if name == "DiceLoss":
        for key, default in (("smooth", 0.0), ("eps", 1e-7)):
            value = config.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (key == "eps" and value == 0)):
                raise ValueError(f"DiceLoss {key} must be finite and {'positive' if key == 'eps' else 'nonnegative'}")
        if config.get("absent_class", "exclude") not in ("exclude", "include"):
            raise ValueError("DiceLoss absent_class must be exclude or include")
