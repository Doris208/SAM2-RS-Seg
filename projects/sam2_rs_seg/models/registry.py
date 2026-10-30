"""Explicit small factories; only the existing SAM2 model/head are registered."""

from copy import deepcopy
import inspect

from torch import nn

from .segmentor import SAM2SemanticSegmentor
from .decode_heads.semantic_head import SemanticSegmentationHead

MODELS = {"SAM2SemanticSegmentor": (SAM2SemanticSegmentor, 3, 32)}
HEADS = {"SemanticSegmentationHead": SemanticSegmentationHead}


def register_model(name, constructor, *, input_channels, size_divisor=1):
    if name in MODELS or not callable(constructor):
        raise ValueError(f"Duplicate model or invalid constructor: {name}")
    if input_channels <= 0 or size_divisor <= 0:
        raise ValueError("Model input_channels and size_divisor must be positive")
    MODELS[name] = (constructor, input_channels, size_divisor)


def register_head(name, constructor):
    if name in HEADS or not callable(constructor):
        raise ValueError(f"Duplicate head or invalid constructor: {name}")
    HEADS[name] = constructor


def _construct(constructor, params, kind):
    try:
        inspect.signature(constructor).bind(**params)
    except TypeError as exc:
        raise ValueError(f"Invalid {kind} parameters: {exc}") from exc
    result = constructor(**params)
    if not isinstance(result, nn.Module):
        raise TypeError(f"{kind} constructor must return nn.Module")
    return result


def build_model(model_config: dict, data_config: dict, *, resume: bool = False) -> nn.Module:
    config = deepcopy(model_config)
    if set(config) - {"name", "params", "decode_head"}:
        raise ValueError("Unknown model configuration keys")
    name = config["name"]
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODELS)}")
    constructor, channels, divisor = MODELS[name]
    if data_config["input_channels"] != channels:
        raise ValueError(f"{name} expects {channels} input channels")
    if any(size < divisor or size % divisor for size in data_config["image_size"]):
        raise ValueError(f"{name} requires image dimensions divisible by {divisor}")
    params = dict(config.get("params", {}))
    classes = data_config["num_classes"]
    if params.get("num_classes", classes) != classes:
        raise ValueError("Model num_classes differs from data num_classes")
    params["num_classes"] = classes
    # Full Lightning restoration is explicit and must not depend on the old
    # initialization file still existing. Trainer subsequently loads all states.
    if resume and name == "SAM2SemanticSegmentor":
        params.update(pretrained=False, checkpoint_path=None)
    head_config = config.get("decode_head")
    if head_config is not None:
        if name != "SAM2SemanticSegmentor" or set(head_config) - {"name", "params"}:
            raise ValueError("Explicit decode_head currently targets the SAM2 segmentor")
        head_name = head_config["name"]
        if head_name not in HEADS:
            raise ValueError(f"Unknown decode head: {head_name}")
        head_params = dict(head_config.get("params", {}))
        if head_params.get("num_classes", classes) != classes:
            raise ValueError("Decode head num_classes differs from data")
        if head_params.get("in_channels", 256) != 256:
            raise ValueError("SAM2.1 Tiny supplies 256 channels per FPN level")
        head_params.update(num_classes=classes, in_channels=256)
        head_params.setdefault("head_channels", params.get("head_channels", 128))
        head = _construct(HEADS[head_name], head_params, "head")
    model = _construct(constructor, params, "model")
    if head_config is not None:
        if model.encoder.out_channels != head_params["in_channels"]:
            raise ValueError("Encoder and decode head channels are incompatible")
        model.decode_head = head.to(next(model.parameters()).device)
    return model
