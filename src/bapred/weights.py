"""Packaged checkpoint presets for BA-Pred."""

from pathlib import Path


DEFAULT_MODEL = "random_seed0"

MODEL_PRESETS = {
    "legacy": "weight/BAPred.pth",
    "random_seed0": "weight/random/cutoff8_seed0_best.pth",
    "random_seed1": "weight/random/cutoff8_seed1_best.pth",
    "random_seed2": "weight/random/cutoff8_seed2_best.pth",
}


def resolve_packaged_weight(model: str = DEFAULT_MODEL) -> Path:
    try:
        relative_path = MODEL_PRESETS[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset: {model}") from exc
    return Path(__file__).resolve().parent / relative_path
