"""Configuration loader for cube operator library.

Reads YAML config files and returns typed config objects used by
benchmarks, models, and compiler passes to query hardware-specific thresholds.
"""

import yaml
import os
from typing import Dict, Any

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE: Dict[str, Dict[str, Any]] = {}


def load_config(name: str) -> Dict[str, Any]:
    """Load a named config file (without .yaml extension).

    Configs are cached after first load so repeated calls are free.

    Example:
        cfg = load_config("rtx4060")
        threshold = cfg["permutation"]["prefer_dense_threshold"]
    """
    if name not in _CACHE:
        path = os.path.join(_CONFIG_DIR, f"{name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[name] = yaml.safe_load(f)
    return _CACHE[name]


def get_active_config() -> Dict[str, Any]:
    """Auto-detect GPU and return the matching config.

    Falls back to rtx4060 if detection fails.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "4060" in name:
                return load_config("rtx4060")
    except Exception:
        pass
    return load_config("rtx4060")
