"""Backend abstraction for device-specific permutation operations.

Usage:
    from backends.nvidia import gather as nv_gather
    from backends.muxi import fused_perm_ln_gelu  # placeholder for MUXI

Or auto-select:
    from config import get_active_config
    cfg = get_active_config()
    if cfg["device"] == "cuda":
        from backends.nvidia import *
    else:
        from backends.muxi import *
"""
