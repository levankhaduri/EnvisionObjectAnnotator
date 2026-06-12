"""
Reproduce Yayun's Singularity error WITHOUT needing SAM2 installed.

The bug: backend/app/processing.py:179 passes an *absolute* path to
build_sam2_video_predictor(). Internally that path is handed to Hydra's
compose(), and Hydra strips the leading '/' from any config_name
because it interprets the argument as a Hydra config key, not a
filesystem path.

This script replays the same compose() call directly so we can see the
error in isolation. On Linux this is exactly what Yayun's container hits.
On Windows we simulate it by passing a POSIX-style absolute path.
"""

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO = Path(__file__).resolve().parent
CONFIGS_DIR = REPO / "configs"

# What backend/app/processing.py:179 effectively passes today.
# On a Linux/Singularity install at /opt/EnvisionObjectAnnotator/, this
# string is what str(REPO_DIR / "configs" / "sam2.1" / "...") produces.
LINUX_ABSOLUTE = "/opt/EnvisionObjectAnnotator/configs/sam2.1/sam2.1_hiera_b+.yaml"

# What the fix would pass instead -- a relative config key.
RELATIVE_KEY = "sam2.1/sam2.1_hiera_b+"  # no extension, Hydra-style


def try_compose(label: str, config_name: str) -> None:
    print(f"\n--- {label} ---")
    print(f"compose(config_name={config_name!r})")
    GlobalHydra.instance().clear()
    # Mimic SAM2's __init__: register the repo's configs/ as the search path.
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        try:
            cfg = compose(config_name=config_name)
            print("OK -- loaded config with keys:", list(cfg.keys())[:3], "...")
        except Exception as exc:
            # Print only the message, not a 50-line Hydra traceback.
            msg = str(exc).splitlines()[0]
            print(f"FAIL [{type(exc).__name__}]: {msg}")


def compose_against_repo(label: str, config_name: str) -> bool:
    """Walk the SAM2-shaped layout: search path = REPO root, key = configs/..."""
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(REPO), version_base=None):
            compose(config_name=config_name)
        print(f"  {label:24s}  {config_name:42s}  OK")
        return True
    except Exception as exc:
        msg = str(exc).splitlines()[0]
        print(f"  {label:24s}  {config_name:42s}  FAIL [{type(exc).__name__}]: {msg}")
        return False


def main() -> int:
    print(f"hydra search path = {CONFIGS_DIR}")
    print(f"configs/sam2.1 exists      = {(CONFIGS_DIR / 'sam2.1').is_dir()}")
    print(f"sam2.1_hiera_b+.yaml exists= {(CONFIGS_DIR / 'sam2.1' / 'sam2.1_hiera_b+.yaml').is_file()}")

    try_compose("BUG SHAPE (Linux absolute path)", LINUX_ABSOLUTE)
    try_compose("FIX SHAPE (relative key, configs/ as search dir)", RELATIVE_KEY)

    # Walk every config_name actually shipped in backend/app/processing.py
    # MODEL_CATALOG. If any reverts to an absolute path, this section fails
    # and exit code is non-zero. We inline the values here to avoid pulling
    # torch in via app.processing.
    catalog = [
        ("sam2.1_hiera_l", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        ("sam2.1_hiera_b+", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        ("sam2.1_hiera_s", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        ("sam2.1_hiera_t", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        ("edgetam", "configs/edgetam.yaml"),
    ]
    print("\n--- POST-FIX CATALOG WALK (each line must say OK) ---")
    failures = sum(0 if compose_against_repo(k, n) else 1 for k, n in catalog)
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {len(catalog) - failures}/{len(catalog)} catalog entries composed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
