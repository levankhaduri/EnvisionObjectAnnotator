"""Guard rails for MODEL_CATALOG entries.

These tests exist because of a real install bug we hit on a Linux/Singularity
deployment: `processing.py` was passing the absolute filesystem path of each
YAML to `build_sam2_video_predictor`, which routes that argument to Hydra's
`compose(config_name=...)`. Hydra strips a leading "/" and then fails with
"Cannot find primary config 'opt/EnvisionObjectAnnotator/configs/...'".

The fix splits each catalog entry into:
  * config_name  -- relative Hydra key string (passed to SAM2)
  * config_path  -- absolute Path used only for the existence check

If anyone reintroduces an absolute path into config_name, the first test here
fails immediately. The Hydra compose() test confirms each name actually
resolves against a configs/ search dir in the shape SAM2 expects.
"""

from pathlib import Path

import pytest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from app.processing import MODEL_CATALOG, REPO_DIR


_IDS = [entry["key"] for entry in MODEL_CATALOG]


@pytest.mark.parametrize("entry", MODEL_CATALOG, ids=_IDS)
def test_config_name_is_relative_hydra_key(entry):
    name = entry["config_name"]
    assert isinstance(name, str), f"config_name must be a str, got {type(name)!r}"
    assert not name.startswith("/"), (
        f"config_name must not start with '/': Hydra strips the leading slash "
        f"and the lookup fails. Got: {name!r}"
    )
    # Windows absolute path (e.g. "C:\\..."). Same hazard once stringified
    # through Path on a non-Windows host.
    assert not (len(name) >= 2 and name[1] == ":"), (
        f"config_name must not be a Windows absolute path. Got: {name!r}"
    )
    assert "\\" not in name, (
        f"config_name must use forward slashes (Hydra is path-separator "
        f"sensitive). Got: {name!r}"
    )
    assert name.endswith(".yaml"), f"config_name should end with .yaml. Got: {name!r}"


@pytest.mark.parametrize("entry", MODEL_CATALOG, ids=_IDS)
def test_config_path_points_at_real_file(entry):
    path = entry["config_path"]
    assert isinstance(path, Path)
    assert path.is_file(), (
        f"config_path for {entry['key']!r} does not exist: {path}. "
        f"list_available_models() uses this to detect broken installs."
    )


@pytest.mark.parametrize("entry", MODEL_CATALOG, ids=_IDS)
def test_hydra_compose_resolves_config_name(entry):
    """Replay what build_sam2_video_predictor does internally.

    SAM2 registers its installed package configs/ dir as the Hydra search
    path. We mirror that by registering REPO_DIR itself, so a config_name of
    "configs/sam2.1/foo.yaml" resolves to REPO_DIR/configs/sam2.1/foo.yaml.
    If the name is shaped wrong (absolute path, missing prefix, wrong
    separator) compose() raises MissingConfigException and this test fails
    with a message that points right back here.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(REPO_DIR), version_base=None):
            cfg = compose(config_name=entry["config_name"])
    finally:
        GlobalHydra.instance().clear()
    assert len(cfg) > 0, f"composed config for {entry['key']} is empty"
