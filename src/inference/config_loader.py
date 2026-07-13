"""Load and merge OmegaConf config fragments for the realtime pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def _project_root() -> Path:
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def _resolve_profile_name(runtime: DictConfig, key: str, default: str,
                          fragments_dir: Path) -> Path:
    """Resolve a profile fragment name from the runtime yaml.

    Looks up `profile.{key}` (e.g. `profile.io`, `profile.model`) with a
    fallback to `default`, then returns the absolute path to the YAML file.
    Raises a helpful error if the file does not exist.

    Args:
        runtime: The runtime yaml loaded as DictConfig.
        key: The key under `profile.` to read (e.g. "io", "model").
        default: Default fragment name if the key is missing.
        fragments_dir: Directory containing the candidate YAML files.

    Returns:
        Absolute Path to the selected fragment YAML.
    """
    name = OmegaConf.select(runtime, f"profile.{key}", default=default)
    path = fragments_dir / f"{name}.yaml"
    if not path.exists():
        choices = sorted(p.stem for p in fragments_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"profile.{key}='{name}' not found at {path}. "
            f"Available: {choices}"
        )
    return path


def load_config(realtime_yaml: Optional[Path] = None) -> DictConfig:
    """Load the realtime config by merging profile fragments + runtime overrides.

    The io and model fragments are chosen dynamically from the runtime yaml's
    `profile.io` and `profile.model` keys (both default to the 2-day input /
    6-hour output GNN+Transformer combo), so any of the 24 × 9 = 216
    combinations can be selected without modifying this loader.

    Merge order (later overrides earlier):
      1. `configs/profile/base.yaml` — normalization, model defaults
      2. `configs/profile/target/{profile.target}.yaml` — input_variables,
         gnn_variable_groups, target_variables (default: ap30)
      3. `configs/profile/io/{profile.io}.yaml` — window indices
      4. `configs/profile/model/{profile.model}.yaml` — model selector
      5. `{realtime_yaml}` — runtime knobs (paths, URLs, analysis toggles)

    Args:
        realtime_yaml: Optional override for the runtime YAML path.

    Returns:
        Merged OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If any referenced fragment YAML is missing.
    """
    root = _project_root()
    profile_dir = root / "configs" / "profile"
    io_dir = profile_dir / "io"
    model_dir = profile_dir / "model"
    target_dir = profile_dir / "target"
    default_runtime = root / "configs" / "realtime.yaml"
    runtime_path = Path(realtime_yaml) if realtime_yaml else default_runtime

    runtime = OmegaConf.load(runtime_path)

    io_path = _resolve_profile_name(runtime, "io", default="in2d_out6h",
                                    fragments_dir=io_dir)
    model_path = _resolve_profile_name(runtime, "model", default="gnn_transformer",
                                       fragments_dir=model_dir)
    target_path = _resolve_profile_name(runtime, "target", default="ap30",
                                        fragments_dir=target_dir)

    base = OmegaConf.load(profile_dir / "base.yaml")
    target_cfg = OmegaConf.load(target_path)
    io_cfg = OmegaConf.load(io_path)
    model_cfg = OmegaConf.load(model_path)

    # target must precede io/model/runtime so its input_variables /
    # gnn_variable_groups / target_variables land intact (base intentionally
    # omits them, so there is no GNN node dict to deep-merge against).
    merged = OmegaConf.merge(base, target_cfg, io_cfg, model_cfg, runtime)
    return merged
