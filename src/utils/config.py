"""
Configuration management for Audio Event Detection system.

Loads YAML config files and provides a unified configuration object.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Hierarchical configuration object with dot-notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with a default fallback."""
        return getattr(self, key, default)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load configuration from YAML file(s).

    Args:
        config_path: Path to a YAML config file. If None, loads default config.
        overrides: Dictionary of overrides to apply on top of file config.

    Returns:
        Config object with all settings.
    """
    # Load default config
    default_config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    with open(default_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Merge with user config if provided
    if config_path is not None:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config_dict = _deep_merge(config_dict, user_config)

    # Apply overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    return Config(config_dict)


def save_config(config: Config, save_path: str) -> None:
    """Save a Config object to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
