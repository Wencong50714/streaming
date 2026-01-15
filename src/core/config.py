"""Configuration loader for video-language model using OmegaConf."""
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class Config:
    """Configuration class that loads settings from config.yaml using OmegaConf."""

    _instance: "Config | None" = None
    _cfg: Any = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from config.yaml using OmegaConf."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if config_path.exists():
            self._cfg = OmegaConf.load(config_path)
        else:
            self._cfg = OmegaConf.create()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __hasattr__(self, name: str) -> bool:
        return name in self._cfg

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._cfg, key, default)

    def to_dict(self) -> dict:
        return OmegaConf.to_container(self._cfg)


# Global config instance
config = Config()
