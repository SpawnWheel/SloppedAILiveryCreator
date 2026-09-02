"""
Configuration manager for AMS2 AI Livery Studio.
Persists settings in AppData / Local storage.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from src.utils.constants import (
    APP_NAME,
    DEFAULT_AI_MODEL,
    DEFAULT_FINISH,
    get_default_ams2_path,
)


class ConfigManager:
    """Manages application persistent configuration and preferences."""

    def __init__(self):
        self.config_dir = (
            Path(os.environ.get("APPDATA", os.path.expanduser("~")))
            / APP_NAME
        )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self._data: Dict[str, Any] = self._load()

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "ai_model": DEFAULT_AI_MODEL,
            "ams2_overrides_path": str(get_default_ams2_path()),
            "default_finish": DEFAULT_FINISH,
            "last_car_folder": "",
            "last_team_name": "My Sim Team",
            "last_driver": "Driver Name",
            "last_livery_id": 51,
        }

    def _load(self) -> Dict[str, Any]:
        defaults = self._get_defaults()
        if not self.config_file.exists():
            return defaults

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                defaults.update(loaded)
                return defaults
        except Exception:
            return defaults

    def save(self) -> None:
        """Persists current configuration to disk."""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=4)
        except Exception as e:
            print(f"[ConfigManager] Error saving configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._data["api_key"] = value.strip()
        self.save()

    @property
    def ams2_overrides_path(self) -> Path:
        raw_path = self._data.get("ams2_overrides_path", "")
        discovered = get_default_ams2_path()

        if raw_path:
            p = Path(raw_path)
            if p.exists():
                # If current path is empty but discovered path has car folders, auto-upgrade
                try:
                    count_saved = sum(1 for _ in p.iterdir())
                    if count_saved > 0:
                        return p
                    if discovered.exists() and sum(1 for _ in discovered.iterdir()) > count_saved:
                        self._data["ams2_overrides_path"] = str(discovered)
                        self.save()
                        return discovered
                except Exception:
                    pass
                return p

        if discovered.exists():
            self._data["ams2_overrides_path"] = str(discovered)
            self.save()
            return discovered
        return Path(raw_path) if raw_path else discovered

    @ams2_overrides_path.setter
    def ams2_overrides_path(self, path: Path) -> None:
        self._data["ams2_overrides_path"] = str(path)
        self.save()


# Singleton instance
config = ConfigManager()
