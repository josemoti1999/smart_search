"""
config.py — Persistent configuration for Smart Search.
Reads and writes ~/.smartsearch/config.json.
"""
import json
from pathlib import Path

CONFIG_DIR = Path.home() / '.smartsearch'
CONFIG_FILE = CONFIG_DIR / 'config.json'

DEFAULT_CONFIG = {
    "scan_folders": [
        str(Path.home() / "Documents"),
        str(Path.home() / "Downloads"),
    ],
    "faiss_index_path": str(CONFIG_DIR / "faiss_index"),
    "max_files": 100,
    "chunk_size": 256,
    "chunk_overlap": 32,
    "groq_api_key": "",
    "groq_model": "llama-3.1-8b-instant",
    "port": 5001,
    "first_run": True,
}


def load_config() -> dict:
    """Load config from disk, merging missing keys with defaults."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    """Persist config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_config() -> dict:
    return load_config()


def update_config(**kwargs) -> dict:
    """Patch specific keys and save."""
    config = load_config()
    config.update(kwargs)
    save_config(config)
    return config
