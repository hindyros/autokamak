from pathlib import Path
from types import SimpleNamespace as NS

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path: str) -> NS:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping/object.")
    return NS(**raw)


def resolve_workspace(workspace: str) -> Path:
    ws = Path(workspace)
    if not ws.is_absolute():
        ws = REPO_ROOT / ws
    return ws
