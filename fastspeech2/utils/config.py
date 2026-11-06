"""NEW FILE (not in ming024/FastSpeech2): one YAML per language with --set overrides (replaces 3 config files).

YAML configuration loading with attribute access and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A ``dict`` whose keys are also readable as attributes (``cfg.audio.sr``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - mirrors attribute semantics
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj: Any) -> Any:
    """Convert a (nested) Config back to plain Python containers."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def _apply_override(tree: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config.

    * Entries under ``paths`` are resolved relative to the config file, so the
      project can be moved or cloned anywhere.
    * ``overrides`` are ``"a.b.c=value"`` strings (value parsed as YAML), e.g.
      ``--set train.total_steps=1000``.
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        _apply_override(raw, key.strip(), value.strip())

    cfg = _wrap(raw)
    root = path.parent
    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str):
            p = Path(value).expanduser()
            cfg.paths[key] = str(p if p.is_absolute() else (root / p).resolve())
    cfg["config_path"] = str(path)
    cfg["project_root"] = str(_find_project_root(path))
    return cfg


def _find_project_root(start: Path) -> Path:
    for parent in [start.parent, *start.parents]:
        if (parent / "fastspeech2").is_dir():
            return parent
    return start.parent


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False, allow_unicode=True)
