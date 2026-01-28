from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Iterable


@dataclass(frozen=True)
class EntryPointPlugin:
    name: str
    object: object
    entry_point: metadata.EntryPoint


def iter_entry_points(group: str) -> Iterable[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group=group)
    return entry_points.get(group, [])


def load_entry_point_plugins(group: str) -> tuple[list[EntryPointPlugin], list[str]]:
    plugins: list[EntryPointPlugin] = []
    errors: list[str] = []
    for entry_point in iter_entry_points(group):
        try:
            obj = entry_point.load()
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{entry_point.name}: {exc}")
            continue
        plugins.append(EntryPointPlugin(name=entry_point.name, object=obj, entry_point=entry_point))
    return plugins, errors
