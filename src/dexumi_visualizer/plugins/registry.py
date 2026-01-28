from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from .base import DataSource, LoaderInfo
from .loader import EntryPointPlugin, load_entry_point_plugins

PLUGIN_ENTRY_POINT_GROUP = "dexumi_visualizer.plugins"
DEFAULT_PLUGIN_NAME = "zarr"

_BUILTIN_MODULES = {
    "zarr": "dexumi_visualizer.plugins.zarr_plugin",
    "hdf5": "dexumi_visualizer.plugins.hdf5_plugin",
}


@dataclass(frozen=True)
class Plugin:
    name: str
    info: LoaderInfo
    can_open: Callable[[Path], bool]
    open: Callable[[Path], DataSource]
    module: object | None = None
    entry_point: str | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    plugins: list[Plugin]
    errors: list[str]


_REGISTRY: dict[str, Plugin] = {}
_CACHE: RegistrySnapshot | None = None


def register_plugin(plugin: Plugin, *, overwrite: bool = False) -> None:
    if not overwrite and plugin.name in _REGISTRY:
        return
    _REGISTRY[plugin.name] = plugin
    _invalidate_cache()


def _invalidate_cache() -> None:
    global _CACHE
    _CACHE = None


def _resolve_plugin(name: str, obj: object, *, entry_point: str | None = None) -> Plugin | None:
    plugin_obj = getattr(obj, "PLUGIN", obj)
    can_open = getattr(plugin_obj, "can_open", None)
    open_fn = getattr(plugin_obj, "open", None)
    describe = getattr(plugin_obj, "describe", None)
    if not callable(open_fn):
        return None
    if not callable(can_open):
        can_open = lambda _path: True
    info = _coerce_info(name, describe)
    return Plugin(
        name=name,
        info=info,
        can_open=can_open,
        open=open_fn,
        module=plugin_obj,
        entry_point=entry_point,
    )


def _coerce_info(name: str, describe: object) -> LoaderInfo:
    if callable(describe):
        try:
            payload = describe()
        except Exception:  # pragma: no cover - defensive
            payload = None
    else:
        payload = None

    if isinstance(payload, LoaderInfo):
        if payload.name:
            return payload
        return LoaderInfo(
            name=name,
            description=payload.description,
            extensions=payload.extensions,
            priority=payload.priority,
            modality_hints=payload.modality_hints,
        )

    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("name", None)
        return LoaderInfo(name=name, **payload)

    return LoaderInfo(name=name)


def discover_plugins(*, force: bool = False) -> RegistrySnapshot:
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    plugins: dict[str, Plugin] = dict(_REGISTRY)
    errors: list[str] = []

    entry_plugins, entry_errors = load_entry_point_plugins(PLUGIN_ENTRY_POINT_GROUP)
    errors.extend(entry_errors)
    for entry in entry_plugins:
        plugin = _resolve_plugin(entry.name, entry.object, entry_point=str(entry.entry_point))
        if plugin is None:
            errors.append(f"{entry.name}: missing required plugin hooks")
            continue
        plugins[plugin.name] = plugin

    for name, module_path in _BUILTIN_MODULES.items():
        if name in plugins:
            continue
        try:
            module = import_module(module_path)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{name}: {exc}")
            continue
        plugin = _resolve_plugin(name, module, entry_point=module_path)
        if plugin is None:
            errors.append(f"{name}: missing required plugin hooks")
            continue
        plugins[plugin.name] = plugin

    ordered = sorted(
        plugins.values(),
        key=lambda item: (-item.info.priority, item.name),
    )
    _CACHE = RegistrySnapshot(plugins=ordered, errors=errors)
    return _CACHE


def list_plugins() -> list[Plugin]:
    return discover_plugins().plugins


def get_plugin(name: str) -> Plugin | None:
    plugins = discover_plugins().plugins
    for plugin in plugins:
        if plugin.name == name:
            return plugin
    return None


def plugin_errors() -> list[str]:
    return discover_plugins().errors
