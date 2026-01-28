from __future__ import annotations

from importlib import import_module

from .registry import (
    DEFAULT_PLUGIN_NAME,
    Plugin,
    RegistrySnapshot,
    discover_plugins,
    get_plugin,
    list_plugins,
    plugin_errors,
    register_plugin,
)

__all__ = [
    "DEFAULT_PLUGIN_NAME",
    "Plugin",
    "RegistrySnapshot",
    "discover_plugins",
    "get_plugin",
    "list_plugins",
    "plugin_errors",
    "register_plugin",
    "list_plugin_names",
    "load_plugin_module",
]


def list_plugin_names() -> list[str]:
    return [plugin.name for plugin in list_plugins()]


def load_plugin_module(name: str):
    return import_module(f"{__name__}.{name}")
