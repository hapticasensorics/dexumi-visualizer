from dexumi_visualizer.plugins import DEFAULT_PLUGIN_NAME, get_plugin, list_plugins


def test_plugin_registry_includes_zarr() -> None:
    names = {plugin.name for plugin in list_plugins()}
    assert "zarr" in names


def test_default_plugin_available() -> None:
    plugin = get_plugin(DEFAULT_PLUGIN_NAME)
    assert plugin is not None
