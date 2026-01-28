from __future__ import annotations

from pathlib import Path

from dexumi_visualizer.plugins import zarr_plugin
from dexumi_visualizer.validation import PluginValidator


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample_data" / "software_go_through"


def test_zarr_plugin_passes_validation() -> None:
    validator = PluginValidator(zarr_plugin, dataset_path=SAMPLE_DATA)
    report = validator.run()
    assert report.passed


class BadPlugin:
    @staticmethod
    def can_open(_path: Path) -> bool:
        return True

    @staticmethod
    def open(_path: Path):
        return object()


def test_invalid_plugin_fails_validation() -> None:
    validator = PluginValidator(BadPlugin, dataset_path=SAMPLE_DATA)
    report = validator.run()
    assert not report.passed
