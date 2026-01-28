from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

import dexumi_visualizer.rerun_loader as rerun_loader
from dexumi_visualizer.rerun_loader import (
    DexumiRerunExporter,
    LogEvent,
    colorize_fsr_values,
    iter_merged_events,
)


class FakeRR:
    class Scalars:
        def __init__(self, values, **kwargs):
            self.values = np.asarray(values)
            self.kwargs = kwargs

    class BarChart:
        def __init__(self, values, **kwargs):
            self.values = np.asarray(values)
            self.kwargs = kwargs

    class Image:
        def __init__(self, values, **kwargs):
            self.values = np.asarray(values)
            self.kwargs = kwargs

    def __init__(self):
        self.saved = None
        self.logged = []
        self.current_time = None

    def init(self, *args, **kwargs):
        return None

    def save(self, path: str):
        self.saved = path

    def set_time_seconds(self, timeline: str, time_seconds: float):
        self.current_time = time_seconds

    def log(self, entity_path: str, component):
        self.logged.append((self.current_time, entity_path, component))


def create_zarr_array(path: Path, data: np.ndarray) -> None:
    array = zarr.open(str(path), mode="w", shape=data.shape, dtype=data.dtype)
    array[:] = data


def build_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir(parents=True)

    numeric_0 = dataset / "numeric_0"
    numeric_0.mkdir()
    create_zarr_array(
        numeric_0 / "joint_angles",
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
    )
    create_zarr_array(
        numeric_0 / "capture_time",
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    numeric_1 = dataset / "numeric_1"
    numeric_1.mkdir()
    create_zarr_array(
        numeric_1 / "fsr_values",
        np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32),
    )
    create_zarr_array(
        numeric_1 / "capture_time",
        np.array([1.5, 2.5], dtype=np.float64),
    )

    camera_0 = dataset / "camera_0"
    camera_0.mkdir()
    create_zarr_array(
        camera_0 / "capture_time",
        np.array([1.2, 2.2], dtype=np.float64),
    )
    (dataset / "camera_0.mp4").touch()

    return dataset


def test_iter_merged_events_orders_by_time():
    stream_a = [
        LogEvent(1.0, "a", "A"),
        LogEvent(3.0, "a", "B"),
    ]
    stream_b = [
        LogEvent(2.0, "b", "C"),
    ]
    times = [event.time_seconds for event in iter_merged_events([stream_a, stream_b])]
    assert times == [1.0, 2.0, 3.0]


def test_colorize_fsr_values_scales_between_blue_and_red():
    colors = colorize_fsr_values(np.array([0.0, 1.0]), max_value=1.0)
    low = colors[0]
    high = colors[1]
    assert low[2] > low[0]
    assert high[0] > high[2]


def test_export_streams_logs_in_time_order(tmp_path: Path, monkeypatch):
    dataset = build_dataset(tmp_path)
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 255, dtype=np.uint8),
    ]

    class FakeCapture:
        def __init__(self, _path: str):
            self._frames = list(frames)

        def read(self):
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def release(self):
            return None

    monkeypatch.setattr(rerun_loader.cv2, "VideoCapture", lambda path: FakeCapture(path))
    fake_rr = FakeRR()
    exporter = DexumiRerunExporter(rr_module=fake_rr)
    exporter.export(dataset, tmp_path / "output.rrd", joint_angles_mode="scalars")

    assert fake_rr.saved is not None
    times = [entry[0] for entry in fake_rr.logged]
    assert times == sorted(times)
    assert len(fake_rr.logged) == 7
    paths = [entry[1] for entry in fake_rr.logged]
    assert "numeric_0/joint_angles" in paths
    assert "numeric_1/fsr_values" in paths
    assert "camera_0/image" in paths
