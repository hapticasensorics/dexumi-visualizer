from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
import rerun as rr
import zarr

TIMELINE_CAPTURE = "capture_time"


@dataclass(frozen=True)
class LogEvent:
    time_seconds: float
    entity_path: str
    component: object


@dataclass(frozen=True)
class CameraStreamConfig:
    name: str
    capture_time_s: np.ndarray
    video_path: Path


@dataclass(frozen=True)
class DexumiDataset:
    root: Path
    joint_angles: zarr.Array | None
    joint_angles_name: str | None
    joint_capture_time_s: np.ndarray | None
    fsr_values: zarr.Array | None
    fsr_values_name: str | None
    fsr_capture_time_s: np.ndarray | None
    cameras: Sequence[CameraStreamConfig]


def open_zarr_array(path: Path) -> zarr.Array:
    if not path.exists():
        raise FileNotFoundError(f"Missing zarr array at {path}")
    return zarr.open(str(path), mode="r")


def read_capture_time_seconds(path: Path, *, time_unit: str = "auto") -> np.ndarray:
    capture = open_zarr_array(path)
    times = np.asarray(capture, dtype=np.float64)
    return normalize_time_seconds(times, time_unit=time_unit)


def normalize_time_seconds(times: np.ndarray, *, time_unit: str = "auto") -> np.ndarray:
    if time_unit == "seconds":
        return times
    if time_unit == "milliseconds":
        return times / 1e3
    if time_unit == "microseconds":
        return times / 1e6
    if time_unit == "nanoseconds":
        return times / 1e9
    if time_unit != "auto":
        raise ValueError(f"Unsupported time_unit: {time_unit}")
    if times.size == 0:
        return times
    max_time = float(np.nanmax(times))
    if max_time > 1e17:
        return times / 1e9
    if max_time > 1e14:
        return times / 1e6
    if max_time > 1e11:
        return times / 1e3
    return times


def iter_merged_events(streams: Sequence[Iterable[LogEvent]]) -> Iterator[LogEvent]:
    heap: list[tuple[float, int, int, LogEvent, Iterator[LogEvent]]] = []
    for stream_index, stream in enumerate(streams):
        iterator = iter(stream)
        try:
            first = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (first.time_seconds, stream_index, 0, first, iterator))
    sequence = 1
    while heap:
        _, _, _, event, iterator = heapq.heappop(heap)
        yield event
        try:
            next_event = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (next_event.time_seconds, stream_index, sequence, next_event, iterator))
        sequence += 1


def colorize_fsr_values(values: np.ndarray, *, max_value: float) -> list[tuple[int, int, int, int]]:
    if max_value <= 0 or np.isnan(max_value):
        return [(128, 128, 128, 255) for _ in values]
    intensities = np.clip(values / max_value, 0.0, 1.0)
    colors: list[tuple[int, int, int, int]] = []
    for intensity in intensities:
        r = int(255 * intensity)
        g = int(64 * (1.0 - intensity))
        b = int(255 * (1.0 - intensity))
        colors.append((r, g, b, 255))
    return colors


def coerce_rr_colors(rr_module, colors: list[tuple[int, int, int, int]]):
    if hasattr(rr_module, "Color"):
        return [rr_module.Color(r, g, b, a) for r, g, b, a in colors]
    if hasattr(rr_module, "Rgba32"):
        return [rr_module.Rgba32(r, g, b, a) for r, g, b, a in colors]
    return colors


def max_in_zarr_array(array: zarr.Array) -> float:
    if array.size == 0:
        return 0.0
    max_value = float("-inf")
    chunk_size = array.chunks[0] if array.chunks else array.shape[0]
    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])
        chunk = np.asarray(array[start:stop])
        if chunk.size == 0:
            continue
        chunk_max = float(np.nanmax(chunk))
        if chunk_max > max_value:
            max_value = chunk_max
    if max_value == float("-inf"):
        return 0.0
    return max_value


def iter_joint_angle_events(
    joint_angles: zarr.Array,
    capture_time_s: np.ndarray,
    *,
    mode: str,
    entity_path: str,
    rr_module=rr,
) -> Iterator[LogEvent]:
    if joint_angles.ndim != 2:
        raise ValueError(f"joint_angles expected 2D array, got shape {joint_angles.shape}")
    count = min(joint_angles.shape[0], capture_time_s.shape[0])
    for index in range(count):
        values = np.asarray(joint_angles[index])
        if mode == "scalars":
            component = rr_module.Scalars(values)
        elif mode == "barchart":
            component = rr_module.BarChart(values)
        else:
            raise ValueError(f"Unsupported joint_angles mode: {mode}")
        yield LogEvent(float(capture_time_s[index]), entity_path, component)


def iter_fsr_events(
    fsr_values: zarr.Array,
    capture_time_s: np.ndarray,
    *,
    entity_path: str,
    max_value: float,
    rr_module=rr,
) -> Iterator[LogEvent]:
    if fsr_values.ndim != 2:
        raise ValueError(f"fsr_values expected 2D array, got shape {fsr_values.shape}")
    count = min(fsr_values.shape[0], capture_time_s.shape[0])
    for index in range(count):
        values = np.asarray(fsr_values[index])
        colors = colorize_fsr_values(values, max_value=max_value)
        component = rr_module.Scalars(values, colors=coerce_rr_colors(rr_module, colors))
        yield LogEvent(float(capture_time_s[index]), entity_path, component)


def iter_camera_events(
    video_path: Path,
    capture_time_s: np.ndarray,
    *,
    entity_path: str,
    rr_module=rr,
) -> Iterator[LogEvent]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_index = 0
        while frame_index < capture_time_s.shape[0]:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            component = rr_module.Image(frame_rgb)
            yield LogEvent(float(capture_time_s[frame_index]), entity_path, component)
            frame_index += 1
    finally:
        cap.release()


def discover_dataset(root: Path, *, time_unit: str = "auto") -> DexumiDataset:
    numeric_0 = root / "numeric_0"
    joint_angles = None
    joint_angles_name = None
    joint_capture_time_s = None
    if numeric_0.exists():
        joint_angles_path = None
        for candidate in ("joint_angles", "joint_angles_interp"):
            candidate_path = numeric_0 / candidate
            if candidate_path.exists():
                joint_angles_path = candidate_path
                joint_angles_name = candidate
                break
        if joint_angles_path is not None:
            joint_angles = open_zarr_array(joint_angles_path)
        capture_time_path = numeric_0 / "capture_time"
        if capture_time_path.exists():
            joint_capture_time_s = read_capture_time_seconds(capture_time_path, time_unit=time_unit)

    numeric_1 = root / "numeric_1"
    fsr_values = None
    fsr_values_name = None
    fsr_capture_time_s = None
    if numeric_1.exists():
        fsr_values_path = None
        for candidate in ("fsr_values", "fsr_values_interp"):
            candidate_path = numeric_1 / candidate
            if candidate_path.exists():
                fsr_values_path = candidate_path
                fsr_values_name = candidate
                break
        if fsr_values_path is not None:
            fsr_values = open_zarr_array(fsr_values_path)
        capture_time_path = numeric_1 / "capture_time"
        if capture_time_path.exists():
            fsr_capture_time_s = read_capture_time_seconds(capture_time_path, time_unit=time_unit)

    cameras: list[CameraStreamConfig] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("camera_"):
            continue
        capture_time_path = entry / "capture_time"
        if not capture_time_path.exists():
            continue
        video_path = root / f"{entry.name}.mp4"
        if not video_path.exists():
            continue
        capture_time_s = read_capture_time_seconds(capture_time_path, time_unit=time_unit)
        cameras.append(CameraStreamConfig(entry.name, capture_time_s, video_path))

    return DexumiDataset(
        root=root,
        joint_angles=joint_angles,
        joint_angles_name=joint_angles_name,
        joint_capture_time_s=joint_capture_time_s,
        fsr_values=fsr_values,
        fsr_values_name=fsr_values_name,
        fsr_capture_time_s=fsr_capture_time_s,
        cameras=cameras,
    )


class DexumiRerunExporter:
    def __init__(
        self,
        *,
        timeline: str = TIMELINE_CAPTURE,
        time_unit: str = "auto",
        rr_module=rr,
    ) -> None:
        self.timeline = timeline
        self.time_unit = time_unit
        self.rr = rr_module

    def export(
        self,
        dataset_path: Path,
        output_path: Path,
        *,
        joint_angles_mode: str = "barchart",
    ) -> None:
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        dataset = discover_dataset(dataset_path, time_unit=self.time_unit)
        self._init_recording(output_path)
        streams: list[Iterable[LogEvent]] = []
        if dataset.joint_angles is not None and dataset.joint_capture_time_s is not None:
            streams.append(
                iter_joint_angle_events(
                    dataset.joint_angles,
                    dataset.joint_capture_time_s,
                    mode=joint_angles_mode,
                    entity_path=f"numeric_0/{dataset.joint_angles_name or 'joint_angles'}",
                    rr_module=self.rr,
                )
            )
        if dataset.fsr_values is not None and dataset.fsr_capture_time_s is not None:
            max_value = max_in_zarr_array(dataset.fsr_values)
            streams.append(
                iter_fsr_events(
                    dataset.fsr_values,
                    dataset.fsr_capture_time_s,
                    entity_path=f"numeric_1/{dataset.fsr_values_name or 'fsr_values'}",
                    max_value=max_value,
                    rr_module=self.rr,
                )
            )
        for camera in dataset.cameras:
            streams.append(
                iter_camera_events(
                    camera.video_path,
                    camera.capture_time_s,
                    entity_path=f"{camera.name}/image",
                    rr_module=self.rr,
                )
            )

        for event in iter_merged_events(streams):
            self._set_time(event.time_seconds)
            self.rr.log(event.entity_path, event.component)

    def _init_recording(self, output_path: Path) -> None:
        output_path = output_path.with_suffix(".rrd")
        self.rr.init("dexumi_visualizer", spawn=False)
        if hasattr(self.rr, "save"):
            self.rr.save(str(output_path))

    def _set_time(self, time_seconds: float) -> None:
        if hasattr(self.rr, "set_time_seconds"):
            self.rr.set_time_seconds(self.timeline, time_seconds)
        elif hasattr(self.rr, "set_time_nanos"):
            self.rr.set_time_nanos(self.timeline, int(time_seconds * 1e9))
        else:
            self.rr.set_time_sequence(self.timeline, int(time_seconds * 1e9))


def export_dataset_to_rrd(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    joint_angles_mode: str = "barchart",
    time_unit: str = "auto",
    rr_module=rr,
) -> None:
    exporter = DexumiRerunExporter(time_unit=time_unit, rr_module=rr_module)
    exporter.export(Path(dataset_path), Path(output_path), joint_angles_mode=joint_angles_mode)
