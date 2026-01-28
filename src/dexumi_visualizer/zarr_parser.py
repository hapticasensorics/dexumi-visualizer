from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr

ZGROUP = ".zgroup"
ZARRAY = ".zarray"


@dataclass(frozen=True)
class SensorSummary:
    name: str
    kind: str
    frames: int | None
    arrays: list[str]
    video_path: Path | None = None


@dataclass(frozen=True)
class EpisodeSummary:
    name: str
    path: Path
    duration_s: float | None
    total_frames: int | None
    sensors: list[SensorSummary]


def discover_episodes(root: Path) -> list[Path]:
    root = Path(root)
    if is_episode_dir(root):
        return [root]

    episodes = [child for child in root.iterdir() if is_episode_dir(child)]
    if episodes:
        return sorted(episodes)

    nested: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        nested.extend([grand for grand in child.iterdir() if is_episode_dir(grand)])
    return sorted(set(nested))


def is_episode_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / ZGROUP).exists():
        return False
    if path.name.startswith(("numeric_", "camera_")):
        return False
    if path.name.startswith("episode_") or path.name == "reference_episode":
        return True
    if any(_is_sensor_group(child) for child in path.iterdir()):
        return True
    if any(child.suffix == ".mp4" for child in path.iterdir() if child.is_file()):
        return True
    if any(_is_zarr_array_dir(child) for child in path.iterdir()):
        return True
    return False


def _is_sensor_group(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / ZGROUP).exists():
        return False
    return path.name.startswith(("numeric_", "camera_"))


def _is_zarr_array_dir(path: Path) -> bool:
    return path.is_dir() and (path / ZARRAY).exists()


def episode_summary(episode_path: Path) -> EpisodeSummary:
    episode_path = Path(episode_path)
    sensors: list[SensorSummary] = []

    sensor_groups = sorted(
        [child for child in episode_path.iterdir() if _is_sensor_group(child)],
        key=lambda p: p.name,
    )
    for group_path in sensor_groups:
        group = zarr.open_group(str(group_path), mode="r")
        arrays = sorted(group.array_keys())
        frames = _frames_from_group(group)
        kind = "camera" if group_path.name.startswith("camera_") else "numeric"
        video_path = None
        if kind == "camera":
            candidate = episode_path / f"{group_path.name}.mp4"
            if candidate.exists():
                video_path = candidate
        sensors.append(
            SensorSummary(
                name=group_path.name,
                kind=kind,
                frames=frames,
                arrays=arrays,
                video_path=video_path,
            )
        )

    for array_dir in sorted([child for child in episode_path.iterdir() if _is_zarr_array_dir(child)]):
        array = zarr.open_array(str(array_dir), mode="r")
        frames = array.shape[0] if array.ndim > 0 else None
        sensors.append(
            SensorSummary(
                name=array_dir.name,
                kind="array",
                frames=frames,
                arrays=[array_dir.name],
            )
        )

    for video_path in sorted(episode_path.glob("*.mp4")):
        if any(sensor.video_path == video_path for sensor in sensors):
            continue
        sensors.append(
            SensorSummary(
                name=video_path.stem,
                kind="video",
                frames=_video_frame_count(video_path),
                arrays=[],
                video_path=video_path,
            )
        )

    duration_s = _episode_duration_seconds(sensor_groups)
    total_frames = _max_frames(sensors)

    return EpisodeSummary(
        name=episode_path.name,
        path=episode_path,
        duration_s=duration_s,
        total_frames=total_frames,
        sensors=sensors,
    )


def _frames_from_group(group: zarr.Group) -> int | None:
    for key in ("valid_indices", "capture_time", "receive_time"):
        if key in group.array_keys():
            return _array_len(group[key])
    lengths = [_array_len(group[key]) for key in group.array_keys()]
    lengths = [length for length in lengths if length is not None]
    return max(lengths) if lengths else None


def _array_len(array: zarr.Array) -> int | None:
    if array.ndim < 1:
        return None
    return int(array.shape[0])


def _episode_duration_seconds(sensor_groups: Iterable[Path]) -> float | None:
    mins: list[float] = []
    maxs: list[float] = []
    for group_path in sensor_groups:
        group = zarr.open_group(str(group_path), mode="r")
        for key in ("capture_time", "receive_time"):
            if key not in group.array_keys():
                continue
            array = group[key]
            if array.size == 0:
                continue
            try:
                start = float(array[0])
                end = float(array[-1])
            except Exception:
                values = np.asarray(array[:], dtype=np.float64)
                if values.size == 0:
                    continue
                start = float(np.nanmin(values))
                end = float(np.nanmax(values))
            mins.append(start)
            maxs.append(end)
    if not mins or not maxs:
        return None
    span = float(np.nanmax(maxs) - np.nanmin(mins))
    if span <= 0 or np.isnan(span):
        return None
    return _normalize_span_seconds(span)


def normalize_timestamps(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    span = float(np.nanmax(values) - np.nanmin(values))
    scale = 1.0
    if span > 1e9:
        scale = 1e-9
    elif span > 1e6:
        scale = 1e-6
    normalized = values * scale
    return normalized - float(np.nanmin(normalized))


def _normalize_span_seconds(span: float) -> float:
    if span > 1e9:
        return span * 1e-9
    if span > 1e6:
        return span * 1e-6
    return span


def _max_frames(sensors: Iterable[SensorSummary]) -> int | None:
    frames = [sensor.frames for sensor in sensors if sensor.frames is not None]
    return max(frames) if frames else None


def _video_frame_count(video_path: Path) -> int | None:
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None
