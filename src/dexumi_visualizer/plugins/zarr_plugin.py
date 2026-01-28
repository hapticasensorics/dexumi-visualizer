from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import zarr

from dexumi_visualizer.zarr_parser import (
    discover_episodes,
    episode_summary,
    is_episode_dir,
    normalize_timestamps,
)

from .base import DataSource, EpisodeRef, LoaderInfo, StreamSpec


@dataclass(frozen=True)
class _StreamDescriptor:
    stream_id: str
    kind: str
    sensor_name: str | None
    array_name: str | None
    array_path: Path | None
    video_path: Path | None
    time_path: Path | None
    time_key: str
    has_time_dim: bool
    dtype: str
    shape: tuple[int, ...]
    modality: str | None
    sample_rate_hz: float | None
    units: str | None
    frame_id: str | None
    calibration: dict | None
    suggested_archetype: str | None


class ZarrDataSource:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._episodes = discover_episodes(self.root)
        self._episode_map = {self._episode_id(path): path for path in self._episodes}
        self._streams_cache: dict[str, dict[str, _StreamDescriptor]] = {}
        self._time_cache: dict[tuple[str, str], np.ndarray] = {}

    def probe(self) -> dict:
        return {
            "format": "zarr",
            "root": str(self.root),
            "episode_count": len(self._episodes),
        }

    def list_episodes(self) -> list[EpisodeRef]:
        episodes: list[EpisodeRef] = []
        for path in self._episodes:
            summary = episode_summary(path)
            end_time_ns = None
            if summary.duration_s is not None:
                end_time_ns = int(summary.duration_s * 1e9)
            episodes.append(
                EpisodeRef(
                    episode_id=self._episode_id(path),
                    start_time_ns=0 if end_time_ns is not None else None,
                    end_time_ns=end_time_ns,
                    tags={"path": str(path)},
                )
            )
        return episodes

    def list_streams(self, episode_id: str) -> list[StreamSpec]:
        descriptors = self._ensure_streams(episode_id)
        return [self._descriptor_to_spec(desc) for desc in descriptors.values()]

    def read_stream(
        self,
        episode_id: str,
        stream_id: str,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> Iterable[dict]:
        descriptor = self._get_descriptor(episode_id, stream_id)
        if descriptor is None:
            raise KeyError(f"Unknown stream_id: {stream_id}")
        if descriptor.kind == "video":
            return self._iter_video_stream(
                descriptor,
                episode_id=episode_id,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
            )
        return self._iter_array_stream(
            descriptor,
            episode_id=episode_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
        )

    def close(self) -> None:
        return

    def _episode_id(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return path.name

    def _episode_path(self, episode_id: str) -> Path:
        if episode_id in self._episode_map:
            return self._episode_map[episode_id]
        candidate = self.root / episode_id
        if candidate.exists():
            return candidate
        raise KeyError(f"Unknown episode_id: {episode_id}")

    def _ensure_streams(self, episode_id: str) -> dict[str, _StreamDescriptor]:
        if episode_id in self._streams_cache:
            return self._streams_cache[episode_id]
        episode_path = self._episode_path(episode_id)
        summary = episode_summary(episode_path)
        descriptors: dict[str, _StreamDescriptor] = {}

        for sensor in summary.sensors:
            if sensor.kind in {"camera", "numeric"}:
                group_path = episode_path / sensor.name
                if not (group_path / ".zgroup").exists():
                    continue
                group = zarr.open_group(str(group_path), mode="r")
                time_key, time_array = _find_time_array(group)
                for key in sorted(group.array_keys()):
                    if key in {"capture_time", "receive_time"}:
                        continue
                    array_path = group_path / key
                    array = group[key]
                    has_time_dim = bool(
                        time_array is not None
                        and array.ndim >= 1
                        and array.shape[0] == time_array.shape[0]
                    )
                    time_path = group_path / time_key if time_key and has_time_dim else None
                    sample_rate = _estimate_sample_rate(time_array) if has_time_dim else None
                    stream_time_key = time_key if time_key and has_time_dim else "index"
                    stream_id = f"{sensor.name}/{key}"
                    modality, archetype = _infer_modality(key, sensor.kind, array)
                    descriptors[stream_id] = _StreamDescriptor(
                        stream_id=stream_id,
                        kind="zarr_array",
                        sensor_name=sensor.name,
                        array_name=key,
                        array_path=array_path,
                        video_path=None,
                        time_path=time_path,
                        time_key=stream_time_key,
                        has_time_dim=has_time_dim,
                        dtype=str(array.dtype),
                        shape=_array_shape(array, has_time_dim=has_time_dim),
                        modality=modality,
                        sample_rate_hz=sample_rate,
                        units=None,
                        frame_id=None,
                        calibration=None,
                        suggested_archetype=archetype,
                    )

                if sensor.video_path:
                    stream_id = f"{sensor.name}/video"
                    width, height, fps = _video_metadata(sensor.video_path)
                    descriptors[stream_id] = _StreamDescriptor(
                        stream_id=stream_id,
                        kind="video",
                        sensor_name=sensor.name,
                        array_name=None,
                        array_path=None,
                        video_path=sensor.video_path,
                        time_path=group_path / time_key if time_key else None,
                        time_key=time_key or "frame",
                        has_time_dim=True,
                        dtype="uint8",
                        shape=_video_shape(width, height),
                        modality="image",
                        sample_rate_hz=fps or _estimate_sample_rate(time_array),
                        units=None,
                        frame_id=None,
                        calibration=None,
                        suggested_archetype="Image",
                    )

            if sensor.kind == "array":
                array_path = episode_path / sensor.name
                if not array_path.exists():
                    continue
                array = zarr.open_array(str(array_path), mode="r")
                has_time_dim = array.ndim >= 1 and array.shape[0] > 1
                stream_id = sensor.name
                modality, archetype = _infer_modality(sensor.name, sensor.kind, array)
                descriptors[stream_id] = _StreamDescriptor(
                    stream_id=stream_id,
                    kind="zarr_array",
                    sensor_name=None,
                    array_name=sensor.name,
                    array_path=array_path,
                    video_path=None,
                    time_path=None,
                    time_key="index",
                    has_time_dim=has_time_dim,
                    dtype=str(array.dtype),
                    shape=_array_shape(array, has_time_dim=has_time_dim),
                    modality=modality,
                    sample_rate_hz=None,
                    units=None,
                    frame_id=None,
                    calibration=None,
                    suggested_archetype=archetype,
                )

            if sensor.kind == "video" and sensor.video_path:
                stream_id = f"{sensor.name}/video"
                width, height, fps = _video_metadata(sensor.video_path)
                descriptors[stream_id] = _StreamDescriptor(
                    stream_id=stream_id,
                    kind="video",
                    sensor_name=sensor.name,
                    array_name=None,
                    array_path=None,
                    video_path=sensor.video_path,
                    time_path=None,
                    time_key="frame",
                    has_time_dim=True,
                    dtype="uint8",
                    shape=_video_shape(width, height),
                    modality="image",
                    sample_rate_hz=fps,
                    units=None,
                    frame_id=None,
                    calibration=None,
                    suggested_archetype="Image",
                )

        self._streams_cache[episode_id] = descriptors
        return descriptors

    def _get_descriptor(self, episode_id: str, stream_id: str) -> _StreamDescriptor | None:
        descriptors = self._ensure_streams(episode_id)
        return descriptors.get(stream_id)

    def _descriptor_to_spec(self, desc: _StreamDescriptor) -> StreamSpec:
        return StreamSpec(
            stream_id=desc.stream_id,
            name=desc.array_name or desc.stream_id,
            modality=desc.modality,
            dtype=desc.dtype,
            shape=desc.shape,
            time_key=desc.time_key,
            sample_rate_hz=desc.sample_rate_hz,
            units=desc.units,
            frame_id=desc.frame_id,
            calibration=desc.calibration,
            suggested_archetype=desc.suggested_archetype,
        )

    def _iter_array_stream(
        self,
        desc: _StreamDescriptor,
        *,
        episode_id: str,
        start_time_ns: int | None,
        end_time_ns: int | None,
    ) -> Iterator[dict]:
        if desc.array_path is None:
            return
        array = zarr.open_array(str(desc.array_path), mode="r")
        time_ns = self._load_time_ns(episode_id, desc.time_path)

        if array.ndim == 0 or not desc.has_time_dim:
            timestamp = 0
            if _within_range(timestamp, start_time_ns, end_time_ns):
                yield {"timestamp_ns": timestamp, "data": np.asarray(array[()])}
            return

        length = array.shape[0]
        if time_ns is not None:
            length = min(length, len(time_ns))
        for index in range(length):
            timestamp = int(time_ns[index]) if time_ns is not None else index
            if not _within_range(timestamp, start_time_ns, end_time_ns):
                continue
            yield {"timestamp_ns": timestamp, "data": np.asarray(array[index])}

    def _iter_video_stream(
        self,
        desc: _StreamDescriptor,
        *,
        episode_id: str,
        start_time_ns: int | None,
        end_time_ns: int | None,
    ) -> Iterator[dict]:
        if desc.video_path is None:
            return
        time_ns = self._load_time_ns(episode_id, desc.time_path)
        try:
            import cv2  # noqa: WPS433
        except Exception:
            return

        cap = cv2.VideoCapture(str(desc.video_path))
        if not cap.isOpened():
            cap.release()
            return
        try:
            index = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                timestamp = int(time_ns[index]) if time_ns is not None and index < len(time_ns) else index
                if _within_range(timestamp, start_time_ns, end_time_ns):
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    yield {"timestamp_ns": timestamp, "data": frame_rgb}
                index += 1
        finally:
            cap.release()

    def _load_time_ns(self, episode_id: str, time_path: Path | None) -> np.ndarray | None:
        if time_path is None:
            return None
        cache_key = (episode_id, str(time_path))
        if cache_key in self._time_cache:
            return self._time_cache[cache_key]
        if not time_path.exists():
            return None
        array = zarr.open_array(str(time_path), mode="r")
        values = np.asarray(array[:], dtype=np.float64)
        if values.size == 0:
            return None
        seconds = normalize_timestamps(values)
        time_ns = (seconds * 1e9).astype(np.int64)
        self._time_cache[cache_key] = time_ns
        return time_ns


def describe() -> LoaderInfo:
    return LoaderInfo(
        name="zarr",
        description="DexUMI Zarr dataset loader",
        extensions=(".zarr",),
        priority=10,
        modality_hints=("image", "depth", "tactile", "pose", "joint"),
    )


def can_open(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    if path.is_dir():
        if is_episode_dir(path):
            return True
        return bool(discover_episodes(path))
    return False


def open(path: Path) -> DataSource:
    return ZarrDataSource(Path(path))


def _find_time_array(group: zarr.Group) -> tuple[str | None, zarr.Array | None]:
    for key in ("capture_time", "receive_time"):
        if key in group.array_keys():
            return key, group[key]
    return None, None


def _estimate_sample_rate(time_array: zarr.Array | None) -> float | None:
    if time_array is None:
        return None
    if time_array.ndim < 1 or time_array.shape[0] < 2:
        return None
    count = min(time_array.shape[0], 1000)
    values = np.asarray(time_array[:count], dtype=np.float64)
    if values.size < 2:
        return None
    times = normalize_timestamps(values)
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    median = float(np.nanmedian(diffs))
    if median <= 0 or np.isnan(median):
        return None
    return 1.0 / median


def _array_shape(array: zarr.Array, *, has_time_dim: bool) -> tuple[int, ...]:
    if array.ndim == 0:
        return ()
    if has_time_dim:
        if array.ndim <= 1:
            return ()
        return tuple(int(dim) for dim in array.shape[1:])
    return tuple(int(dim) for dim in array.shape)


def _infer_modality(
    name: str,
    sensor_kind: str,
    array: zarr.Array | None,
) -> tuple[str | None, str | None]:
    lower = name.lower()
    if "depth" in lower:
        return "depth", "DepthImage"
    if "seg" in lower or "mask" in lower:
        return "segmentation", "SegmentationImage"
    if "pose" in lower:
        return "pose", "Transform3D"
    if "joint" in lower:
        return "joint", "Scalars"
    if "fsr" in lower or "tactile" in lower or "pressure" in lower:
        return "tactile", "Scalars"
    if sensor_kind == "camera":
        return "image", "Image"
    if array is not None and array.ndim >= 3:
        return "tensor", "Tensor"
    return None, None


def _video_metadata(path: Path) -> tuple[int | None, int | None, float | None]:
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return None, None, None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width <= 0 or height <= 0:
        width, height = None, None
    if fps <= 0 or np.isnan(fps):
        fps = None
    return width, height, fps


def _video_shape(width: int | None, height: int | None) -> tuple[int, ...]:
    if width is None or height is None:
        return ()
    return (int(height), int(width), 3)


def _within_range(timestamp: int, start_time_ns: int | None, end_time_ns: int | None) -> bool:
    if start_time_ns is not None and timestamp < start_time_ns:
        return False
    if end_time_ns is not None and timestamp > end_time_ns:
        return False
    return True
