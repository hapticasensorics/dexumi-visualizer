"""DexUMI web integration helpers for HapticaGUI-style dataset browsing."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dexumi_visualizer.zarr_parser import discover_episodes

DEFAULT_SESSION_SERVICE_URL = os.environ.get("HAPTICA_SESSION_SERVICE_URL", "http://localhost:8080")
DEFAULT_DATASETS_PATH = os.environ.get(
    "DEXUMI_DATASETS_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "sample_data"),
)
DEFAULT_PREVIEW_SAMPLES = int(os.environ.get("DEXUMI_PREVIEW_SAMPLES", "12"))
DEFAULT_FRAME_RATE = float(os.environ.get("DEXUMI_DEFAULT_FPS", "30"))

STREAM_ALIASES = {
    "joint_angles": "joint_angles",
    "joint_angles_interp": "joint_angles",
    "fsr_values": "fsr",
    "fsr_values_interp": "fsr",
    "raw_voltage": "fsr_voltage",
    "hand_motor_value": "hand_motor",
    "pose": "pose",
    "pose_interp": "pose",
    "intrinsics": "intrinsics",
    "valid_indices": "valid_indices",
}


try:  # Optional dependency for real preview values.
    import zarr  # type: ignore
except Exception:  # pragma: no cover - optional
    zarr = None

try:  # Optional dependency used when zarr is available.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    np = None


try:  # Reflex is optional so core logic can be tested without it.
    import reflex as rx  # type: ignore
except Exception:  # pragma: no cover - optional
    rx = None


def _format_count(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _format_duration(seconds: float | None, *, estimated: bool = False) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        suffix = " (est)" if estimated else ""
        return f"{seconds:.1f}s{suffix}"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        suffix = " (est)" if estimated else ""
        return f"{int(minutes)}m {remainder:04.1f}s{suffix}"
    hours, minutes = divmod(minutes, 60)
    suffix = " (est)" if estimated else ""
    return f"{int(hours)}h {int(minutes)}m{suffix}"


def _read_zarray_shape(zarray_path: Path) -> tuple[int, ...] | None:
    try:
        data = json.loads(zarray_path.read_text())
    except Exception:
        return None
    shape = data.get("shape")
    if not isinstance(shape, list):
        return None
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return tuple(shape)


def _find_arrays_by_name(root: Path, names: Iterable[str]) -> list[Path]:
    wanted = set(names)
    matches: list[Path] = []
    for zarray_path in root.rglob(".zarray"):
        if zarray_path.parent.name in wanted:
            matches.append(zarray_path.parent)
    return matches


def _has_zarr_group(path: Path) -> bool:
    return (path / ".zgroup").exists() or (path / ".zarray").exists()


def _episode_sensor_dirs(episode_path: Path) -> list[Path]:
    sensors: list[Path] = []
    for child in episode_path.iterdir():
        if child.name.startswith(".") or not child.is_dir():
            continue
        if _has_zarr_group(child):
            sensors.append(child)
            continue
        if any(grandchild.name == ".zgroup" for grandchild in child.iterdir()):
            sensors.append(child)
    return sensors


def _detect_streams(episode_path: Path) -> list[str]:
    streams: set[str] = set()
    for sensor in _episode_sensor_dirs(episode_path):
        if sensor.name.startswith("camera"):
            streams.add("camera")
        if sensor.name.startswith("numeric"):
            streams.add("numeric")
        for zarray_path in sensor.rglob(".zarray"):
            name = zarray_path.parent.name
            streams.add(STREAM_ALIASES.get(name, name))
    return sorted(streams)


def _extract_frame_count(episode_path: Path) -> int | None:
    candidates = _find_arrays_by_name(
        episode_path,
        [
            "capture_time",
            "receive_time",
            "joint_angles",
            "joint_angles_interp",
            "fsr_values",
            "fsr_values_interp",
            "raw_voltage",
            "valid_indices",
        ],
    )
    max_frames: int | None = None
    for array_path in candidates:
        shape = _read_zarray_shape(array_path / ".zarray")
        if not shape:
            continue
        frame_count = shape[0] if shape else None
        if frame_count is None:
            continue
        if max_frames is None or frame_count > max_frames:
            max_frames = int(frame_count)
    return max_frames


def _read_duration_from_arrays(episode_path: Path) -> tuple[float | None, bool]:
    """Return (duration_s, estimated_flag)."""
    if zarr is None or np is None:
        return None, False
    candidates = _find_arrays_by_name(episode_path, ["capture_time", "receive_time"])
    for array_path in candidates:
        try:
            arr = zarr.open(str(array_path), mode="r")
            if arr.size < 2:
                continue
            start = float(arr[0])
            end = float(arr[-1])
            duration = max(end - start, 0.0)
            if duration > 0:
                return duration, False
        except Exception:
            continue
    return None, False


def extract_episode_metadata(episode_path: str | Path) -> dict:
    episode = Path(episode_path)
    sensor_dirs = _episode_sensor_dirs(episode)
    sensor_count = len(sensor_dirs) if sensor_dirs else None
    frame_count = _extract_frame_count(episode)
    duration_s, estimated = _read_duration_from_arrays(episode)
    if duration_s is None and frame_count is not None and DEFAULT_FRAME_RATE > 0:
        duration_s = frame_count / DEFAULT_FRAME_RATE
        estimated = True

    streams = _detect_streams(episode)

    return {
        "sensor_count": sensor_count,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "duration_estimated": estimated,
        "streams": streams,
        "sensor_count_display": _format_count(sensor_count),
        "frame_count_display": _format_count(frame_count),
        "duration_display": _format_duration(duration_s, estimated=estimated),
    }


def _find_episode_dirs(dataset_path: Path) -> list[Path]:
    return discover_episodes(dataset_path)


def scan_dexumi_datasets(root_path: str | Path) -> list[dict]:
    root = Path(root_path).expanduser()
    if not root.exists():
        return []
    datasets: list[dict] = []
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        episodes = _find_episode_dirs(dataset_dir)
        if not episodes:
            continue
        episode_entries: list[dict] = []
        total_duration: float = 0.0
        total_duration_estimated = False
        for episode in episodes:
            metadata = extract_episode_metadata(episode)
            try:
                relative_path = str(episode.relative_to(root))
            except ValueError:
                relative_path = str(episode)
            duration_s = metadata.get("duration_s")
            if isinstance(duration_s, (int, float)):
                total_duration += float(duration_s)
                total_duration_estimated = total_duration_estimated or bool(
                    metadata.get("duration_estimated")
                )
            episode_entries.append(
                {
                    "name": episode.name,
                    "path": str(episode.resolve()),
                    "relative_path": relative_path,
                    **metadata,
                }
            )
        dataset_duration = total_duration if total_duration else None
        datasets.append(
            {
                "name": dataset_dir.name,
                "path": str(dataset_dir.resolve()),
                "relative_path": dataset_dir.name,
                "episode_count": len(episode_entries),
                "episodes": episode_entries,
                "duration_s": dataset_duration,
                "duration_estimated": total_duration_estimated,
                "duration_display": _format_duration(
                    dataset_duration, estimated=total_duration_estimated
                ),
            }
        )
    return datasets


def _safe_mean(values: Any) -> float | None:
    if np is None:
        return None
    try:
        array = np.asarray(values)
        if array.size == 0:
            return None
        return float(array.mean())
    except Exception:
        return None


def _load_preview_series(episode_path: Path) -> dict[str, Any]:
    series: dict[str, Any] = {"time": None, "joint": None, "fsr": None}
    if zarr is None:
        return series
    time_candidates = _find_arrays_by_name(episode_path, ["capture_time", "receive_time"])
    joint_candidates = _find_arrays_by_name(
        episode_path, ["joint_angles", "joint_angles_interp"]
    )
    fsr_candidates = _find_arrays_by_name(episode_path, ["fsr_values", "fsr_values_interp"])
    if time_candidates:
        series["time"] = time_candidates[0]
    if joint_candidates:
        series["joint"] = joint_candidates[0]
    if fsr_candidates:
        series["fsr"] = fsr_candidates[0]
    return series


def generate_preview_frames(
    episode_path: str | Path, sample_count: int = DEFAULT_PREVIEW_SAMPLES
) -> list[dict]:
    episode = Path(episode_path)
    metadata = extract_episode_metadata(episode)
    frame_count = metadata.get("frame_count") or 0
    if frame_count == 0:
        return []

    indices = list(range(frame_count))
    if frame_count > sample_count:
        if np is not None:
            indices = sorted(
                set(np.linspace(0, frame_count - 1, sample_count).astype(int))
            )
        else:
            step = max(1, frame_count // sample_count)
            indices = list(range(0, frame_count, step))
            if indices[-1] != frame_count - 1:
                indices.append(frame_count - 1)

    duration = metadata.get("duration_s")
    duration = float(duration) if isinstance(duration, (int, float)) else None

    preview_series = _load_preview_series(episode)
    frames: list[dict] = []

    for idx in indices:
        time_s: float | None = None
        joint_mean = None
        fsr_mean = None
        if zarr is not None and np is not None:
            try:
                if preview_series.get("time") is not None:
                    arr = zarr.open(str(preview_series["time"]), mode="r")
                    if idx < arr.shape[0]:
                        time_s = float(arr[idx])
            except Exception:
                time_s = None
            try:
                if preview_series.get("joint") is not None:
                    arr = zarr.open(str(preview_series["joint"]), mode="r")
                    if idx < arr.shape[0]:
                        joint_mean = _safe_mean(arr[idx])
            except Exception:
                joint_mean = None
            try:
                if preview_series.get("fsr") is not None:
                    arr = zarr.open(str(preview_series["fsr"]), mode="r")
                    if idx < arr.shape[0]:
                        fsr_mean = _safe_mean(arr[idx])
            except Exception:
                fsr_mean = None

        if time_s is None and duration is not None and frame_count > 1:
            time_s = duration * (idx / (frame_count - 1))
        if time_s is None:
            time_s = idx / DEFAULT_FRAME_RATE if DEFAULT_FRAME_RATE > 0 else float(idx)

        frames.append(
            {
                "index": int(idx),
                "time_s": float(time_s),
                "timecode": _format_duration(float(time_s)),
                "joint_mean": joint_mean,
                "fsr_mean": fsr_mean,
            }
        )
    return frames


def _request_json(base_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {"ok": True, "data": {}}
            return {"ok": True, "data": json.loads(body)}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = ""
        return {"ok": False, "error": f"{exc.code} {exc.reason}", "detail": detail}
    except Exception as exc:  # pragma: no cover - network errors are environment-specific
        return {"ok": False, "error": str(exc)}


@dataclass
class DexUMIBrowserModel:
    datasets_root: str = DEFAULT_DATASETS_PATH
    session_service_url: str = DEFAULT_SESSION_SERVICE_URL
    datasets: list[dict] = field(default_factory=list)
    selected_dataset: str = ""
    selected_episode: str = ""
    preview_frames: list[dict] = field(default_factory=list)
    preview_index: int = 0
    preview_running: bool = False
    session_id: str = ""
    session_status: str = "idle"
    session_message: str = ""
    session_state: dict = field(default_factory=dict)
    available_streams: list[str] = field(default_factory=list)
    stream_toggles: dict[str, bool] = field(default_factory=dict)
    loading: bool = False

    @property
    def current_preview(self) -> dict:
        if not self.preview_frames:
            return {}
        return self.preview_frames[self.preview_index % len(self.preview_frames)]

    def load_datasets(self) -> None:
        self.datasets = scan_dexumi_datasets(self.datasets_root)
        if not self.datasets:
            self.selected_dataset = ""
            self.selected_episode = ""
            self.preview_frames = []
            self.preview_index = 0
            self.available_streams = []
            self.stream_toggles = {}
            return
        if self.selected_dataset:
            for dataset in self.datasets:
                if dataset.get("path") == self.selected_dataset:
                    self._set_selected_dataset(dataset)
                    return
        self._set_selected_dataset(self.datasets[0])

    def _set_selected_dataset(self, dataset: dict) -> None:
        self.selected_dataset = dataset.get("path", "")
        episodes = dataset.get("episodes", [])
        if episodes:
            self._set_selected_episode(episodes[0])
        else:
            self.selected_episode = ""
            self.preview_frames = []
            self.preview_index = 0
            self.available_streams = []
            self.stream_toggles = {}

    def select_dataset(self, dataset_path: str) -> None:
        for dataset in self.datasets:
            if dataset.get("path") == dataset_path:
                self._set_selected_dataset(dataset)
                return

    def _set_selected_episode(self, episode: dict) -> None:
        self.selected_episode = episode.get("path", "")
        streams = list(episode.get("streams", []))
        if not streams:
            streams = ["camera", "joint_angles", "fsr"]
        self.available_streams = streams
        self.stream_toggles = {stream: True for stream in streams}
        self.refresh_preview()

    def select_episode(self, episode_path: str) -> None:
        for dataset in self.datasets:
            for episode in dataset.get("episodes", []):
                if episode.get("path") == episode_path:
                    self._set_selected_episode(episode)
                    return

    def refresh_preview(self) -> None:
        if not self.selected_episode:
            self.preview_frames = []
            self.preview_index = 0
            return
        self.preview_frames = generate_preview_frames(self.selected_episode)
        self.preview_index = 0

    def advance_preview(self) -> None:
        if not self.preview_frames:
            return
        self.preview_index = (self.preview_index + 1) % len(self.preview_frames)

    def toggle_preview(self) -> None:
        self.preview_running = not self.preview_running

    def toggle_stream(self, stream: str, enabled: bool) -> None:
        updated = dict(self.stream_toggles)
        updated[stream] = enabled
        self.stream_toggles = updated

    def _handle_session_error(self, response: dict, fallback: str) -> None:
        detail = response.get("detail") or response.get("error") or fallback
        self.session_message = str(detail)
        self.session_status = "error"

    def open_in_rerun(self) -> None:
        if self.loading:
            return
        if not self.selected_episode:
            self.session_message = "Select an episode first."
            self.session_status = "error"
            return
        self.loading = True
        self.session_status = "loading"
        self.session_message = ""
        payload: dict[str, Any] = {"path": self.selected_episode}
        streams = [name for name, enabled in self.stream_toggles.items() if enabled]
        if streams:
            payload["streams"] = streams
        payload["metadata"] = {
            "episode": Path(self.selected_episode).name,
            "dataset": Path(self.selected_episode).parent.name,
        }

        try:
            response = _request_json(self.session_service_url, "POST", "/sessions", payload)
            if not response.get("ok"):
                self._handle_session_error(response, "Unable to open session.")
                return
            data = response.get("data", {})
            self.session_state = data
            if isinstance(data, dict):
                self.session_id = (
                    data.get("id") or data.get("session_id") or data.get("session") or ""
                )
                self.session_status = data.get("status", "ready")
            else:
                self.session_id = ""
                self.session_status = "ready"
            self.session_message = ""
        finally:
            self.loading = False

    def refresh_session_state(self) -> None:
        if not self.session_id:
            self.session_message = "Session not started yet."
            return
        response = _request_json(
            self.session_service_url,
            "GET",
            f"/sessions/{self.session_id}/state",
        )
        if not response.get("ok"):
            self._handle_session_error(response, "Unable to fetch session state.")
            return
        data = response.get("data", {})
        self.session_state = data
        if isinstance(data, dict):
            self.session_status = data.get("status", self.session_status)
        self.session_message = ""


if rx is not None:  # pragma: no cover - UI integration only when Reflex is installed

    class DexUMIBrowserState(rx.State):
        datasets_root: str = DEFAULT_DATASETS_PATH
        session_service_url: str = DEFAULT_SESSION_SERVICE_URL
        datasets: list[dict] = []
        selected_dataset: str = ""
        selected_episode: str = ""
        preview_frames: list[dict] = []
        preview_index: int = 0
        preview_running: bool = False
        session_id: str = ""
        session_status: str = "idle"
        session_message: str = ""
        session_state: dict = {}
        available_streams: list[str] = []
        stream_toggles: dict[str, bool] = {}
        loading: bool = False

        @rx.var
        def has_datasets(self) -> bool:
            return len(self.datasets) > 0

        @rx.var
        def has_preview(self) -> bool:
            return len(self.preview_frames) > 0

        @rx.var
        def current_preview(self) -> dict:
            if not self.preview_frames:
                return {}
            return self.preview_frames[self.preview_index % len(self.preview_frames)]

        @rx.var
        def preview_progress(self) -> str:
            if not self.preview_frames:
                return "0%"
            percent = (self.preview_index + 1) / max(len(self.preview_frames), 1)
            return f"{percent * 100:.0f}%"

        @rx.var
        def session_state_json(self) -> str:
            if not self.session_state:
                return "No session state yet."
            return json.dumps(self.session_state, indent=2, default=str)

        @rx.var
        def has_session_message(self) -> bool:
            return bool(self.session_message)

        @rx.var
        def current_episodes(self) -> list[dict]:
            for dataset in self.datasets:
                if dataset.get("path") == self.selected_dataset:
                    return dataset.get("episodes", [])
            return []

        def load_datasets(self) -> None:
            self.datasets = scan_dexumi_datasets(self.datasets_root)
            if not self.datasets:
                self.selected_dataset = ""
                self.selected_episode = ""
                self.preview_frames = []
                self.preview_index = 0
                self.available_streams = []
                self.stream_toggles = {}
                return
            if self.selected_dataset:
                for dataset in self.datasets:
                    if dataset.get("path") == self.selected_dataset:
                        self._set_selected_dataset(dataset)
                        return
            self._set_selected_dataset(self.datasets[0])

        def _set_selected_dataset(self, dataset: dict) -> None:
            self.selected_dataset = dataset.get("path", "")
            episodes = dataset.get("episodes", [])
            if episodes:
                self._set_selected_episode(episodes[0])
            else:
                self.selected_episode = ""
                self.preview_frames = []
                self.preview_index = 0
                self.available_streams = []
                self.stream_toggles = {}

        def select_dataset(self, dataset_path: str) -> None:
            for dataset in self.datasets:
                if dataset.get("path") == dataset_path:
                    self._set_selected_dataset(dataset)
                    return

        def _set_selected_episode(self, episode: dict) -> None:
            self.selected_episode = episode.get("path", "")
            streams = list(episode.get("streams", []))
            if not streams:
                streams = ["camera", "joint_angles", "fsr"]
            self.available_streams = streams
            self.stream_toggles = {stream: True for stream in streams}
            self.refresh_preview()

        def select_episode(self, episode_path: str) -> None:
            for dataset in self.datasets:
                for episode in dataset.get("episodes", []):
                    if episode.get("path") == episode_path:
                        self._set_selected_episode(episode)
                        return

        def refresh_preview(self) -> None:
            if not self.selected_episode:
                self.preview_frames = []
                self.preview_index = 0
                return
            self.preview_frames = generate_preview_frames(self.selected_episode)
            self.preview_index = 0

        def advance_preview(self) -> None:
            if not self.preview_frames:
                return
            self.preview_index = (self.preview_index + 1) % len(self.preview_frames)

        def toggle_preview(self) -> None:
            self.preview_running = not self.preview_running

        def toggle_stream(self, stream: str, enabled: bool) -> None:
            updated = dict(self.stream_toggles)
            updated[stream] = enabled
            self.stream_toggles = updated

        def _handle_session_error(self, response: dict, fallback: str) -> None:
            detail = response.get("detail") or response.get("error") or fallback
            self.session_message = str(detail)
            self.session_status = "error"

        def open_in_rerun(self) -> None:
            if self.loading:
                return
            if not self.selected_episode:
                self.session_message = "Select an episode first."
                self.session_status = "error"
                return
            self.loading = True
            self.session_status = "loading"
            self.session_message = ""
            payload: dict[str, Any] = {"path": self.selected_episode}
            streams = [name for name, enabled in self.stream_toggles.items() if enabled]
            if streams:
                payload["streams"] = streams
            payload["metadata"] = {
                "episode": Path(self.selected_episode).name,
                "dataset": Path(self.selected_episode).parent.name,
            }

            try:
                response = _request_json(self.session_service_url, "POST", "/sessions", payload)
                if not response.get("ok"):
                    self._handle_session_error(response, "Unable to open session.")
                    return
                data = response.get("data", {})
                self.session_state = data
                if isinstance(data, dict):
                    self.session_id = (
                        data.get("id")
                        or data.get("session_id")
                        or data.get("session")
                        or ""
                    )
                    self.session_status = data.get("status", "ready")
                else:
                    self.session_id = ""
                    self.session_status = "ready"
                self.session_message = ""
            finally:
                self.loading = False

        def refresh_session_state(self) -> None:
            if not self.session_id:
                self.session_message = "Session not started yet."
                return
            response = _request_json(
                self.session_service_url,
                "GET",
                f"/sessions/{self.session_id}/state",
            )
            if not response.get("ok"):
                self._handle_session_error(response, "Unable to fetch session state.")
                return
            data = response.get("data", {})
            self.session_state = data
            if isinstance(data, dict):
                self.session_status = data.get("status", self.session_status)
            self.session_message = ""


    def _metric_chip(label: str, value: str) -> rx.Component:
        return rx.vstack(
            rx.text(label, font_size="0.65rem", color="#7b8794", text_transform="uppercase"),
            rx.text(value, font_size="0.95rem", font_weight="600"),
            spacing="1",
            align_items="flex-start",
        )


    def _dataset_card(dataset: rx.Var) -> rx.Component:
        selected_border = rx.cond(
            dataset["path"] == DexUMIBrowserState.selected_dataset,
            "1px solid #00d4aa",
            "1px solid rgba(255,255,255,0.08)",
        )
        header = rx.hstack(
            rx.vstack(
                rx.text(dataset["name"], font_size="1.05rem", font_weight="600"),
                rx.text(
                    dataset["relative_path"],
                    font_size="0.75rem",
                    color="#8c9aa5",
                ),
                spacing="1",
                align_items="flex-start",
            ),
            rx.button(
                "Select",
                on_click=DexUMIBrowserState.select_dataset(dataset["path"]),
                size="sm",
                variant="outline",
            ),
            justify="between",
            align="center",
            width="100%",
        )

        stats = rx.flex(
            _metric_chip("Episodes", str(dataset["episode_count"])),
            _metric_chip("Duration", dataset["duration_display"]),
            gap="2rem",
            wrap="wrap",
            width="100%",
        )

        return rx.box(
            rx.vstack(header, stats, spacing="3", width="100%"),
            padding="1rem",
            border=selected_border,
            border_radius="14px",
            background="rgba(12, 14, 18, 0.65)",
            width="100%",
        )


    def _episode_card(episode: rx.Var) -> rx.Component:
        selected_border = rx.cond(
            episode["path"] == DexUMIBrowserState.selected_episode,
            "1px solid #00d4aa",
            "1px solid rgba(255,255,255,0.08)",
        )
        header = rx.hstack(
            rx.vstack(
                rx.text(episode["name"], font_size="0.95rem", font_weight="600"),
                rx.text(
                    episode["relative_path"],
                    font_size="0.7rem",
                    color="#8c9aa5",
                ),
                spacing="1",
                align_items="flex-start",
            ),
            rx.button(
                "Preview",
                on_click=DexUMIBrowserState.select_episode(episode["path"]),
                size="sm",
                variant="ghost",
            ),
            justify="between",
            align="center",
            width="100%",
        )
        stats = rx.flex(
            _metric_chip("Sensors", episode["sensor_count_display"]),
            _metric_chip("Frames", episode["frame_count_display"]),
            _metric_chip("Duration", episode["duration_display"]),
            gap="1.5rem",
            wrap="wrap",
            width="100%",
        )
        return rx.box(
            rx.vstack(header, stats, spacing="3", width="100%"),
            padding="0.9rem",
            border=selected_border,
            border_radius="12px",
            background="rgba(18, 20, 24, 0.7)",
            width="100%",
        )


    def _preview_panel() -> rx.Component:
        preview = DexUMIBrowserState.current_preview
        progress_track = rx.box(
            rx.box(
                width=DexUMIBrowserState.preview_progress,
                height="6px",
                background="linear-gradient(90deg, #00d4aa 0%, #00a2ff 100%)",
                border_radius="999px",
            ),
            width="100%",
            height="6px",
            background="rgba(255,255,255,0.08)",
            border_radius="999px",
        )

        return rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("Streaming preview", font_weight="600"),
                    rx.box(flex="1"),
                    rx.button(
                        rx.cond(
                            DexUMIBrowserState.preview_running, "Pause", "Play"
                        ),
                        on_click=DexUMIBrowserState.toggle_preview,
                        size="sm",
                        variant="outline",
                    ),
                    rx.button(
                        "Step",
                        on_click=DexUMIBrowserState.advance_preview,
                        size="sm",
                        variant="ghost",
                    ),
                    spacing="2",
                    width="100%",
                    align="center",
                ),
                rx.text(
                    rx.cond(
                        DexUMIBrowserState.has_preview,
                        preview["timecode"],
                        "No preview loaded",
                    ),
                    font_size="0.85rem",
                    color="#8c9aa5",
                ),
                progress_track,
                rx.flex(
                    rx.vstack(
                        rx.text("Joint mean", font_size="0.65rem", color="#7b8794"),
                        rx.text(
                            rx.cond(
                                DexUMIBrowserState.has_preview,
                                rx.cond(
                                    preview["joint_mean"] != None,
                                    preview["joint_mean"],
                                    "N/A",
                                ),
                                "N/A",
                            ),
                            font_size="0.9rem",
                            font_weight="600",
                        ),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.vstack(
                        rx.text("FSR mean", font_size="0.65rem", color="#7b8794"),
                        rx.text(
                            rx.cond(
                                DexUMIBrowserState.has_preview,
                                rx.cond(
                                    preview["fsr_mean"] != None,
                                    preview["fsr_mean"],
                                    "N/A",
                                ),
                                "N/A",
                            ),
                            font_size="0.9rem",
                            font_weight="600",
                        ),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    gap="2rem",
                    wrap="wrap",
                    width="100%",
                ),
                spacing="3",
                align_items="flex-start",
                width="100%",
            ),
            padding="1rem",
            border_radius="16px",
            border="1px solid rgba(255,255,255,0.08)",
            background=(
                "linear-gradient(135deg, rgba(0, 212, 170, 0.18) 0%, "
                "rgba(8, 12, 20, 0.95) 60%)"
            ),
            width="100%",
        )


    def _stream_toggle(stream: rx.Var) -> rx.Component:
        return rx.checkbox(
            stream,
            checked=DexUMIBrowserState.stream_toggles[stream],
            on_change=DexUMIBrowserState.toggle_stream(stream),
            size="2",
        )


    def _session_panel() -> rx.Component:
        return rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("Rerun session", font_weight="600"),
                    rx.box(flex="1"),
                    rx.text(
                        DexUMIBrowserState.session_status,
                        font_size="0.8rem",
                        color="#8c9aa5",
                    ),
                    align="center",
                    width="100%",
                ),
                rx.text(
                    DexUMIBrowserState.session_service_url,
                    font_size="0.75rem",
                    color="#8c9aa5",
                ),
                rx.flex(
                    rx.button(
                        "Open in Rerun",
                        on_click=DexUMIBrowserState.open_in_rerun,
                        size="sm",
                        variant="solid",
                        disabled=DexUMIBrowserState.loading,
                    ),
                    rx.button(
                        "Refresh state",
                        on_click=DexUMIBrowserState.refresh_session_state,
                        size="sm",
                        variant="ghost",
                    ),
                    gap="0.75rem",
                    wrap="wrap",
                ),
                rx.vstack(
                    rx.text("Streams", font_size="0.8rem", font_weight="600"),
                    rx.flex(
                        rx.foreach(DexUMIBrowserState.available_streams, _stream_toggle),
                        gap="0.75rem",
                        wrap="wrap",
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.cond(
                    DexUMIBrowserState.has_session_message,
                    rx.text(
                        DexUMIBrowserState.session_message,
                        color="#ff657a",
                        font_size="0.85rem",
                    ),
                    rx.box(),
                ),
                rx.text(
                    DexUMIBrowserState.session_state_json,
                    font_family="monospace",
                    font_size="0.75rem",
                    color="#8c9aa5",
                    white_space="pre-wrap",
                ),
                spacing="3",
                align_items="flex-start",
                width="100%",
            ),
            padding="1rem",
            border_radius="16px",
            border="1px solid rgba(255,255,255,0.08)",
            background="rgba(12, 16, 22, 0.85)",
            width="100%",
        )


    def dexumi_browser_page() -> rx.Component:
        header = rx.vstack(
            rx.text("DexUMI Dataset Browser", font_size="1.6rem", font_weight="700"),
            rx.text(
                "Browse episodes, preview signals, and launch Rerun sessions.",
                color="#8c9aa5",
            ),
            spacing="2",
            align_items="flex-start",
            width="100%",
        )

        controls = rx.flex(
            rx.vstack(
                rx.text("Dataset root", font_size="0.7rem", color="#8c9aa5"),
                rx.input(
                    value=DexUMIBrowserState.datasets_root,
                    on_change=DexUMIBrowserState.set_datasets_root,
                    width="100%",
                ),
                spacing="1",
                align_items="flex-start",
                width="100%",
            ),
            rx.button(
                "Rescan",
                on_click=DexUMIBrowserState.load_datasets,
                size="sm",
                variant="outline",
            ),
            gap="1rem",
            wrap="wrap",
            width="100%",
        )

        dataset_list = rx.vstack(
            rx.foreach(DexUMIBrowserState.datasets, _dataset_card),
            spacing="3",
            width="100%",
        )

        episode_list = rx.vstack(
            rx.cond(
                DexUMIBrowserState.selected_dataset,
                rx.foreach(DexUMIBrowserState.current_episodes, _episode_card),
                rx.text("Select a dataset to see episodes.", color="#8c9aa5"),
            ),
            spacing="3",
            width="100%",
        )

        main_grid = rx.flex(
            rx.vstack(
                rx.text("Datasets", font_weight="600"),
                dataset_list,
                spacing="3",
                width="100%",
            ),
            rx.vstack(
                rx.text("Episodes", font_weight="600"),
                episode_list,
                spacing="3",
                width="100%",
            ),
            gap="2rem",
            wrap="wrap",
            width="100%",
        )

        right_panel = rx.vstack(
            _preview_panel(),
            _session_panel(),
            spacing="4",
            width="100%",
        )

        layout = rx.flex(
            rx.box(main_grid, width="60%"),
            rx.box(right_panel, width="40%"),
            gap="2rem",
            wrap="wrap",
            width="100%",
        )

        interval = rx.box()
        if hasattr(rx, "interval"):
            interval = rx.interval(
                interval=1000,
                on_tick=DexUMIBrowserState.advance_preview,
                is_running=DexUMIBrowserState.preview_running,
            )

        return rx.box(
            interval,
            rx.vstack(
                header,
                controls,
                layout,
                spacing="5",
                width="100%",
            ),
            padding="2rem",
            background="radial-gradient(circle at top, #0b1020 0%, #04060c 55%)",
            color="#f5f7fa",
            width="100%",
            min_height="100vh",
        )


__all__ = [
    "DEFAULT_DATASETS_PATH",
    "DEFAULT_SESSION_SERVICE_URL",
    "DexUMIBrowserModel",
    "extract_episode_metadata",
    "generate_preview_frames",
    "scan_dexumi_datasets",
]

if rx is not None:  # pragma: no cover - conditional export
    __all__.append("DexUMIBrowserState")
    __all__.append("dexumi_browser_page")
