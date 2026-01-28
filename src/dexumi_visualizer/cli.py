from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

import numpy as np
import typer
import zarr
from rich.console import Console
from rich.table import Table

from dexumi_visualizer.zarr_parser import (
    EpisodeSummary,
    SensorSummary,
    discover_episodes,
    episode_summary,
    is_episode_dir,
    normalize_timestamps,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("list")
def list_cmd(path: Path, absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths")):
    """List all episodes in a dataset directory."""
    episodes = discover_episodes(path)
    if not episodes:
        typer.echo("No episodes found.")
        raise typer.Exit(code=1)
    for episode in episodes:
        label = str(episode) if absolute else _format_relative(episode, path)
        typer.echo(label)


@app.command()
def info(episode: Path):
    """Show episode metadata (duration, sensors, frames)."""
    if not episode.exists():
        typer.echo(f"Episode not found: {episode}")
        raise typer.Exit(code=1)
    if not is_episode_dir(episode):
        typer.echo(f"Not an episode directory: {episode}")
        raise typer.Exit(code=1)
    summary = episode_summary(episode)
    _render_summary(summary)


@app.command()
def convert(
    episode: Path,
    output: Path = typer.Option(..., "-o", "--output", help="Output .rrd file"),
):
    """Convert a single episode into Rerun format."""
    if not is_episode_dir(episode):
        typer.echo(f"Not an episode directory: {episode}")
        raise typer.Exit(code=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    _convert_episode(episode, output_path=output, spawn=False)
    typer.echo(f"Wrote {output}")


@app.command()
def view(episode: Path):
    """Launch the Rerun viewer for a single episode."""
    if not is_episode_dir(episode):
        typer.echo(f"Not an episode directory: {episode}")
        raise typer.Exit(code=1)
    _convert_episode(episode, output_path=None, spawn=True)


@app.command()
def batch(
    directory: Path,
    output_dir: Path = typer.Option(..., "-o", "--output", help="Output directory"),
):
    """Batch convert all episodes in a directory."""
    episodes = discover_episodes(directory)
    if not episodes:
        typer.echo("No episodes found.")
        raise typer.Exit(code=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        rel = _safe_relative(episode, directory)
        target_dir = output_dir / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{rel.name}.rrd"
        _convert_episode(episode, output_path=output_path, spawn=False)
        typer.echo(f"Wrote {output_path}")


@app.command()
def serve(
    directory: Path,
    port: int = typer.Option(8080, "--port", help="Port for the session service"),
):
    """Start a lightweight session service for HapticaGUI."""
    episodes = discover_episodes(directory)
    if not episodes:
        typer.echo("No episodes found.")
        raise typer.Exit(code=1)
    handler = _make_session_handler(directory, episodes)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    typer.echo(f"Serving {len(episodes)} episodes on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Shutting down.")


def _format_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _safe_relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def _render_summary(summary: EpisodeSummary) -> None:
    console.print(f"Episode: {summary.name}")
    console.print(f"Path: {summary.path}")
    duration = f"{summary.duration_s:.3f}s" if summary.duration_s is not None else "unknown"
    total_frames = str(summary.total_frames) if summary.total_frames is not None else "unknown"
    console.print(f"Duration: {duration}")
    console.print(f"Frames: {total_frames}")

    table = Table(title="Sensors")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Frames", justify="right")
    table.add_column("Arrays")
    table.add_column("Video")

    for sensor in summary.sensors:
        arrays = ", ".join(sensor.arrays) if sensor.arrays else "-"
        frames = str(sensor.frames) if sensor.frames is not None else "-"
        video = str(sensor.video_path) if sensor.video_path else "-"
        table.add_row(sensor.name, sensor.kind, frames, arrays, video)
    console.print(table)


def _convert_episode(
    episode_path: Path,
    output_path: Path | None,
    spawn: bool,
    streams: set[str] | None = None,
) -> None:
    import rerun as rr

    rr.init("dexumi-viz", spawn=spawn)
    if output_path is not None:
        rr.save(str(output_path))

    summary = episode_summary(episode_path)
    rr.log("episode/summary", rr.TextDocument(_summary_to_text(summary)), static=True)

    time_sync = _compute_time_sync(episode_path, summary.sensors)
    _log_calibration_points(episode_path)
    _log_sensor_groups(episode_path, summary.sensors, streams=streams, time_sync=time_sync)

    # rr.init(spawn=True) already launches the viewer.


def _summary_to_text(summary: EpisodeSummary) -> str:
    payload = {
        "episode": summary.name,
        "path": str(summary.path),
        "duration_s": summary.duration_s,
        "total_frames": summary.total_frames,
        "sensors": [
            {
                "name": sensor.name,
                "kind": sensor.kind,
                "frames": sensor.frames,
                "arrays": sensor.arrays,
                "video_path": str(sensor.video_path) if sensor.video_path else None,
            }
            for sensor in summary.sensors
        ],
    }
    return json.dumps(payload, indent=2)


@dataclass(frozen=True)
class TimeSync:
    origin: float
    scale: float


def _infer_time_scale(max_time: float) -> float:
    if max_time > 1e17:
        return 1e-9
    if max_time > 1e14:
        return 1e-6
    if max_time > 1e11:
        return 1e-3
    return 1.0


def _compute_time_sync(episode_path: Path, sensors: Iterable[SensorSummary]) -> TimeSync | None:
    arrays: list[np.ndarray] = []
    for sensor in sensors:
        if sensor.kind not in {"camera", "numeric"}:
            continue
        group_path = episode_path / sensor.name
        if not (group_path / ".zgroup").exists():
            continue
        group = zarr.open_group(str(group_path), mode="r")
        for key in ("capture_time", "receive_time"):
            if key in group.array_keys():
                values = np.asarray(group[key][:], dtype=np.float64)
                if values.size:
                    arrays.append(values)
                break
    if not arrays:
        return None
    min_time = min(float(np.nanmin(values)) for values in arrays if values.size)
    max_time = max(float(np.nanmax(values)) for values in arrays if values.size)
    scale = _infer_time_scale(max(abs(min_time), abs(max_time)))
    return TimeSync(origin=min_time, scale=scale)


def _normalize_times(values: np.ndarray, time_sync: TimeSync | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if time_sync is None:
        return normalize_timestamps(values)
    scaled = values * time_sync.scale
    return scaled - time_sync.origin * time_sync.scale


def _log_sensor_groups(
    episode_path: Path,
    sensors: Iterable[SensorSummary],
    streams: set[str] | None = None,
    time_sync: TimeSync | None = None,
) -> None:
    for sensor in sensors:
        if streams and not _sensor_matches_streams(sensor, streams):
            continue
        if sensor.kind in {"camera", "video"} and sensor.video_path:
            _log_camera_sensor(sensor, episode_path, time_sync=time_sync)
            continue
        if sensor.kind in {"numeric", "array"}:
            _log_numeric_sensor(sensor, episode_path, time_sync=time_sync)


def _log_camera_sensor(
    sensor: SensorSummary,
    episode_path: Path,
    *,
    time_sync: TimeSync | None,
) -> None:
    video_path = sensor.video_path
    if video_path is None:
        return
    camera_path = f"sensors/{sensor.name}"
    image_path = f"{camera_path}/image"

    capture_times = _load_capture_times(episode_path, sensor.name)
    times = _normalize_times(capture_times, time_sync) if capture_times is not None else None

    group = None
    group_path = episode_path / sensor.name
    if (group_path / ".zgroup").exists():
        group = zarr.open_group(str(group_path), mode="r")

    if group is not None:
        if "intrinsics" in group.array_keys():
            intrinsics = np.asarray(group["intrinsics"][:], dtype=np.float64)
            _log_camera_intrinsics(intrinsics, image_path, video_path, times)

        pose_keys = [key for key in ("pose", "pose_interp") if key in group.array_keys()]
        primary_pose = pose_keys[0] if pose_keys else None
        for key in pose_keys:
            pose_array = group[key]
            entity_path = camera_path if key == primary_pose else f"{camera_path}/{key}"
            _log_pose_series(pose_array, times, entity_path)

        ignored = {"capture_time", "receive_time", "pose", "pose_interp", "intrinsics"}
        for key in sorted(group.array_keys()):
            if key in ignored:
                continue
            array = group[key]
            _log_zarr_array_series(
                array,
                array_name=key,
                times=times,
                sensor_name=sensor.name,
                base_path=camera_path,
            )

    _log_video_frames(video_path, image_path, times, episode_path, time_sync=time_sync)


def _log_video_frames(
    video_path: Path,
    image_path: str,
    times: np.ndarray | None,
    episode_path: Path,
    *,
    time_sync: TimeSync | None,
) -> None:
    import cv2
    import rerun as rr

    if times is None:
        frame_count = _video_frame_count(video_path)
        if frame_count is not None:
            matching = _find_matching_capture_times(episode_path, frame_count)
            if matching is not None:
                times = _normalize_times(matching, time_sync)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return
    try:
        frame_index = 0
        max_frames = len(times) if times is not None else None
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if times is not None and frame_index < len(times):
                rr.set_time("time", duration=float(times[frame_index]))
            else:
                rr.set_time("frame", sequence=frame_index)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rr.log(image_path, rr.Image(frame_rgb))
            frame_index += 1
    finally:
        cap.release()


def _read_video_resolution(video_path: Path) -> tuple[int | None, int | None]:
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _log_pose_series(pose_array: zarr.Array, times: np.ndarray | None, entity_path: str) -> None:
    import rerun as rr

    if pose_array.ndim < 2:
        return
    length = pose_array.shape[0]
    use_times = times if times is not None and len(times) == length else None
    for index in range(length):
        if use_times is not None:
            rr.set_time("time", duration=float(use_times[index]))
        else:
            rr.set_time("frame", sequence=index)
        matrix = np.asarray(pose_array[index], dtype=np.float64)
        _log_transform_from_matrix(matrix, entity_path)


def _log_transform_from_matrix(matrix: np.ndarray, entity_path: str) -> None:
    import rerun as rr

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (4, 4):
        rotation = mat[:3, :3]
        translation = mat[:3, 3]
    elif mat.shape == (3, 4):
        rotation = mat[:3, :3]
        translation = mat[:3, 3]
    elif mat.shape == (3, 3):
        rotation = mat
        translation = np.zeros(3, dtype=np.float64)
    else:
        return
    rr.log(entity_path, rr.Transform3D(mat3x3=rotation, translation=translation))


def _log_camera_intrinsics(
    intrinsics: np.ndarray,
    image_path: str,
    video_path: Path,
    times: np.ndarray | None,
) -> None:
    import rerun as rr

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    width, height = _read_video_resolution(video_path)
    if intrinsics.ndim == 3 and times is not None and len(times) == intrinsics.shape[0]:
        for index, matrix in enumerate(intrinsics):
            if matrix.shape != (3, 3):
                continue
            rr.set_time("time", duration=float(times[index]))
            if width is not None and height is not None:
                rr.log(
                    image_path,
                    rr.Pinhole(image_from_camera=matrix, width=width, height=height),
                )
            else:
                rr.log(image_path, rr.Pinhole(image_from_camera=matrix))
        return
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    if intrinsics.shape != (3, 3):
        return
    if width is not None and height is not None:
        rr.log(
            image_path,
            rr.Pinhole(image_from_camera=intrinsics, width=width, height=height),
            static=True,
        )
    else:
        rr.log(image_path, rr.Pinhole(image_from_camera=intrinsics), static=True)


def _find_matching_capture_times(episode_path: Path, frame_count: int) -> np.ndarray | None:
    for entry in sorted(episode_path.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / ".zgroup").exists():
            continue
        group = zarr.open_group(str(entry), mode="r")
        for key in ("capture_time", "receive_time"):
            if key not in group.array_keys():
                continue
            array = group[key]
            if array.ndim < 1 or array.shape[0] != frame_count:
                continue
            return np.asarray(array[:], dtype=np.float64)
    return None


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


def _load_capture_times(episode_path: Path, sensor_name: str) -> np.ndarray | None:
    group_path = episode_path / sensor_name
    if not group_path.exists():
        return None
    if not (group_path / ".zgroup").exists():
        return None
    group = zarr.open_group(str(group_path), mode="r")
    for key in ("capture_time", "receive_time"):
        if key in group.array_keys():
            return np.asarray(group[key][:], dtype=np.float64)
    return None


def _log_calibration_points(episode_path: Path) -> None:
    import pickle

    import rerun as rr

    roots = [episode_path, episode_path.parent]
    seen: set[Path] = set()
    for root in roots:
        for pkl_path in sorted(root.glob("*.pkl")):
            if pkl_path in seen:
                continue
            seen.add(pkl_path)
            try:
                with pkl_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception:
                continue
            if not isinstance(payload, dict) or "points" not in payload:
                continue
            points = _coerce_points_3d(payload.get("points"))
            if points is None:
                continue
            entity_path = f"calibration/{pkl_path.stem}"
            rr.log(entity_path, rr.Points3D(points), static=True)


def _coerce_points_3d(points: object) -> np.ndarray | None:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return None
    if array.shape[1] == 2:
        zeros = np.zeros((array.shape[0], 1), dtype=array.dtype)
        return np.concatenate([array, zeros], axis=1)
    if array.shape[1] >= 3:
        return array[:, :3]
    return None


def _log_numeric_sensor(
    sensor: SensorSummary,
    episode_path: Path,
    *,
    time_sync: TimeSync | None,
) -> None:
    if sensor.kind == "numeric":
        group_path = episode_path / sensor.name
        if not (group_path / ".zgroup").exists():
            return
        group = zarr.open_group(str(group_path), mode="r")
        time_values = None
        for key in ("capture_time", "receive_time"):
            if key in group.array_keys():
                time_values = np.asarray(group[key][:], dtype=np.float64)
                break
        times = _normalize_times(time_values, time_sync) if time_values is not None else None
        base_path = f"sensors/{sensor.name}"
        for key in sorted(group.array_keys()):
            if key in {"capture_time", "receive_time"}:
                continue
            array = group[key]
            _log_zarr_array_series(
                array,
                array_name=key,
                times=times,
                sensor_name=sensor.name,
                base_path=base_path,
            )
        return

    array_path = episode_path / sensor.name
    if not array_path.exists():
        return
    array = zarr.open_array(str(array_path), mode="r")
    _log_zarr_array_series(
        array,
        array_name=sensor.name,
        times=None,
        sensor_name=None,
        base_path="sensors",
    )


def _log_zarr_array_series(
    array: zarr.Array,
    *,
    array_name: str,
    times: np.ndarray | None,
    sensor_name: str | None,
    base_path: str,
) -> None:
    import rerun as rr

    if array.ndim == 0:
        entity_path = _default_entity_path(base_path, array_name)
        rr.log(entity_path, rr.Scalars(float(array[()])))
        return
    length = int(array.shape[0])
    use_times = times if times is not None and len(times) == length else None
    max_value = _max_in_zarr_array(array) if _is_fsr_array(array_name) else None
    for index in range(length):
        if use_times is not None:
            rr.set_time("time", duration=float(use_times[index]))
        else:
            rr.set_time("frame", sequence=index)
        sample = np.asarray(array[index])
        _log_array_sample(
            sample,
            array_name=array_name,
            sensor_name=sensor_name,
            base_path=base_path,
            max_value=max_value,
        )


def _log_array_sample(
    sample: np.ndarray,
    *,
    array_name: str,
    sensor_name: str | None,
    base_path: str,
    max_value: float | None,
) -> None:
    import rerun as rr

    values = np.asarray(sample)
    if _is_pose_array(array_name, values):
        pose_path = f"sensors/pose/{array_name}" if sensor_name is None else f"sensors/{sensor_name}/{array_name}"
        _log_transform_from_matrix(values, pose_path)
        return
    if _is_joint_angles_array(array_name):
        joint_base = f"robot/joints/{array_name}"
        joint_values = values.astype(np.float64).ravel()
        rr.log(f"{joint_base}/bar", rr.BarChart(joint_values))
        for idx, value in enumerate(joint_values):
            rr.log(f"{joint_base}/joint_{idx}", rr.Scalars(float(value)))
        return
    if _is_fsr_array(array_name):
        fsr_values = values.astype(np.float64).ravel()
        colors = _colorize_fsr_values(fsr_values, max_value=max_value or 0.0)
        entity_path = _fsr_entity_path(sensor_name, array_name)
        positions = np.column_stack(
            (np.arange(fsr_values.shape[0], dtype=np.float32), fsr_values.astype(np.float32))
        )
        rr.log(
            entity_path,
            rr.Points2D(positions, colors=_coerce_rr_colors(rr, colors)),
        )
        rr.log(f"{entity_path}/values", rr.Scalars(fsr_values))
        return
    entity_path = _default_entity_path(base_path, array_name)
    rr.log(entity_path, rr.Scalars(values.ravel()))


def _default_entity_path(base_path: str, array_name: str) -> str:
    if base_path:
        return f"{base_path}/{array_name}"
    return array_name


def _fsr_entity_path(sensor_name: str | None, array_name: str) -> str:
    suffix = sensor_name or "fsr"
    return f"sensors/fsr/{suffix}/{array_name}"


def _is_joint_angles_array(name: str) -> bool:
    return "joint_angles" in name


def _is_fsr_array(name: str) -> bool:
    return "fsr" in name or "raw_voltage" in name


def _is_pose_array(name: str, values: np.ndarray) -> bool:
    if "pose" not in name:
        return False
    return values.shape in {(4, 4), (3, 4), (3, 3)}


def _colorize_fsr_values(values: np.ndarray, *, max_value: float) -> list[tuple[int, int, int, int]]:
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


def _coerce_rr_colors(rr_module, colors: list[tuple[int, int, int, int]]):
    if hasattr(rr_module, "Color"):
        return [rr_module.Color(r, g, b, a) for r, g, b, a in colors]
    if hasattr(rr_module, "Rgba32"):
        return [rr_module.Rgba32(r, g, b, a) for r, g, b, a in colors]
    return colors


def _max_in_zarr_array(array: zarr.Array) -> float:
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


def _make_session_handler(root: Path, episodes: list[Path]) -> type[BaseHTTPRequestHandler]:
    root = Path(root)
    episode_paths = list(episodes)

    class SessionHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - external API
            if self.path in {"/", "/health"}:
                self._send_json({"status": "ok"})
                return
            if self.path.rstrip("/") == "/episodes":
                self._send_json(_episodes_payload(root, episode_paths))
                return
            if self.path.startswith("/episodes/"):
                parts = self.path.strip("/").split("/")
                if len(parts) == 2:
                    episode = _episode_by_id(parts[1], episode_paths)
                    if episode is None:
                        self._send_json({"error": "episode not found"}, status=404)
                        return
                    summary = episode_summary(episode)
                    self._send_json(_summary_payload(summary))
                    return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802 - external API
            if self.path.rstrip("/") != "/sessions":
                self._send_json({"error": "not found"}, status=404)
                return
            payload = self._read_json()
            episode_path = Path(payload.get("path", "")) if payload else None
            if episode_path is None or not episode_path.exists() or not is_episode_dir(episode_path):
                self._send_json({"error": "invalid episode path"}, status=400)
                return
            streams = payload.get("streams") if isinstance(payload, dict) else None
            stream_set = set(streams) if isinstance(streams, list) else None
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            thread = threading.Thread(
                target=_convert_episode,
                args=(episode_path, None, True),
                kwargs={"streams": stream_set},
                daemon=True,
            )
            thread.start()
            self._send_json({"id": session_id, "status": "launching"})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            body = self.rfile.read(length)
            try:
                return json.loads(body.decode("utf-8"))
            except Exception:
                return {}

    return SessionHandler


def _episodes_payload(root: Path, episodes: list[Path]) -> dict:
    return {
        "root": str(root),
        "episodes": [
            {
                "id": str(index),
                "name": episode.name,
                "path": str(episode),
                "relative": str(_safe_relative(episode, root)),
            }
            for index, episode in enumerate(episodes)
        ],
    }


def _episode_by_id(raw_id: str, episodes: list[Path]) -> Path | None:
    try:
        index = int(raw_id)
    except ValueError:
        return None
    if index < 0 or index >= len(episodes):
        return None
    return episodes[index]


def _summary_payload(summary: EpisodeSummary) -> dict:
    return {
        "episode": summary.name,
        "path": str(summary.path),
        "duration_s": summary.duration_s,
        "total_frames": summary.total_frames,
        "sensors": [
            {
                "name": sensor.name,
                "kind": sensor.kind,
                "frames": sensor.frames,
                "arrays": sensor.arrays,
                "video_path": str(sensor.video_path) if sensor.video_path else None,
            }
            for sensor in summary.sensors
        ],
    }


def _sensor_matches_streams(sensor: SensorSummary, streams: set[str]) -> bool:
    if not streams:
        return True
    tags = {sensor.kind, sensor.name}
    tags.update(sensor.arrays)
    if sensor.kind in {"camera", "video"}:
        tags.add("camera")
    for array_name in sensor.arrays:
        if "joint_angles" in array_name:
            tags.add("joint_angles")
        if "fsr" in array_name or "raw_voltage" in array_name:
            tags.add("fsr")
        if "pose" in array_name:
            tags.add("pose")
        if "hand_motor" in array_name:
            tags.add("hand_motor")
        if "valid_indices" in array_name:
            tags.add("valid_indices")
    return bool(tags.intersection(streams))
