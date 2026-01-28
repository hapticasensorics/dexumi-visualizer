from __future__ import annotations

import json
import threading
import uuid
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

    _log_sensor_groups(episode_path, summary.sensors, streams=streams)

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


def _log_sensor_groups(
    episode_path: Path, sensors: Iterable[SensorSummary], streams: set[str] | None = None
) -> None:
    import rerun as rr

    for sensor in sensors:
        if streams and not _sensor_matches_streams(sensor, streams):
            continue
        if sensor.kind == "camera" and sensor.video_path:
            _log_video_asset(sensor, episode_path)
            continue
        if sensor.kind == "video" and sensor.video_path:
            _log_video_asset(sensor, episode_path)
            continue
        if sensor.kind in {"numeric", "array"}:
            _log_numeric_sensor(sensor, episode_path)


def _log_video_asset(sensor: SensorSummary, episode_path: Path) -> None:
    import rerun as rr

    video_path = sensor.video_path
    if video_path is None:
        return
    entity_path = f"video/{sensor.name}"
    video_asset = rr.AssetVideo(path=str(video_path))
    rr.log(entity_path, video_asset, static=True)

    capture_times = _load_capture_times(episode_path, sensor.name)
    try:
        frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    except Exception:
        return

    if capture_times is not None:
        times = normalize_timestamps(capture_times)
        length = min(len(times), len(frame_timestamps_ns))
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn("time", duration=times[:length])],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns[:length]),
        )
    else:
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn("frame", sequence=np.arange(len(frame_timestamps_ns)))],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )


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


def _log_numeric_sensor(sensor: SensorSummary, episode_path: Path) -> None:
    import rerun as rr

    if sensor.kind == "numeric":
        group_path = episode_path / sensor.name
        group = zarr.open_group(str(group_path), mode="r")
        timestamps = None
        for key in ("capture_time", "receive_time"):
            if key in group.array_keys():
                timestamps = normalize_timestamps(np.asarray(group[key][:], dtype=np.float64))
                break
        arrays = [key for key in group.array_keys() if key not in {"capture_time", "receive_time"}]
        if not arrays:
            return
        length = min(_array_length(group[key]) for key in arrays)
        for idx in range(length):
            if timestamps is not None:
                rr.set_time("time", duration=float(timestamps[idx]))
            else:
                rr.set_time("frame", sequence=idx)
            for key in arrays:
                rr.log(f"{sensor.name}/{key}", rr.Scalars(np.asarray(group[key][idx]).ravel()))
        return

    array_path = episode_path / sensor.name
    if not array_path.exists():
        return
    array = zarr.open_array(str(array_path), mode="r")
    if array.ndim == 0:
        rr.log(f"{sensor.name}", rr.Scalars(float(array[()])))
        return
    length = array.shape[0]
    for idx in range(length):
        rr.set_time("frame", sequence=idx)
        rr.log(f"{sensor.name}", rr.Scalars(np.asarray(array[idx]).ravel()))


def _array_length(array: zarr.Array) -> int:
    if array.ndim == 0:
        return 1
    return int(array.shape[0])


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
