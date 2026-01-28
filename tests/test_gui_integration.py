from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dexumi_visualizer.gui_integration import (
    DexUMIBrowserModel,
    extract_episode_metadata,
    generate_preview_frames,
    scan_dexumi_datasets,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "sample_data"


class _SessionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        payload = json.loads(body.decode("utf-8")) if body else {}
        self.server.last_request = {"path": self.path, "payload": payload}

        if getattr(self.server, "respond_with_error", False):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"boom"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {"id": "session-123", "status": "ready"}
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - stdlib signature
        return


def _start_session_server(respond_with_error: bool = False) -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), _SessionHandler)
    server.respond_with_error = respond_with_error
    server.last_request = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_session_server(server: HTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_scan_datasets_finds_sample_data() -> None:
    datasets = scan_dexumi_datasets(DATA_ROOT)
    names = {dataset["name"] for dataset in datasets}
    assert "software_go_through" in names
    assert "xhand_reference_5_14" in names

    dataset = next(d for d in datasets if d["name"] == "software_go_through")
    episode_names = {episode["name"] for episode in dataset["episodes"]}
    assert "episode_0" in episode_names


def test_episode_metadata_extracts_counts() -> None:
    episode = DATA_ROOT / "software_go_through" / "episode_0"
    metadata = extract_episode_metadata(episode)
    assert metadata["sensor_count"] is not None
    assert metadata["sensor_count"] > 0
    assert metadata["frame_count"] is not None
    assert metadata["frame_count"] > 0
    assert metadata["duration_s"] is not None
    assert metadata["duration_s"] > 0


def test_generate_preview_frames_returns_samples() -> None:
    episode = DATA_ROOT / "software_go_through" / "episode_0"
    frames = generate_preview_frames(episode, sample_count=5)
    assert len(frames) > 0
    assert frames[0]["index"] == 0
    assert "time_s" in frames[0]


def test_open_in_rerun_posts_session() -> None:
    server, thread = _start_session_server()
    try:
        base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"
        model = DexUMIBrowserModel(session_service_url=base_url)
        model.selected_episode = "/tmp/sample_episode"
        model.stream_toggles = {"camera": True, "fsr": False}

        model.open_in_rerun()

        request = server.last_request
        assert request is not None
        assert request["path"] == "/sessions"
        assert request["payload"]["path"] == "/tmp/sample_episode"
        assert request["payload"]["streams"] == ["camera"]
        assert model.session_id == "session-123"
        assert model.session_status == "ready"
        assert model.session_message == ""
        assert model.loading is False
    finally:
        _stop_session_server(server, thread)


def test_open_in_rerun_handles_error() -> None:
    server, thread = _start_session_server(respond_with_error=True)
    try:
        base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"
        model = DexUMIBrowserModel(session_service_url=base_url)
        model.selected_episode = "/tmp/sample_episode"

        model.open_in_rerun()

        assert model.session_status == "error"
        assert model.session_message
        assert model.loading is False
    finally:
        _stop_session_server(server, thread)
