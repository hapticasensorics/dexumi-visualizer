from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import DataSource, EpisodeRef, LoaderInfo, StreamSpec


class Hdf5DataSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def probe(self) -> dict:
        return {
            "format": "hdf5",
            "path": str(self.path),
            "status": "stub",
        }

    def list_episodes(self) -> list[EpisodeRef]:
        return []

    def list_streams(self, episode_id: str) -> list[StreamSpec]:
        return []

    def read_stream(
        self,
        episode_id: str,
        stream_id: str,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> Iterable[dict]:
        if False:
            yield {}
        return

    def close(self) -> None:
        return


def describe() -> LoaderInfo:
    return LoaderInfo(
        name="hdf5",
        description="HDF5 dataset loader (stub)",
        extensions=(".h5", ".hdf5"),
        priority=1,
        modality_hints=(),
    )


def can_open(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return True
    if path.is_dir():
        return any(child.suffix.lower() in {".h5", ".hdf5"} for child in path.iterdir())
    return False


def open(path: Path) -> DataSource:
    return Hdf5DataSource(Path(path))
