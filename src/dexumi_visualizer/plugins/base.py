from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    start_time_ns: int | None
    end_time_ns: int | None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    name: str
    modality: str | None
    dtype: str
    shape: tuple[int, ...]
    time_key: str
    sample_rate_hz: float | None
    units: str | None
    frame_id: str | None
    calibration: dict | None
    suggested_archetype: str | None


@dataclass(frozen=True)
class LoaderInfo:
    name: str
    description: str = ""
    extensions: tuple[str, ...] = ()
    priority: int = 0
    modality_hints: tuple[str, ...] = ()


class DataSource(Protocol):
    def probe(self) -> dict: ...

    def list_episodes(self) -> list[EpisodeRef]: ...

    def list_streams(self, episode_id: str) -> list[StreamSpec]: ...

    def read_stream(
        self,
        episode_id: str,
        stream_id: str,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> Iterable[dict]:
        """Yield dicts with {"timestamp_ns": int, "data": Any, ...}."""

    def close(self) -> None: ...
