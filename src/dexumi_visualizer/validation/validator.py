from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dexumi_visualizer.plugins.base import DataSource, EpisodeRef, StreamSpec
from dexumi_visualizer.validation.report import ValidationCheckResult, ValidationReport


class PluginValidator:
    def __init__(
        self,
        plugin: object,
        *,
        dataset_path: Path,
        episode_id: str | None = None,
        max_streams: int = 5,
        max_samples_per_stream: int = 3,
    ) -> None:
        self.plugin = plugin
        self.dataset_path = Path(dataset_path)
        self.episode_id = episode_id
        self.max_streams = max_streams
        self.max_samples_per_stream = max_samples_per_stream

    def run(self) -> ValidationReport:
        checks: list[ValidationCheckResult] = []
        metrics: dict[str, Any] = {}
        plugin_name = self._plugin_name()

        if not self._check_plugin_interface(checks):
            return ValidationReport.now(
                plugin_name=plugin_name,
                dataset_path=str(self.dataset_path),
                checks=checks,
                metrics=metrics,
            )

        datasource = None
        try:
            datasource = self.plugin.open(self.dataset_path)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ValidationCheckResult(
                    name="open",
                    status="fail",
                    details=f"plugin.open failed: {exc}",
                )
            )
            return ValidationReport.now(
                plugin_name=plugin_name,
                dataset_path=str(self.dataset_path),
                checks=checks,
                metrics=metrics,
            )

        if not self._check_datasource(datasource, checks):
            return ValidationReport.now(
                plugin_name=plugin_name,
                dataset_path=str(self.dataset_path),
                checks=checks,
                metrics=metrics,
            )

        try:
            episodes = self._safe_list_episodes(datasource, checks)
            if not episodes:
                return ValidationReport.now(
                    plugin_name=plugin_name,
                    dataset_path=str(self.dataset_path),
                    checks=checks,
                    metrics=metrics,
                )
            episode_id = self._select_episode_id(episodes)
            streams = self._safe_list_streams(datasource, episode_id, checks)
            if not streams:
                return ValidationReport.now(
                    plugin_name=plugin_name,
                    dataset_path=str(self.dataset_path),
                    checks=checks,
                    metrics=metrics,
                )
            sample_bundle, ttf_metrics = self._iterate_streams(
                datasource,
                episode_id,
                streams,
                checks,
            )
            if ttf_metrics:
                checks.append(
                    ValidationCheckResult(
                        name="performance",
                        status="pass",
                        details="time to first frame collected",
                        metrics=ttf_metrics,
                    )
                )
            else:
                checks.append(
                    ValidationCheckResult(
                        name="performance",
                        status="skip",
                        details="no samples to benchmark",
                    )
                )
            if ttf_metrics:
                metrics["time_to_first_frame_ms"] = ttf_metrics
                times = [value for value in ttf_metrics.values() if value is not None]
                if times:
                    metrics["time_to_first_frame_summary_ms"] = {
                        "min": min(times),
                        "max": max(times),
                        "avg": sum(times) / len(times),
                    }
            self._check_rerun_archetypes(sample_bundle, checks)
            self._check_missing_data(datasource, episode_id, streams, checks)
        finally:
            if datasource is not None:
                try:
                    datasource.close()
                except Exception:  # noqa: BLE001
                    pass

        return ValidationReport.now(
            plugin_name=plugin_name,
            dataset_path=str(self.dataset_path),
            checks=checks,
            metrics=metrics,
        )

    def _plugin_name(self) -> str:
        return (
            getattr(self.plugin, "name", None)
            or getattr(self.plugin, "PLUGIN_NAME", None)
            or getattr(self.plugin, "__name__", None)
            or self.plugin.__class__.__name__
        )

    def _check_plugin_interface(self, checks: list[ValidationCheckResult]) -> bool:
        missing: list[str] = []
        for name in ("open",):
            if not callable(getattr(self.plugin, name, None)):
                missing.append(name)
        if missing:
            checks.append(
                ValidationCheckResult(
                    name="plugin_interface",
                    status="fail",
                    details=f"missing callables: {', '.join(missing)}",
                )
            )
            return False

        if not callable(getattr(self.plugin, "can_open", None)):
            checks.append(
                ValidationCheckResult(
                    name="plugin_interface",
                    status="warn",
                    details="missing can_open(path) helper",
                )
            )
            return True

        try:
            can_open = bool(self.plugin.can_open(self.dataset_path))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ValidationCheckResult(
                    name="plugin_interface",
                    status="warn",
                    details=f"can_open raised {exc}",
                )
            )
            return True

        status = "pass" if can_open else "warn"
        details = None if can_open else "can_open returned False"
        checks.append(ValidationCheckResult(name="plugin_interface", status=status, details=details))
        return True

    def _check_datasource(self, datasource: object, checks: list[ValidationCheckResult]) -> bool:
        required = ("probe", "list_episodes", "list_streams", "read_stream", "close")
        missing = [name for name in required if not callable(getattr(datasource, name, None))]
        if missing:
            checks.append(
                ValidationCheckResult(
                    name="datasource_protocol",
                    status="fail",
                    details=f"missing methods: {', '.join(missing)}",
                )
            )
            return False

        checks.append(ValidationCheckResult(name="datasource_protocol", status="pass"))
        return True

    def _safe_list_episodes(
        self,
        datasource: DataSource,
        checks: list[ValidationCheckResult],
    ) -> list[EpisodeRef]:
        try:
            episodes = datasource.list_episodes()
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ValidationCheckResult(
                    name="list_episodes",
                    status="fail",
                    details=f"list_episodes failed: {exc}",
                )
            )
            return []
        if not isinstance(episodes, list):
            checks.append(
                ValidationCheckResult(
                    name="list_episodes",
                    status="fail",
                    details="list_episodes did not return a list",
                )
            )
            return []
        if not episodes:
            checks.append(
                ValidationCheckResult(
                    name="list_episodes",
                    status="warn",
                    details="no episodes returned",
                )
            )
            return []
        checks.append(ValidationCheckResult(name="list_episodes", status="pass"))
        return episodes

    def _safe_list_streams(
        self,
        datasource: DataSource,
        episode_id: str,
        checks: list[ValidationCheckResult],
    ) -> list[StreamSpec]:
        try:
            streams = datasource.list_streams(episode_id)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ValidationCheckResult(
                    name="list_streams",
                    status="fail",
                    details=f"list_streams failed: {exc}",
                )
            )
            return []
        if not isinstance(streams, list):
            checks.append(
                ValidationCheckResult(
                    name="list_streams",
                    status="fail",
                    details="list_streams did not return a list",
                )
            )
            return []
        if not streams:
            checks.append(
                ValidationCheckResult(
                    name="list_streams",
                    status="warn",
                    details="no streams returned",
                )
            )
            return []
        checks.append(ValidationCheckResult(name="list_streams", status="pass"))
        return streams

    def _iterate_streams(
        self,
        datasource: DataSource,
        episode_id: str,
        streams: list[StreamSpec],
        checks: list[ValidationCheckResult],
    ) -> tuple[dict[str, dict], dict[str, float | None]]:
        sample_bundle: dict[str, dict] = {}
        ttf_metrics: dict[str, float | None] = {}
        errors: list[str] = []

        for stream in streams[: self.max_streams]:
            stream_id = self._stream_id(stream)
            start_time = perf_counter()
            try:
                iterator = iter(datasource.read_stream(episode_id, stream_id))
                first = next(iterator, None)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{stream_id}: {exc}")
                continue
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            ttf_metrics[stream_id] = elapsed_ms if first is not None else None
            if first is None:
                continue
            sample_bundle[stream_id] = {
                "spec": stream,
                "sample": first,
            }
            # Drain a few more samples to confirm iterability.
            remaining = 0
            for _ in iterator:
                remaining += 1
                if remaining >= self.max_samples_per_stream - 1:
                    break

        if errors:
            checks.append(
                ValidationCheckResult(
                    name="iterate_streams",
                    status="fail",
                    details="; ".join(errors),
                )
            )
            return sample_bundle, ttf_metrics

        if not sample_bundle:
            checks.append(
                ValidationCheckResult(
                    name="iterate_streams",
                    status="warn",
                    details="streams returned no samples",
                )
            )
        else:
            checks.append(ValidationCheckResult(name="iterate_streams", status="pass"))
        return sample_bundle, ttf_metrics

    def _check_rerun_archetypes(
        self,
        sample_bundle: dict[str, dict],
        checks: list[ValidationCheckResult],
    ) -> None:
        if not sample_bundle:
            checks.append(
                ValidationCheckResult(
                    name="rerun_archetypes",
                    status="skip",
                    details="no samples to validate",
                )
            )
            return

        failures: list[str] = []
        for stream_id, payload in sample_bundle.items():
            try:
                _build_rerun_archetype(payload["spec"], payload["sample"])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{stream_id}: {exc}")

        if failures:
            checks.append(
                ValidationCheckResult(
                    name="rerun_archetypes",
                    status="fail",
                    details="; ".join(failures),
                )
            )
        else:
            checks.append(ValidationCheckResult(name="rerun_archetypes", status="pass"))

    def _check_missing_data(
        self,
        datasource: DataSource,
        episode_id: str,
        streams: list[StreamSpec],
        checks: list[ValidationCheckResult],
    ) -> None:
        missing_id = "__missing_stream__"
        if streams:
            missing_id = f"{self._stream_id(streams[0])}__missing"
        try:
            iterator = iter(datasource.read_stream(episode_id, missing_id))
            next(iterator, None)
        except (KeyError, ValueError):
            checks.append(
                ValidationCheckResult(
                    name="missing_data",
                    status="pass",
                    details="missing stream handled with explicit error",
                )
            )
            return
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ValidationCheckResult(
                    name="missing_data",
                    status="fail",
                    details=f"unexpected error for missing stream: {exc}",
                )
            )
            return

        checks.append(
            ValidationCheckResult(
                name="missing_data",
                status="pass",
                details="missing stream returned empty or handled gracefully",
            )
        )

    def _select_episode_id(self, episodes: list[EpisodeRef]) -> str:
        if self.episode_id:
            return self.episode_id
        first = episodes[0]
        if isinstance(first, EpisodeRef):
            return first.episode_id
        if isinstance(first, dict):
            return str(first.get("episode_id") or first.get("id") or first.get("name"))
        return str(first)

    @staticmethod
    def _stream_id(stream: StreamSpec) -> str:
        if isinstance(stream, StreamSpec):
            return stream.stream_id
        if isinstance(stream, dict):
            return str(stream.get("stream_id") or stream.get("id") or stream.get("name"))
        return str(stream)


def _build_rerun_archetype(stream_spec: StreamSpec, sample: dict) -> object:
    import rerun as rr

    if "archetype" in sample:
        return sample["archetype"]
    if "rerun_archetype" in sample:
        return sample["rerun_archetype"]

    data = sample.get("data")
    if data is None:
        raise ValueError("sample missing data payload")

    if isinstance(data, str):
        return rr.TextDocument(data)

    array = np.asarray(data)
    suggested = None
    if isinstance(stream_spec, StreamSpec):
        suggested = stream_spec.suggested_archetype
    if isinstance(stream_spec, dict):
        suggested = stream_spec.get("suggested_archetype")

    mapping: dict[str, object] = {
        "Image": getattr(rr, "Image", None),
        "DepthImage": getattr(rr, "DepthImage", None),
        "SegmentationImage": getattr(rr, "SegmentationImage", None),
        "Points3D": getattr(rr, "Points3D", None),
        "Points2D": getattr(rr, "Points2D", None),
        "Scalars": getattr(rr, "Scalars", None),
        "Tensor": getattr(rr, "Tensor", None),
        "TextDocument": getattr(rr, "TextDocument", None),
    }
    archetype = mapping.get(suggested)
    if archetype is not None:
        return archetype(array)

    if array.ndim == 3 and array.shape[-1] in (3, 4) and array.dtype == np.uint8:
        return rr.Image(array)
    if array.ndim == 2:
        return rr.Tensor(array)
    if array.ndim <= 1:
        return rr.Scalars(array.ravel())
    return rr.Tensor(array)
