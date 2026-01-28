# Generalized Dataset Ingestion Architecture (HapticaGUI + Rerun)

## Goals
- Ingest any robotics or tactile dataset through a consistent pipeline.
- Separate **storage formats** from **semantic meaning** so new datasets can be added without rewriting the GUI or visualization logic.
- Auto-map common modalities to Rerun archetypes while allowing overrides.
- Preserve provenance, calibration, and time alignment in a structured manifest.

## Standards Snapshot (inputs to the design)
- **RLDS (Reinforcement Learning Datasets)** defines datasets as *episodes* composed of a *sequence of steps*, with optional episode-level metadata for identification and context.
- **LeRobot v3** uses a file-based layout: tabular data in Parquet, visuals in MP4, and metadata in `meta/` (info, stats, tasks, episodes) to describe datasets and episode boundaries.
- **Open X-Embodiment** aggregates robotics data as a *sequence of RLDS episodes* for cross-robot generalization.

These ideas suggest a canonical ingestion model that is **episode- and stream-centric** while remaining format-agnostic.

## Unified Canonical Model

**Dataset -> Episodes -> Streams -> Samples**

- **Dataset**: registry-level descriptor (name, version, license, contact, etc.).
- **Episode**: a temporally bounded slice with a stable timeline and optional task labels.
- **Stream**: a time-aligned sequence of samples for a single modality (image, tactile grid, pose, joint state, etc.).
- **Sample**: data payload plus timestamp and optional frame/transform.

This model is storage-neutral and can represent RLDS, LeRobot, Open X-Embodiment, ROS bags, Zarr, HDF5, and custom layouts.

## 1. Abstract Data Source Interface

Define a **DataSource** contract that any loader must implement. The GUI never touches raw files directly; it only speaks DataSource.

```python
from dataclasses import dataclass
from typing import Iterable, Protocol

@dataclass
class EpisodeRef:
    episode_id: str
    start_time_ns: int | None
    end_time_ns: int | None
    tags: dict[str, str]

@dataclass
class StreamSpec:
    stream_id: str
    name: str
    modality: str | None           # e.g. "image", "depth", "tactile"
    dtype: str                     # numpy-style dtype string
    shape: tuple[int, ...]         # sample shape (not including time)
    time_key: str                  # "timestamp", "index", etc
    sample_rate_hz: float | None
    units: str | None              # e.g. "rad", "N", "mm"
    frame_id: str | None           # coordinate frame name
    calibration: dict | None       # intrinsics, extrinsics, etc
    suggested_archetype: str | None

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
```

**Key principles**
- Provide **random access** at the stream level (chunked reads for large arrays).
- Support **lazy decoding** (load video frames on demand; decode MCAP messages lazily).
- Attach **calibration and frame info** so the downstream Rerun logging can be spatially consistent.

## 2. Modality Detection and Auto-Mapping to Rerun Archetypes

### Detection Pipeline
1. **Explicit metadata** from the manifest or source-specific attributes wins.
2. **Name-based heuristics** (e.g., `rgb`, `depth`, `seg`, `joint`, `pose`).
3. **Shape/dtype inference** for unknown streams.
4. **Fallback** to generic tensor/time-series if still ambiguous.

### Heuristic Rules (examples)
- `uint8` `(H, W, 3|4)` -> RGB(A) image
- `float32` `(H, W)` + name contains `depth` -> depth image
- integer `(H, W)` + name contains `seg` or `mask` -> segmentation
- `(N, 3)` or `(N, 6)` -> point cloud (XYZ or XYZ+RGB)
- `(4, 4)` or `(7,)` -> pose / transform
- `(J,)` or `(T, J)` where J is joint count -> joint states
- `(C, H, W)` for tactile grids or pressure maps

### Default Mapping to Rerun Archetypes
(Overrideable per stream in the manifest.)

| Modality | Rerun Archetype | Notes |
|---|---|---|
| RGB/RGBA image | `Image` | Visual cameras |
| Depth image | `DepthImage` | Depth in meters or mm |
| Segmentation mask | `SegmentationImage` | Class or instance IDs |
| Camera intrinsics | `Pinhole` | `fx, fy, cx, cy` |
| Pose / extrinsics | `Transform3D` | World <-> sensor frames |
| Point cloud | `Points3D` | XYZ (+ color/intensity) |
| 2D/3D trajectories | `LineStrips2D/3D` | Paths or polylines |
| 2D keypoints | `Points2D` | Finger/keypoint tracking |
| 3D keypoints | `Points3D` | Skeleton or fingertip tracks |
| Scalars / time-series | `Scalars` or `SeriesLines` | Tactile channels, torques |
| Generic tensor | `Tensor` | Fallback for arbitrary arrays |
| Text annotations | `TextLog` / `TextDocument` | Human labels or comments |
| Videos | `VideoFrameReference` / `VideoStream` | Prefer time-aligned frame refs |

### MCAP / ROS Bags
If the dataset is a ROS bag or MCAP, prefer using Rerun's MCAP data loader and the ROS2 message conversion pipeline when available, then fall back to DataSource-based decoding.

## 3. Metadata Schema for Dataset Registration

A dataset is registered by a **manifest** (YAML/JSON). It can live at the dataset root or be generated on the fly.

```yaml
schema_version: 1
id: haptica.sample
name: Sample Dataset
version: 0.1.0
license: CC-BY-4.0
source:
  organization: Example Lab
  contact: research@example.org
  citation: "Doe et al. 2025"
format:
  primary: zarr           # zarr | hdf5 | rlds | lerobot | mcap | rosbag
  storage: filesystem
  location: /data/sample

timebase:
  type: timestamp
  units: ns
  epoch: unix

coordinate_frames:
  - id: world
    parent: null
    transform: null
  - id: wrist_cam
    parent: world
    transform: {type: rigid, units: m}

episodes:
  index:
    kind: directory
    pattern: "episode_*"
  fields:
    episode_id: "${dirname}"
    task: "${metadata.task_name}"

streams:
  - id: cam_rgb
    name: camera_0
    modality: image
    shape: [H, W, 3]
    dtype: uint8
    time_key: capture_time
    sample_rate_hz: 30
    frame_id: wrist_cam
    calibration:
      intrinsics: {fx: 600, fy: 600, cx: 320, cy: 240}
    rerun:
      archetype: Image

  - id: wrist_pose
    name: camera_1/pose
    modality: pose
    shape: [4, 4]
    dtype: float64
    time_key: capture_time
    frame_id: world
    rerun:
      archetype: Transform3D

  - id: fsr
    name: numeric_1/fsr_values
    modality: tactile
    shape: [3]
    dtype: int64
    time_key: capture_time
    units: "N"
    rerun:
      archetype: Scalars
```

**Notes**
- Aligns with RLDS-style episode metadata (episode IDs, tasks, environment context).
- Mirrors LeRobot's separation of data files and metadata in `meta/` (stats, episodes, tasks).
- Allows **stream-level overrides** so ambiguous modalities can be explicitly mapped.

## 4. Plugin System for Custom Dataset Loaders

Support three tiers of loaders:
1. **Built-ins**: Zarr, HDF5, RLDS, LeRobot, MCAP/ROS bag.
2. **External loaders**: discoverable executables (e.g., `rerun-loader-*`) that can translate a dataset into a live Rerun session.
3. **Project-specific plugins**: Python entry points implementing `DataSource`.

**Suggested plugin contract**
- `can_open(path) -> bool`
- `open(path) -> DataSource`
- `describe() -> LoaderInfo` (supported extensions, priority, modality hints)

**Discovery strategy**
- Scan Python entry points (`haptica.loaders`).
- Scan PATH for `rerun-loader-*` to reuse Rerun's external loader ecosystem.
- Allow explicit configuration in a GUI settings file.

## 5. Example Implementations for Common Formats

### Zarr
- Treat **groups** as dataset/episode boundaries.
- Treat **arrays** as streams; attributes provide modality hints.
- Use chunked reads for large arrays and time slicing.

### HDF5
- Map **groups** to episodes and **datasets** to streams.
- Leverage dataset attributes for units, frames, and calibration.

### RLDS (TFDS)
- Use `tfds.builder` to enumerate episodes.
- Map `steps` to time-ordered samples per stream (observation, action, reward).
- Carry episode metadata (task, environment, etc.) into the manifest.

### LeRobot v3
- Parse `meta/info.json`, `meta/episodes.*`, `meta/tasks.*`.
- Load tabular signals from Parquet (actions, states, timestamps).
- Load videos from MP4 and align with the tabular timeline.

### Open X-Embodiment
- Treat each RLDS episode as a Haptica episode.
- Use per-dataset configs to normalize field names into canonical streams.

### ROS bag / MCAP
- Prefer Rerun's MCAP data loader for ROS2 messages when available.
- For raw rosbag2, use storage plugins (mcap/sqlite) and decode messages into streams.

## Operational Flow (HapticaGUI + Rerun)
1. **Dataset registration**: user drops a dataset folder; the loader probes and writes a manifest.
2. **Episode browsing**: GUI lists episodes from `DataSource.list_episodes()`.
3. **Stream preview**: GUI shows modality hints, sample rate, and inferred archetypes.
4. **Rerun launch**: GUI calls a `RerunExporter` that maps streams to archetypes and logs them.
5. **Overrides**: user can edit the manifest to adjust mappings or calibrations.

## References
- RLDS (Reinforcement Learning Datasets): https://github.com/google-research/rlds
- LeRobot Dataset v3 format: https://huggingface.co/docs/lerobot/en/dataset
- Open X-Embodiment: https://github.com/google-deepmind/open_x_embodiment
- Rerun DataLoaders: https://rerun.io/docs/reference/data-loaders/overview
- Rerun Archetypes: https://rerun.io/docs/reference/types/archetypes
- Rerun MCAP support: https://rerun.io/docs/howto/mcap
- Rerun ROS2 message formats: https://rerun.io/docs/reference/data-loaders/ros2msg/supported-formats
- MCAP format: https://mcap.dev
- rosbag2 storage plugins: https://github.com/ros2/rosbag2
- Zarr format: https://zarr.dev
- Zarr documentation: https://zarr.readthedocs.io
- HDF5 groups and datasets: https://portal.hdfgroup.org/display/HDF5/Groups+in+HDF5
