# Rerun Plugin Development Best Practices (Robotics Datasets)

This guide summarizes proven patterns for building Rerun plugins and loaders for robotics datasets. It focuses on the two extension paths (external loaders and Rust DataLoader implementations), blueprint-driven multi-modal layouts, time synchronization, efficient streaming, and viewer trade-offs.

## 1) DataLoader trait and how to implement it

Rerun routes file opening through the DataLoader system. All loaders are notified for every open, and a loader should return `DataLoaderError::Incompatible` quickly if it does not support the file. On native platforms, loaders execute in parallel. The trait supports both filesystem paths and in-memory contents (for drag-and-drop and web viewer). This behavior is documented in the Rust DataLoader trait reference. 
Sources: DataLoader trait (docs.rs). 

### Option A: External loader (fastest path, any language)

External loaders are executables named `rerun-loader-<something>` on your `PATH`. The Viewer/SDK calls them with a file path and a set of recommended CLI arguments (application id, recording id, entity path prefix, time args, and static flag). The loader must log to stdout (use the stdout sink in the SDK) and exit with the dedicated incompatible exit code if it does not support the file. 
Sources: Data loader overview, Operating modes (stdout). 

Minimal Python skeleton:

```python
#!/usr/bin/env python3
import argparse
import sys

import rerun as rr


def parse_kv_list(pairs):
    out = {}
    for item in pairs or []:
        name, value = item.split("=", 1)
        out[name] = int(value)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--application-id", required=True)
    ap.add_argument("--recording-id", required=True)
    ap.add_argument("--entity-path-prefix", default="")
    ap.add_argument("--time_sequence", action="append", default=[])
    ap.add_argument("--time_duration_nanos", action="append", default=[])
    ap.add_argument("--time_timestamp_nanos", action="append", default=[])
    ap.add_argument("--static", action="store_true")
    ap.add_argument("path")
    return ap.parse_args()


def join_path(prefix, suffix):
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}/{suffix}"


def main():
    args = parse_args()

    # Fast reject: not our file type.
    if not args.path.endswith(".mydataset"):
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    rr.init(args.application_id, recording_id=args.recording_id)
    rr.stdout()

    seq_times = parse_kv_list(args.time_sequence)
    ts_times = parse_kv_list(args.time_timestamp_nanos)
    dur_times = parse_kv_list(args.time_duration_nanos)

    for timeline, seq in seq_times.items():
        rr.set_time(timeline, sequence=seq)
    for timeline, ts_ns in ts_times.items():
        rr.set_time(timeline, timestamp=ts_ns)
    for timeline, dur_ns in dur_times.items():
        rr.set_time(timeline, duration=dur_ns)

    # Example logging
    entity_root = join_path(args.entity_path_prefix, "robot")
    rr.log(join_path(entity_root, "status"), rr.TextLog("loaded"), static=args.static)


if __name__ == "__main__":
    main()
```

Notes:
- Use the `--entity-path-prefix` argument to avoid collisions when multiple loaders contribute to the same recording.
- Honor the time arguments (`--time_sequence`, `--time_timestamp_nanos`, `--time_duration_nanos`) so the Viewer can align timelines for drag-and-drop and file open. 
- When the loader is not compatible, exit with the dedicated incompatible exit code; do not print errors to stdout. 

### Option B: Custom Rust DataLoader (in-process)

Implement the Rust `DataLoader` trait and register it in a custom Viewer/SDK build. This gives deep control and works on native and web (Wasm) if compiled appropriately. The trait provides both path-based and in-memory entry points and a `Sender<LoadedData>` to push data into the Viewer. 
Source: DataLoader trait (docs.rs).

Minimal Rust skeleton:

```rust
use re_data_loader::{DataLoader, DataLoaderError, DataLoaderName, DataLoaderSettings, LoadedData};
use std::{borrow::Cow, path::PathBuf, sync::mpsc::Sender};

struct MyLoader;

impl DataLoader for MyLoader {
    fn name(&self) -> DataLoaderName {
        "my_loader".into()
    }

    fn load_from_path(
        &self,
        _settings: &DataLoaderSettings,
        path: PathBuf,
        _tx: Sender<LoadedData>,
    ) -> Result<(), DataLoaderError> {
        if path.extension().and_then(|s| s.to_str()) != Some("mydataset") {
            return Err(DataLoaderError::Incompatible);
        }
        // Parse file and send LoadedData through tx.
        Ok(())
    }

    fn load_from_file_contents(
        &self,
        _settings: &DataLoaderSettings,
        _filepath: PathBuf,
        _contents: Cow<'_, [u8]>,
        _tx: Sender<LoadedData>,
    ) -> Result<(), DataLoaderError> {
        // Used for drag-and-drop and web viewer.
        Err(DataLoaderError::Incompatible)
    }
}
```

## 2) Blueprint customization for multi-modal data

Blueprints are just data, stored in a separate blueprint timeline, and can be generated programmatically or saved from the Viewer for reuse. Use them to pin an opinionated multi-modal layout (e.g., 3D world + camera feeds + plots + logs). 
Sources: Blueprints concept, build blueprints programmatically, blueprint API reference.

Example layout (Python):

```python
import rerun as rr
import rerun.blueprint as rrb

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(),
        rrb.Vertical(
            rrb.Spatial2DView(),
            rrb.Spatial2DView(),
        ),
    ),
    rrb.TimeSeriesView(),
    rrb.TextLogView(),
    rrb.TimePanel(state="collapsed", timeline="frame"),
)

rr.init("robot_dataset", spawn=True)
rr.send_blueprint(blueprint)
```

Tips for robotics datasets:
- Use stable entity paths per modality: `/world`, `/sensors/cam_left`, `/sensors/lidar`, `/state/imu`, `/logs`.
- Keep 3D + 2D views separated so dense point clouds do not dominate camera panels.
- Collapse the time panel by default if the dataset has a single driving timeline.

## 3) Timeline synchronization across modalities

Rerun always logs to `log_tick` and `log_time`. You can associate data with custom timelines using `set_time`. Rerun supports sequence, timestamp (ns since Unix epoch), and duration (ns) indices, and you can set multiple timelines for the same log call. 
Sources: Events and Timelines, timeline functions reference.

Example pattern:

```python
import rerun as rr

rr.init("robot_sync", spawn=True)

for frame in dataset:
    rr.set_time("frame", sequence=frame.idx)
    rr.set_time("sensor_time", timestamp=frame.ts_ns)

    rr.log("sensors/cam_left", rr.Image(frame.left_image))
    rr.log("sensors/lidar", rr.Points3D(frame.lidar_points))
    rr.log("state/imu", rr.Scalar(frame.imu_z))
```

Best practices:
- Pick a primary timeline for playback (often `frame` or `sensor_time`) and log all modalities against it.
- If your loader is triggered by file open or drag-and-drop, honor the CLI time arguments so the Viewer can align timelines.
- Use `static=True` for coordinate frames, static meshes, and annotation context that should not depend on time. 
Source: Events and Timelines (static data).

## 4) Memory-efficient streaming for large recordings

Key levers:

1) Batch your data. Most archetypes accept arrays, so log a single batch instead of per-point/per-box logging. This dramatically reduces overhead. 
Source: Component Batches.

2) Let micro-batching work for you. The SDK compacts small log calls into larger chunks; by default it flushes roughly every 200 ms when logging to file or 8 ms when streaming to the Viewer, or at about 1 MiB per batch. This reduces network and CPU overhead. 
Source: Optimize chunk count.

3) Pre-compact offline for repeated analysis. Use `rerun rrd stats` and `rerun rrd compact` on large recordings to reduce chunk counts and improve viewer performance. 
Source: Optimize chunk count.

Batch logging example:

```python
# Log the full point cloud as one batch
rr.log("sensors/lidar", rr.Points3D(points_xyz, colors=colors_rgb))
```

## 5) Web viewer vs native viewer trade-offs

- The native viewer is best for large datasets and heavy 3D scenes.
- The web viewer runs as 32-bit Wasm and is limited to roughly 2 GiB of memory in practice, so large recordings will hit limits sooner. 
Source: How does Rerun work (viewer section).

Loader implications:
- Drag-and-drop and web viewer use `load_from_file_contents`, which provides in-memory data and may not have filesystem access; the path is informational only. Design your loader to handle both path-based and content-based loading. 
Source: DataLoader trait (docs.rs).

## 6) Real-time vs batch logging patterns

Rerun supports streaming to a Viewer or saving to `.rrd` files. Choose a pattern based on latency and offline analysis needs. 
Source: How does Rerun work (streaming vs save), Operating Modes.

Real-time pattern (stream):

```python
import rerun as rr

rr.init("robot_rt")
rr.connect_grpc()  # or rr.spawn() for local viewer

for frame in live_stream:
    rr.set_time("frame", sequence=frame.idx)
    rr.log("sensors/cam_left", rr.Image(frame.left_image))
```

Batch pattern (save for later):

```python
import rerun as rr

rr.init("robot_offline")
rr.save("dataset.rrd")

for frame in dataset:
    rr.set_time("frame", sequence=frame.idx)
    rr.log("sensors/lidar", rr.Points3D(frame.lidar_points))
```

## Reference examples to study

- URDF loader (Python external loader example). 
- TFRecord loader (Python external loader example). 
- Data loader overview (external loader CLI parameters, stdout sink, incompatible exit code). 

## Sources

- https://docs.rs/re_data_loader/latest/re_data_loader/trait.DataLoader.html
- https://rerun.io/docs/reference/data-loaders/overview
- https://rerun.io/docs/reference/sdk-operating-modes
- https://rerun.io/docs/concepts/timelines
- https://ref.rerun.io/docs/python/main/common/timeline_functions/
- https://rerun.io/docs/howto/optimize-chunks
- https://rerun.io/docs/concepts/batches
- https://rerun.io/docs/concepts/app-model
- https://rerun.io/docs/concepts/blueprints
- https://rerun.io/docs/howto/build-a-blueprint-programmatically
- https://ref.rerun.io/docs/python/main/common/blueprint_apis/
- https://rerun.io/examples/robotics/urdf_loader
- https://rerun.io/examples/integrations/tfrecord_loader
