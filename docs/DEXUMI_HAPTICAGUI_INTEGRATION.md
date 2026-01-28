# DexUMI -> HapticaGUI Integration Guide

This guide explains how the DexUMI visualizer in this repo connects DexUMI Zarr
datasets to the HapticaGUIPlugin session flow and Rerun visualization.
It is grounded in:

- `src/dexumi_visualizer/cli.py`
- `src/dexumi_visualizer/gui_integration.py`
- `MODALITIES.md`

## Architecture at a glance

There are three cooperating layers:

1. **Dataset parsing** (Zarr + metadata)
   - `zarr_parser.discover_episodes()` and `episode_summary()` identify episodes
     and enumerate sensors (camera_*, numeric_*, standalone arrays, video-only mp4).
2. **Rerun logging** (per episode)
   - `_convert_episode()` in `cli.py` loads the episode, normalizes time, and
     emits Rerun entities for video, poses, scalars, and calibration points.
3. **HapticaGUI integration** (dataset browsing + sessions)
   - `DexUMIBrowserModel` in `gui_integration.py` scans datasets, provides
     preview metadata, and calls the session service at `/sessions`.
   - `dexumi-viz serve` starts a lightweight session service which launches
     Rerun viewers on demand.

You can use these pieces independently, or wire them together via the HTTP
session service to drive HapticaGUI.

## 1) Data flow: Zarr arrays -> Rerun entities

### Episode discovery and metadata
- `discover_episodes(path)` detects episode folders (e.g., `episode_*`,
  `reference_episode`) or any folder with Zarr groups / mp4 files.
- `episode_summary(path)` computes:
  - `duration_s` from the min/max of capture/receive times.
  - `total_frames` from the max frame count across sensors.
  - `sensors`: list of camera/numeric/array/video groups.

This summary is logged into Rerun as a static text blob at:
- `episode/summary` (via `rr.TextDocument`) in `_convert_episode()`.

### Time normalization
`_convert_episode()` calls `_compute_time_sync()` which:
- Scans `capture_time` or `receive_time` across camera + numeric groups.
- Infers a scale (sec/ms/us/ns) from magnitude.
- Builds a `TimeSync(origin, scale)` to normalize timestamps.

All frame logging uses:
- `rr.set_time("time", duration=<seconds>)` when timestamps exist.
- `rr.set_time("frame", sequence=<index>)` as fallback.

This means sensors can be logged in any order; the Rerun viewer aligns them
by timestamp.

### Sensor-to-entity mapping

#### Cameras and videos
For sensors with `kind in {camera, video}` and an `.mp4` file:

- **Frames**
  - Entity: `sensors/<sensor_name>/image`
  - Rerun type: `rr.Image(frame_rgb)`
  - Timestamping: capture/receive time if available; else frame index

- **Intrinsics** (if Zarr array `intrinsics` exists)
  - Entity: `sensors/<sensor_name>/image`
  - Rerun type: `rr.Pinhole(image_from_camera=K, width, height)`
  - If intrinsics are per-frame (shape `[T,3,3]`), they are logged at each
    time step. Otherwise logged as static.

- **Poses** (`pose` or `pose_interp` arrays)
  - Entity: `sensors/<sensor_name>` (primary) or
    `sensors/<sensor_name>/<pose_key>` (secondary)
  - Rerun type: `rr.Transform3D(mat3x3=R, translation=t)`

- **Extra arrays** (any other arrays inside the Zarr group)
  - Entity: `sensors/<sensor_name>/<array_name>`
  - Rerun type: `rr.Scalars(values.ravel())` or special handling (see below)

Video-only sensors (mp4 without Zarr group) still log image frames under the
same path, but omit intrinsics and pose.

#### Numeric groups and standalone arrays
Numeric sensors (`numeric_*`) are logged via `_log_numeric_sensor()`:

- For each array in the numeric group (except capture/receive time):
  - `sensors/<numeric_name>/<array_name>` with time from capture/receive time.

Standalone Zarr arrays (episode-level directories with a `.zarray` file) are
logged under `sensors/<array_name>`.

#### Specialized numeric interpretations
When `_log_array_sample()` sees specific array names it creates richer visuals:

- **Pose matrices** (`pose`, `pose_interp`, etc.)
  - Entity: `sensors/pose/<array_name>` or `sensors/<numeric>/<array_name>`
  - Rerun type: `rr.Transform3D`

- **Joint angles** (`*joint_angles*`)
  - Entity: `robot/joints/<array_name>/bar` (bar chart)
  - Entity: `robot/joints/<array_name>/joint_<i>` (scalar per joint)
  - Rerun types: `rr.BarChart`, `rr.Scalars`

- **FSR / tactile** (`*fsr*` or `*raw_voltage*`)
  - Entity: `sensors/fsr/<numeric_or_fsr>/<array_name>`
  - Rerun type: `rr.Points2D` (x=index, y=value) + `rr.Scalars` values
  - Colors are normalized against the max value across the array to highlight
    pressure intensity.

### Calibration points
Any `*.pkl` at the episode level or parent directory containing a dict with
`"points"` is logged as:
- Entity: `calibration/<pkl_name>`
- Rerun type: `rr.Points3D` (static)

This supports point prompts in DexUMI reference episodes.

## 2) Video stream handling (10 cameras per episode)

The current pipeline assumes each camera is represented by:
- A Zarr group `camera_<id>` with capture/receive times and optional intrinsics/pose.
- A matching mp4 file `camera_<id>.mp4` in the episode directory.

With 10 cameras per episode, the key behaviors are:

- **Independent decoding per camera**: `_log_video_frames()` creates its own
  `cv2.VideoCapture` and logs all frames for that camera before moving to the
  next. Rerun timestamps keep global alignment even when logs are not interleaved.

- **Timestamp sync**:
  - Primary: use `capture_time` / `receive_time` in the camera group.
  - Fallback: if missing, match a capture_time array from *any* sensor with the
    same frame count (`_find_matching_capture_times()`), else use frame index.

- **Intrinsics + resolution**: `rr.Pinhole` is logged if intrinsics exist.
  For per-frame intrinsics (`[T,3,3]`), logging scales with frame count and can
  be heavy when multiplied by 10 cameras.

### Recommended handling for 10 cameras
- Use stream filters to avoid loading all cameras at once (see Section 4).
- If per-frame intrinsics are available but static in practice, consider
  pre-collapsing them to one frame and log static intrinsics to reduce log size.
- If some cameras are auxiliary, log them in a separate session or convert them
  to `.rrd` files via `dexumi-viz convert` and load on demand.

## 3) Tactile/FSR sensor visualization

FSR channels show up as either:
- `fsr_values` / `fsr_values_interp` arrays in `numeric_*` groups, or
- `raw_voltage` arrays (treated as FSR in `_is_fsr_array`).

The visualizer logs FSR data as:

- **Scatter points**
  - Entity: `sensors/fsr/<numeric_or_fsr>/<array_name>`
  - X-axis: channel index
  - Y-axis: sensor value
  - Color: normalized intensity against global max

- **Scalar series**
  - Entity: `sensors/fsr/<...>/<array_name>/values`
  - Values: raw channel values

### Best practices for tactile visualization
- Use the scatter points for quick health checks (channels spiking, dropouts).
- If a grid or finger layout is known, consider replacing `Points2D` with a
  custom 2D layout for more intuitive spatial mapping.
- If the dataset contains interpolated FSR streams (`*_interp`), prefer those
  when synchronizing with video playback.

## 4) Session management for multiple episodes

### Session service (CLI)
`dexumi-viz serve <dataset_dir>` starts an HTTP service:

- `GET /health` -> `{ "status": "ok" }`
- `GET /episodes` -> list of episodes + paths
- `GET /episodes/<id>` -> detailed episode summary
- `POST /sessions` -> launches a Rerun viewer for the episode

`POST /sessions` expects JSON:

```
{
  "path": "/abs/path/to/episode",
  "streams": ["camera", "joint_angles", "fsr"],
  "metadata": {"episode": "...", "dataset": "..."}
}
```

The service spawns `_convert_episode()` in a background thread and returns:
`{ "id": "session-<short-id>", "status": "launching" }`.

### HapticaGUI integration model
`DexUMIBrowserModel` in `gui_integration.py` provides a Python-side path for
HapticaGUI plugins to:

1. Scan datasets with `scan_dexumi_datasets()`.
2. Select dataset and episode.
3. Toggle stream filters (camera/joint_angles/fsr/etc).
4. Call `open_in_rerun()` which POSTs to `/sessions`.

**Important gap:** The model references `/sessions/<id>/state`, but the
server in `cli.py` does not implement that endpoint. If you rely on session
state polling, you will need to extend the server or handle a missing endpoint
in the UI.

### Session strategies for multiple episodes
- **One viewer per episode**: the current server spawns a new Rerun session per
  POST. Useful for side-by-side comparison, but can be heavy for large episodes.
- **Pre-convert then load**: use `dexumi-viz convert` or `dexumi-viz batch` to
  create `.rrd` files, then load them on demand in Rerun without re-decoding.
- **Stream filtering**: send `streams` to limit logged data per session.
  The server uses `_sensor_matches_streams()` to include cameras, joint angles,
  FSR, pose, valid_indices, etc.

## 5) Best practices for large file handling

### Use stream filters aggressively
`_sensor_matches_streams()` allows subset selection by sensor name, kind, or
array tags such as `camera`, `joint_angles`, `fsr`, `pose`, `valid_indices`.
Send a reduced stream set in the `/sessions` payload to avoid logging everything
at once.

### Prefer batch conversion for repeated viewing
- `dexumi-viz convert <episode> -o episode.rrd`
- `dexumi-viz batch <dataset_dir> -o output_dir/`

This produces portable `.rrd` files and avoids repeated video decoding.

### Avoid loading full arrays in UI previews
`generate_preview_frames()` samples a small number of frames and uses only
simple statistics (mean of joints/FSR). Keep `DEXUMI_PREVIEW_SAMPLES` small
for large datasets.

### Exploit Zarr chunking
Large arrays are read lazily. When you add new summaries or statistics:
- Prefer chunked reads rather than `array[:]`.
- For global max values, `_max_in_zarr_array()` already iterates by chunk.

### Video decoding considerations
- Video decoding is fully sequential; for 10 cameras this is CPU-heavy.
- If you only need keyframes, add frame skipping logic or decode at lower FPS.
- Consider dedicated sessions for subsets of cameras.

### Timestamp normalization
DexUMI timestamps do not encode units. The visualizer infers scale by value
magnitude. If your dataset mixes units, normalize upstream or patch
`_infer_time_scale()` / `normalize_timestamps()` to prevent misalignment.

## Implementation checklist for a HapticaGUI plugin

- [ ] Use `scan_dexumi_datasets()` (local) or `/episodes` (remote) to list episodes.
- [ ] Render available streams from `extract_episode_metadata()['streams']`.
- [ ] POST to `/sessions` with `path` + `streams` to open Rerun.
- [ ] Handle missing `/sessions/<id>/state` endpoint gracefully.
- [ ] Use `.rrd` caching for repeat viewing and large episodes.

## Reference: data modalities (sample)

See `MODALITIES.md` for concrete examples of:
- camera_0/1 video + intrinsics + pose
- numeric joint_angles + raw_voltage
- reference episodes with fsr_values and point prompts
- replay episodes with interpolated arrays

These modalities map directly to the logging behavior described above.
