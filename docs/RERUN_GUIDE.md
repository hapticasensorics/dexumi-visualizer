# Rerun + Python: Complete Guide for Robotics Data Visualization

> **All agents should consult this guide** for Rerun best practices and DexUMI visualization patterns.

Below is a practical, end-to-end guide to Rerun (open-source) + Python, with a concrete "how to visualize DexUMI episodes" recipe at the end.

Rerun, in spirit, is "printf debugging for robotics/vision," except the printf is typed, time-aware, and comes with a viewer that can juggle images + 3D + plots without crying. You log structured data (images, point clouds, transforms, scalars, …) from your app, and Rerun can stream it live to the Viewer or save it to a file.

---

## 1) Install Rerun in a Python project

### Minimal install

Rerun's Python package is `rerun-sdk`.

```bash
pip install rerun-sdk
```

The current stable release on PyPI is 0.28.2, and it requires Python ≥ 3.10.

### Project dependency patterns (recommended)

If you're integrating into an existing codebase (training loop, robotics stack, etc.), treat Rerun as an optional dependency:

**pyproject.toml (extras)**

```toml
[project.optional-dependencies]
viz = ["rerun-sdk>=0.28.2"]
```

Then users can do:

```bash
pip install -e ".[viz]"
```

This keeps prod / deployment slim, while your debug tooling stays one pip-flag away.

---

## 2) The mental model that makes Rerun "click"

### Recordings + entity paths + archetypes

- You log to a **recording** (think: a time-indexed multimodal log).
- Data is organized by **entity paths** (hierarchical strings like `world/camera/image`).
- You log **archetypes** (typed payloads like `Image`, `Points3D`, `Transform3D`, `Scalars`, …).

### Time is not an afterthought

Rerun's viewer is basically a time machine:
- You choose a **timeline** (e.g. "frame", "time", "step"),
- Set time each iteration,
- Log data at that time.

Example pattern:

```python
rr.set_time("frame", sequence=frame_idx)
rr.log("entity/path", rr.SomeArchetype(...))
```

### Static vs temporal

Some data is best logged once (camera intrinsics, class labels, coordinate conventions) as **static**. Other data changes over time (frames, poses, force readings).

```python
rr.log("labels", rr.AnnotationContext([...]), static=True)
```

---

## 3) The 20-second "Hello Rerun" (3D points)

```python
import numpy as np
import rerun as rr

rr.init("points", spawn=True)   # starts a recording + opens the viewer
rr.log("random", rr.Points3D(np.random.rand(100, 3), colors=[255, 0, 0]))
```

This is essentially the canonical "I installed it correctly" test.

---

## 4) Timelines: frame, step, seconds (pick your poison)

### The canonical loop shape

```python
import rerun as rr

rr.init("my_app", spawn=True)

for frame_idx in range(1000):
    rr.set_time("frame", sequence=frame_idx)
    # rr.set_time("time", seconds=frame_idx / fps)  # optional second timeline
    rr.log("log/status", rr.TextLog(f"frame={frame_idx}"))
```

`TextLog` is useful for breadcrumbs ("what phase am I in?").

### High-performance logging for big timeseries

If you have lots of timeseries (joint angles, force sensors, etc.), looping `rr.log(...)` per timestep can be okay, but Rerun also supports sending whole columns at once via `rr.send_columns(...)`. The docs show exactly this for scalars.

---

## 5) Operating modes: live, saved, shareable

Rerun can:
- stream live to the Viewer
- and/or save to an `.rrd` file for later playback

### Save to .rrd (useful for CI artifacts + sharing)

```python
rr.init("my_app")
rr.save("output.rrd")
# ... log data ...
```

Rerun's "save" mode writes to disk; you can open saved files with the `rerun` CLI. The docs note that `.rrd` compatibility is only guaranteed across adjacent minor versions (e.g. 0.24 opens 0.23).

**In practice:** if you plan to archive `.rrd` files long-term, archive the viewer version too.

---

## 6) Logging the stuff you actually care about

### 6.1 Images (numpy)

```python
import rerun as rr
import numpy as np

rr.log("cam/front", rr.Image(np.zeros((480, 640, 3), dtype=np.uint8)))
```

### 6.2 Video streams: four approaches, different tradeoffs

Rerun's video docs lay out the menu:

1. **Uncompressed frames**: log many `Images` (simple, big).
2. **Compressed frames**: log many `EncodedImages` (smaller).
3. **MP4 container**: log one `AssetVideo` + per-frame `VideoFrameReference` (compact, more setup).
4. **Raw streaming chunks**: `VideoStream` (powerful, "here be dragons").

For most Python projects, start with option (1) or (2). Only go (3)/(4) when you're fighting bandwidth/storage.

### 6.3 3D coordinate frames + transforms (this is where Rerun shines)

Rerun has a full transform hierarchy model: entities live in a tree of coordinate frames.

Typical pattern:
- log a `Transform3D` at `world/camera`
- log camera intrinsics `Pinhole` at `world/camera`
- log images at `world/camera/image`

### 6.4 Plots / timeseries (Scalars + styling)

**Important gotcha:** modern Rerun uses plural archetypes: `Scalars`, `SeriesLines`, `SeriesPoints`, etc. (Older `Scalar` existed; it was deprecated.)

**Basic scalar over time (direct from docs):**

```python
import math
import rerun as rr

rr.init("scalar_demo", spawn=True)

for step in range(64):
    rr.set_time("step", sequence=step)
    rr.log("signals/sin", rr.Scalars(math.sin(step / 10.0)))
```

**Multiple series with styling:**

```python
import numpy as np
import rerun as rr
from math import sin, cos

rr.init("multi_series", spawn=True)

# Static styling: names/colors for the plotted line series
rr.log("trig/sin", rr.SeriesLines(colors=[255, 0, 0], names="sin"), static=True)
rr.log("trig/cos", rr.SeriesLines(colors=[0, 255, 0], names="cos"), static=True)

for t in range(500):
    rr.set_time("step", sequence=t)
    rr.log("trig/sin", rr.Scalars(sin(t / 50.0)))
    rr.log("trig/cos", rr.Scalars(cos(t / 50.0)))
```

This pattern is straight from the Scalars reference.

### 6.5 Segmentation masks (properly) with AnnotationContext

If you want segmentation masks to show up as colored labeled classes, use `SegmentationImage` + an `AnnotationContext`.

**Minimal working example from the docs:**

```python
import numpy as np
import rerun as rr

rr.init("seg_demo", spawn=True)

# A global annotation context mapping class ids -> (label, color)
rr.log("/", rr.AnnotationContext([
    (1, "hand", (255, 0, 0)),
    (2, "exo",  (0, 255, 0)),
]), static=True)

mask = np.zeros((200, 300), dtype=np.uint8)
mask[50:100, 50:120] = 1
mask[100:180, 130:280] = 2

rr.log("segmentation/mask", rr.SegmentationImage(mask))
```

---

## 7) Blueprints: make the viewer layout deterministic

Blueprints let you programmatically decide:
- which views exist (2D, 3D, time series, logs, …),
- what each view shows,
- how the UI is laid out.

They're first-class data: you can save/share them and generate them without manual UI clicking.

**A minimal "side by side" blueprint:**

```python
import rerun as rr
import rerun.blueprint as rrb

blueprint = rrb.Horizontal(
    rrb.Spatial3DView(origin="/world", contents=["/..."]),
    rrb.Spatial2DView(origin="/world/camera/image", contents=["/..."]),
    column_shares=[2, 1],
)

rr.send_blueprint(blueprint)
```

You don't need blueprints, but they're the difference between:
- "debugger vibes"
- and "repeatable instrumentation product."

---

## 8) Integrating Rerun into real Python projects (without turning your codebase into soup)

Here's the pattern I recommend (boring, reliable, easy to disable):

### 8.1 A tiny wrapper module

Create `myproj/viz/rerun_logger.py`:

```python
# myproj/viz/rerun_logger.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RerunConfig:
    enabled: bool = bool(int(os.environ.get("RERUN", "0")))
    app_id: str = os.environ.get("RERUN_APP", "myproj")
    spawn: bool = bool(int(os.environ.get("RERUN_SPAWN", "1")))

_rr_ready = False

def init(config: Optional[RerunConfig] = None) -> bool:
    global _rr_ready
    if _rr_ready:
        return True

    config = config or RerunConfig()
    if not config.enabled:
        return False

    import rerun as rr
    rr.init(config.app_id, spawn=config.spawn)
    _rr_ready = True
    return True

def rr():
    # late import so prod doesn't need rerun installed
    import rerun as rr
    return rr
```

**Usage anywhere else:**

```python
from myproj.viz import rerun_logger as rlog

if rlog.init():
    rr = rlog.rr()
    rr.log("log/status", rr.TextLog("hello from my training loop"))
```

### 8.2 If you want multiple recordings

Rerun exposes explicit recording streams (`rr.new_recording(...)`) for more control (multiple recordings, non-global usage).

---

## 9) Visualizing DexUMI with Rerun (the "make it real" section)

DexUMI's repo spells out the episode artifacts at each pipeline stage. In particular, the generated replay data includes videos like `dex_camera_0.mp4`, `exo_camera_0.mp4`, combined videos, segmentation mask folders, and numeric streams like `fsr_values_interp`, `joint_angles_interp`, `pose_interp`, etc.

### 9.1 What to visualize (high value / low effort)

For a single episode directory (post-generation), I'd log:

**Visual streams**
- `exo_camera_0.mp4` → `exo/cam0/rgb`
- `dex_camera_0.mp4` → `dex/cam0/rgb`
- `combined.mp4` or `debug_combined.mp4` → `combined/rgb`
- (optional) `maskout_baseline.mp4` → `baseline/rgb`

**Segmentation**
- `exo_seg_mask/*` → `exo/cam0/seg`
- `dex_seg_mask/*` → `dex/cam0/seg`
(and thumb/finger masks similarly)

Use `SegmentationImage` + `AnnotationContext` so class ids render as stable colors/labels.

**Numeric streams (plots)**
- `joint_angles_interp` → `signals/joint_angles`
- `hand_motor_value` → `signals/motor`
- `fsr_values_interp*` → `signals/fsr/*`
- `pose_interp` (wrist pose) → `signals/pose` (plot) and/or `world/wrist` (transform)

Use `Scalars` for time series; style via `SeriesLines` if you want names/colors.

---

### 9.2 A complete DexUMI episode visualizer script (robust to unknown file formats)

This is designed to be practical rather than "perfectly matched to DexUMI internals," because your exact mask filenames and numeric file extensions can vary.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

import rerun as rr
import rerun.blueprint as rrb


def sorted_files(pattern: str) -> list[str]:
    files = glob.glob(pattern)
    files.sort()
    return files


def try_load_array(stem: Path) -> Optional[np.ndarray]:
    """
    Try to load numeric arrays from a path stem.
    Handles:
      - stem.npy
      - stem.npz (first array)
      - stem.csv / stem.txt (via np.loadtxt)
    Returns None if nothing matches.
    """
    # Exact file
    if stem.exists() and stem.is_file():
        # best-effort
        try:
            return np.load(stem)
        except Exception:
            pass

    # Common extensions
    for ext in [".npy", ".npz", ".csv", ".txt"]:
        p = stem.with_suffix(ext)
        if not p.exists():
            continue

        if ext == ".npy":
            return np.load(p)
        if ext == ".npz":
            z = np.load(p)
            # take first array in archive
            for k in z.files:
                return z[k]
            return None
        if ext in [".csv", ".txt"]:
            return np.loadtxt(p, delimiter="," if ext == ".csv" else None)

    return None


def log_video_stream(entity_path: str, mp4_path: Path, timeline: str = "frame") -> float:
    """
    Decode an mp4 with OpenCV and log frames as rr.Image.
    Returns inferred FPS (or 0.0 if unknown).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        rr.set_time(timeline, sequence=frame_idx)

        # OpenCV gives BGR; Rerun expects RGB for typical image arrays
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log(entity_path, rr.Image(frame_rgb))

        frame_idx += 1

    cap.release()
    return fps


def log_image_sequence_as_segmentation(entity_path: str, folder: Path, timeline: str = "frame") -> None:
    """
    Log a folder of mask images (png/jpg/etc) as rr.SegmentationImage.

    Assumes: sorted order = time order.
    """
    files = sorted_files(str(folder / "*"))
    if not files:
        return

    for i, f in enumerate(files):
        rr.set_time(timeline, sequence=i)

        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # If mask is RGB, convert to single channel by taking first channel
        if img.ndim == 3:
            img = img[:, :, 0]

        rr.log(entity_path, rr.SegmentationImage(img.astype(np.uint8)))


def log_timeseries(entity_path: str, values: np.ndarray, timeline: str = "frame") -> None:
    """
    Log a 1D or 2D array as rr.Scalars over time.
    - values shape (T,) or (T, D)
    """
    values = np.asarray(values)

    # If it's a single vector (D,), treat it as one timestep (not that useful).
    if values.ndim == 1:
        T = values.shape[0]
        for t in range(T):
            rr.set_time(timeline, sequence=t)
            rr.log(entity_path, rr.Scalars(float(values[t])))
        return

    if values.ndim == 2:
        T = values.shape[0]
        for t in range(T):
            rr.set_time(timeline, sequence=t)
            rr.log(entity_path, rr.Scalars(values[t]))
        return

    # fallback
    rr.log(entity_path, rr.TextLog(f"Unsupported timeseries shape: {values.shape}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir", type=Path, help="Path to one DexUMI episode directory (generated replay data).")
    args = ap.parse_args()

    ep = args.episode_dir
    if not ep.exists():
        raise SystemExit(f"Episode dir not found: {ep}")

    rr.init(f"dexumi:{ep.name}", spawn=True)

    # Optional: deterministic layout (2D + plots)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(origin="/exo", contents=["/exo/**"]),
        rrb.Spatial2DView(origin="/dex", contents=["/dex/**"]),
        rrb.Spatial2DView(origin="/combined", contents=["/combined/**"]),
        rrb.TimeSeriesView(origin="/signals", contents=["/signals/**"]),
        column_shares=[1, 1, 1, 1],
    )
    rr.send_blueprint(blueprint)

    # Annotation context for segmentation ids (edit these IDs to match your dataset)
    rr.log(
        "/",
        rr.AnnotationContext([
            (0, "bg",   (0, 0, 0)),
            (1, "hand", (255, 0, 0)),
            (2, "exo",  (0, 255, 0)),
            (3, "obj",  (0, 0, 255)),
        ]),
        static=True,
    )

    # ---- Videos ----
    # These filenames come from the DexUMI README (generated episode structure).
    # You may not have all of them.
    video_map = {
        "exo_camera_0.mp4": "/exo/cam0/rgb",
        "dex_camera_0.mp4": "/dex/cam0/rgb",
        "combined.mp4": "/combined/rgb",
        "debug_combined.mp4": "/combined/debug",
        "maskout_baseline.mp4": "/baseline/rgb",
    }

    inferred_fps = 0.0
    for fname, path in video_map.items():
        p = ep / fname
        if p.exists():
            rr.log("/log", rr.TextLog(f"Logging video {fname} -> {path}"))
            fps = log_video_stream(path, p, timeline="frame")
            inferred_fps = inferred_fps or fps

    if inferred_fps:
        rr.log("/log", rr.TextLog(f"Inferred FPS: {inferred_fps:.2f}"))

    # ---- Segmentation folders ----
    seg_map = {
        "exo_seg_mask": "/exo/cam0/seg",
        "dex_seg_mask": "/dex/cam0/seg",
        "exo_thumb_seg_mask": "/exo/cam0/seg_thumb",
        "dex_thumb_seg_mask": "/dex/cam0/seg_thumb",
        "exo_finger_seg_mask": "/exo/cam0/seg_fingers",
        "dex_finger_seg_mask": "/dex/cam0/seg_fingers",
    }

    for folder_name, path in seg_map.items():
        folder = ep / folder_name
        if folder.exists() and folder.is_dir():
            rr.log("/log", rr.TextLog(f"Logging segmentation folder {folder_name} -> {path}"))
            log_image_sequence_as_segmentation(path, folder, timeline="frame")

    # ---- Numeric timeseries ----
    # These names appear in the DexUMI README; actual stored files may have extensions like .npy.
    numeric_stems = {
        "joint_angles_interp": "/signals/joint_angles",
        "hand_motor_value": "/signals/motor",
        "pose_interp": "/signals/pose",
        "fsr_values_interp": "/signals/fsr/all",
        "fsr_values_interp_1": "/signals/fsr/1",
        "fsr_values_interp_2": "/signals/fsr/2",
        "fsr_values_interp_3": "/signals/fsr/3",
    }

    for stem_name, entity_path in numeric_stems.items():
        arr = try_load_array(ep / stem_name)
        if arr is None:
            continue
        rr.log("/log", rr.TextLog(f"Logging {stem_name} shape={arr.shape} -> {entity_path}"))
        log_timeseries(entity_path, arr, timeline="frame")

    rr.log("/log", rr.TextLog("Done."))


if __name__ == "__main__":
    main()
```

### Why this script is aligned with Rerun + DexUMI's structure

- DexUMI's README lists the relevant episode artifacts (`dex_camera_0.mp4`, `exo_camera_0.mp4`, segmentation folders, `joint_angles_interp`, `pose_interp`, `fsr_values_interp*`, etc.).
- `SegmentationImage` + `AnnotationContext` is the intended way to render class-id masks with stable labels/colors.
- `Scalars` is the intended archetype for time series plots, and the docs show both per-step logging and the high-performance `send_columns` approach.
- Blueprint usage (`Spatial2DView`, `TimeSeriesView`, etc.) is the intended way to enforce a consistent layout.

---

### 9.3 What you'll probably want to tweak for "real" DexUMI

1. **The segmentation class IDs.**
   The script uses placeholder IDs (0 bg, 1 hand, 2 exo, 3 obj). Replace with whatever your masks actually encode.

2. **Pose interpretation.**
   If `pose_interp` is a 7-vector (xyz + quat) or 4×4 matrices, the next step is to log it as `Transform3D` under something like `world/wrist`, and optionally show axes via `TransformAxes3D`. (Transforms are a core concept in Rerun.)

3. **Video performance.**
   Decoding mp4 and logging raw frames works, but it's bandwidth-heavy. When you hit scale pain, move to Rerun's video options (`EncodedImage`, `AssetVideo` + `VideoFrameReference`, or `VideoStream`) depending on your constraints.

---

## 10) The "one level deeper" moves (when you're ready)

- **Send columns instead of per-frame loops for numeric arrays**
  This reduces Python overhead significantly for large timeseries.

- **Use the entity hierarchy intentionally**
  Example: put all episode data under `/episode_0001/...` so you can compare multiple episodes in one recording.

- **Log your model outputs alongside DexUMI signals**
  For debugging policies, log:
  - action outputs as `Scalars`
  - value estimates / confidence as `Scalars`
  - intermediate embeddings as `Tensor`
  and time-align them to the same "frame" or "step" timeline.

- **Blueprint your "debug cockpit"**
  One "golden layout" per debugging scenario is unbelievably effective at keeping teams sane.

---

## Troubleshooting

If you implement the DexUMI visualizer and it loads everything except your numeric files, that's almost always just "extension / serialization mismatch" (e.g., `.npy` vs `.npz` vs raw binary). The wrapper `try_load_array()` is where you'd adapt it to your exact storage format, and the rest stays stable.

---

## Quick Reference

| Data Type | Archetype | Entity Path Convention |
|-----------|-----------|----------------------|
| RGB Image | `rr.Image` | `/camera_name/rgb` |
| Segmentation Mask | `rr.SegmentationImage` | `/camera_name/seg` |
| Time Series | `rr.Scalars` | `/signals/sensor_name` |
| 3D Points | `rr.Points3D` | `/world/point_cloud` |
| Transform | `rr.Transform3D` | `/world/frame_name` |
| Camera Intrinsics | `rr.Pinhole` | `/camera_name` |
| Text Log | `rr.TextLog` | `/log` or `/debug` |
| Video Asset | `rr.AssetVideo` | `/video/name` |

---

## Related Documentation

- [Rerun Python Quick Start](https://rerun.io/docs/getting-started/quick-start/python)
- [Rerun Archetypes Reference](https://rerun.io/docs/reference/types)
- [Rerun Blueprints Guide](https://rerun.io/docs/concepts/blueprint)
- [DexUMI Dataset](https://dex-umi.github.io/)
