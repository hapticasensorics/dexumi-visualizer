# Agent Reference Guide

**All agents working on this repository MUST consult these resources:**

## Required Reading

1. **[RERUN_GUIDE.md](./RERUN_GUIDE.md)** - Complete Rerun + Python guide with DexUMI visualization patterns
   - Installation patterns
   - Mental model (recordings, entity paths, archetypes)
   - Timelines and time management
   - All archetype usage (Images, Scalars, Segmentation, Transforms)
   - Blueprint configuration
   - Complete DexUMI episode visualizer script

2. **[MODALITIES.md](./MODALITIES.md)** - DexUMI dataset modality documentation
   - All video streams (10 cameras per episode)
   - Zarr array specifications
   - Pickle file formats
   - Data shapes and dtypes

## Key Patterns to Follow

### Entity Path Conventions
```
/video/{camera_name}/rgb     - RGB video frames
/video/{camera_name}/seg     - Segmentation masks
/signals/{sensor_name}       - Time series data
/world/{frame_name}          - 3D transforms
/log                         - Debug text logs
```

### Archetype Usage
| Data Type | Archetype | Notes |
|-----------|-----------|-------|
| RGB frames | `rr.Image` | Convert BGR→RGB from OpenCV |
| Segmentation | `rr.SegmentationImage` | Requires `AnnotationContext` |
| Time series | `rr.Scalars` | Use `SeriesLines` for styling |
| 3D points | `rr.Points3D` | Supports colors, radii |
| Transforms | `rr.Transform3D` | For camera/world poses |

### Timeline Management
```python
rr.set_time("frame", sequence=frame_idx)  # Integer timeline
rr.set_time("time", seconds=timestamp)     # Float timeline
```

### Static vs Temporal
```python
# Static (logged once)
rr.log("labels", rr.AnnotationContext([...]), static=True)

# Temporal (logged per frame)
rr.set_time("frame", sequence=i)
rr.log("camera/rgb", rr.Image(frame))
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HapticaGUIPlugin                         │
│  (Reflex web UI for browsing/selecting datasets)            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP POST /sessions
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Session Service                           │
│  (FastAPI server managing Rerun viewer instances)           │
└─────────────────────┬───────────────────────────────────────┘
                      │ Spawns
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Rerun Viewer                              │
│  (Native or web viewer displaying .rrd recordings)          │
└─────────────────────────────────────────────────────────────┘
                      ▲
                      │ Loads
┌─────────────────────────────────────────────────────────────┐
│                   dexumi-visualizer                         │
│  (CLI tool: list, info, convert, batch, serve)              │
│  Converts DexUMI Zarr/MP4 → .rrd files                      │
└─────────────────────────────────────────────────────────────┘
```

## Common Pitfalls

1. **Don't use deprecated `Scalar`** - Use `Scalars` (plural)
2. **Don't forget `AnnotationContext`** - Required for SegmentationImage colors
3. **Don't ignore BGR→RGB conversion** - OpenCV uses BGR, Rerun expects RGB
4. **Don't create massive .rrd files** - Use blueprints to limit what's logged
5. **Don't mix timeline types** - Use either sequence or seconds consistently

## File Structure

```
dexumi-visualizer/
├── src/dexumi_visualizer/
│   ├── cli.py              # Main CLI entry point
│   ├── zarr_parser.py      # Episode discovery
│   ├── rerun_loader.py     # Streaming export
│   └── gui_integration.py  # HapticaGUI web integration
├── docs/
│   ├── RERUN_GUIDE.md      # ← START HERE
│   ├── MODALITIES.md       # Data format reference
│   └── AGENTS_README.md    # This file
└── output/                 # Converted .rrd files
```
