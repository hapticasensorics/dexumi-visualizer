# DexUMI Visualizer

Visualize DexUMI dexterous manipulation datasets using Rerun and HapticaGUI.

## Features

- Load and parse DexUMI dataset formats (video, tactile, IMU, joint angles)
- Visualize synchronized multimodal data streams in Rerun
- Web-based dataset browser via HapticaGUI integration
- CLI for batch processing and conversion
- 10 camera streams per episode with full video support
- FSR tactile sensor visualization with color gradients

## Documentation

- **[Rerun Guide](docs/RERUN_GUIDE.md)** - Complete Rerun + Python guide with DexUMI patterns
- **[Modalities Reference](docs/MODALITIES.md)** - All DexUMI data streams documented
- **[Agent Reference](docs/AGENTS_README.md)** - Integration architecture and patterns

## Data Sources

- **Sample Data**: https://real.stanford.edu/dexumi/sample_data.zip (519MB, 5 episodes)
- **Full Dataset**: https://umi-data.github.io/
- **DexUMI Paper**: https://arxiv.org/abs/2505.21864

## Installation

```bash
uv venv && uv pip install -e .
```

## Usage

```bash
# List available episodes
dexumi-viz list data/sample_data/

# Show episode info (cameras, sensors, shapes)
dexumi-viz info data/sample_data/software_go_through/episode_0

# Convert single episode to .rrd
dexumi-viz convert data/sample_data/software_go_through/episode_0 -o episode.rrd

# Batch convert all episodes
dexumi-viz batch data/sample_data/ -o output/

# View in Rerun (native viewer)
dexumi-viz view output/software_go_through/episode_0.rrd

# Serve via web viewer
dexumi-viz serve output/software_go_through/episode_0.rrd --port 9090
```

## Output Structure

After batch conversion:
```
output/
├── software_go_through/
│   ├── episode_0.rrd (2.0GB with video)
│   └── episode_1.rrd (1.8GB with video)
├── software_go_through_replay/
│   ├── episode_0.rrd
│   └── episode_1.rrd
└── xhand_reference_5_14/
    └── reference_episode.rrd
```

## Integration with HapticaGUI

This visualizer integrates with the HapticaGUI dataset browser for web-based exploration.

```
HapticaGUIPlugin → Session Service → Rerun Viewer
       ↓                 ↓
  Browse datasets   Spawn viewers
```

See [docs/AGENTS_README.md](docs/AGENTS_README.md) for the full integration architecture.
