# DexUMI Visualizer

Visualize DexUMI dexterous manipulation datasets using Rerun and HapticaGUI.

## Features

- Load and parse DexUMI dataset formats (video, tactile, IMU, joint angles)
- Visualize synchronized multimodal data streams in Rerun
- Web-based dataset browser via HapticaGUI integration
- CLI for batch processing and conversion

## Data Sources

- **Sample Data**: https://real.stanford.edu/dexumi/sample_data.zip
- **Full Dataset**: https://umi-data.github.io/
- **DexUMI Paper**: https://arxiv.org/abs/2505.21864

## Installation

```bash
uv venv && uv pip install -e .
```

## Usage

```bash
# Visualize a dataset
dexumi-viz visualize data/sample_data/

# Convert to Rerun format
dexumi-viz convert data/sample_data/ -o output.rrd
```

## Integration with HapticaGUI

This visualizer integrates with the HapticaGUI dataset browser for web-based exploration.
