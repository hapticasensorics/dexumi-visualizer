# Tactile Dataset Standardization Patterns

## Scope and goals
- Provide a consistent vocabulary for tactile modalities and data products.
- Make datasets interoperable for visualization (Rerun), learning, and fusion with vision and proprioception.
- Preserve raw data and calibration metadata so downstream users can reprocess or re-normalize.

## 1. Common tactile modalities
- Capacitive: measures capacitance change from deformation; often grid taxels or sparse channels. Raw values are capacitance or ADC counts; sensitive to drift and humidity.
- Resistive (FSR): resistance changes under load; commonly 1D channels or sparse grids; nonlinear response and hysteresis.
- Piezoresistive: resistance changes in piezoresistive materials; commonly pressure maps; often needs per-taxel calibration for linearization.
- Optical (vision-based, e.g., GelSight/DIGIT): camera observes gel deformation; outputs RGB or grayscale images; derived maps (height, normal, contact) are computed via calibration.

## 2. Data formats across datasets (examples)

### OpenTouch (full-hand tactile + egocentric vision)
- Modalities: synchronized egocentric video, full-hand tactile maps, and hand-pose trajectories; about 5 hours total with a 3-hour annotated subset. [1]
- Time sync: hardware synchronized at 30 Hz with reported ~2 ms latency. [1]
- Release status: project page lists code and dataset as "coming", so file-level format is not yet public. [1]

### DexUMI (this repo sample)
- Modalities: mp4 videos, Zarr v2 arrays (.zgroup/.zarray) for numeric streams, and .pkl point prompts.
- Timing: per-stream capture_time and receive_time arrays; interpolated streams (_interp) for synchronized replay.
- Numeric: joint angles, raw voltage/FSR streams, and pose arrays.

### DIGIT (optical fingertip sensor)
- YCB-Slide provides DIGIT images, sensor poses, RGB video feed, and ground-truth mesh models; simulation includes ground-truth heightmaps and contact masks. [3]
- YCB-Slide data directories include frames/, webcam_frames/, digit_data.npy, synced_data.npy, and related visualization artifacts. [3]
- Sparsh DIGIT pretraining data (digitv1) uses per-object dataset_*.pkl plus background images in bgs/. [2]

### GelSight (optical fingertip sensor)
- Sparsh GelSight pretraining data stores Touch and Go and ObjectFolder-Real sequences as .pkl files under gelsight/touch_go and gelsight/object_folder. [2]
- TactileTracking (GelSight Mini) includes gelsight.avi, webcam.avi, pose arrays, contact_masks.npy, and gradient_maps.npy. [4]

## 3. Sensor geometry and spatial mapping
Standardize geometry so tactile values can be placed in a consistent 2D or 3D frame.

Recommended geometry metadata:
- sensor_frame: name and axis convention (right-handed, units in meters).
- taxel_positions_m: Nx3 array of taxel centers in sensor_frame.
- taxel_normals: Nx3 normals (optional for curved surfaces).
- taxel_areas_m2: N array (optional but helpful for force integration).
- grid_shape: [H, W] and spacing_m for regular grids.
- pixel_to_taxel_map: mapping from optical pixels to taxel index (if derived).

Example:
- OpenTouch provides full-hand tactile maps, so mapping to a hand surface or mesh is necessary for 3D visualization and cross-modal alignment. [1]

## 4. Time synchronization with visual and proprioceptive data
Standardize timestamping and alignment so multi-modal fusion is reliable.

Recommended timing metadata:
- capture_time: sensor capture timestamp per sample or frame.
- receive_time: host receive timestamp per sample or frame.
- clock_domain: sensor_clock or host_clock plus offset and scale info.
- sync_events: explicit sync markers (visual flashes, trigger pulses).
- resampled streams: store *_interp (or equivalent) when resampling onto a shared timeline.

Examples:
- OpenTouch synchronizes tactile, vision, and pose at 30 Hz with low latency. [1]
- YCB-Slide uses OptiTrack for time-synced sensor poses and provides aligned data arrays. [3]
- DexUMI uses capture_time and receive_time arrays and provides interpolated streams for replay alignment.

## 5. Normalization and calibration standards
Keep raw data and add calibrated or normalized products with explicit metadata.

Baseline recommendations:
- Always store raw values alongside calibrated values.
- Record per-session calibration data, temperature, and material parameters.
- Track sensor configuration (gel hardness, marker pattern, camera intrinsics).

Optical sensors:
- Provide background images and illumination normalization (flat-fielding).
- Provide calibration artifacts (ball and cube presses) to map RGB to surface gradients or height. [8]

Resistive, capacitive, and piezoresistive sensors:
- Provide per-taxel calibration curves (ADC or voltage to force or pressure).
- Store linearization coefficients and hysteresis compensation where available.
- Include noise floor and saturation thresholds for masking.

Normalization outputs (recommended):
- pressure_pa (calibrated pressure map)
- force_n (integrated normal force)
- contact_mask (binary or probability)
- height_m (optical depth or height map)
- normal_map (surface normals)

## 6. Proposed unified schema for tactile data in Rerun
Use Rerun archetypes to make datasets comparable and easy to visualize.

Entity layout (example):
- /tactile/<sensor_id>/
  - extrinsics (Transform3D) [14]
  - intrinsics (Pinhole, for optical) [13]
  - geometry/taxels (Points3D) [12]
  - raw/image (Image) [9]
  - raw/pressure (Tensor) [11]
  - calibrated/pressure (Tensor) [11]
  - derived/height (DepthImage) [10]
  - derived/normal (Tensor) [11]
  - derived/contact_mask (SegmentationImage) [15]
  - signals/force (Scalars) [16]

Schema notes:
- Use Image for RGB tactile images from optical sensors. [9]
- Use DepthImage for heightmaps (meter scale explicitly provided). [10]
- Use Tensor for grid maps (pressure or normal) or 1D taxel arrays with dim_names. [11]
- Use Points3D for taxel center locations or point clouds in sensor_frame. [12]
- Use Transform3D to place the sensor in world or hand frames. [14]
- Use Pinhole for camera intrinsics when optical sensors are used. [13]
- Use SegmentationImage for contact masks. [15]
- Use Scalars for single-channel time series (force, temperature). [16]

Recommended Tensor dim_names:
- Grid sensors: ["height", "width"] or ["height", "width", "channel"]
- Taxel arrays: ["taxel"] or ["taxel", "channel"]
- Time series: ["time", "taxel"] if pre-batched (otherwise log per-timestep)

## 7. Foundation model data requirements (Sparsh, UniT, and others)

Sparsh:
- Pretraining uses large-scale unlabeled tactile images from DIGIT and GelSight sensors.
- DIGIT pretraining mixes YCB-Slide and Touch-Slide; digitv1 uses per-object dataset_*.pkl plus background images. [2]
- GelSight pretraining uses Touch and Go and ObjectFolder-Real, stored as .pkl files. [2]

UniT:
- Representation learning uses tactile images collected from a single simple object. [6]
- The released datasets contain 200, 180, and 170 demonstrations for three tasks. [6]
- Pretrained models assume GelSight Mini with markers and a specific crop; other sensors or crops require retraining or matching the crop. [6]

AnyTouch:
- TacQuad is a multi-sensor aligned dataset with GelSight Mini, DIGIT, DuraGel, and Tac3D, including fine-grained spatiotemporal alignment and coarse spatial alignment. [5]

Transferable Tactile Transformers (T3):
- FoTa aggregates multiple datasets into a unified WebDataset format, covering 13 sensors and 11 tasks. [7]

SITR:
- Uses calibration images (ball and cube presses) to characterize sensor-specific artifacts. [8]
- Trains on a large synthetic dataset spanning many sensor configurations using PBR. [8]

## References
[1] OpenTouch project page: https://opentouch-tactile.github.io/
[2] Sparsh repository (datasets section): https://github.com/facebookresearch/sparsh
[3] YCB-Slide dataset README: https://github.com/rpl-cmu/YCB-Slide
[4] TactileTracking dataset card: https://huggingface.co/datasets/joehjhuang/TactileTracking
[5] AnyTouch repository (TacQuad details): https://github.com/GeWu-Lab/AnyTouch
[6] UniT repository: https://github.com/ZhengtongXu/UniT
[7] T3 project page (FoTa details): https://t3.alanz.info/
[8] SITR project page: https://hgupt3.github.io/sitr/
[9] Rerun Image archetype: https://rerun.io/docs/reference/types/archetypes/image
[10] Rerun DepthImage archetype: https://rerun.io/docs/reference/types/archetypes/depth_image
[11] Rerun Tensor archetype: https://rerun.io/docs/reference/types/archetypes/tensor
[12] Rerun Points3D archetype: https://rerun.io/docs/reference/types/archetypes/points3d
[13] Rerun Pinhole archetype: https://rerun.io/docs/reference/types/archetypes/pinhole
[14] Rerun Transform3D archetype: https://rerun.io/docs/reference/types/archetypes/transform3d
[15] Rerun SegmentationImage archetype: https://rerun.io/docs/reference/types/archetypes/segmentation_image
[16] Rerun Scalars archetype: https://rerun.io/docs/reference/types/archetypes/scalars
