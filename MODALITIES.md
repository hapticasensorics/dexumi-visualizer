# DexUMI Modalities (sample_data)

Scope: This document enumerates every data stream present under
`data/sample_data/` in this repo. It is a sample subset of the DexUMI dataset
and follows the same naming conventions described in DexUMI docs (data_raw,
data_replay, reference/point prompts).

## File types present
- `.mp4`: video streams.
- `.pkl`: point-prompt data (labeled 2D points for segmentation/inpainting).
- `.py`: constants used for scaling/adjustments.
- `.zgroup`, `.zarray`: Zarr v2 metadata files.
- Zarr chunk files (no extension, e.g., `0`, `0.0`).

## Dataset collections in sample_data

### software_go_through (exo demonstration; data_raw-style)
- `camera_0.mp4`: exoskeleton wrist RGB camera video.
- `camera_1.mp4`: Record3D/iPhone video with per-frame intrinsics + 4x4 poses.
- `numeric_0`: exoskeleton joint angles (12 DOF) + raw analog voltages.
- Per-sensor timestamps: `capture_time` and `receive_time` arrays.
- `valid_indices`: boolean masks for valid camera frames.

### software_go_through_replay (robot replay; data_replay-style)
- `exo_camera_0.mp4`: exoskeleton camera video from the demo.
- `dex_camera_0.mp4`: dexterous robot camera video from replay.
- `hand_motor_value`: 12-channel robot hand motor commands/positions used to
  replay the demo.
- `joint_angles_interp` + `pose_interp`: resampled exoskeleton joint angles and
  wrist poses aligned to the replay timeline.
- `valid_indices`: boolean mask for valid exo frames used in alignment.

### xhand_reference_5_14 (reference + point prompts)
- `reference_episode/`: reference capture similar to data_raw (camera_0/1,
  numeric_0), plus multiple FSR sensor groups (numeric_1/2/3).
- `*_points.pkl`: labeled 2D point prompts for dex/exo fingers and thumb (used
  in segmentation/inpainting prompt generation).

## Docs-only modalities (not present in this sample_data tree)
- DexUMI docs describe data_replay episodes that also store interpolated FSR
  streams (`fsr_values_interp_1/2/3`) alongside `hand_motor_value`,
  `joint_angles_interp`, `pose_interp`, and `valid_indices`.
- The post-processing step that generates the final dataset adds composite and
  debug videos plus dex/exo segmentation masks and image folders (e.g.,
  `combined.mp4`, `debug_combined.mp4`, `dex_*_seg_mask`, `exo_*_seg_mask`,
  `dex_img`, `exo_img`, and interpolated FSR values).

## Video streams (all)
| Video | Resolution | FPS | Duration (s) | Frames |
|---|---|---|---|---|
| `software_go_through/episode_0/camera_0.mp4` | `1280x800` | `45.000` | `13.888889` | `625` |
| `software_go_through/episode_0/camera_1.mp4` | `720x960` | `45.000` | `13.888889` | `625` |
| `software_go_through/episode_1/camera_0.mp4` | `1280x800` | `45.000` | `12.622222` | `568` |
| `software_go_through/episode_1/camera_1.mp4` | `720x960` | `45.000` | `12.622222` | `568` |
| `software_go_through_replay/episode_0/dex_camera_0.mp4` | `1280x800` | `30.000` | `6.766667` | `203` |
| `software_go_through_replay/episode_0/exo_camera_0.mp4` | `1280x800` | `45.000` | `13.888889` | `625` |
| `software_go_through_replay/episode_1/dex_camera_0.mp4` | `1280x800` | `30.000` | `6.066667` | `182` |
| `software_go_through_replay/episode_1/exo_camera_0.mp4` | `1280x800` | `45.000` | `12.622222` | `568` |
| `xhand_reference_5_14/reference_episode/camera_0.mp4` | `1280x800` | `45.000` | `16.244444` | `731` |
| `xhand_reference_5_14/reference_episode/camera_1.mp4` | `720x960` | `45.000` | `16.244444` | `731` |

## Pickle files (point prompts)
| File | Keys | points shape/dtype | labels shape/dtype |
|---|---|---|---|
| `xhand_reference_5_14/dex_finger_points.pkl` | `['points', 'labels']` | `(15, 2) / float32` | `(15,) / int32` |
| `xhand_reference_5_14/dex_thumb_points.pkl` | `['points', 'labels']` | `(13, 2) / float32` | `(13,) / int32` |
| `xhand_reference_5_14/exo_finger_points.pkl` | `['points', 'labels']` | `(20, 2) / float32` | `(20,) / int32` |
| `xhand_reference_5_14/exo_pinky_points.pkl` | `['points', 'labels']` | `(16, 2) / float32` | `(16,) / int32` |
| `xhand_reference_5_14/exo_thumb_points.pkl` | `['points', 'labels']` | `(26, 2) / float32` | `(26,) / int32` |

## Other files
- `xhand_reference_5_14/constants.py`: scale factors and reference values used
  for hand motor adjustments and rescaling (not a data stream).

## Zarr arrays (all)

### software_go_through/episode_0
| Zarr array | Shape | Dtype | Description |
|---|---|---|---|
| `software_go_through/episode_0/camera_0/capture_time` | `[625]` | `<f8` | Capture timestamp per frame for camera_0. |
| `software_go_through/episode_0/camera_0/receive_time` | `[625]` | `<f8` | Receive timestamp for camera_0 frames. |
| `software_go_through/episode_0/camera_0/valid_indices` | `[625]` | `|b1` | Boolean mask indicating valid camera_0 frames. |
| `software_go_through/episode_0/camera_1/capture_time` | `[625]` | `<f8` | Capture timestamp per frame for camera_1. |
| `software_go_through/episode_0/camera_1/intrinsics` | `[625, 3, 3]` | `<f8` | Camera_1 intrinsics (3x3) per frame. |
| `software_go_through/episode_0/camera_1/pose` | `[625, 4, 4]` | `<f8` | Camera/wrist pose (4x4) per frame from Record3D. |
| `software_go_through/episode_0/camera_1/pose_interp` | `[202, 4, 4]` | `<f8` | Interpolated camera/wrist pose on synced timeline. |
| `software_go_through/episode_0/camera_1/receive_time` | `[625]` | `<f8` | Receive timestamp for camera_1 frames. |
| `software_go_through/episode_0/numeric_0/capture_time` | `[403]` | `<f8` | Capture timestamp per sample for numeric_0. |
| `software_go_through/episode_0/numeric_0/joint_angles` | `[403, 12]` | `<f8` | Exoskeleton joint angles (12 DOF). |
| `software_go_through/episode_0/numeric_0/joint_angles_interp` | `[202, 12]` | `<i4` | Interpolated joint angles on synced timeline. |
| `software_go_through/episode_0/numeric_0/raw_voltage` | `[403, 14]` | `<f8` | Raw analog voltage channels (14), likely uncalibrated tactile/FSR data. |
| `software_go_through/episode_0/numeric_0/receive_time` | `[403]` | `<f8` | Receive timestamp for numeric_0 samples. |

### software_go_through/episode_1
| Zarr array | Shape | Dtype | Description |
|---|---|---|---|
| `software_go_through/episode_1/camera_0/capture_time` | `[568]` | `<f8` | Capture timestamp per frame for camera_0. |
| `software_go_through/episode_1/camera_0/receive_time` | `[568]` | `<f8` | Receive timestamp for camera_0 frames. |
| `software_go_through/episode_1/camera_0/valid_indices` | `[568]` | `|b1` | Boolean mask indicating valid camera_0 frames. |
| `software_go_through/episode_1/camera_1/capture_time` | `[568]` | `<f8` | Capture timestamp per frame for camera_1. |
| `software_go_through/episode_1/camera_1/intrinsics` | `[568, 3, 3]` | `<f8` | Camera_1 intrinsics (3x3) per frame. |
| `software_go_through/episode_1/camera_1/pose` | `[568, 4, 4]` | `<f8` | Camera/wrist pose (4x4) per frame from Record3D. |
| `software_go_through/episode_1/camera_1/pose_interp` | `[181, 4, 4]` | `<f8` | Interpolated camera/wrist pose on synced timeline. |
| `software_go_through/episode_1/camera_1/receive_time` | `[568]` | `<f8` | Receive timestamp for camera_1 frames. |
| `software_go_through/episode_1/numeric_0/capture_time` | `[365]` | `<f8` | Capture timestamp per sample for numeric_0. |
| `software_go_through/episode_1/numeric_0/joint_angles` | `[365, 12]` | `<f8` | Exoskeleton joint angles (12 DOF). |
| `software_go_through/episode_1/numeric_0/joint_angles_interp` | `[181, 12]` | `<i4` | Interpolated joint angles on synced timeline. |
| `software_go_through/episode_1/numeric_0/raw_voltage` | `[365, 14]` | `<f8` | Raw analog voltage channels (14), likely uncalibrated tactile/FSR data. |
| `software_go_through/episode_1/numeric_0/receive_time` | `[365]` | `<f8` | Receive timestamp for numeric_0 samples. |

### software_go_through_replay/episode_0
| Zarr array | Shape | Dtype | Description |
|---|---|---|---|
| `software_go_through_replay/episode_0/hand_motor_value` | `[202, 12]` | `<f8` | Robot hand motor commands/positions (12) used during replay. |
| `software_go_through_replay/episode_0/joint_angles_interp` | `[202, 12]` | `<i4` | Exoskeleton joint angles resampled to replay timeline. |
| `software_go_through_replay/episode_0/pose_interp` | `[202, 4, 4]` | `<f8` | Wrist pose resampled to replay timeline. |
| `software_go_through_replay/episode_0/valid_indices` | `[625]` | `|b1` | Boolean mask for aligned exo frames during replay. |

### software_go_through_replay/episode_1
| Zarr array | Shape | Dtype | Description |
|---|---|---|---|
| `software_go_through_replay/episode_1/hand_motor_value` | `[181, 12]` | `<f8` | Robot hand motor commands/positions (12) used during replay. |
| `software_go_through_replay/episode_1/joint_angles_interp` | `[181, 12]` | `<i4` | Exoskeleton joint angles resampled to replay timeline. |
| `software_go_through_replay/episode_1/pose_interp` | `[181, 4, 4]` | `<f8` | Wrist pose resampled to replay timeline. |
| `software_go_through_replay/episode_1/valid_indices` | `[568]` | `|b1` | Boolean mask for aligned exo frames during replay. |

### xhand_reference_5_14/reference_episode
| Zarr array | Shape | Dtype | Description |
|---|---|---|---|
| `xhand_reference_5_14/reference_episode/camera_0/capture_time` | `[731]` | `<f8` | Capture timestamp per frame for camera_0. |
| `xhand_reference_5_14/reference_episode/camera_0/receive_time` | `[731]` | `<f8` | Receive timestamp for camera_0 frames. |
| `xhand_reference_5_14/reference_episode/camera_0/valid_indices` | `[731]` | `|b1` | Boolean mask indicating valid camera_0 frames. |
| `xhand_reference_5_14/reference_episode/camera_1/capture_time` | `[731]` | `<f8` | Capture timestamp per frame for camera_1. |
| `xhand_reference_5_14/reference_episode/camera_1/intrinsics` | `[731, 3, 3]` | `<f8` | Camera_1 intrinsics (3x3) per frame. |
| `xhand_reference_5_14/reference_episode/camera_1/pose` | `[731, 4, 4]` | `<f8` | Camera/wrist pose (4x4) per frame from Record3D. |
| `xhand_reference_5_14/reference_episode/camera_1/pose_interp` | `[235, 4, 4]` | `<f8` | Interpolated camera/wrist pose on synced timeline. |
| `xhand_reference_5_14/reference_episode/camera_1/receive_time` | `[731]` | `<f8` | Receive timestamp for camera_1 frames. |
| `xhand_reference_5_14/reference_episode/numeric_0/capture_time` | `[474]` | `<f8` | Capture timestamp per sample for numeric_0. |
| `xhand_reference_5_14/reference_episode/numeric_0/joint_angles` | `[474, 12]` | `<f8` | Exoskeleton joint angles (12 DOF). |
| `xhand_reference_5_14/reference_episode/numeric_0/joint_angles_interp` | `[235, 12]` | `<i4` | Interpolated joint angles on synced timeline. |
| `xhand_reference_5_14/reference_episode/numeric_0/raw_voltage` | `[474, 14]` | `<f8` | Raw analog voltage channels (14), likely uncalibrated tactile/FSR data. |
| `xhand_reference_5_14/reference_episode/numeric_0/receive_time` | `[474]` | `<f8` | Receive timestamp for numeric_0 samples. |
| `xhand_reference_5_14/reference_episode/numeric_1/capture_time` | `[379]` | `<f8` | Capture timestamp per sample for numeric_1. |
| `xhand_reference_5_14/reference_episode/numeric_1/fsr_values` | `[379, 3]` | `<i8` | FSR sensor values (3 channels). |
| `xhand_reference_5_14/reference_episode/numeric_1/fsr_values_interp` | `[235, 3]` | `<f8` | Interpolated FSR values on synced timeline. |
| `xhand_reference_5_14/reference_episode/numeric_1/receive_time` | `[379]` | `<f8` | Receive timestamp for numeric_1 samples. |
| `xhand_reference_5_14/reference_episode/numeric_2/capture_time` | `[474]` | `<f8` | Capture timestamp per sample for numeric_2. |
| `xhand_reference_5_14/reference_episode/numeric_2/fsr_values` | `[474, 3]` | `<i8` | FSR sensor values (3 channels). |
| `xhand_reference_5_14/reference_episode/numeric_2/fsr_values_interp` | `[235, 3]` | `<f8` | Interpolated FSR values on synced timeline. |
| `xhand_reference_5_14/reference_episode/numeric_2/receive_time` | `[474]` | `<f8` | Receive timestamp for numeric_2 samples. |
| `xhand_reference_5_14/reference_episode/numeric_3/capture_time` | `[474]` | `<f8` | Capture timestamp per sample for numeric_3. |
| `xhand_reference_5_14/reference_episode/numeric_3/fsr_values` | `[474, 3]` | `<i8` | FSR sensor values (3 channels). |
| `xhand_reference_5_14/reference_episode/numeric_3/fsr_values_interp` | `[235, 3]` | `<f8` | Interpolated FSR values on synced timeline. |
| `xhand_reference_5_14/reference_episode/numeric_3/receive_time` | `[474]` | `<f8` | Receive timestamp for numeric_3 samples. |

## Exoskeleton vs. robot data relationship (context)
- Exoskeleton demos are recorded first with synchronized wrist camera video,
  joint angles, wrist pose (Record3D), and tactile/FSR sensors.
- Replay data uses those exoskeleton actions to drive the dexterous robot hand;
  the robot camera video (dex) is recorded alongside the original exo video.
- `hand_motor_value` captures the robot-side motor values produced from
  exoskeleton inputs, while `joint_angles_interp` and `pose_interp` are the
  resampled exoskeleton streams aligned to the replay timeline.

## Notes on timestamps and interpolation
- All timestamp arrays are float64 (`capture_time`, `receive_time`). Their unit
  is not explicitly encoded in metadata; the DexUMI tooling normalizes based on
  value scale (seconds/milliseconds/microseconds/nanoseconds).
- Arrays ending in `_interp` are resampled onto a common timeline, typically
  used to align different streams for replay or visualization.
