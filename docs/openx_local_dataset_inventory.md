# Local Open X-Embodiment dataset mirror

Root path (remote): `/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/open-x-embodiment`

## Using this mirror in this repo

Set the TF OXE adapter to prefer the local mirror (and fall back to GCS when not present):

- `data.adapter.tf.tfds_read.source=auto`
- `data.adapter.tf.tfds_read.local_root=/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/open-x-embodiment`

Note: the loader expects the TFDS version directories to match between local and `gs://gresearch/robotics/...`
(i.e., `<dataset_dirname>/<version>` exists in both places). If you update the mirror or GCS versions, keep
`packages/common/adapters/oxe.py` (`OXE_DATASETS[*].gcs_path`) consistent.

## On-disk structure

Each dataset is stored as a TFDS prepared dataset directory (RLDS episodes in TFRecord shards).

```text
/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/open-x-embodiment/<dataset_name>/
  <version>/
    dataset_info.json      # split metadata (shards + episode counts)
    features.json          # RLDS/TFDS feature schema
    <dataset_name>-<split>.tfrecord-00000-of-<NNNN>
    ...
  (optional extra directories)
```

## Splits

This mirror uses the split names defined by TFDS in each `dataset_info.json` (commonly `train`, sometimes `val` and/or `test`).

Split patterns observed:
- `train`: 47 datasets
- `test, train`: 11 datasets
- `train, val`: 11 datasets

Datasets by split pattern:
- `train`: `aloha_mobile`, `asu_table_top_converted_externally_to_rlds`, `austin_buds_dataset_converted_externally_to_rlds`, `austin_sailor_dataset_converted_externally_to_rlds`, `austin_sirius_dataset_converted_externally_to_rlds`, `berkeley_fanuc_manipulation`, `berkeley_gnm_cory_hall`, `berkeley_gnm_recon`, `berkeley_gnm_sac_son`, `berkeley_mvp_converted_externally_to_rlds`, `berkeley_rpt_converted_externally_to_rlds`, `cmu_franka_exploration_dataset_converted_externally_to_rlds`, `cmu_play_fusion`, `cmu_playing_with_food`, `cmu_stretch`, `dlr_edan_shared_control_converted_externally_to_rlds`, `dlr_sara_grid_clamp_converted_externally_to_rlds`, `dlr_sara_pour_converted_externally_to_rlds`, `dobbe`, `droid`, `eth_agent_affordances`, `fmb`, `fractal20220817_data`, `furniture_bench_dataset_converted_externally_to_rlds`, `iamlab_cmu_pickup_insert_converted_externally_to_rlds`, `imperialcollege_sawyer_wrist_cam`, `io_ai_tech`, `kaist_nonprehensile_converted_externally_to_rlds`, `kuka`, `language_table`, `maniskill_dataset_converted_externally_to_rlds`, `mimic_play`, `nyu_rot_dataset_converted_externally_to_rlds`, `qut_dexterous_manpulation`, `robo_net`, `robo_set`, `stanford_hydra_dataset_converted_externally_to_rlds`, `stanford_kuka_multimodal_dataset_converted_externally_to_rlds`, `stanford_robocook_converted_externally_to_rlds`, `tidybot`, `tokyo_u_lsmo_converted_externally_to_rlds`, `ucsd_kitchen_dataset_converted_externally_to_rlds`, `ucsd_pick_and_place_dataset_converted_externally_to_rlds`, `uiuc_d3field`, `utaustin_mutex`, `utokyo_saytap_converted_externally_to_rlds`, `vima_converted_externally_to_rlds`
- `test, train`: `berkeley_autolab_ur5`, `berkeley_cable_routing`, `bridge`, `columbia_cairlab_pusht_real`, `jaco_play`, `nyu_door_opening_surprising_effectiveness`, `robot_vqa`, `roboturk`, `taco_play`, `toto`, `viola`
- `train, val`: `bc_z`, `conq_hose_manipulation`, `nyu_franka_play_dataset_converted_externally_to_rlds`, `plex_robosuite`, `spoc`, `stanford_mask_vit_converted_externally_to_rlds`, `usc_cloth_sim_converted_externally_to_rlds`, `utokyo_pr2_opening_fridge_converted_externally_to_rlds`, `utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds`, `utokyo_xarm_bimanual_converted_externally_to_rlds`, `utokyo_xarm_pick_and_place_converted_externally_to_rlds`

## Feature schema (RLDS)

Across all datasets here, each TFRecord example is an *episode* with a `steps` sequence. Common per-step fields:
- `is_first`, `is_last`, `is_terminal` (bool)
- `observation` (dict; typically contains one or more camera images + robot state)
- `action` (either a tensor or a dict of tensors)
- `reward` (float) and often `discount` (float)

The exact observation/action keys vary by dataset; the lists below are extracted from each dataset’s `features.json`.

## Aggregate feature stats

Step keys (how many datasets contain each key):
- `is_first`: 69
- `is_last`: 69
- `is_terminal`: 69
- `observation`: 69
- `action`: 68
- `reward`: 68
- `discount`: 54
- `language_instruction`: 53
- `language_embedding`: 49
- `action_angle`: 3
- `action_delta`: 1
- `action_inst`: 1
- `goal_object`: 1
- `ground_truth_states`: 1
- `action_mode`: 1
- `intv_label`: 1
- `structured_action`: 1
- `action_dict`: 1
- `language_instruction_2`: 1
- `language_instruction_3`: 1
- `skill_completion`: 1
- `is_dense`: 1
- `multimodal_instruction`: 1
- `multimodal_instruction_assets`: 1

Action representation:
- `tensor_or_other`: 55
- `dict`: 14

Most common image observation keys (count across datasets):
- `image`: 52
- `wrist_image`: 15
- `hand_image`: 4
- `depth`: 2
- `image_wrist`: 2
- `image2`: 2
- `image_1`: 2
- `image_2`: 2
- `image_3`: 2
- `image_4`: 2
- `cam_high`: 1
- `cam_left_wrist`: 1
- `cam_right_wrist`: 1
- `image_with_depth`: 1
- `top_image`: 1
- `wrist225_image`: 1
- `wrist45_image`: 1
- `highres_image`: 1
- `finger_vision_1`: 1
- `finger_vision_2`: 1
- `frontleft_fisheye_image`: 1
- `frontright_fisheye_image`: 1
- `hand_color_image`: 1
- `exterior_image_1_left`: 1
- `exterior_image_2_left`: 1

Most common language/text-like fields (count across datasets):
- `step.language_instruction`: 53
- `step.language_embedding`: 49
- `observation.natural_language_embedding`: 13
- `observation.natural_language_instruction`: 13
- `step.language_instruction_2`: 1
- `step.language_instruction_3`: 1
- `observation.instruction`: 1
- `observation.raw_text_answer`: 1
- `observation.raw_text_question`: 1
- `observation.structured_language_instruction`: 1
- `step.multimodal_instruction`: 1
- `step.multimodal_instruction_assets`: 1

Most common action dict keys (count across datasets):
- `terminate_episode`: 12
- `world_vector`: 11
- `rotation_delta`: 10
- `gripper_closedness_action`: 8
- `open_gripper`: 2
- `base_displacement_vector`: 2
- `base_displacement_vertical_rotation`: 2
- `future/axis_angle_residual`: 1
- `future/target_close`: 1
- `future/xyz_residual`: 1
- `actions`: 1
- `rel_actions_gripper`: 1
- `rel_actions_world`: 1
- `pose0_position`: 1
- `pose0_rotation`: 1
- `pose1_position`: 1
- `pose1_rotation`: 1

## Inventory (storage + splits)

| Dataset | Version | Splits (episodes; shards) | TFRecord files | Extra dirs |
|---|---:|---|---:|---|
| `aloha_mobile` | `0.0.1` | train=276 eps (166 shards, 47.83 GiB) | 160 | `` |
| `asu_table_top_converted_externally_to_rlds` | `0.1.0` | train=110 eps (8 shards, 737.60 MiB) | 8 | `` |
| `austin_buds_dataset_converted_externally_to_rlds` | `0.1.0` | train=50 eps (16 shards, 1.49 GiB) | 16 | `` |
| `austin_sailor_dataset_converted_externally_to_rlds` | `0.1.0` | train=240 eps (109 shards, 18.85 GiB) | 109 | `` |
| `austin_sirius_dataset_converted_externally_to_rlds` | `0.1.0` | train=559 eps (64 shards, 6.55 GiB) | 64 | `` |
| `bc_z` | `0.1.0` | train=39350 eps (1024 shards, 73.25 GiB); val=3914 eps (64 shards, 7.29 GiB) | 1088 | `` |
| `berkeley_autolab_ur5` | `0.1.0` | test=104 eps (50 shards, 7.93 GiB); train=896 eps (412 shards, 68.46 GiB) | 462 | `` |
| `berkeley_cable_routing` | `0.1.0` | test=165 eps (4 shards, 463.14 MiB); train=1482 eps (64 shards, 4.22 GiB) | 68 | `` |
| `berkeley_fanuc_manipulation` | `0.1.0` | train=415 eps (124 shards, 8.85 GiB) | 124 | `` |
| `berkeley_gnm_cory_hall` | `0.1.0` | train=7331 eps (16 shards, 1.39 GiB) | 16 | `` |
| `berkeley_gnm_recon` | `0.1.0` | train=11834 eps (256 shards, 18.73 GiB) | 256 | `` |
| `berkeley_gnm_sac_son` | `0.1.0` | train=2955 eps (64 shards, 7.00 GiB) | 64 | `` |
| `berkeley_mvp_converted_externally_to_rlds` | `0.1.0` | train=480 eps (124 shards, 12.34 GiB) | 124 | `` |
| `berkeley_rpt_converted_externally_to_rlds` | `0.1.0` | train=908 eps (441 shards, 40.64 GiB) | 441 | `` |
| `bridge` | `0.1.0` | test=3475 eps (512 shards, 46.65 GiB); train=25460 eps (1024 shards, 340.85 GiB) | 1536 | `` |
| `cmu_franka_exploration_dataset_converted_externally_to_rlds` | `0.1.0` | train=199 eps (8 shards, 602.24 MiB) | 8 | `` |
| `cmu_play_fusion` | `0.1.0` | train=576 eps (64 shards, 6.68 GiB) | 64 | `` |
| `cmu_playing_with_food` | `1.0.0` | train=4200 eps (1024 shards, 259.30 GiB) | 1024 | `` |
| `cmu_stretch` | `0.1.0` | train=135 eps (8 shards, 728.06 MiB) | 8 | `` |
| `columbia_cairlab_pusht_real` | `0.1.0` | test=14 eps (4 shards, 296.78 MiB); train=122 eps (32 shards, 2.51 GiB) | 36 | `` |
| `conq_hose_manipulation` | `0.0.1` | train=113 eps (32 shards, 2.07 GiB); val=26 eps (8 shards, 656.21 MiB) | 66 | `` |
| `dlr_edan_shared_control_converted_externally_to_rlds` | `0.1.0` | train=104 eps (29 shards, 3.09 GiB) | 29 | `` |
| `dlr_sara_grid_clamp_converted_externally_to_rlds` | `0.1.0` | train=107 eps (16 shards, 1.65 GiB) | 16 | `` |
| `dlr_sara_pour_converted_externally_to_rlds` | `0.1.0` | train=100 eps (31 shards, 2.92 GiB) | 31 | `` |
| `dobbe` | `0.0.1` | train=5208 eps (256 shards, 21.10 GiB) | 1016 | `` |
| `droid` | `1.0.1` | train=95658 eps (2048 shards, 1.70 TiB) | 2048 | `` |
| `eth_agent_affordances` | `0.1.0` | train=118 eps (53 shards, 17.27 GiB) | 53 | `` |
| `fmb` | `0.0.1` | train=8611 eps (2017 shards, 1.17 TiB) | 2017 | `` |
| `fractal20220817_data` | `0.1.0` | train=87212 eps (1024 shards, 111.07 GiB) | 1024 | `` |
| `furniture_bench_dataset_converted_externally_to_rlds` | `0.1.0` | train=5100 eps (1016 shards, 115.00 GiB) | 1016 | `` |
| `iamlab_cmu_pickup_insert_converted_externally_to_rlds` | `0.1.0` | train=631 eps (369 shards, 50.29 GiB) | 369 | `` |
| `imperialcollege_sawyer_wrist_cam` | `0.1.0` | train=170 eps (1 shards, 81.87 MiB) | 1 | `` |
| `io_ai_tech` | `0.0.1` | train=3847 eps (1001 shards, 89.33 GiB) | 999 | `` |
| `jaco_play` | `0.1.0` | test=109 eps (8 shards, 957.31 MiB); train=976 eps (128 shards, 8.30 GiB) | 136 | `` |
| `kaist_nonprehensile_converted_externally_to_rlds` | `0.1.0` | train=201 eps (101 shards, 11.71 GiB) | 101 | `` |
| `kuka` | `0.1.0` | train=580392 eps (1024 shards, 778.02 GiB) | 1024 | `` |
| `language_table` | `0.1.0` | train=442226 eps (1024 shards, 399.87 GiB) | 1024 | `captions, long_horizon` |
| `maniskill_dataset_converted_externally_to_rlds` | `0.1.0` | train=30213 eps (1024 shards, 151.05 GiB) | 1024 | `` |
| `mimic_play` | `0.0.1` | train=378 eps (64 shards, 7.13 GiB) | 204 | `` |
| `nyu_door_opening_surprising_effectiveness` | `0.1.0` | test=49 eps (8 shards, 862.90 MiB); train=435 eps (64 shards, 6.28 GiB) | 72 | `` |
| `nyu_franka_play_dataset_converted_externally_to_rlds` | `0.1.0` | train=365 eps (32 shards, 3.98 GiB); val=91 eps (16 shards, 1.20 GiB) | 48 | `` |
| `nyu_rot_dataset_converted_externally_to_rlds` | `0.1.0` | train=14 eps (1 shards, 5.33 MiB) | 1 | `` |
| `plex_robosuite` | `0.0.1` | train=402 eps (16 shards, 1.13 GiB); val=48 eps (2 shards, 138.06 MiB) | 36 | `` |
| `qut_dexterous_manpulation` | `0.1.0` | train=200 eps (105 shards, 35.00 GiB) | 105 | `` |
| `robo_net` | `0.1.0` | train=82775 eps (1024 shards, 218.71 GiB) | 1024 | `` |
| `robo_set` | `0.0.1` | train=18250 eps (1024 shards, 178.65 GiB) | 1024 | `` |
| `robot_vqa` | `0.1.0` | test=3774 eps (16 shards, 1.55 GiB); train=3331523 eps (2048 shards, 1.29 TiB) | 2064 | `` |
| `roboturk` | `0.1.0` | test=199 eps (60 shards, 4.66 GiB); train=1796 eps (494 shards, 40.73 GiB) | 554 | `` |
| `spoc` | `0.0.1` | train=212043 eps (1024 shards, 697.31 GiB); val=21108 eps (1024 shards, 68.36 GiB) | 2048 | `` |
| `stanford_hydra_dataset_converted_externally_to_rlds` | `0.1.0` | train=570 eps (350 shards, 72.48 GiB) | 350 | `` |
| `stanford_kuka_multimodal_dataset_converted_externally_to_rlds` | `0.1.0` | train=3000 eps (256 shards, 31.98 GiB) | 256 | `` |
| `stanford_mask_vit_converted_externally_to_rlds` | `0.1.0` | train=9109 eps (1023 shards, 75.41 GiB); val=91 eps (8 shards, 770.63 MiB) | 1031 | `` |
| `stanford_robocook_converted_externally_to_rlds` | `0.1.0` | train=2460 eps (923 shards, 124.62 GiB) | 923 | `` |
| `taco_play` | `0.1.0` | test=361 eps (64 shards, 4.79 GiB); train=3242 eps (511 shards, 42.98 GiB) | 575 | `` |
| `tidybot` | `0.0.1` | train=24 eps (1 shards, 20.10 MiB) | 1 | `` |
| `tokyo_u_lsmo_converted_externally_to_rlds` | `0.1.0` | train=50 eps (4 shards, 335.71 MiB) | 4 | `` |
| `toto` | `0.1.0` | test=101 eps (52 shards, 12.24 GiB); train=902 eps (415 shards, 115.42 GiB) | 467 | `` |
| `ucsd_kitchen_dataset_converted_externally_to_rlds` | `0.1.0` | train=150 eps (16 shards, 1.33 GiB) | 16 | `` |
| `ucsd_pick_and_place_dataset_converted_externally_to_rlds` | `0.1.0` | train=1355 eps (32 shards, 3.53 GiB) | 32 | `` |
| `uiuc_d3field` | `0.1.0` | train=192 eps (98 shards, 15.82 GiB) | 98 | `` |
| `usc_cloth_sim_converted_externally_to_rlds` | `0.1.0` | train=800 eps (2 shards, 203.61 MiB); val=200 eps (1 shards, 50.90 MiB) | 3 | `` |
| `utaustin_mutex` | `0.1.0` | train=1500 eps (256 shards, 20.79 GiB) | 256 | `` |
| `utokyo_pr2_opening_fridge_converted_externally_to_rlds` | `0.1.0` | train=64 eps (4 shards, 286.12 MiB); val=16 eps (1 shards, 74.46 MiB) | 5 | `` |
| `utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds` | `0.1.0` | train=192 eps (8 shards, 668.53 MiB); val=48 eps (2 shards, 160.84 MiB) | 10 | `` |
| `utokyo_saytap_converted_externally_to_rlds` | `0.1.0` | train=20 eps (1 shards, 55.34 MiB) | 1 | `` |
| `utokyo_xarm_bimanual_converted_externally_to_rlds` | `0.1.0` | train=64 eps (1 shards, 126.89 MiB); val=6 eps (1 shards, 11.55 MiB) | 2 | `` |
| `utokyo_xarm_pick_and_place_converted_externally_to_rlds` | `0.1.0` | train=92 eps (16 shards, 1.17 GiB); val=10 eps (1 shards, 122.30 MiB) | 17 | `` |
| `vima_converted_externally_to_rlds` | `0.0.1` | train=660103 eps (2048 shards, 1.39 TiB) | 2048 | `` |
| `viola` | `0.1.0` | test=15 eps (7 shards, 1.01 GiB); train=135 eps (81 shards, 9.39 GiB) | 88 | `` |

## Per-dataset features

### aloha_mobile (0.0.1)

- Splits: `train`: 276 episodes, 166 shards (47.83 GiB)
- TFRecord shard files in version dir: 160
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_instruction`
- Observation keys: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `state`
- Images: `cam_high`: 480x640x3 uint8, jpeg; `cam_left_wrist`: 480x640x3 uint8, jpeg; `cam_right_wrist`: 480x640x3 uint8, jpeg
- Language/text-like fields: step: `language_instruction` (string scalar)
- Action: float32 16

### asu_table_top_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 110 episodes, 8 shards (737.60 MiB)
- TFRecord shard files in version dir: 8
- Step keys: `action`, `action_delta`, `action_inst`, `discount`, `goal_object`, `ground_truth_states`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `action_delta`, `action_inst`, `goal_object`, `ground_truth_states`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `state_vel`
- Images: `image`: 224x224x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### austin_buds_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 50 episodes, 16 shards (1.49 GiB)
- TFRecord shard files in version dir: 16
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 128x128x3 uint8, png; `wrist_image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### austin_sailor_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 240 episodes, 109 shards (18.85 GiB)
- TFRecord shard files in version dir: 109
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `state_ee`, `state_gripper`, `state_joint`, `wrist_image`
- Images: `image`: 128x128x3 uint8, png; `wrist_image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### austin_sirius_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 559 episodes, 64 shards (6.55 GiB)
- TFRecord shard files in version dir: 64
- Step keys: `action`, `action_mode`, `discount`, `intv_label`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `action_mode`, `intv_label`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `state_ee`, `state_gripper`, `state_joint`, `wrist_image`
- Images: `image`: 84x84x3 uint8, png; `wrist_image`: 84x84x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### bc_z (0.1.0)

- Splits: `train`: 39350 episodes, 1024 shards (73.25 GiB), `val`: 3914 episodes, 64 shards (7.29 GiB)
- TFRecord shard files in version dir: 1088
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `episode_success`, `image`, `natural_language_embedding`, `natural_language_instruction`, `present/autonomous`, `present/axis_angle`, `present/intervention`, `present/sensed_close`, `present/xyz`, `sequence_length`
- Images: `image`: 171x213x3 uint8, jpeg
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `future/axis_angle_residual` (float32 30), `future/target_close` (int64 10), `future/xyz_residual` (float32 30)

### berkeley_autolab_ur5 (0.1.0)

- Splits: `test`: 104 episodes, 50 shards (7.93 GiB), `train`: 896 episodes, 412 shards (68.46 GiB)
- TFRecord shard files in version dir: 462
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `hand_image`, `image`, `image_with_depth`, `natural_language_embedding`, `natural_language_instruction`, `robot_state`
- Images: `hand_image`: 480x640x3 uint8; `image`: 480x640x3 uint8; `image_with_depth`: 480x640x1 float32
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 scalar), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### berkeley_cable_routing (0.1.0)

- Splits: `test`: 165 episodes, 4 shards (463.14 MiB), `train`: 1482 episodes, 64 shards (4.22 GiB)
- TFRecord shard files in version dir: 68
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `image`, `natural_language_embedding`, `natural_language_instruction`, `robot_state`, `top_image`, `wrist225_image`, `wrist45_image`
- Images: `image`: 128x128x3 uint8; `top_image`: 128x128x3 uint8; `wrist225_image`: 128x128x3 uint8; `wrist45_image`: 128x128x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### berkeley_fanuc_manipulation (0.1.0)

- Splits: `train`: 415 episodes, 124 shards (8.85 GiB)
- TFRecord shard files in version dir: 124
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `end_effector_state`, `image`, `state`, `wrist_image`
- Images: `image`: 224x224x3 uint8, png; `wrist_image`: 224x224x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 6

### berkeley_gnm_cory_hall (0.1.0)

- Splits: `train`: 7331 episodes, 16 shards (1.39 GiB)
- TFRecord shard files in version dir: 16
- Step keys: `action`, `action_angle`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `action_angle`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `position`, `state`, `yaw`
- Images: `image`: 64x85x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float64 2

### berkeley_gnm_recon (0.1.0)

- Splits: `train`: 11834 episodes, 256 shards (18.73 GiB)
- TFRecord shard files in version dir: 256
- Step keys: `action`, `action_angle`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `action_angle`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `position`, `state`, `yaw`
- Images: `image`: 120x160x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float64 2

### berkeley_gnm_sac_son (0.1.0)

- Splits: `train`: 2955 episodes, 64 shards (7.00 GiB)
- TFRecord shard files in version dir: 64
- Step keys: `action`, `action_angle`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `action_angle`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `position`, `state`, `yaw`
- Images: `image`: 120x160x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float64 2

### berkeley_mvp_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 480 episodes, 124 shards (12.34 GiB)
- TFRecord shard files in version dir: 124
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `gripper`, `hand_image`, `joint_pos`, `pose`
- Images: `hand_image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### berkeley_rpt_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 908 episodes, 441 shards (40.64 GiB)
- TFRecord shard files in version dir: 441
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `gripper`, `hand_image`, `joint_pos`
- Images: `hand_image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### bridge (0.1.0)

- Splits: `test`: 3475 episodes, 512 shards (46.65 GiB), `train`: 25460 episodes, 1024 shards (340.85 GiB)
- TFRecord shard files in version dir: 1536
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `image`, `natural_language_embedding`, `natural_language_instruction`, `state`
- Images: `image`: 480x640x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `open_gripper` (bool scalar), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### cmu_franka_exploration_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 199 episodes, 8 shards (602.24 MiB)
- TFRecord shard files in version dir: 8
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`, `structured_action`
- Step extra keys: `language_embedding`, `language_instruction`, `structured_action`
- Observation keys: `highres_image`, `image`
- Images: `highres_image`: 480x640x3 uint8, png; `image`: 64x64x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### cmu_play_fusion (0.1.0)

- Splits: `train`: 576 episodes, 64 shards (6.68 GiB)
- TFRecord shard files in version dir: 64
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 9

### cmu_playing_with_food (1.0.0)

- Splits: `train`: 4200 episodes, 1024 shards (259.30 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `finger_vision_1`, `finger_vision_2`, `image`, `state`
- Images: `finger_vision_1`: 480x640x3 uint8, png; `finger_vision_2`: 480x640x3 uint8, png; `image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### cmu_stretch (0.1.0)

- Splits: `train`: 135 episodes, 8 shards (728.06 MiB)
- TFRecord shard files in version dir: 8
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### columbia_cairlab_pusht_real (0.1.0)

- Splits: `test`: 14 episodes, 4 shards (296.78 MiB), `train`: 122 episodes, 32 shards (2.51 GiB)
- TFRecord shard files in version dir: 36
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `image`, `natural_language_embedding`, `natural_language_instruction`, `robot_state`, `wrist_image`
- Images: `image`: 240x320x3 uint8; `wrist_image`: 240x320x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 scalar), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### conq_hose_manipulation (0.0.1)

- Splits: `train`: 113 episodes, 32 shards (2.07 GiB), `val`: 26 episodes, 8 shards (656.21 MiB)
- TFRecord shard files in version dir: 66
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `frontleft_fisheye_image`, `frontright_fisheye_image`, `hand_color_image`, `state`
- Images: `frontleft_fisheye_image`: 726x604x3 uint8, jpeg; `frontright_fisheye_image`: 726x604x3 uint8, jpeg; `hand_color_image`: 480x640x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float32 7

### dlr_edan_shared_control_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 104 episodes, 29 shards (3.09 GiB)
- TFRecord shard files in version dir: 29
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 360x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### dlr_sara_grid_clamp_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 107 episodes, 16 shards (1.65 GiB)
- TFRecord shard files in version dir: 16
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### dlr_sara_pour_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 100 episodes, 31 shards (2.92 GiB)
- TFRecord shard files in version dir: 31
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### dobbe (0.0.1)

- Splits: `train`: 5208 episodes, 256 shards (21.10 GiB)
- TFRecord shard files in version dir: 1016
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `gripper`, `quat`, `rot`, `state`, `wrist_image`, `xyz`
- Images: `wrist_image`: 256x256x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float32 7

### droid (1.0.1)

- Splits: `train`: 95658 episodes, 2048 shards (1.70 TiB)
- TFRecord shard files in version dir: 2048
- Step keys: `action`, `action_dict`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_instruction`, `language_instruction_2`, `language_instruction_3`, `observation`, `reward`
- Step extra keys: `action_dict`, `language_instruction`, `language_instruction_2`, `language_instruction_3`
- Observation keys: `cartesian_position`, `exterior_image_1_left`, `exterior_image_2_left`, `gripper_position`, `joint_position`, `wrist_image_left`
- Images: `exterior_image_1_left`: 180x320x3 uint8, jpeg; `exterior_image_2_left`: 180x320x3 uint8, jpeg; `wrist_image_left`: 180x320x3 uint8, jpeg
- Language/text-like fields: step: `language_instruction` (string scalar), `language_instruction_2` (string scalar), `language_instruction_3` (string scalar)
- Action: float64 7

### eth_agent_affordances (0.1.0)

- Splits: `train`: 118 episodes, 53 shards (17.27 GiB)
- TFRecord shard files in version dir: 53
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `input_point_cloud`, `state`
- Images: `image`: 64x64x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 6

### fmb (0.0.1)

- Splits: `train`: 8611 episodes, 2017 shards (1.17 TiB)
- TFRecord shard files in version dir: 2017
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `color_id`, `eef_force`, `eef_pose`, `eef_torque`, `eef_vel`, `image_side_1`, `image_side_1_depth`, `image_side_2`, `image_side_2_depth`, `image_wrist_1`, `image_wrist_1_depth`, `image_wrist_2`, `image_wrist_2_depth`, `joint_pos`, `joint_vel`, `length`, `object_id`, `primitive`, `shape_id`, `size`, `state_gripper_pose`
- Images: `image_side_1`: 256x256x3 uint8, jpeg; `image_side_2`: 256x256x3 uint8, jpeg; `image_wrist_1`: 256x256x3 uint8, jpeg; `image_wrist_2`: 256x256x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float32 7

### fractal20220817_data (0.1.0)

- Splits: `train`: 87212 episodes, 1024 shards (111.07 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `base_pose_tool_reached`, `gripper_closed`, `gripper_closedness_commanded`, `height_to_bottom`, `image`, `natural_language_embedding`, `natural_language_instruction`, `orientation_box`, `orientation_start`, `robot_orientation_positions_box`, `rotation_delta_to_go`, `src_rotation`, `vector_to_go`, `workspace_bounds`
- Images: `image`: 256x320x3 uint8, jpeg
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `base_displacement_vector` (float32 2), `base_displacement_vertical_rotation` (float32 1), `gripper_closedness_action` (float32 1), `rotation_delta` (float32 3), `terminate_episode` (int32 3), `world_vector` (float32 3)

### furniture_bench_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 5100 episodes, 1016 shards (115.00 GiB)
- TFRecord shard files in version dir: 1016
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`, `skill_completion`
- Step extra keys: `language_embedding`, `language_instruction`, `skill_completion`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 224x224x3 uint8, jpeg; `wrist_image`: 224x224x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### iamlab_cmu_pickup_insert_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 631 episodes, 369 shards (50.29 GiB)
- TFRecord shard files in version dir: 369
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 360x640x3 uint8, png; `wrist_image`: 240x320x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### imperialcollege_sawyer_wrist_cam (0.1.0)

- Splits: `train`: 170 episodes, 1 shards (81.87 MiB)
- TFRecord shard files in version dir: 1
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 64x64x3 uint8, png; `wrist_image`: 64x64x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### io_ai_tech (0.0.1)

- Splits: `train`: 3847 episodes, 1001 shards (89.33 GiB)
- TFRecord shard files in version dir: 999
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `depth`, `fisheye_camera_extrinsic`, `fisheye_camera_intrinsic`, `image`, `image_fisheye`, `image_left_side`, `image_right_side`, `left_camera_extrinsic`, `left_camera_intrinsic`, `main_camera_intrinsic`, `right_camera_extrinsic`, `right_camera_intrinsic`, `state`
- Images: `depth`: 720x1280x1 uint8, jpeg; `image`: 360x640x3 uint8, jpeg; `image_fisheye`: 640x800x3 uint8, jpeg; `image_left_side`: 360x640x3 uint8, jpeg; `image_right_side`: 360x640x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float32 7

### jaco_play (0.1.0)

- Splits: `test`: 109 episodes, 8 shards (957.31 MiB), `train`: 976 episodes, 128 shards (8.30 GiB)
- TFRecord shard files in version dir: 136
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `end_effector_cartesian_pos`, `end_effector_cartesian_velocity`, `image`, `image_wrist`, `joint_pos`, `natural_language_embedding`, `natural_language_instruction`
- Images: `image`: 224x224x3 uint8; `image_wrist`: 224x224x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 1), `terminate_episode` (int32 3), `world_vector` (float32 3)

### kaist_nonprehensile_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 201 episodes, 101 shards (11.71 GiB)
- TFRecord shard files in version dir: 101
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `partial_pointcloud`, `state`
- Images: `image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 20

### kuka (0.1.0)

- Splits: `train`: 580392 episodes, 1024 shards (778.02 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `clip_function_input/base_pose_tool_reached`, `clip_function_input/workspace_bounds`, `gripper_closed`, `height_to_bottom`, `image`, `natural_language_embedding`, `natural_language_instruction`, `task_id`
- Images: `image`: 512x640x3 uint8, jpeg
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `base_displacement_vector` (float32 2), `base_displacement_vertical_rotation` (float32 1), `gripper_closedness_action` (float32 1), `rotation_delta` (float32 3), `terminate_episode` (int32 3), `world_vector` (float32 3)

### language_table (0.1.0)

- Splits: `train`: 442226 episodes, 1024 shards (399.87 GiB)
- TFRecord shard files in version dir: 1024
- Extra non-version dirs: `captions`, `long_horizon`
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `effector_target_translation`, `effector_translation`, `instruction`, `rgb`
- Images: `rgb`: 360x640x3 uint8, jpeg
- Language/text-like fields: observation: `instruction` (int32 512)
- Action: float32 2

### maniskill_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 30213 episodes, 1024 shards (151.05 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `base_pose`, `depth`, `image`, `main_camera_cam2world_gl`, `main_camera_extrinsic_cv`, `main_camera_intrinsic_cv`, `state`, `target_object_or_part_final_pose`, `target_object_or_part_final_pose_valid`, `target_object_or_part_initial_pose`, `target_object_or_part_initial_pose_valid`, `tcp_pose`, `wrist_camera_cam2world_gl`, `wrist_camera_extrinsic_cv`, `wrist_camera_intrinsic_cv`, `wrist_depth`, `wrist_image`
- Images: `depth`: 256x256x1 uint16, png; `image`: 256x256x3 uint8, png; `wrist_depth`: 256x256x1 uint16, png; `wrist_image`: 256x256x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### mimic_play (0.0.1)

- Splits: `train`: 378 episodes, 64 shards (7.13 GiB)
- TFRecord shard files in version dir: 204
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float32 7

### nyu_door_opening_surprising_effectiveness (0.1.0)

- Splits: `test`: 49 episodes, 8 shards (862.90 MiB), `train`: 435 episodes, 64 shards (6.28 GiB)
- TFRecord shard files in version dir: 72
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `image`, `natural_language_embedding`, `natural_language_instruction`
- Images: `image`: 720x960x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 1), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### nyu_franka_play_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 365 episodes, 32 shards (3.98 GiB), `val`: 91 episodes, 16 shards (1.20 GiB)
- TFRecord shard files in version dir: 48
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `depth`, `depth_additional_view`, `image`, `image_additional_view`, `state`
- Images: `image`: 128x128x3 uint8, png; `image_additional_view`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 15

### nyu_rot_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 14 episodes, 1 shards (5.33 MiB)
- TFRecord shard files in version dir: 1
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 84x84x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### plex_robosuite (0.0.1)

- Splits: `train`: 402 episodes, 16 shards (1.13 GiB), `val`: 48 episodes, 2 shards (138.06 MiB)
- TFRecord shard files in version dir: 36
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 128x128x3 uint8, jpeg; `wrist_image`: 128x128x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: float64 7

### qut_dexterous_manpulation (0.1.0)

- Splits: `train`: 200 episodes, 105 shards (35.00 GiB)
- TFRecord shard files in version dir: 105
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 480x640x3 uint8, jpeg; `wrist_image`: 480x640x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### robo_net (0.1.0)

- Splits: `train`: 82775 episodes, 1024 shards (218.71 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `image1`, `image2`, `state`
- Images: `image`: 240x320x3 uint8, jpeg; `image1`: 240x320x3 uint8, jpeg; `image2`: 240x320x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 5

### robo_set (0.0.1)

- Splits: `train`: 18250 episodes, 1024 shards (178.65 GiB)
- TFRecord shard files in version dir: 1024
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_instruction`
- Observation keys: `image_left`, `image_right`, `image_top`, `image_wrist`, `state`, `state_velocity`
- Images: `image_left`: 240x424x3 uint8, jpeg; `image_right`: 240x424x3 uint8, jpeg; `image_top`: 240x424x3 uint8, jpeg; `image_wrist`: 240x424x3 uint8, jpeg
- Language/text-like fields: step: `language_instruction` (string scalar)
- Action: float32 8

### robot_vqa (0.1.0)

- Splits: `test`: 3774 episodes, 16 shards (1.55 GiB), `train`: 3331523 episodes, 2048 shards (1.29 TiB)
- TFRecord shard files in version dir: 2064
- Step keys: `is_first`, `is_last`, `is_terminal`, `observation`
- Observation keys: `images`, `raw_text_answer`, `raw_text_question`
- Language/text-like fields: observation: `raw_text_answer` (string scalar), `raw_text_question` (string scalar)
- Action:  scalar

### roboturk (0.1.0)

- Splits: `test`: 199 episodes, 60 shards (4.66 GiB), `train`: 1796 episodes, 494 shards (40.73 GiB)
- TFRecord shard files in version dir: 554
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `front_rgb`, `natural_language_embedding`, `natural_language_instruction`
- Images: `front_rgb`: 480x640x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 1), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### spoc (0.0.1)

- Splits: `train`: 212043 episodes, 1024 shards (697.31 GiB), `val`: 21108 episodes, 1024 shards (68.36 GiB)
- TFRecord shard files in version dir: 2048
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_instruction`
- Observation keys: `an_object_is_in_hand`, `house_index`, `hypothetical_task_success`, `image`, `image_manipulation`, `last_action_is_random`, `last_action_str`, `last_action_success`, `last_agent_location`, `manip_object_bbox`, `minimum_l2_target_distance`, `minimum_visible_target_alignment`, `nav_object_bbox`, `relative_arm_location_metadata`, `room_current_seen`, `rooms_seen`, `visible_target_4m_count`
- Images: `image`: 224x384x3 uint8, jpeg; `image_manipulation`: 224x384x3 uint8, jpeg
- Language/text-like fields: step: `language_instruction` (string scalar)
- Action: float32 9

### stanford_hydra_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 570 episodes, 350 shards (72.48 GiB)
- TFRecord shard files in version dir: 350
- Step keys: `action`, `discount`, `is_dense`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `is_dense`, `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 240x320x3 uint8, png; `wrist_image`: 240x320x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### stanford_kuka_multimodal_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 3000 episodes, 256 shards (31.98 GiB)
- TFRecord shard files in version dir: 256
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `contact`, `depth_image`, `ee_forces_continuous`, `ee_orientation`, `ee_orientation_vel`, `ee_position`, `ee_vel`, `ee_yaw`, `ee_yaw_delta`, `image`, `joint_pos`, `joint_vel`, `optical_flow`, `state`
- Images: `image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 4

### stanford_mask_vit_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 9109 episodes, 1023 shards (75.41 GiB), `val`: 91 episodes, 8 shards (770.63 MiB)
- TFRecord shard files in version dir: 1031
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `end_effector_pose`, `finger_sensors`, `high_bound`, `image`, `low_bound`, `state`
- Images: `image`: 480x480x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 5

### stanford_robocook_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 2460 episodes, 923 shards (124.62 GiB)
- TFRecord shard files in version dir: 923
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `depth_1`, `depth_2`, `depth_3`, `depth_4`, `image_1`, `image_2`, `image_3`, `image_4`, `state`
- Images: `image_1`: 256x256x3 uint8, jpeg; `image_2`: 256x256x3 uint8, jpeg; `image_3`: 256x256x3 uint8, jpeg; `image_4`: 256x256x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### taco_play (0.1.0)

- Splits: `test`: 361 episodes, 64 shards (4.79 GiB), `train`: 3242 episodes, 511 shards (42.98 GiB)
- TFRecord shard files in version dir: 575
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `depth_gripper`, `depth_static`, `natural_language_embedding`, `natural_language_instruction`, `rgb_gripper`, `rgb_static`, `robot_obs`, `structured_language_instruction`
- Images: `rgb_gripper`: 84x84x3 uint8; `rgb_static`: 150x200x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar), `structured_language_instruction` (string scalar)
- Action: dict with keys `actions` (float32 7), `rel_actions_gripper` (float32 7), `rel_actions_world` (float32 7), `terminate_episode` (float32 scalar)

### tidybot (0.0.1)

- Splits: `train`: 24 episodes, 1 shards (20.10 MiB)
- TFRecord shard files in version dir: 1
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `object`, `receptacles`
- Images: `image`: 360x640x3 uint8, jpeg
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` (string scalar)
- Action: string scalar

### tokyo_u_lsmo_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 50 episodes, 4 shards (335.71 MiB)
- TFRecord shard files in version dir: 4
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 120x120x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### toto (0.1.0)

- Splits: `test`: 101 episodes, 52 shards (12.24 GiB), `train`: 902 episodes, 415 shards (115.42 GiB)
- TFRecord shard files in version dir: 467
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `image`, `natural_language_embedding`, `natural_language_instruction`, `state`
- Images: `image`: 480x640x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `open_gripper` (bool scalar), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)

### ucsd_kitchen_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 150 episodes, 16 shards (1.33 GiB)
- TFRecord shard files in version dir: 16
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 480x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### ucsd_pick_and_place_dataset_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 1355 episodes, 32 shards (3.53 GiB)
- TFRecord shard files in version dir: 32
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 224x224x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 4

### uiuc_d3field (0.1.0)

- Splits: `train`: 192 episodes, 98 shards (15.82 GiB)
- TFRecord shard files in version dir: 98
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `depth_1`, `depth_2`, `depth_3`, `depth_4`, `image_1`, `image_2`, `image_3`, `image_4`, `state`
- Images: `depth_1`: 360x640x1 uint16, png; `depth_2`: 360x640x1 uint16, png; `depth_3`: 360x640x1 uint16, png; `depth_4`: 360x640x1 uint16, png; `image_1`: 360x640x3 uint8, png; `image_2`: 360x640x3 uint8, png; `image_3`: 360x640x3 uint8, png; `image_4`: 360x640x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 3

### usc_cloth_sim_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 800 episodes, 2 shards (203.61 MiB), `val`: 200 episodes, 1 shards (50.90 MiB)
- TFRecord shard files in version dir: 3
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`
- Images: `image`: 32x32x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 4

### utaustin_mutex (0.1.0)

- Splits: `train`: 1500 episodes, 256 shards (20.79 GiB)
- TFRecord shard files in version dir: 256
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`, `wrist_image`
- Images: `image`: 128x128x3 uint8, png; `wrist_image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### utokyo_pr2_opening_fridge_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 64 episodes, 4 shards (286.12 MiB), `val`: 16 episodes, 1 shards (74.46 MiB)
- TFRecord shard files in version dir: 5
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 192 episodes, 8 shards (668.53 MiB), `val`: 48 episodes, 2 shards (160.84 MiB)
- TFRecord shard files in version dir: 10
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `image`, `state`
- Images: `image`: 128x128x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 8

### utokyo_saytap_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 20 episodes, 1 shards (55.34 MiB)
- TFRecord shard files in version dir: 1
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `desired_pattern`, `desired_vel`, `image`, `prev_act`, `proj_grav_vec`, `state`, `wrist_image`
- Images: `image`: 64x64x3 uint8, png; `wrist_image`: 64x64x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 12

### utokyo_xarm_bimanual_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 64 episodes, 1 shards (126.89 MiB), `val`: 6 episodes, 1 shards (11.55 MiB)
- TFRecord shard files in version dir: 2
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `action_l`, `action_r`, `image`, `pose_l`, `pose_r`
- Images: `image`: 256x256x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 14

### utokyo_xarm_pick_and_place_converted_externally_to_rlds (0.1.0)

- Splits: `train`: 92 episodes, 16 shards (1.17 GiB), `val`: 10 episodes, 1 shards (122.30 MiB)
- TFRecord shard files in version dir: 17
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `language_embedding`, `language_instruction`, `observation`, `reward`
- Step extra keys: `language_embedding`, `language_instruction`
- Observation keys: `end_effector_pose`, `hand_image`, `image`, `image2`, `joint_state`, `joint_trajectory`
- Images: `hand_image`: 224x224x3 uint8, png; `image`: 224x224x3 uint8, png; `image2`: 224x224x3 uint8, png
- Language/text-like fields: step: `language_embedding` (float32 512), `language_instruction` ( scalar)
- Action: float32 7

### vima_converted_externally_to_rlds (0.0.1)

- Splits: `train`: 660103 episodes, 2048 shards (1.39 TiB)
- TFRecord shard files in version dir: 2048
- Step keys: `action`, `discount`, `is_first`, `is_last`, `is_terminal`, `multimodal_instruction`, `multimodal_instruction_assets`, `observation`, `reward`
- Step extra keys: `multimodal_instruction`, `multimodal_instruction_assets`
- Observation keys: `ee`, `frontal_image`, `frontal_segmentation`, `image`, `segmentation`, `segmentation_obj_info`
- Language/text-like fields: step: `multimodal_instruction` (string scalar), `multimodal_instruction_assets` (FeaturesDict )
- Action: dict with keys `pose0_position` (float32 3), `pose0_rotation` (float32 4), `pose1_position` (float32 3), `pose1_rotation` (float32 4)

### viola (0.1.0)

- Splits: `test`: 15 episodes, 7 shards (1.01 GiB), `train`: 135 episodes, 81 shards (9.39 GiB)
- TFRecord shard files in version dir: 88
- Step keys: `action`, `is_first`, `is_last`, `is_terminal`, `observation`, `reward`
- Observation keys: `agentview_rgb`, `ee_states`, `eye_in_hand_rgb`, `gripper_states`, `joint_states`, `natural_language_embedding`, `natural_language_instruction`
- Images: `agentview_rgb`: 224x224x3 uint8; `eye_in_hand_rgb`: 224x224x3 uint8
- Language/text-like fields: observation: `natural_language_embedding` (float32 512), `natural_language_instruction` (string scalar)
- Action: dict with keys `gripper_closedness_action` (float32 scalar), `rotation_delta` (float32 3), `terminate_episode` (float32 scalar), `world_vector` (float32 3)
