# Training and test scenes: exact area and dates

All scenes are NISAR L2 GCOV, provisional, L-band, HH + VV, 80 m pixels. Machine-readable list: `config/dataset.json`.

## Area

- Search box: **70 - 78 N, 150 - 180 W** (Chukchi Sea and western Beaufort Sea).
- Only scenes without land (under 0.5 % land in the AMSR2 map).

- A scene is taken when it overlaps the box, so the chosen scenes reach a little further: about 67 - 78 N,
  and in the west across the 180 degree line to about 170 E.

## Dates

| Set | Period | Chosen | Used | Skipped (empty channel) |
|---|---|---|---|---|
| Training | 14 Sep - 28 Oct 2026 | 24 | 22 | 2 |
| Test (locked) | 29 - 31 Oct 2026 | 10 | 4 | 6 |

- The test period starts after the training period ends. No test scene was opened before the model was frozen.
- Scene type comes from the AMSR2 map of that day: *edge* = over 10 % clear water and over 10 % clear ice; *all ice* / *all water* = over 95 %.
- Each used scene gives up to 20,000 random labelled pixels per class (`tables/train.npz`, `tables/test.npz`).

## Training scenes

| Date | Time (UTC) | Type | Centre | Orbit | Used | Scene ID (track_direction_frame) |
|---|---|---|---|---|---|---|
| 15 Sep | 10:13 | edge | 75.9 N, 170.5 W | descending | yes | `133_D_047` |
| 16 Sep | 07:52 | all water | 74.4 N, 148.5 W | descending | yes | `146_D_049` |
| 16 Sep | 11:09 | edge | 76.5 N, 167.1 W | ascending | yes | `148_A_045` |
| 17 Sep | 10:29 | edge | 76.3 N, 166.7 W | descending | yes | `162_D_046` |
| 18 Sep | 09:47 | edge | 76.3 N, 156.3 W | descending | yes | `003_D_046` |
| 18 Sep | 13:03 | all water | 73.3 N, 159.1 W | ascending | yes | `005_A_040` |
| 20 Sep | 10:05 | edge | 75.2 N, 175.4 E | descending | yes | `032_D_048` |
| 21 Sep | 07:44 | all water | 74.4 N, 146.4 W | descending | yes | `045_D_049` |
| 23 Sep | 09:40 | edge | 75.2 N, 169.2 W | descending | **no**, empty channel | `075_D_048` |
| 25 Sep | 08:19 | all water | 69.2 N, 174.8 W | descending | yes | `103_D_053` |
| 29 Sep | 08:50 | edge | 74.4 N, 163.1 W | descending | yes | `161_D_049` |
| 03 Oct | 07:44 | edge | 73.3 N, 152.2 W | descending | yes | `045_D_050` |
| 03 Oct | 09:22 | all ice | 76.3 N, 150.1 W | descending | yes | `046_D_046` |
| 05 Oct | 12:55 | edge | 73.3 N, 157.0 W | ascending | **no**, empty channel | `077_A_040` |
| 10 Oct | 07:53 | edge | 73.3 N, 154.3 W | descending | yes | `146_D_050` |
| 11 Oct | 13:44 | all water | 71.9 N, 163.9 W | ascending | yes | `164_A_039` |
| 13 Oct | 10:46 | all ice | 75.9 N, 178.8 E | descending | yes | `018_D_047` |
| 19 Oct | 08:18 | edge | 73.3 N, 160.5 W | descending | yes | `103_D_050` |
| 19 Oct | 11:34 | all ice | 76.5 N, 173.2 W | ascending | yes | `105_A_045` |
| 21 Oct | 06:55 | edge | 72.0 N, 145.3 W | descending | yes | `131_D_051` |
| 27 Oct | 11:01 | all ice | 76.5 N, 164.9 W | ascending | yes | `047_A_045` |
| 27 Oct | 14:18 | edge | 71.9 N, 172.1 W | ascending | yes | `049_A_039` |
| 28 Oct | 08:43 | edge | 72.0 N, 172.4 W | descending | yes | `060_D_051` |
| 28 Oct | 10:20 | all ice | 76.3 N, 164.6 W | descending | yes | `061_D_046` |

## Test scenes

| Date | Time (UTC) | Type | Centre | Orbit | Used | Scene ID (track_direction_frame) |
|---|---|---|---|---|---|---|
| 29 Oct | 08:00 | all ice | 74.4 N, 150.5 W | descending | **no**, empty channel | `074_D_049` |
| 29 Oct | 08:01 | edge | 73.3 N, 156.4 W | descending | **no**, empty channel | `074_D_050` |
| 29 Oct | 09:39 | all ice | 76.3 N, 153.6 W | descending | yes | `075_D_046` |
| 29 Oct | 12:55 | edge | 73.3 N, 156.9 W | ascending | **no**, empty channel | `077_A_040` |
| 29 Oct | 12:56 | edge | 74.3 N, 162.8 W | ascending | **no**, empty channel | `077_A_041` |
| 29 Oct | 14:33 | all water | 69.2 N, 168.1 W | ascending | **no**, empty channel | `078_A_037` |
| 30 Oct | 12:13 | edge | 73.3 N, 146.5 W | ascending | yes | `091_A_040` |
| 31 Oct | 08:19 | edge | 69.2 N, 174.8 W | descending | yes | `103_D_053` |
| 31 Oct | 13:12 | edge | 73.2 N, 161.1 W | ascending | yes | `106_A_040` |
| 31 Oct | 14:50 | all water | 68.8 N, 171.4 W | ascending | **no**, empty channel | `107_A_037` |

## Full file names

```
NISAR_L2_PR_GCOV_030_133_D_047_0505_SVSH_A_20260915T101301_20260915T101332_P05023_N_F_J_001
NISAR_L2_PR_GCOV_030_146_D_049_0505_SVSH_A_20260916T075232_20260916T075303_P05023_N_F_J_001
NISAR_L2_PR_GCOV_030_148_A_045_0505_SVSH_A_20260916T110959_20260916T111044_P05023_N_F_J_001
NISAR_L2_PR_GCOV_030_162_D_046_0505_SVSH_A_20260917T102905_20260917T102941_P05023_N_F_J_001
NISAR_L2_PR_GCOV_031_003_D_046_0505_SVSH_A_20260918T094728_20260918T094804_P05023_F_F_J_001
NISAR_L2_PR_GCOV_031_005_A_040_0505_SVSH_A_20260918T130352_20260918T130427_P05023_F_F_J_001
NISAR_L2_PR_GCOV_031_032_D_048_0505_SVSH_A_20260920T100511_20260920T100544_P05023_F_F_J_001
NISAR_L2_PR_GCOV_031_045_D_049_0505_SVSH_A_20260921T074413_20260921T074443_P05023_N_F_J_001
NISAR_L2_PR_GCOV_031_075_D_048_0505_SHSV_A_20260923T094013_20260923T094046_P05023_N_F_J_001
NISAR_L2_PR_GCOV_031_103_D_053_0505_SVSH_A_20260925T081945_20260925T082012_P05023_N_P_J_001
NISAR_L2_PR_GCOV_031_161_D_049_0505_SVSH_A_20260929T085048_20260929T085119_P05023_N_F_J_001
NISAR_L2_PR_GCOV_032_045_D_050_0505_SVSH_A_20261003T074442_20261003T074518_P05023_N_F_J_001
NISAR_L2_PR_GCOV_032_046_D_046_0505_SVSH_A_20261003T092230_20261003T092306_P05023_N_F_J_001
NISAR_L2_PR_GCOV_032_077_A_040_0505_SVSH_A_20261005T125533_20261005T125608_P05023_N_F_J_001
NISAR_L2_PR_GCOV_032_146_D_050_0505_SVSH_A_20261010T075302_20261010T075337_P05023_N_F_J_001
NISAR_L2_PR_GCOV_032_164_A_039_0505_SVSH_A_20261011T134453_20261011T134530_P05023_N_F_J_001
NISAR_L2_PR_GCOV_033_018_D_047_0505_SVSH_A_20261013T104619_20261013T104649_P05023_N_F_J_001
NISAR_L2_PR_GCOV_033_103_D_050_0505_SVSH_A_20261019T081800_20261019T081835_P05023_N_F_J_001
NISAR_L2_PR_GCOV_033_105_A_045_0505_SVSH_A_20261019T113458_20261019T113542_P05023_N_F_J_001
NISAR_L2_PR_GCOV_033_131_D_051_0505_SVSH_A_20261021T065520_20261021T065557_P05023_N_F_J_001
NISAR_L2_PR_GCOV_034_047_A_045_0505_SVSH_A_20261027T110140_20261027T110225_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_049_A_039_0505_SVSH_A_20261027T141811_20261027T141848_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_060_D_051_0505_SVSH_A_20261028T084333_20261028T084410_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_061_D_046_0505_SVSH_A_20261028T102047_20261028T102122_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_074_D_049_0505_SHSV_A_20261029T080052_20261029T080123_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_074_D_050_0505_SHSV_A_20261029T080122_20261029T080157_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_075_D_046_0505_SVSH_A_20261029T093910_20261029T093940_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_077_A_040_0505_SVSH_A_20261029T125533_20261029T125608_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_077_A_041_0505_SVSH_A_20261029T125607_20261029T125638_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_078_A_037_0505_SVSH_A_20261029T143356_20261029T143416_P05036_N_P_J_001
NISAR_L2_PR_GCOV_034_091_A_040_0505_SVSH_A_20261030T121356_20261030T121431_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_103_D_053_0505_SVSH_A_20261031T081946_20261031T082013_P05036_N_P_J_001
NISAR_L2_PR_GCOV_034_106_A_040_0505_SVSH_A_20261031T131212_20261031T131247_P05036_N_F_J_001
NISAR_L2_PR_GCOV_034_107_A_037_0505_SVSH_A_20261031T145018_20261031T145055_P05036_N_F_J_001
```
