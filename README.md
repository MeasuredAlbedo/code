# Measured Albedo in the Wild

This repository contains evaluation code for the Measured Albedo in the Wild
(MAW) dataset.

## Environment

Build the evaluator image locally:

```bash
docker build -t measured-albedo .
```

Or pull the shared evaluator environment:

```bash
docker pull public.ecr.aws/z8e4h4q6/measured-albedo/evaluator
```

The Docker image includes the numeric dependencies used by the MAW scripts and
by MAW 2.0. LPIPS runs with CPU PyTorch by default; pass `--use_gpu` to
`texture_score_lpips.py` only in an environment with a CUDA PyTorch install.

For a local Python environment, install CPU PyTorch first, then the shared
requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## MAW Dataset

Download the MAW dataset:

```text
https://umd.box.com/s/rzuzf12ooqnaxyojjgam09zctgp8fr2c
```

Unzip `labels.zip` into this repository so `labels/` is next to the evaluation
scripts. `images_png.zip` contains PNG images. `images_raw.zip` contains camera
RAW images and is needed for shading-label generation.

## MAW Albedo Evaluation

By default, `utils.py` looks for the MAW images and method outputs under:

```text
output/images
output/ours
output/ravi
output/cgintrinsics
output/bigtime
output/soumyadip
output/usi3d
output/revisit
output/bell2014
output/niid
output/retinex
output/nestmeyer
output/l1trans
```

Set `MAW_OUTPUT_ROOT` to use another root, or set a per-method path such as
`MAW_IMGS_PATH`, `MAW_OURS_PATH`, or `MAW_RAVI_PATH`. Missing configured
directories fail immediately when they are used.

Edit the `names` lists in `run_cmp.py` and `texture_score_lpips.py` to choose
which methods to evaluate. The method keys are defined in `utils.py`.

Albedo intensity:

```bash
python3 run_cmp.py --meta meta.csv --loss si --metric mean --type metric
```

Albedo chromaticity:

```bash
python3 run_cmp.py --meta meta.csv --loss per_si --metric deltae --type metric
```

WHDR:

```bash
python3 run_cmp.py --meta meta.csv --type whdr
```

Texture LPIPS:

```bash
python3 texture_score_lpips.py maw --meta meta.csv --imgs_dir /path/to/images_png --use_gpu
```

## MAW Shading Evaluation

With `images_png/` and `images_raw/` in the same folder, generate shading
labels:

```bash
python3 compute_shading.py meta.csv --imgs_dir /path/to/images_png
```

Then edit the `names` list in `run_shading_cmp.py` and run:

```bash
python3 run_shading_cmp.py --meta meta.csv --imgs_dir /path/to/images_png
```

## Notes on the Paper Release

The public MAW dataset removes 14 images from `scene_0` because those images
revealed personal credit cards. We will consider restore those images by blackout sensitive area.

The dataset release counts scenes differently from the paper: some released
`scene_*` folders contain images from multiple physical areas or rooms. The
affected folders are:

```text
<scene_0>: contains 2 scenes.
<scene_2>: contains 3 scenes.
<scene_2>: contains 2 scenes.
<scene_31>: contains 4 scenes.
<scene_34>: contains 2 scenes.
```

## MAW 2.0 Update for GLOW

MAW 2.0 evaluates measured albedo on public [GLOW](https://glow-inverse-rendering.github.io/) real-scene validation splits.
The GLOW image data is distributed with the GLOW dataset release;
the MAW 2.0 archive contains only measurement annotations.

## Download

[Dataset Link](https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/glow_maw2_measurements_release.zip)

The public MAW 2.0 contains measurements for the following splits:

```text
coffee_table_colocated_val
coffee_table_natural_val
window_sill_colocated_val
window_sill_natural_val
shoe_rack_colocated_val
shoe_rack_natural_val
```

Each split has a metadata file:

```text
meta_2_0/<split>.csv
```

and measurements under:

```text
phase_2_0/masks/<split>/
```

## Prediction Format

Evaluate method albedo predictions as files named:

```text
{sorted_image_index}_{method}.exr
```

For example:

```text
0_nerad.exr
1_nerad.exr
```

## Evaluation

Evaluate albedo intensity:

```bash
python evaluate_maw2_glow.py \
  --measurements-root /path/to/glow_maw2_measurements_release \
  --meta meta_2_0/coffee_table_colocated_val.csv \
  --method nerad \
  --loss si \
  --metric mean \
  /path/to/predictions \
  coffee_table_colocated_val_nerad_mse.csv
```

Evaluate albedo chromaticity:

```bash
python evaluate_maw2_glow.py \
  --measurements-root /path/to/glow_maw2_measurements_release \
  --meta meta_2_0/coffee_table_colocated_val.csv \
  --method nerad \
  --loss per_si \
  --metric deltae \
  /path/to/predictions \
  coffee_table_colocated_val_nerad_deltae.csv
```

Docker example:

```bash
docker run --rm \
  -v /path/to/glow_maw2_measurements_release:/measurements:ro \
  -v /path/to/predictions:/predictions:ro \
  -v "$PWD":/outputs \
  public.ecr.aws/z8e4h4q6/measured-albedo/evaluator \
  -lc 'python evaluate_maw2_glow.py \
    --measurements-root /measurements \
    --meta meta_2_0/coffee_table_colocated_val.csv \
    --method nerad \
    --loss si \
    --metric mean \
    /predictions \
    /outputs/coffee_table_colocated_val_nerad_mse.csv'
```
