# Measured Albedo In the Wild

## Dependencies
* numpy
* html4vision
* shapely
* colour-science
* lensfunpy
* pytorch
* largestinteriorrectangle

## Dataset

[Dataset Link](https://umd.box.com/s/rzuzf12ooqnaxyojjgam09zctgp8fr2c)

Download and unzip `labels.zip` to the code folder so that `labels` is at same level as various scripts. 

`images_png.zip` and `images_raw.zip` are png format images and camera raw format images respectively. `images_raw` will be needed for shading computation. 

## Albedo evaluation

### Preparation

In `utils.py`, change paths in `ALGORITHM_PATHS` to results output by each algorithm, and `IMGS_PATH` to path of MAW images. 

In `run_cmp.py`, update `names` on line 30 to algorithms you want to evaluate. Similarly in `texture_score_lpips.py`, update `names` on line 750 to algothims you want to evaluate. (possible `names` are in utils.py starting at line 159.)

### Evaluation

To evaluate all algorimths for albedo intensity, run:
```
python3 run_cmp.py --meta meta.csv --loss="si" --metric "mean" --type=metric
```

To evaluate all algorimths for albedo chromaticity, run:
```
python3 run_cmp.py --meta meta.csv --loss="per_si" --metric "deltae" --type=metric
```

To evaluate all algorimths for WHDR on MAW, run:
```
python3 run_cmp.py --meta meta.csv --type=whdr
```

To evaluate all algorithms for texture, run:

```
python3 texture_score_lpips.py maw --meta meta.csv --imgs_dir <MAW PNG IMG PATH>  --use_gpu

```

## Shading Evaluation

### Preparation

* With `images_png` and `images_raw` in the same folder, run `python3 compute_shading.py meta.csv --imgs_dir <MAW PNG IMG PATH>` to generate shading labels.
* In `run_shading_cmp.py` update `names` on line 54 to algorithms you want to evaluate.

To evaluate all algorimths for shading, run:

```
python3 run_shading_cmp.py --meta meta.csv --imgs_dir <MAW PNG IMG PATH>
```


## Changes from paper release

Unfortunately, we have to remove 14 images from scene_0 as those images reveal personal credit cards. We will consider restore those images by blackout sensitive area.

The way number of scenes are counted in the dataset release is different from the paper. In the dataset release, some `scene` can contain multiple scenes as counted by the paper, as pictures in those `scenes` come from physically separate areas/different rooms. `scene` containing multiple scenes are listed below:
* `<scene_0>`: contains 2 scenes.
* `<scene_2>`: contains 3 scenes.
* `<scene_2>`: contains 2 scenes.
* `<scene_31>`: contains 4 scenes.
* `<scene_34>`: contains 2 scenes.

