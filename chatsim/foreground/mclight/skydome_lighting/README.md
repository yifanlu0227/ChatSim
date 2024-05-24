# Multi-camera images to HDR skydome

This is an auxiliary library for ChatSim, predicting skydome environment HDR map from single or multi-view images.

## Installation

```bash
conda activate chatsim
pip install -r requirements.txt
```

**imageio requirement:** To read .exr file, you need FreeImage library. You can obtain it with either:

- download in terminal:
 
    ```imageio_download_bin freeimage```
- download in python CLI: 

    ```import imageio; imageio.plugins.freeimage.download()```

### install this repo as a package

clone the repository and setup this repo.

```
python setup.py develop
mkdir dataset # create a folder for dataset
```

## Data
### HDRI data
We use HDRI data crawled from [Poly Hevean](https://polyhaven.com). All HDRIs subject to [CC0 License](https://polyhaven.com/license). 

You can download our selected and divided HDRIs from [Google Drive](https://drive.google.com/file/d/1dU4Bce3dpcr6lnBJkyG2OKl1GmU8BviQ/view?usp=drive_link), and put it under the `dataset` folder, naming it `HDRi_download`

Or you want to download the HDRIs yourself, you can use the following command.
```
python mc_to_sky/utils/downloader.py 1k outdoor exr
```
This will download all outdoor category HDRI in 1k resolution with extension `.exr`. Not all outdoor HDRIs are suitable for training,
we mannually select and split them into train / val set. See [here](https://drive.google.com/drive/folders/1ossgXhGBwnJ5CpMP8B7Ngm0ZmahUu3nM?usp=drive_link).


### HoliCity
We select outdoor samples from [HoliCity](https://holicity.io/) Panorama dataset. Please download the [resized panorama](https://drive.google.com/file/d/1XkEydyPePKODRUNeWhFcOQ5g-fgqbpgk/view?usp=drive_link), [resized panorama mask](https://drive.google.com/file/d/1qzF8w67qiqg_Im53xuKf6WirBOFfj3oq/view?usp=drive_link), [cropped images](https://drive.google.com/file/d/1I97TtkGXPCjMOUr4RD1L115XnEd4q6iI/view?usp=drive_link), [meta info](https://drive.google.com/drive/folders/1zbgwNBT-4Pvp-kgXXrOqpC0KhcwPyQ8J?usp=drive_link) and unzip them into `dataset` folder.

If you want to process the panorama data yourself, you can download the original [HoliCity](https://holicity.io/) Panorama Dataset from their [Google Drive](https://drive.google.com/file/d/1Qhy2axPtcYG6lKwalE3CStY_eLpUj9nR/edit). Then, you need to resize the panorama and crop perspective view images with `mc_to_sky/tools/holicity/holicity_preprocess.py`. 

Finally, an expected data structure should be like this:
```
dataset
├── HDRi_download 
│   ├── train
│   └── val
├── holicity_meta_info
│   └── selected_sample.json
├── holicity_crop_multiview 
├── holicity_pano_sky_resized_64 
└── holicity_pano_sky_resized_64_mask 
```

## Usage

### Stage 1: Train LDR to HDR autoencoder
**Train**

```bash
python mc_to_sky/tool/train.py -y mc_to_sky/config/stage1/skymodel_peak_enhanced.yaml
```
Then we use the trained model to predict pseduo HDRI GT for HoliCity dataset. You can find a `config.yaml` and the best checkpoint inside the log folder, we denote them `STAGE_1_CONFIG` and `STAGE_1_BEST_CKPT` respectively.

```bash
python mc_to_sky/tools/holicity/holicity_generate_gt.py -y STAGE_1_CONFIG -c STAGE_1_BEST_CKPT --target_dir dataset/holicity_pano_hdr 
```
Now `dataset/holicity_pano_hdr` stores the pseduo HDRI GT.

**Test**

You can validate and test your checkpoint by

```bash
python mc_to_sky/tools/test.py -y STAGE_1_CONFIG -c STAGE_1_BEST_CKPT
```

### Stage 2: Train HDRI predictor from multiview images
**Train**

First edit line 43 of `mc_to_sky/config/stage2/multi_view_avg.yaml`, put `STAGE_1_BEST_CKPT` as the value. Then conduct the stage 2 training. 
```bash
python mc_to_sky/tool/train.py -y mc_to_sky/config/stage2/multi_view_avg.yaml
```
**Test**

You can also validate and test your checkpoint by
```bash
python mc_to_sky/tools/test.py -y STAGE_2_CONFIG -c STAGE_2_BEST_CKPT
```
where `STAGE_2_CONFIG` and `STAGE_2_BEST_CKPT` refers to the new training log.

**Infer**

We directly adopt the model trained on HoliCity for the inference on Waymo dataset.
```bash
python mc_to_sky/tools/infer.py -y STAGE_2_CONFIG -c STAGE_2_BEST_CKPT -i IMAGE_FOLDER -o OUPUT_FOLDER
```

The `IMAGE_FOLDER` should contain continuous image data, for example, pictures from three views should be put together and follow the order in `STAGE_2_CONFIG['view_setting']['view_dis']`. Note that `IMAGE_FOLDER` can include multiple frames, an examplar image sequence can be `[frame_1_front, frame_1_front_left, frame_1_front_right, frame_2_front, frame_2_front_left, frame_2_front_right, ...]`


We further provide `mc_to_sky/tools/infer_waymo_batch.py` which add another loop on the scene-level. 
```bash
python mc_to_sky/tools/infer.py -y STAGE_2_CONFIG -c STAGE_2_BEST_CKPT -waymo_scenes_dir WAYMO_SCENE_DIR -o OUPUT_FOLDER
```
where we suppose the `WAYMO_SCENE_DIR` have the following structure
```
WAYMO_SCENE_DIR
├── segment-10061305430875486848_1080_000_1100_000_with_camera_labels
│   ├── ...
│   └── images # also follows the specific image order
├── segment-10247954040621004675_2180_000_2200_000_with_camera_labels
│   ├── ...
│   └── images
├── segment-10275144660749673822_5755_561_5775_561_with_camera_labels
│   ├── ...
│   └── images
└── ...
```

## Pretrain
download the pretrain from [Google Drive](https://drive.google.com/file/d/1vc8LeChk-wH4YTYEOGbxfng8TB6RBYL7/view?usp=drive_link)
