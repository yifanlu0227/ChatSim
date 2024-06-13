"""
@author Jianfei Guo, Shanghai AI Lab

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9;
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer


    suppose such a structure:

    ├── skyseg.py
    └── SegFormer/

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir

    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.

    The 'segformer.b5.1024x1024.city.160k.py' checkpoint can be found here 
    https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw

    Download and put it in SegFormer/

    inference example:
    python skyseg.py -i <DATA_ROOT>/waymo_multi_view/segment-1172406780360799916_1660_000_1680_000_with_camera_labels/colmap/sparse_undistorted/images
"""

from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import click
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import glob


def semantic_seg(image_data_input, semantic_folder_name, sky_folder_name):
    sky_masks_dir = image_data_input.rstrip("/") + '_' + sky_folder_name

    segformer_path = Path(__file__).parent / "SegFormer"
    segformer_path = segformer_path.as_posix()

    config = os.path.join(segformer_path, 'local_configs', 'segformer', 'B5',
                          'segformer.b5.1024x1024.city.160k.py')
    checkpoint = os.path.join(segformer_path,
                              'segformer.b5.1024x1024.city.160k.pth')
    model = init_segmentor(config, checkpoint, device='cuda')

    # iterate through the waymo data
    for filename in os.listdir(image_data_input):
        image_path = os.path.join(image_data_input, filename)
        result = inference_segmentor(model, image_path)

        # get the semantic mask
        semantic_mask = result[0].astype(np.uint8)

        # get the sky mask. 0 is sky, 1 is not sky
        sky_mask = (semantic_mask == 10).astype(np.uint8)
        sky_mask = (1 - sky_mask) * 255

        # should imitate images' filenames and finally impersonate it
        # use png for losslessly preserving the semantic mask!
        # CAN NOT use jpg, because jpg is lossy!
        sky_mask_path = os.path.join(sky_masks_dir, filename) + ".png" # for merge mask

        os.makedirs(os.path.dirname(sky_mask_path), exist_ok=True)
        cv2.imwrite(sky_mask_path, sky_mask)


@click.command()
@click.option("--image_data_input",
              "-i",
              default="/home/jiahuih/workspace/yiflu-workspace/video_for_xcube++/000",
              help="The directory of the waymo data")
@click.option("--semantic_folder_name",
              default="semantic_masks",
              help="The name of folder to save the semantic masks (same directory as images)")
@click.option("--sky_mask_folder_name",
              default="sky_masks",
              help="The name of folder to save the sky masks (same directory as images)")
@click.option("--overwrite",
              "-o",
              is_flag=True,
              help="Whether to overwrite the existing masks")
def main(image_data_input, semantic_folder_name, sky_mask_folder_name, overwrite):
    semantic_seg(image_data_input, semantic_folder_name, sky_mask_folder_name)


if __name__ == "__main__":
    main()
