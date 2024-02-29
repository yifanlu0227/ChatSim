"""
Infer from single image. Like waymo camera data

python '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/tools/infer.py' -y '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/pred_hdr_pano_from_single_1012_195149/config.yaml' -c '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/pred_hdr_pano_from_single_1012_195149/lightning_logs/version_0/checkpoints/epoch=329-val_loss=0.06.ckpt' -i /home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/images_front -o /home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/images_front_hdr

python '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/tools/infer.py' -y '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/unet_recon_pano_mc_to_sky_1011_131303/config.yaml' -c '/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/unet_recon_pano_mc_to_sky_1011_131303/lightning_logs/version_0/checkpoints/epoch=459-val_loss=0.03.ckpt' -i /home/yfl/workspace/f2-nerf/exp/front_3_cam_1346_metashape2waymo/wanjinyou_1012/panorama -o /home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/nerf_pano
"""

import argparse
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from mc_to_sky.data_utils import build_dataset
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction
from mc_to_sky.utils.train_utils import (build_model, check_and_mkdirs,
                                          get_exp_dir)
from mc_to_sky.utils.yaml_utils import read_yaml

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

def get_parser():
    parser = argparse.ArgumentParser(description='Example argparse program')
    parser.add_argument("--config", "-y", type=str, help="path to config file")
    parser.add_argument("--ckpt_path", "-c", type=str, default=None, help="path to ckpt file for restore training")
    parser.add_argument("--image_dir", "-i", type=str, help="path to image directory") 
    parser.add_argument("--output_dir", "-o", type=str, help="path to store hdr panorama")
    args = parser.parse_args()
    return args

def infer_sky(sky_pred, img_crop):
    """
    Args:
        img_crop: torch.tensor
            [1, 3, H, W], range 0-1
    """
    with torch.no_grad():
        peak_vector, latent_vector = sky_pred.latent_predictor(img_crop)
        hdr_skypano, ldr_skypano_pred, _ = sky_pred.decode_forward(latent_vector, peak_vector)
        hdr_skypano = hdr_skypano.permute(0,2,3,1).squeeze().cpu().numpy().astype(np.float32) # [H, W, 3]
        return hdr_skypano

if __name__ == "__main__":
    args = get_parser()
    check_and_mkdirs(args.output_dir)
    hypes = read_yaml(args.config)
    model = build_model(hypes, return_cls=True).load_from_checkpoint(args.ckpt_path, hypes=hypes).to('cuda')
    model.eval()

    model_name = hypes['model']['name']

    if model_name == 'skyunet':
        skip = 1
    elif model_name == 'skypred':
        skip = hypes['dataset']['view_setting']['view_num']

    filename_list = sorted(os.listdir(args.image_dir))

    for idx in tqdm(range(0, len(filename_list), skip)):
        if model_name == 'skyunet':
            filename = filename_list[idx]
            image_path = os.path.join(args.image_dir, filename)
            image = imageio.imread(image_path).astype(np.float32) / 255.0

            input_W = hypes['dataset']['image_W'] // hypes['dataset']['downsample']
            input_H = hypes['dataset']['image_H'] // hypes['dataset']['downsample'] // 2

            image_skypano = cv2.resize(image, (input_W, input_H))
            image_input = torch.from_numpy(image_skypano).permute(2,0,1).unsqueeze(0).to('cuda')

            with torch.no_grad():
                hdr_skypano = model.unet(image_input)

            hdr_skypano = hdr_skypano.permute(0,2,3,1).squeeze().cpu().numpy().astype(np.float32) # [H, W, 3]
            imageio.imwrite(os.path.join(args.output_dir, image_path.split("/")[-1].replace('png', 'exr')), hdr_skypano)

        elif model_name == 'skypred': # skip > 1
            input_W = hypes['dataset']['view_setting']['camera_W'] // hypes['dataset']['view_setting']['downsample_for_crop']
            input_H = hypes['dataset']['view_setting']['camera_H'] // hypes['dataset']['view_setting']['downsample_for_crop']

            image_paths = [os.path.join(args.image_dir, filename_list[idx + ii]) for ii in range(skip)]
            images = [imageio.imread(image_path).astype(np.float32) / 255.0 for image_path in image_paths]
            images_crop = [cv2.resize(image, (input_W, input_H)) for image in images]
            images_input = [torch.from_numpy(image_crop).permute(2,0,1).unsqueeze(0).to('cuda') for image_crop in images_crop]
            
            inputs = torch.stack(images_input, dim=1)
            hdr_skypano = infer_sky(model, inputs)
            imageio.imwrite(os.path.join(args.output_dir, image_paths[0].split("/")[-1].replace('.png', '_sky.exr')), hdr_skypano)

            hdr_fullpano = np.zeros((hdr_skypano.shape[0]*2, hdr_skypano.shape[1], 3), dtype=np.float32)
            hdr_fullpano[:hdr_skypano.shape[0]] = hdr_skypano
            imageio.imwrite(os.path.join(args.output_dir, image_paths[0].split("/")[-1].replace('.png', '.exr')), hdr_fullpano)

            SAVE_FULL_NPZ = False

            if SAVE_FULL_NPZ:
                poses_bounds = image_paths[0].split('images')[0] + 'poses_bounds_waymo.npy'
                waymo_ext_int = np.load(poses_bounds)[:, :15].reshape(-1, 3, 5)
                waymo_ext = waymo_ext_int[idx ,:3,:4]

                waymo_ext_opencv = np.stack([waymo_ext[:,1], waymo_ext[:,0], -waymo_ext[:,2], waymo_ext[:,3]], axis=-1)
                waymo_ext_pad = np.identity(4)
                waymo_ext_pad[:3,:4] = waymo_ext_opencv # 4, 4

                waymo_int = waymo_ext_int[idx ,:3, 4]

                np.savez(os.path.join(args.output_dir, image_paths[0].split("/")[-1].replace('png', 'npz')),
                         H = int(waymo_int[0]),
                         W = int(waymo_int[1]),
                         focal = waymo_int[2],
                         rgb = imageio.imread(image_paths[0]),
                         depth = np.full((int(waymo_int[0]), int(waymo_int[1])), 1e4),
                         extrinsic = waymo_ext_pad
                )
                raise

