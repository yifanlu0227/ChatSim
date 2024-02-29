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
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction, adjust_rotation
from mc_to_sky.utils.train_utils import (build_model, check_and_mkdirs,
                                          get_exp_dir)
from mc_to_sky.utils.yaml_utils import read_yaml

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

def get_parser():
    parser = argparse.ArgumentParser(description='Example argparse program')
    parser.add_argument("--config", "-y", type=str, help="path to config file")
    parser.add_argument("--ckpt_path", "-c", type=str, default=None, help="path to ckpt file for restore training")
    parser.add_argument("--waymo_scenes_dir", "-w", type=str, default="/home/yfl/workspace/f2-nerf/data/waymo_multi_view", help="path to image directory") 
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
    # with torch.no_grad():
    #     latent_vector_pred, azimuth_logit = sky_pred.latent_predictor(img_crop)
    #     decoded_hdr_pred = sky_pred.decode_forward(latent_vector_pred)
    #     hdr_skypano = decoded_hdr_pred.permute(0,2,3,1).squeeze().cpu().numpy().astype(np.float32) # [H, W, 3]
    #     num_bins = 16
    #     bin_size = 2 * np.pi / num_bins
    #     bin_edges = torch.arange(0, 2*np.pi, bin_size)  # 计算每个bin的边缘角度
    #     bin_index = torch.argmax(azimuth_logit, dim=1)  # 找到概率最高的bin
    #     angle = bin_edges[bin_index]

    #     azimuth_angle = angle.squeeze().cpu().numpy().tolist()
    #     hdr_skypano = adjust_rotation(hdr_skypano, azimuth=-azimuth_angle).astype(np.float32)

    #     return hdr_skypano

def main():
    args = get_parser()
    hypes = read_yaml(args.config)
    model = build_model(hypes, return_cls=True).load_from_checkpoint(args.ckpt_path, hypes=hypes).to('cuda')
    model.eval()

    model_name = hypes['model']['name']

    skip = hypes['dataset']['view_setting']['view_num']
    skip = 3
    all_waymo = args.waymo_scenes_dir
    scenes = os.listdir(all_waymo)
    # scenes.remove('1172','1308', '1433', '1522', '1534')

    for scene in tqdm(scenes):
        scene_image_dir = os.path.join(all_waymo, scene, 'images') 
        scene_output_dir = os.path.join(args.output_dir, scene)
        check_and_mkdirs(scene_output_dir)

        filename_list = sorted(os.listdir(scene_image_dir))

        for idx in range(0, len(filename_list), skip):
            input_W = hypes['dataset']['view_setting']['camera_W'] // hypes['dataset']['view_setting']['downsample_for_crop']
            input_H = hypes['dataset']['view_setting']['camera_H'] // hypes['dataset']['view_setting']['downsample_for_crop']

            image_paths = [os.path.join(scene_image_dir, filename_list[idx + ii]) for ii in range(skip)]
            images = [imageio.imread(image_path).astype(np.float32) / 255.0 for image_path in image_paths]
            images_crop = [cv2.resize(image, (input_W, input_H)) for image in images]
            images_input = [torch.from_numpy(image_crop).permute(2,0,1).unsqueeze(0).to('cuda') for image_crop in images_crop]
            
            inputs = torch.stack(images_input, dim=1)
            hdr_skypano = infer_sky(model, inputs)
            imageio.imwrite(os.path.join(scene_output_dir, image_paths[0].split("/")[-1].replace('.png', '_sky.exr')), hdr_skypano)

            hdr_fullpano = np.zeros((hdr_skypano.shape[0]*2, hdr_skypano.shape[1], 3), dtype=np.float32)
            hdr_fullpano[:hdr_skypano.shape[0]] = hdr_skypano
            imageio.imwrite(os.path.join(scene_output_dir, image_paths[0].split("/")[-1].replace('.png', '.exr')), hdr_fullpano)

            SAVE_FULL_NPZ = False

            if SAVE_FULL_NPZ:
                poses_bounds = image_paths[0].split('images')[0] + 'poses_bounds.npy'
                waymo_ext_int = np.load(poses_bounds)[:, :15].reshape(-1, 3, 5)
                waymo_ext = waymo_ext_int[idx ,:3,:4]

                waymo_ext_opencv = np.stack([waymo_ext[:,1], waymo_ext[:,0], -waymo_ext[:,2], waymo_ext[:,3]], axis=-1)
                waymo_ext_pad = np.identity(4)
                waymo_ext_pad[:3,:4] = waymo_ext_opencv # 4, 4

                waymo_int = waymo_ext_int[idx ,:3, 4]

                print(os.path.join(scene_output_dir, image_paths[0].split("/")[-1].replace('png', 'npz')))

                np.savez(os.path.join(scene_output_dir, image_paths[0].split("/")[-1].replace('png', 'npz')),
                        H = int(waymo_int[0]),
                        W = int(waymo_int[1]),
                        focal = waymo_int[2],
                        rgb = imageio.imread(image_paths[0]),
                        depth = np.full((int(waymo_int[0]), int(waymo_int[1])), 1e4),
                        extrinsic = waymo_ext_pad
                )


if __name__ == "__main__":
    main()