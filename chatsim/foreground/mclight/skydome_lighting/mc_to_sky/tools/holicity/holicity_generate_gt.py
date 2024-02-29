import os
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from mc_to_sky.utils.yaml_utils import read_yaml
from mc_to_sky.model.skymodel import SkyModel
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction
from mc_to_sky.utils.train_utils import (build_model, check_and_mkdirs,
                                          get_exp_dir)
import imageio.v2 as imageio
import torch
import cv2
from tqdm import tqdm
import json

def infer_and_save(sky_model, ldr_skypano, save_dir, filename):
    """
    Args:
        ldr_skypano: torch.tensor
            [1, 3, H, W]
    """
    with torch.no_grad():
        peak_vector, latent_vector = sky_model.encode_forward(ldr_skypano)
        hdr_skypano, ldr_skypano, _ = sky_model.decode_forward(latent_vector, peak_vector, peak_vector)

        sample_pseudo_gt = {
            'peak_vector': peak_vector.squeeze().cpu().numpy(), # [6,]
            'latent_vector': latent_vector.squeeze().cpu().numpy(), # [64,]
            'hdr_skypano': hdr_skypano.permute(0,2,3,1).squeeze().cpu().numpy().astype(np.float32)
        }
        np.savez(os.path.join(save_dir, filename.replace('.jpg', '.npz')), **sample_pseudo_gt)

        # save gamma corrected for visualization
        hdr_skypano = hdr_skypano.permute(0,2,3,1).squeeze().cpu().numpy().astype(np.float32) # [H, W, 3]
        
        return hdr_skypano

def get_parser():
    parser = argparse.ArgumentParser(description='Example argparse program')
    parser.add_argument("--config", "-y", type=str, help="path to config file")
    parser.add_argument('--ckpt_path', '-c', type=str, help='checkpoint path', required=True)
    parser.add_argument('--target_dir', type=str, help='output directory for estimated hdri', required=True)
    parser.add_argument('--holicity_sky_data', type=str, help='directory of resized sky holicity dataset', default='dataset/holicity_pano_sky_resized_64')
    parser.add_argument('--selected_sample_json', type=str, help='path to selected sample json file', default='dataset/holicity_meta_info/selected_sample.json')
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_parser()
    hypes = read_yaml(args.config)

    holicity_sky_dir = args.holicity_sky_data
    target_dir = args.target_dir
    target_vis_dir = args.target_dir + '_vis'
    selected_sample_json = args.selected_sample_json

    model = build_model(hypes, return_cls=True).load_from_checkpoint(args.ckpt_path, hypes=hypes).cuda()
    model.eval()

    with open(selected_sample_json) as f:
        selected_sample = json.load(f)

    record_dates = os.listdir(holicity_sky_dir)
    for record_date in tqdm(record_dates):
        source_date_dir = os.path.join(holicity_sky_dir, record_date)
        target_date_dir = os.path.join(target_dir, record_date)
        target_vis_date_dir = os.path.join(target_vis_dir, record_date)
        check_and_mkdirs(target_date_dir)
        check_and_mkdirs(target_vis_date_dir)

        filenames = os.listdir(source_date_dir)
        for filename in filenames:

            # we already select the samples.
            # if not, comment these lines
            if os.path.join(record_date, filename) not in selected_sample:
                continue

            sky_pano_image = imageio.imread(os.path.join(source_date_dir, filename)).astype(np.float32) # [H, W, 3]
            sky_pano_image /= 255

            sky_pano_image_input = cv2.resize(sky_pano_image, (256, 64)).astype(np.float32)
            sky_pano_tensor = torch.from_numpy(sky_pano_image_input).permute(2,0,1).unsqueeze(0).to('cuda')
            hdr_skypano_np = infer_and_save(model, sky_pano_tensor, target_date_dir, filename)
            vis_hdr = True

            if vis_hdr:
                hdr_skypano_np = cv2.resize(hdr_skypano_np, (sky_pano_image.shape[1], sky_pano_image.shape[0]))
                ldr_skypano_np = srgb_gamma_correction(hdr_skypano_np)
                sky_and_hdr_sky = np.concatenate([ldr_skypano_np, sky_pano_image], axis=0).clip(0, 1)*255
                sky_and_hdr_sky = sky_and_hdr_sky.astype(np.uint8)
                imageio.imsave(os.path.join(target_vis_date_dir, filename.replace('.jpg','.exr')), hdr_skypano_np)
                imageio.imsave(os.path.join(target_vis_date_dir, filename), sky_and_hdr_sky)