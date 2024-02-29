import argparse
import os
import torch
import imageio.v2 as imageio
import numpy as np
import pytorch_lightning as pl
from icecream import ic
from torch.utils.data import DataLoader, Dataset, Subset

from mc_to_sky.data_utils import build_dataset
from mc_to_sky.utils.train_utils import (build_model, check_and_mkdirs,
                                          get_exp_dir)
from mc_to_sky.utils.yaml_utils import read_yaml

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

def get_parser():
    parser = argparse.ArgumentParser(description='Example argparse program')
    parser.add_argument("--config", "-y", type=str, help="path to config file")
    parser.add_argument("--ckpt_path", "-c", type=str, default=None, help="path to ckpt file for restore training")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    hypes = read_yaml(args.config)

    test_set = build_dataset(hypes, split='val')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    model = build_model(hypes, return_cls=True).load_from_checkpoint(args.ckpt_path, hypes=hypes)
    
    trainer = pl.Trainer(accelerator=hypes['train_conf']['accelerator'],
                         devices=1,
                         logger=False) # no default_root_dir
    predictions = trainer.predict(model, test_loader)
    
    # create path for visualiaztion
    vis_dir = args.ckpt_path.split("checkpoints")[0] + "visualization"
    check_and_mkdirs(vis_dir)

    for pred in predictions:
        batch_idx = pred['batch_idx']

        hdr_skypano_pred = pred['hdr_skypano_pred'].squeeze().cpu().numpy().astype(np.float32)
        hdr_skypano_gt = pred['hdr_skypano_gt'].squeeze().cpu().numpy().astype(np.float32)

        imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_hdr_pred.exr"), hdr_skypano_pred)
        imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_hdr_gt.exr"), hdr_skypano_gt)

        if 'ldr_skypano_input' in pred:
            ldr_skypano_input = (pred['ldr_skypano_input'].squeeze().cpu().numpy()*255).astype(np.uint8) 
            imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_ldr_input.png"), ldr_skypano_input)

        if 'ldr_skypano_pred' in pred:
            ldr_skypano_pred = (pred['ldr_skypano_pred'].squeeze().cpu().numpy()*255).astype(np.uint8) 
            imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_ldr_pred.png"), ldr_skypano_pred)

        if 'mask_env' in pred:
            mask = (pred['mask_env'][0].cpu().numpy()*255).astype(np.uint8)
            imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_mask.png"), mask)

        if 'image_crops' in pred:
            image_crops = (pred['image_crops'].flatten(0,2).cpu().numpy()*255).astype(np.uint8) # B * N_view * H, W, C
            imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_img_crops.png"), image_crops)

        if 'azimuth_angle' in pred:
            # Predict from image with center aligned training. use azimuth to rotate hdr pano
            from envmap import EnvironmentMap, rotation_matrix
            from mc_to_sky.utils.hdr_utils import adjust_rotation

            azimuth_angle = pred['azimuth_angle'].squeeze().cpu().numpy().tolist()
            hdr_skypano_pred_rotated = adjust_rotation(hdr_skypano_pred, azimuth=-azimuth_angle)
            imageio.imsave(os.path.join(vis_dir, f"{batch_idx:04}_hdr_pred_rotated.exr"), hdr_skypano_pred_rotated.astype(np.float32))


