"""
This dataset provide ldr data input for pretrained sky model
"""

import json
import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from envmap import EnvironmentMap, rotation_matrix
import random
from imageio.v2 import imread
import time
from line_profiler import LineProfiler
from random import choice


def totensor(x: np.ndarray):
    if len(x.shape) == 3:
        return torch.from_numpy(x.astype(np.float32)).permute(2,0,1)
    return torch.from_numpy(x.astype(np.float32))

class HoliCitySDRDataset(Dataset):
    def __init__(self, args, split='train'):
        self.multicrop_dir = args['multicrop_dir'] # holicity pano multi view crop 
        self.skymask_dir = args['skymask_dir'] # 64*256 sky mask for holicity sky pano
        self.skyldr_dir = args['skyldr_dir'] # 64*256 sky mask for holicity sky pano
        self.skyhdr_dir = args['skyhdr_dir'] # pt file for predicted holicity hdr sky

        selected_sample_json = args['selected_sample_json'] # json
        view_args = args['view_setting']
        self.crop_H = view_args['camera_H'] // view_args['downsample_for_crop']
        self.crop_W = view_args['camera_W'] // view_args['downsample_for_crop']
        self.camera_vfov = np.degrees(np.arctan2(view_args['camera_H']/2, view_args['focal'])) * 2
        self.aspect_ratio = view_args['camera_W'] / view_args['camera_H']

        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis'] # deg

        self.sky_pano_H = args['sky_pano_H']
        self.sky_pano_W = args['sky_pano_W']

        with open(selected_sample_json,'r') as f:
            self.select_sample = json.load(f)

        random.seed(303)
        random.shuffle(self.select_sample)
        all_sample_num = len(self.select_sample)
        train_ratio = 0.8
        self.train_file_list = self.select_sample[:int(all_sample_num*train_ratio)]
        self.val_file_list = self.select_sample[int(all_sample_num*train_ratio):]

        self.is_train = split=='train'
        if self.is_train:
            self.file_list = self.train_file_list
        else:
            self.file_list = self.val_file_list

        self.aug_rotation = True if self.is_train else False

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sky_ldr_path = os.path.join(self.skyldr_dir, self.file_list[idx])
        sky_mask_path = os.path.join(self.skymask_dir, self.file_list[idx])
        sky_hdr_path = os.path.join(self.skyhdr_dir, self.file_list[idx].replace('.jpg', '.npz'))

        ldr_skypano = imread(sky_ldr_path) / 255
        sky_mask = imread(sky_mask_path).astype(np.float32) / 255
        sky_hdr_dict = np.load(sky_hdr_path)

        peak_vector = sky_hdr_dict['peak_vector'] # np
        latent_vector = sky_hdr_dict['latent_vector'] # np
        hdr_skypano = sky_hdr_dict['hdr_skypano'] # np, [H, W, 3]

        ldr_envmap = EnvironmentMap(ldr_skypano, 'skylatlong')
        hdr_envmap = EnvironmentMap(hdr_skypano, 'skylatlong')
        mask_envmap = EnvironmentMap(sky_mask, 'skylatlong')

        if self.aug_rotation:
            azimuth_deg = choice(range(0, 360, 45))
            azimuth_rad = np.radians(azimuth_deg)
            rotation_mat = rotation_matrix(azimuth = azimuth_rad, elevation = 0)
            inv_rotation_mat = rotation_matrix(azimuth = -azimuth_rad, elevation = 0)
        else:
            azimuth_deg = 0

        img_crops_tensor_list = []
        for i in range(self.view_num):
            azimuth_deg_i = (azimuth_deg + self.view_dis_deg[i]) % 360
            azimuth_deg_i = int(azimuth_deg_i)
            img_crop_path = os.path.join(self.multicrop_dir, str(azimuth_deg_i), self.file_list[idx])
            img_crop = imread(img_crop_path) / 255
            img_crops_tensor_list.append(totensor(img_crop))
        
        if self.aug_rotation:
            hdr_envmap.rotate(dcm = inv_rotation_mat) # can be slow
            mask_envmap.rotate(dcm = inv_rotation_mat) # can be slow
            ldr_envmap.rotate(dcm = inv_rotation_mat) # can be slow

            # peak dir vector should be rotated.
            peak_vector[:3] = (rotation_mat @ peak_vector[:3].reshape(3,1)).flatten()

        img_crops_tensor = torch.stack(img_crops_tensor_list)
        peak_vector_tensor = totensor(peak_vector)
        latent_vector_tensor = totensor(latent_vector)
        mask_envmap_tensor = totensor(mask_envmap.data)
        hdr_envmap_tensor = totensor(hdr_envmap.data)
        ldr_envmap_tensor = totensor(ldr_envmap.data)

        return (img_crops_tensor,  # [N_view, 3, H, W]
                peak_vector_tensor, 
                latent_vector_tensor, 
                mask_envmap_tensor, 
                hdr_envmap_tensor,
                ldr_envmap_tensor)

if __name__ == "__main__":
    from mc_to_sky.utils.yaml_utils import read_yaml
    yaml_path = "mc_to_sky/config/pred_from_single/demo.yaml"
    hypes = read_yaml(yaml_path)
    args = hypes['dataset']
    dataset = HoliCitySDRDataset(args, 'train')

    lp = LineProfiler()
    lp_wrapper = lp(dataset.__getitem__)
    lp_wrapper(0) # arg to dataset.__getitem__
    lp.print_stats()
