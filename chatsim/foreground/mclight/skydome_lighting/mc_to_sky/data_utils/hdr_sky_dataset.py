import os
import cv2
import numpy as np
import torch
from torch import nn
from imageio.v2 import imread
from torch.utils.data import Dataset
from envmap import EnvironmentMap
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction, \
    adjust_exposure, adjust_flip, adjust_rotation, adjust_color_temperature

class HDRSkyDataset(Dataset):
    def __init__(self, args, split='train'):
        root_dir = args['root_dir']
        downsample = args['downsample']

        self.sky_H = args['image_H'] // downsample // 2 
        self.sky_W = args['image_W'] // downsample

        self.root_dir = os.path.join(root_dir, split)
        self.downsample = downsample
        self.file_list = sorted(os.listdir(self.root_dir))
        self.is_train = split=='train'

        self.env_template = EnvironmentMap(self.sky_H, 'skylatlong')

        self.center_align = args.get('center_align', False)
        self.normalize = args.get('normalize', None)
        
        # data augmentation
        self.aug_exposure_range = args.get('aug_exposure_range', [-2.5, 0.5])
        self.aug_temperature_range = args.get('aug_temperature_range', [1, 1])
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
            
        # Load hdr file, RGB order
        hdr_pano = imread(file_path)
        hdr_skypano = hdr_pano[:hdr_pano.shape[0]//2, :, :] # get sky

        # resize the image. sky_H, sky_W, 3
        hdr_skypano = cv2.resize(hdr_skypano, (self.sky_W, self.sky_H))[:,:,:3]

        # augmentation
        if self.is_train:
            hdr_skypano = adjust_exposure(hdr_skypano, self.aug_exposure_range)
            
            hdr_skypano = adjust_flip(hdr_skypano)

            if not self.center_align: # no need to rotate
                hdr_skypano = adjust_rotation(hdr_skypano)
            
            hdr_skypano = adjust_color_temperature(hdr_skypano, self.aug_temperature_range)

        # get peak dir, peak intensity. sRGB color space.
        illumination = 0.2126*hdr_skypano[...,0] + 0.7152*hdr_skypano[...,1] + 0.0722*hdr_skypano[...,2]
        max_index = np.argmax(illumination, axis=None)
        max_index_2d = np.unravel_index(max_index, illumination.shape)
        peak_int_v, peak_int_u = max_index_2d

        # center align the sun
        if self.center_align:
            azimuth = ((self.sky_W // 2 - peak_int_u) % self.sky_W) / self.sky_W * 2 * np.pi
            hdr_skypano = adjust_rotation(hdr_skypano, azimuth)

            # update peak_int_u
            peak_int_u = self.sky_W // 2

        # get peak intensity. R G B 3 channels
        peak_int = hdr_skypano[peak_int_v, peak_int_u]

        # get peak direction
        peak_dir_w_flag = self.env_template.pixel2world(peak_int_u, peak_int_v) # in the [-1, 1] interval, unit sphere
        peak_dir = np.array([peak_dir_w_flag[0], peak_dir_w_flag[1], peak_dir_w_flag[2]])
        
        # gamma correction for LDR.
        ldr_skypano = srgb_gamma_correction(hdr_skypano)

        # if normalize with some percentage  [0, max_intensity * normalize_percentage] -> [0,1] 
        if self.normalize:
            peak_int_R = np.percentile(hdr_skypano[..., 0], self.normalize * 100) # 0.99 -> 99
            peak_int_G = np.percentile(hdr_skypano[..., 1], self.normalize * 100)
            peak_int_B = np.percentile(hdr_skypano[..., 2], self.normalize * 100)
            peak_int = np.array([peak_int_R, peak_int_G, peak_int_B])
            
            # normalize the hdr pano
            hdr_skypano = hdr_skypano / peak_int
            hdr_skypano = hdr_skypano.clip(0, 1)

        # cat direction and intensity to one vector
        peak_vector = np.concatenate([peak_dir, peak_int], axis=-1) # [6,]

        peak_vector_tensor = torch.from_numpy(peak_vector.astype(np.float32))

        # Convert to tensor
        hdr_skypano_tensor = torch.from_numpy(hdr_skypano.astype(np.float32)).permute(2, 0, 1) # C, H, W
        ldr_skypano_tensor = torch.from_numpy(ldr_skypano.astype(np.float32)).permute(2, 0, 1) # C, H, W

        return ldr_skypano_tensor, hdr_skypano_tensor, peak_vector_tensor