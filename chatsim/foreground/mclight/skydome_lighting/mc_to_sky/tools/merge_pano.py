import torch
import cv2
import numpy as np
import os
import imageio.v2 as imageio
from icecream import ic

def merge(last_trans, sky_hdri, sur_hdri):
    """
        Merge the sky hdri and the surrounding env hdri to the final hdri

        last_trans : torch.Tensor
            shape [H, W, 1]

        sky_hdri : torch.Tensor
            shape [H, W, 3]

        sur_hdri: torch.Tensor
            shape [H, W, 3]
    """
    return sur_hdri * (1 - last_trans) + sky_hdri * last_trans

if __name__ == "__main__":
    last_trans = torch.jit.load('/home/yfl/workspace/f2-nerf/panorama/tmp/last_trans.pt').state_dict()['0']
    ic(last_trans.shape) # 1280, 5120, 1
    
    # sky_hdri = imageio.imread("/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/images_front_hdr/000.exr")
    sky_hdri = imageio.imread("/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/images_front_hdr/000.exr")

    sur_hdri = imageio.imread("/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/nerf_pano/20000_000.exr")

    last_trans = cv2.resize(last_trans.cpu().numpy(), (sky_hdri.shape[1], sky_hdri.shape[0]))[..., np.newaxis]

    ic(sky_hdri.shape)
    ic(sur_hdri.shape)
    ic(last_trans.shape)

    merged_hdri = merge(last_trans, sky_hdri, sur_hdri) # sky pano, H, W, 3

    # sky pano to full pano
    env_skypano = np.zeros((merged_hdri.shape[0]*2, merged_hdri.shape[1], 3)).astype(np.float32)
    env_skypano[:merged_hdri.shape[0],:,:] = merged_hdri

    imageio.imsave("/home/yfl/workspace/LDR_to_HDR/mc_to_sky/logs/waymo_1346/merged_hdri/000_step12.exr", env_skypano)


    
    