from envmap import EnvironmentMap, rotation_matrix
import numpy as np
import imageio.v2 as imageio

def test_interpolate():
    e = EnvironmentMap("/home/yfl/workspace/dataset_ln/holicity_pano(full_resolution)/2008-07/8heFyix0weuW7Kzd6A_BLg.jpg", 'latlong')
    outpath = "/home/yfl/workspace/LDR_to_HDR/crop_scipy.png"
    rotation_mat = rotation_matrix(azimuth=np.pi/6, elevation=0) # rad,
    crop = (e.project(vfov=60, ar=16/9, resolution=(640, 360), rotation_matrix=rotation_mat)*255).astype(np.uint8)
    imageio.imsave(outpath, crop)

if __name__ == "__main__":
    test_interpolate()