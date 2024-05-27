# modified from https://github.com/LightwheelAI/street-gaussians-ns/blob/main/scripts/pythons/transform2colmap.py
import collections
import json
import os
import shutil
import random
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
import numpy as np

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

        
if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default="")
    parser.add_argument("--cams_meta_file", default="cams_meta_waymo.npy")
    parser.add_argument("--cam_num", default=3, help="number of cameras.")

    args = parser.parse_args()

    # 读取cams_meta.npy
    input_path = args.input_path
    point3D_txt = input_path + "/colmap/sparse/origin/points3D.txt"
    cameras_txt = input_path + "/colmap/sparse/origin/cameras.txt"
    images_txt = input_path + "/colmap/sparse/origin/images.txt"
    folder_path = input_path + "/colmap/sparse/origin"

    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)

    if os.path.exists(point3D_txt):  
        print("文件已存在")  
    else:  
        with open(point3D_txt, 'w') as file:  
            pass 
    
    cams_meta_file = os.path.join(args.input_path, args.cams_meta_file)
    cams_meta = np.load(cams_meta_file)
    # cams_meta.shape = (N, 27)
    # cams_meta[:, 0 :12]: flatten camera poses in RUB
    # cams_meta[:, 12:21]: flatten camse intrinsics
    # cams_meta[:, 21:25]: distortion params [k1, k2, p1, p2]
    # cams_meta[:, 25:27]: bounds [z_near, z_far] (not used.)


    cameras_content={}
    images_content={}
    
    # the order of images are the same as the folder. frame1_cam1, frame1_cam2, frame1_cam3, frame2_cam1, frame2_cam2, frame2_cam3, ...
    for id, camera_data in enumerate(cams_meta):  

        # camera convention RUB -> RDF
        c2w_RUB = camera_data[:12].reshape(3, 4)
        c2w_RUB = np.concatenate([c2w_RUB, np.array([[0, 0, 0, 1]])], axis=0)
        c2w_RDF = np.concatenate([c2w_RUB[:, 0:1], -c2w_RUB[:, 1:2], -c2w_RUB[:, 2:3], c2w_RUB[:, 3:4]], axis=1)

        # In COLMAP, the reconstructed pose of an image is specified as the projection from 
        # world to camera coordinate system (world-to-camera matrix) of an image using a quaternion (QW, QX, QY, QZ) 
        # and a translation vector (TX, TY, TZ). 
        # This is both display int images.txt and the GUI. Be careful!
        w2c = np.linalg.inv(c2w_RDF)
        quaternion = rotmat2qvec(w2c[:3,:3])
        translation = (w2c[:3, 3]).tolist()
        cam_index = id % args.cam_num + 1 # start from 1. Note that is designed for 3 front cameras. If you have more cameras, you need to modify this.
        
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_paras = [quaternion[0], quaternion[1], quaternion[2], quaternion[3], translation[0], translation[1], translation[2], cam_index, f"{id:03d}.png"]
        image_paras = [str(i) for i in image_paras]  
        
        img_index = id + 1
        images_content[img_index] = image_paras
        image_data = imageio.imread(os.path.join(args.input_path, "images", f"{id:03d}.png"))
        w = image_data.shape[1]
        h = image_data.shape[0]
        K = camera_data[12:21].reshape(3,3)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        k1 = camera_data[21]
        k2 = camera_data[22]
        p1 = camera_data[23]
        p2 = camera_data[24]

        if (cam_index in cameras_content):
            continue
        else:
            paras=["OPENCV", w, h, fx, fy, cx, cy, k1, k2, p1, p2]
            paras =[str(i) for i in paras]  
            cameras_content[cam_index] = paras


    with open(cameras_txt, 'w') as f:  
        for cam_index in cameras_content:
            f.write(str(cam_index)+" ")
            f.write(' '.join(cameras_content[cam_index]) + '\n')
            
    with open(images_txt, 'w') as f:  
        for image_index in images_content:
            f.write(str(image_index)+" ")
            f.write(' '.join(images_content[image_index]) + '\n\n')

