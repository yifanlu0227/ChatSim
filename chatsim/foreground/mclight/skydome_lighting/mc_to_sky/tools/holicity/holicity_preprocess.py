# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

from envmap import EnvironmentMap, rotation_matrix
import imageio.v2 as imageio
import numpy as np
import os
import cv2
import time
import glob
from tqdm import tqdm
from mc_to_sky.utils.train_utils import check_and_mkdirs
from multiprocessing import Process, Manager, Pool
import json

def test_crop():
    e = EnvironmentMap("dataset/holicity_pano/2008-07/8heFyix0weuW7Kzd6A_BLg.jpg", 'latlong')
    outpath = "crop.png"
    rotation_mat = rotation_matrix(azimuth=np.pi/6, elevation=0) # rad,
    # crop is also rotating right
    # if crop with a positive azimuth rotation alpha
    # sky env short rotate with a negative azimuth -alpha to match the perspective
    crop = (e.project(vfov=60, ar=16/9, resolution=(640, 360), rotation_matrix=rotation_mat)*255).astype(np.uint8) 
   
    imageio.imsave(outpath, crop)


def test_resolution():
    pano_image = imageio.imread("dataset/holicity_pano/2008-07/8heFyix0weuW7Kzd6A_BLg.jpg")
    H = pano_image.shape[0]

    for level in range(8): # level = 2 better quality, level 3 better speed
        time_begin = time.time()
        pano_image_downsample = cv2.resize(pano_image, (H//(2**level) * 2, H//(2**level)))#, interpolation=cv2.INTER_CUBIC)
        e = EnvironmentMap(pano_image_downsample, 'latlong')

        for j in range(3):
            outpath = f"crop_down{level}x_{j}.png"
            rotation_mat = rotation_matrix(azimuth=2*np.pi/10, elevation=0) # rad
            e.rotate(rotation_mat)
            crop = (e.project(vfov=60, ar=16/9, resolution=(640, 360), rotation_matrix=rotation_mat)).astype(np.uint8)
            imageio.imsave(outpath, crop)
        print(f"Level {level} using time: {time.time()-time_begin}")


def resize_all(source_dir='dataset/holicity_pano',
               target_dir='dataset/holicity_pano_resized_800'):
    record_dates = os.listdir(source_dir)
    for record_date in tqdm(record_dates):
        source_date_dir = os.path.join(source_dir, record_date)
        target_date_dir = os.path.join(target_dir, record_date)
        if not os.path.exists(target_date_dir):
            os.mkdir(target_date_dir)
        
        # grab all images
        pano_filenames = os.listdir(source_date_dir)
        for pano_filename in pano_filenames:
            image = imageio.imread(os.path.join(source_date_dir, pano_filename))
            image_resize = cv2.resize(image, (1600, 800))
            imageio.imsave(os.path.join(target_date_dir, pano_filename), image_resize)

def resize_sky(source_dir='dataset/holicity_pano_resized_800',
               target_dir='dataset/holicity_pano_sky_resized_64'):
    record_dates = os.listdir(source_dir)
    for record_date in tqdm(record_dates):
        source_date_dir = os.path.join(source_dir, record_date)
        target_date_dir = os.path.join(target_dir, record_date)
        if not os.path.exists(target_date_dir):
            os.mkdir(target_date_dir)
        
        # grab all images
        pano_filenames = os.listdir(source_date_dir)
        for pano_filename in pano_filenames:
            image = imageio.imread(os.path.join(source_date_dir, pano_filename))
            image_resize = cv2.resize(image[:image.shape[0]//2, :, :], (256, 64))
            imageio.imsave(os.path.join(target_date_dir, pano_filename), image_resize)

def crop_pano_single(ns, record_date):
    # crop the image with 45 (for waymo) degrees as interval
    print(f"In this process, we handle {record_date}")
    source_date_dir = os.path.join(ns['source_dir'], record_date)
    pano_filenames = os.listdir(source_date_dir)

    for azimuth_deg in range(0, 360, ns['degree_interval']):
        check_and_mkdirs(os.path.join(ns['target_dir'], str(azimuth_deg), record_date))

    for pano_filename in ns['selected_sample_dict'][record_date]:
        azimuth_deg = range(0, 360, ns['degree_interval'])[-1]
        pass_flag = os.path.exists(os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename))
        
        if pass_flag: 
            continue

        image = cv2.imread(os.path.join(source_date_dir, pano_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255 
        pano_envmap = EnvironmentMap(image, 'latlong')
        for azimuth_deg in range(0, 360, ns['degree_interval']):
            if os.path.exists(os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename)):
                continue

            azimuth_rad = np.radians(azimuth_deg)
            rotation_mat_i = rotation_matrix(azimuth=azimuth_rad, elevation=0) # rad,
            img_crop = pano_envmap.project(vfov=ns['camera_vfov'], 
                                            ar=ns['aspect_ratio'], 
                                            resolution=(ns['crop_W'], ns['crop_H']), 
                                            rotation_matrix=rotation_mat_i) # can be slow. H, W, 3
            img_crop = (img_crop*255).astype(np.uint8)

            imageio.imsave(
                os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename), 
                img_crop
            )
            print(f"save to {os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename)}")


def crop_pano(source_dir='dataset/holicity_pano',
             target_dir='dataset/holicity_crop_multiview',
             selected_sample_json='dataset/holicity_meta_info/selected_sample.json',
             camera_H=1280,
             camera_W=1920,
             focal=2088.465,
             downsample_for_crop=4,
             degree_interval=45,
             multiprocess=-1,
             ):

    crop_H = camera_H // downsample_for_crop
    crop_W = camera_W // downsample_for_crop
    camera_vfov = np.degrees(np.arctan2(camera_H, camera_W)) * 2
    aspect_ratio = camera_W / camera_H

    with open(selected_sample_json) as f:
        selected_sample = json.load(f)

    sample_dict = {}
    for selected_sample_name in selected_sample:
        date, filename = selected_sample_name.split('/')
        if date in sample_dict:
            sample_dict[date].append(filename)
        else:
            sample_dict[date] = [filename]

    info_dict = {}
    info_dict['crop_H'] = crop_H
    info_dict['crop_W'] = crop_W
    info_dict['camera_vfov'] = camera_vfov
    info_dict['aspect_ratio'] = aspect_ratio
    info_dict['degree_interval'] = degree_interval
    info_dict['source_dir'] = source_dir
    info_dict['target_dir'] = target_dir
    info_dict['selected_sample_dict'] = sample_dict

    record_dates = sorted(sample_dict.keys())

    if multiprocess <= 0:
        for record_date in record_dates:
            crop_pano_single(info_dict, record_date)
    else:
        pool = Pool(multiprocess) # 会使用机器上的CPU核心数作为进程数。
        #  异步地提交任务到进程池。这意味着，对于record_dates中的每一个record_date，你都提交了一个任务。但是，这并不意味着会为每一个任务启动一个新的进程。相反，这些任务会被分配给已经存在的工作进程去执行。如果所有的工作进程都在忙，新的任务会等待直到有一个工作进程可用。
        for record_date in record_dates:
            pool.apply_async(func=crop_pano_single, args=(info_dict, record_date))
        pool.close()
        pool.join()

if __name__ == "__main__":
    Holicity_dataset_dir = 'dataset/holicity_pano' # change to your path
    Holicity_valid_sample_json = 'dataset/holicity_meta_info/selected_sample.json' # change to your path

    Holicity_dataset_resized_1600x800_dir = Holicity_dataset_dir.replace('holicity_pano', 'holicity_pano_resize_800')
    Holicity_dataset_sky_256x64_dir = Holicity_dataset_dir.replace('holicity_pano', 'holicity_pano_sky_resized_64')
    Holicity_dataset_crop_multiview_dir = Holicity_dataset_dir.replace('holicity_pano', 'holicity_crop_multiview')


    resize_all(Holicity_dataset_dir, Holicity_dataset_resized_1600x800_dir)

    resize_sky(Holicity_dataset_resized_1600x800_dir, Holicity_dataset_sky_256x64_dir)

    crop_pano(Holicity_dataset_dir, Holicity_dataset_crop_multiview_dir, Holicity_valid_sample_json)