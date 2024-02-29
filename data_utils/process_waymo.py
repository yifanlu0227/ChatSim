import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import imageio
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import pickle
from collections import defaultdict
from copy import deepcopy
# import waymo
import cv2
import open3d as o3d
import copy

SAVE_INTRINSIC = True
SINGLE_TRACK_INFO_FILE = True
DEBUG = False
MULTIPLE_DIRS = False

# CAMERAS = {
#     'FRONT': 0,
#     'FRONT_LEFT': 1,
#     'FRONT_RIGHT': 2,
#     'SIDE_LEFT': 3,
#     'SIDE_RIGHT': 4,
# }
CAMERAS = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_dir', type=str)
    parser.add_argument('--nerf_data_dir', type=str, default='/path/to/nerf/data/dir')
    parser.add_argument('-nd', '--no_data', action='store_true')
    parser.add_argument('--frame_nums', type=int, default=60)
    parser.add_argument('--start_frame', type=int, default=0)

    return parser.parse_args()


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])


def extract_label_fields(l,dims):
    assert dims in [2,3]
    label_dict = {'c_x':l.box.center_x,'c_y':l.box.center_y,'width':l.box.width,'length':l.box.length,'type':l.type}
    if dims==3:
        label_dict['c_z'] = l.box.center_z
        label_dict['height'] = l.box.height
        label_dict['heading'] = l.box.heading
    return label_dict


def read_intrinsic(intrinsic_params_vector):
    return dict(zip(['f_u', 'f_v', 'c_u', 'c_v', 'k_1', 'k_2', 'p_1', 'p_2', 'k_3'], intrinsic_params_vector))


def get_shutter(filename, save_path, start_frame = 0, end_frame = 40):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    dataset_iter = list(dataset.as_numpy_iterator())

    frame = open_dataset.Frame()
    print(f"{filename}")
    shutter_save = []
    for i, frame_data in tqdm(enumerate(dataset_iter)):
        if (i < start_frame) or (i >= end_frame):
            continue
        frame.ParseFromString(frame_data)
        for image in frame.images:
            if open_dataset.CameraName.Name.Name(image.name) in CAMERAS:
                shutter_save.append(image.shutter)
    shutter_save = np.array(shutter_save)

    mean = shutter_save.mean()
    std = shutter_save.std()

    shutter_save = (shutter_save - mean) / std

    for i in range(len(shutter_save)):
        filename = f"{save_path}/{str(i).zfill(3)}.txt"
        with open(filename, 'w') as f:
            f.write(str(shutter_save[i]))


def main():
    args = parse_args()
    datadirs = args.tfrecord_dir
    export_data = not args.no_data
    scene_name = datadirs.split('/')[-1][:-9]
    nerf_data_dir = os.path.join(args.nerf_data_dir, scene_name)
    saving_dir = '/'.join(datadirs.split('/')[:-1])

    if '.tfrecord' not in datadirs:
        saving_dir = 1*datadirs
        datadirs = glob.glob(datadirs+'/*.tfrecord',recursive=True)
        datadirs = sorted([f for f in datadirs if '.tfrecord' in f])
        MULTIPLE_DIRS = True

    if not isinstance(datadirs,list):   datadirs = [datadirs]
    if not os.path.isdir(saving_dir):   os.mkdir(saving_dir)
    if not os.path.isdir(nerf_data_dir): os.mkdir(nerf_data_dir)

    isotropic_focal = lambda intrinsic_dict: intrinsic_dict['f_u']==intrinsic_dict['f_v']

    for file_num,file in enumerate(datadirs):
        if SINGLE_TRACK_INFO_FILE:
            tracking_info = {}
        if file_num > 0 and DEBUG:   break
        file_name = file.split('/')[-1].split('.')[0]
        print('Processing file ',file_name)
        if not os.path.isdir(os.path.join(saving_dir, file_name)):   os.mkdir(os.path.join(saving_dir, file_name))
        if not os.path.isdir(os.path.join(saving_dir,file_name, 'images')):   os.mkdir(os.path.join(saving_dir,file_name, 'images'))
        if not os.path.isdir(os.path.join(saving_dir, file_name, 'point_cloud')):   os.mkdir(os.path.join(saving_dir, file_name, 'point_cloud'))
        if not SINGLE_TRACK_INFO_FILE:
            if not os.path.isdir(os.path.join(saving_dir,file_name, 'tracking')):   os.mkdir(os.path.join(saving_dir,file_name, 'tracking'))
        dataset = tf.data.TFRecordDataset(file, compression_type='')
        for f_num, data in enumerate(tqdm(dataset)):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            pose = np.zeros([len(frame.images), 4, 4])
            im_paths = {}
            pcd_paths = {}
            if SAVE_INTRINSIC:
                intrinsic = np.zeros([len(frame.images),9])
            extrinsic = np.zeros_like(pose)
            width,height,camera_labels = np.zeros([len(frame.images)]),np.zeros([len(frame.images)]),defaultdict(dict)
            for im in frame.images:
                saving_name = os.path.join(saving_dir,file_name, 'images','%03d_%s.png'%(f_num,open_dataset.CameraName.Name.Name(im.name)))
                if not DEBUG and export_data:
                    im_array = tf.image.decode_jpeg(im.image).numpy()
                    imageio.imwrite(saving_name, im_array, compress_level=3)
                pose[im.name-1, :, :] = np.reshape(im.pose.transform, [4, 4])
                im_paths[im.name] = saving_name
                extrinsic[im.name-1, :, :] = np.reshape(frame.context.camera_calibrations[im.name-1].extrinsic.transform, [4, 4])
                if SAVE_INTRINSIC:
                    intrinsic[im.name-1, :] = frame.context.camera_calibrations[im.name-1].intrinsic
                    assert isotropic_focal(read_intrinsic(intrinsic[im.name-1, :])),'Unexpected difference between f_u and f_v.'
                width[im.name-1] = frame.context.camera_calibrations[im.name-1].width
                height[im.name-1] = frame.context.camera_calibrations[im.name-1].height
                for obj_label in frame.projected_lidar_labels[im.name-1].labels:
                    camera_labels[im.name][obj_label.id.replace('_'+open_dataset.CameraName.Name.Name(im.name),'')] = extract_label_fields(obj_label,2)
            # Extract point cloud data from stored range images
            laser_calib = np.zeros([len(frame.lasers), 4,4])
            if export_data:
                (range_images, camera_projections, range_image_top_pose) = \
                    frame_utils.parse_range_image_and_camera_projection(frame)
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                                                range_images,
                                                                                camera_projections,
                                                                                range_image_top_pose)
            else:
                points =np.empty([len(frame.lasers), 1])

            laser_mapping = {}
            for (laser, pts) in zip(frame.lasers, points):
                saving_name = os.path.join(saving_dir, file_name, 'point_cloud', '%03d_%s.ply' % (f_num, open_dataset.LaserName.Name.Name(laser.name)))
                if export_data:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    o3d.io.write_point_cloud(saving_name, pcd)
                calib_id = int(np.where(np.array([cali.name for cali in frame.context.laser_calibrations[:5]]) == laser.name)[0])
                laser_calib[laser.name-1, :, :] = np.reshape(frame.context.laser_calibrations[calib_id].extrinsic.transform, [4, 4])
                pcd_paths[laser.name] = saving_name
                laser_mapping.update({open_dataset.LaserName.Name.Name(laser.name): calib_id})

            if 'intrinsic' in tracking_info:
                assert np.all(tracking_info['intrinsic']==intrinsic) and np.all(tracking_info['width']==width) and np.all(tracking_info['height']==height)
            else:
                tracking_info['intrinsic'],tracking_info['width'],tracking_info['height'] = intrinsic,width,height
            dict_2_save = {'per_cam_veh_pose':pose,'cam2veh':extrinsic,'im_paths':im_paths,'width':width,'height':height,
                        'veh2laser':laser_calib, 'pcd_paths': pcd_paths, 'focal': intrinsic[:, 0]}
            if SAVE_INTRINSIC and SINGLE_TRACK_INFO_FILE:
                dict_2_save['intrinsic'] = intrinsic
            lidar_labels = {}
            for obj_label in frame.laser_labels:
                lidar_labels[obj_label.id] = extract_label_fields(obj_label,3)
            dict_2_save['lidar_labels'] = lidar_labels
            dict_2_save['camera_labels'] = camera_labels
            dict_2_save['veh_pose'] = np.reshape(frame.pose.transform,[4,4])
            dict_2_save['timestamp'] = frame.timestamp_micros
            if SINGLE_TRACK_INFO_FILE:
                tracking_info[(file_num,f_num)] = deepcopy(dict_2_save)
            else:
                with open(os.path.join(saving_dir,file_name, 'tracking','%03d.pkl'%(f_num)),'wb') as f:
                    pickle.dump(dict_2_save,f)

        if SINGLE_TRACK_INFO_FILE:
            with open(os.path.join(saving_dir, file_name, 'tracking_info%s.pkl'%('_debug' if DEBUG else '')), 'wb') as f:
                pickle.dump(tracking_info, f)


    print("Generating poses_bounds_waymo.npy")
    with open(os.path.join(saving_dir, scene_name) + '/tracking_info.pkl', 'rb') as file:
        data = pickle.load(file)

    all_veh_poses_per_cam = []
    all_cam2veh = []
    all_veh2world = []
    hwf = []
    for i in range(args.start_frame, args.start_frame + args.frame_nums):
        frame = data[(0, i)]
        all_veh_poses_per_cam.append(frame['per_cam_veh_pose'][None, ...])
        all_cam2veh.append(frame['cam2veh'][None, ...])
        all_veh2world.append(np.stack([frame['veh_pose'] for j in range(len(CAMERAS))])[None, ...])
        height = np.ones((len(CAMERAS), 1, 1)) * 1280
        width = np.ones((len(CAMERAS), 1, 1)) * 1920
        focal = frame['focal'][:3, None, None]
        hwf_ = np.concatenate((height, width, focal), axis = 1)[None, ...]
        hwf.append(hwf_)

    all_veh_poses_per_cam = np.concatenate(all_veh_poses_per_cam, 0)
    all_cam2veh = np.concatenate(all_cam2veh, 0)
    all_veh2world = np.concatenate(all_veh2world, 0)
    hwf = np.concatenate(hwf, 0)

    extrinsics = []
    all_vehi2veh0 = []

    veh2world_per_cam = all_veh_poses_per_cam[:, 0]
    world2veh_per_cam = np.stack([invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2world_per_cam])
    print(world2veh_per_cam.shape)
    cam2veh = all_cam2veh[:, 0]
    veh2world = all_veh2world[:, 0]

    cam2veh = np.matmul(world2veh_per_cam, np.matmul(veh2world, cam2veh))

    veh2world_per_cam_0 = copy.deepcopy(veh2world_per_cam[0])
    world2veh_per_cam_0 = invert_transformation(veh2world_per_cam_0[:3, :3], veh2world_per_cam_0[:3, 3])

    for cam_i in range(len(CAMERAS)):
        veh2world_per_cam = all_veh_poses_per_cam[:, cam_i]
        world2veh_per_cam = np.stack([invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2world_per_cam])
        print(world2veh_per_cam.shape)
        cam2veh = all_cam2veh[:, cam_i]
        veh2world = all_veh2world[:, cam_i]

        cam2veh = np.matmul(world2veh_per_cam, np.matmul(veh2world, cam2veh))

        vehi2veh0 = []
        for i in range(1, len(veh2world_per_cam)):
            veh2world_i = veh2world_per_cam[i]
            vehi2veh0.append(world2veh_per_cam_0.dot(veh2world_i))

        vehi2veh0 = np.stack(vehi2veh0)
        
        pose_0 = np.eye(4)[None, ...]
        vehi2veh0 = np.concatenate((pose_0, vehi2veh0))
        all_vehi2veh0.append(vehi2veh0)
        extrinsics_ = np.matmul(vehi2veh0, cam2veh)

        trans_mat = np.array([[[ 0.,  0.,  1., 0.],
                        [-1., -0., -0., 0.],
                        [-0., -1., -0., 0.],
                        [-0., 0., -0., 1.]]])

        extrinsics_ = np.matmul(extrinsics_, trans_mat)
        extrinsics.append(extrinsics_[:, None, ...])

    all_vehi2veh0 = np.stack(all_vehi2veh0)
    np.save(nerf_data_dir + '/vehi2veh0.npy', all_vehi2veh0)

    extrinsics = np.concatenate(extrinsics, axis=1)

    ########################## opencv to llff coordinates ###########################################
    extrinsics = extrinsics[:, :, :3, :]
    extrinsics = np.concatenate([extrinsics[:, :, :, 1:2], extrinsics[:, :, :, 0:1], -extrinsics[:, :, :, 2:3], extrinsics[:, :, :, 3:]], 3)
    poses = np.concatenate((extrinsics, hwf), 3)
    poses = poses.reshape(args.frame_nums * len(CAMERAS), 3, 5)   
    poses = poses.reshape(args.frame_nums * len(CAMERAS), -1)
    poses_bounds = np.zeros((args.frame_nums * len(CAMERAS), 17))
    poses_bounds[:, :15] = poses
    poses_bounds[:, 15] = 0.1
    poses_bounds[:, 16] = 600.

    np.save(nerf_data_dir + '/poses_bounds_waymo.npy', poses_bounds)

    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, 4]

    ########################## llff to nerf coordinates ###########################################
    # print("Converting LLFF coordinates to NeRF coordinates")
    # poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    # bounds = poses_bounds[:, 15: 17]
    # n_poses = len(poses)
    # intri = np.zeros([n_poses, 3, 3])
    # intri[:, :3, :3] = np.eye(3)
    # intri[:, 0, 0] = hwf[:, 2] 
    # intri[:, 1, 1] = hwf[:, 2] 
    # intri[:, 0, 2] = hwf[:, 1] * .5 
    # intri[:, 1, 2] = hwf[:, 0] * .5

    # data = np.concatenate([
    #     poses.reshape(n_poses, -1),
    #     intri.reshape(n_poses, -1),
    #     np.zeros([n_poses, 4]),
    #     bounds.reshape(n_poses, -1)
    # ], -1)

    # data = np.ascontiguousarray(np.array(data).astype(np.float64))
    # np.save(os.path.join(nerf_data_dir, 'cams_meta.npy'), data)

    ########################## Getting shutter times from tfrecord ###########################################
    print("Getting Shutter Times from tfrecord files")
    save_path = os.path.join(args.nerf_data_dir, args.tfrecord_dir.split('/')[-1].rstrip(".tfrecord") ,'shutters')
    get_shutter(args.tfrecord_dir, save_path, start_frame=args.start_frame, end_frame=args.start_frame + args.frame_nums)

    print("Moving Images and Point cloud")
    imgs_path = os.path.join(saving_dir, scene_name) + '/images'
    new_img_path = nerf_data_dir + '/images'
    if not os.path.isdir(new_img_path): os.mkdir(new_img_path)
    for save_idx, img_index in enumerate(range(args.start_frame, args.start_frame + args.frame_nums)):
        for key, value in CAMERAS.items():
            img_path = os.path.join(imgs_path, '%03d_%s.png'%(img_index, key))
            img = cv2.imread(img_path)
            img = cv2.resize(img, (1920, 1280))
            cv2.imwrite(os.path.join(new_img_path, '%03d.png'%(save_idx*len(CAMERAS) + value)), img)

    pc_path = os.path.join(saving_dir, scene_name) + '/point_cloud'
    os.system('cp -r {} {}'.format(pc_path, nerf_data_dir))

    tracking_info_path = os.path.join(saving_dir, scene_name) + '/tracking_info.pkl'
    os.system('cp {} {}'.format(tracking_info_path, nerf_data_dir))


if __name__ == '__main__':
    
    main()