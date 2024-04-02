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
import cv2
import shutil
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
    parser.add_argument('--nerf_data_dir', type=str)
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
    tfrecord_path = args.tfrecord_dir
    export_data = not args.no_data
    scene_name = tfrecord_path.split('/')[-1].split('.')[0]
    scene_name = scene_name[scene_name.find('segment'):]
    saving_dir = os.path.join(args.nerf_data_dir, scene_name)

    if not isinstance(tfrecord_path, list):   
        tfrecord_path = [tfrecord_path]
    if not os.path.isdir(saving_dir):   
        os.makedirs(saving_dir, exist_ok=True)

    isotropic_focal = lambda intrinsic_dict: intrinsic_dict['f_u']==intrinsic_dict['f_v']

    if SINGLE_TRACK_INFO_FILE:
        tracking_info = {}

    print('Processing file ', tfrecord_path)

    if not os.path.isdir(os.path.join(saving_dir, 'images_all')):   
        os.mkdir(os.path.join(saving_dir, 'images_all'))
    if not os.path.isdir(os.path.join(saving_dir, 'images')):   
        os.mkdir(os.path.join(saving_dir, 'images'))
    if not os.path.isdir(os.path.join(saving_dir, 'point_cloud')):   
        os.mkdir(os.path.join(saving_dir, 'point_cloud'))
    if not SINGLE_TRACK_INFO_FILE:
        if not os.path.isdir(os.path.join(saving_dir, 'tracking')):   
            os.mkdir(os.path.join(saving_dir, 'tracking'))
            
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    frames = []
    for f_num, data in enumerate(tqdm(dataset)):
        frame = open_dataset.Frame()
        frames.append(frame)
        frame.ParseFromString(bytearray(data.numpy()))
        pose = np.zeros([len(frame.images), 4, 4])
        im_paths = {}
        pcd_paths = {}
        if SAVE_INTRINSIC:
            intrinsic = np.zeros([len(frame.images),9])
        extrinsic = np.zeros_like(pose)
        width,height,camera_labels = np.zeros([len(frame.images)]),np.zeros([len(frame.images)]),defaultdict(dict)
        for im in frame.images:
            saving_name = os.path.join(saving_dir, 'images_all','%03d_%s.png'%(f_num,open_dataset.CameraName.Name.Name(im.name)))
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
            (range_images, camera_projections, seg_labels, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                                            range_images,
                                                                            camera_projections,
                                                                            range_image_top_pose)
        else:
            points = np.empty([len(frame.lasers), 1])

        laser_mapping = {}
        for (laser, pts) in zip(frame.lasers, points):
            saving_name = os.path.join(saving_dir, 'point_cloud', '%03d_%s.ply' % (f_num, open_dataset.LaserName.Name.Name(laser.name)))
            if export_data and f_num == 0:
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

        tracking_info[(0,f_num)] = deepcopy(dict_2_save)

    with open(os.path.join(saving_dir, 'tracking_info%s.pkl'%('_debug' if DEBUG else '')), 'wb') as f:
        pickle.dump(tracking_info, f)

    transform = np.reshape(np.array(frames[args.start_frame].pose.transform), [4, 4])
    transform = np.linalg.inv(transform)
    road_edges = []
    lanes = []   
    for i in range(len(frames[0].map_features)):
        if len(frames[0].map_features[i].lane.polyline) > 0:
            curr_lane = []
            for node in frames[0].map_features[i].lane.polyline:
                node_position = np.ones(4)
                node_position[0] = node.x
                node_position[1] = node.y
                node_position[2] = node.z
                curr_lane.append(node_position)
            curr_lane = np.stack(curr_lane)
            curr_lane = np.transpose(np.matmul(transform, np.transpose(curr_lane)))[:, 0:3]
            lanes.append(curr_lane)
        
        if len(frames[0].map_features[i].road_edge.polyline) > 0:
            curr_edge = []
            for node in frames[0].map_features[i].road_edge.polyline:
                node_position = np.ones(4)
                node_position[0] = node.x
                node_position[1] = node.y
                node_position[2] = node.z
                curr_edge.append(node_position)
            curr_edge = np.stack(curr_edge)
            curr_edge = np.transpose(np.matmul(transform, np.transpose(curr_edge)))[:, 0:3]
            road_edges.append(curr_edge)

    x_min = -30
    x_max = 50
    y_min = -20
    y_max = 20
    cropped_road_edges = []
    for edge in road_edges:
        new_road_edge = []
        for i in range(edge.shape[0]):
            if edge[i,0] < x_min or edge[i,0] > x_max or edge[i,1] < y_min or edge[i,1] > y_max:
                continue
            new_road_edge.append(edge[i])
        if len(new_road_edge) > 0:
            new_road_edge = np.stack(new_road_edge)
            cropped_road_edges.append(new_road_edge)

    cropped_lanes = []
    for lane in lanes:
        new_lane = []
        for i in range(lane.shape[0]):
            if lane[i,0] < x_min or lane[i,0] > x_max or lane[i,1] < y_min or lane[i,1] > y_max:
                continue
            new_lane.append(lane[i])
        if len(new_lane) > 0:
            new_lane = np.stack(new_lane)
            cropped_lanes.append(new_lane)
    output_map = {"centerline":cropped_lanes,"boundary":cropped_road_edges}

    with open(os.path.join(saving_dir, 'map.pkl'), 'wb') as f:
        pickle.dump(output_map, f)
                
    with open(os.path.join(saving_dir, 'tracking_info.pkl'), 'rb') as file:
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
    np.save(os.path.join(saving_dir, "vehi2veh0.npy"), all_vehi2veh0)

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

    np.save(os.path.join(saving_dir, 'poses_bounds_waymo.npy'), poses_bounds)

    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, 4]


    ########################## Getting shutter times from tfrecord ###########################################
    print("Getting Shutter Times from tfrecord files")
    save_path = os.path.join(saving_dir ,'shutters')
    get_shutter(args.tfrecord_dir, save_path, start_frame=args.start_frame, end_frame=args.start_frame + args.frame_nums)

    # rename image files
    for save_idx, img_index in enumerate(range(args.start_frame, args.start_frame + args.frame_nums)):
        for key, value in CAMERAS.items():
            img_path_from = os.path.join(saving_dir, "images_all", '%03d_%s.png'%(img_index, key))
            img_path_to = os.path.join(saving_dir, "images", '%03d.png'%(save_idx*len(CAMERAS) + value))
            shutil.copyfile(img_path_from, img_path_to)

    valid_vehicles = data[(0,args.start_frame)]['camera_labels'][1].keys()
    valid_vehicles = [key  for key in valid_vehicles if data[(0,args.start_frame)]['camera_labels'][1][key]['type'] == 1]

    bboxes_dict = {}

    for i,key in enumerate(valid_vehicles):
        bboxes_dict[str(i)] = data[(0,args.start_frame)]['lidar_labels'][key]
        bboxes_dict[str(i)]['cx'] = bboxes_dict[str(i)].pop('c_x')
        bboxes_dict[str(i)]['cy'] = bboxes_dict[str(i)].pop('c_y')
        bboxes_dict[str(i)]['cz'] = bboxes_dict[str(i)].pop('c_z')
    
    np.save(os.path.join(saving_dir, '3d_boxes.npy'), bboxes_dict)



if __name__ == '__main__':
    
    main()