import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from copy import deepcopy
from glob import glob
import click
import camera_utils
import cv2
from colmap_warpper.pycolmap.scene_manager import SceneManager
from typing import Mapping, Optional, Sequence, Text, Tuple, Union
import struct
import open3d as o3d
from collections import OrderedDict
from termcolor import colored

def load_colmap_sparse_points(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)

    """
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error

    points = OrderedDict()
    points['xyz'] = xyzs
    points['rgb'] = rgbs
    points['error'] = errors

    return points

# Metashape


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])


def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)

    # leads to bad nerf training results, set to zero
    # cx = width/2 + float(calibration.find('cx').text)
    # cy = height/2 + float(calibration.find('cy').text)
    cx = width/2 
    cy = height/2

    # leads to bad nerf training results, set to zero
    # dist_params = (float(calibration.find('k1').text), 
    #                float(calibration.find('k2').text), 
    #                float(calibration.find('p1').text), 
    #                float(calibration.find('p2').text))
    dist_params = (0., 0., 0., 0.)

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ], dtype=np.float32)

    return K, (width, height), dist_params


def extrinsics_from_xml(xml_file, verbose=False):
    """
    Metashape return RDF convention camera poses
    """
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):

        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x)
                             for x in transforms[label].split()]).reshape(4, 4)

        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def read_xml_save_npy(data_dir):
    """
    We will save `cams_meta_metashape.npy` (RUB convention)
    not `poses_bounds_metashape.npy` (DRB) now.
    """
    print("Parsing Metashape results")
    intrinsic, (width, height), dist_params = intrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    poses_RDF, labels_sort = extrinsics_from_xml(os.path.join(data_dir, 'camera.xml')) # (RDF convention. check it)
    poses_RDF = np.stack(poses_RDF, axis=0) # (N, 4, 4)

    poses_RUB = np.concatenate(
        (poses_RDF[:, :, 0:1], -poses_RDF[:, :, 1:2], -poses_RDF[:, :, 2:3], poses_RDF[:, :, 3:]), axis=-1) # (N, 4, 4)
    poses_RUB = poses_RUB[:, :3, :] # (N, 3, 4)

    N = poses_RUB.shape[0]
    intrinsic = intrinsic.reshape(1,3,3).repeat(N, axis=0) # [N, 3, 3]
    dist_params = np.array(dist_params).reshape(1,4).repeat(N, axis=0) # [N, 4]
    bounds = np.array([0.1, 999]).reshape(1,2).repeat(N, axis=0) # [N, 2]

    cams_meta = np.concatenate([
        poses_RUB.reshape(N, -1),
        intrinsic.reshape(N, -1),
        dist_params.reshape(N, -1),
        bounds.reshape(N, -1)
    ],
    axis = 1)

    np.save(os.path.join(data_dir, 'cams_meta_metashape.npy'), cams_meta)


class NeRFSceneManager(SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def __init__(self, data_dir, use_undistorted=False):
        """
        use_undistorted: bool
            gaussians splatting needs undistorted camera intrinsics,
            McNeRF does not need undistorted camera intrinsics.

            But the images in the root folder is distorted. The undistorted version is in data_dir/colmap/sparse_undistorted/images
        """
        # COLMAP, undistorted
        if use_undistorted:
            sfm_dir = pjoin(data_dir, 'colmap/sparse_undistorted/sparse')
        # COLMAP, distorted
        else:
            sfm_dir = pjoin(data_dir, 'colmap/sparse/not_align/0')

        assert os.path.exists(sfm_dir)
        super(NeRFSceneManager, self).__init__(sfm_dir)

    def process(
            self
    ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, Optional[Mapping[
            Text, float]], camera_utils.ProjectionType]:
        """Applies NeRF-specific postprocessing to the loaded pose data.

        Returns:
          a tuple [image_names, poses, pixtocam, distortion_params].
          image_names:  contains the only the basename of the images.
          poses: [N, 4, 4] array containing the camera to world matrices.
          pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
          distortion_params: mapping of distortion param name to distortion
            parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
        """

        self.load_cameras()
        self.load_images()
        self.load_points3D()

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate(
                [np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])
        # pixtocam = np.diag([1, -1, -1]) @ pixtocam

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


class Colamp_Dataset:

    def __init__(self, data_dir):
        scene_manager = NeRFSceneManager(data_dir)
        self.names, self.poses, self.pix2cam, self.params, self.camtype = scene_manager.process(
        )
        self.cam2pix = np.linalg.inv(self.pix2cam)
        self.n_images = len(self.poses)

        # re-permute images by name
        sorted_image_names = sorted(deepcopy(self.names))
        sort_img_idx = []
        for i in range(self.n_images):
            sort_img_idx.append(self.names.index(sorted_image_names[i]))
        img_idx = np.array(sort_img_idx, dtype=np.int32)
        self.poses = self.poses[sort_img_idx]

        # Move all to numpy
        def proc(x):
            return np.ascontiguousarray(np.array(x).astype(np.float64))

        self.poses = proc(self.poses)
        self.cam2pix = proc(
            np.tile(self.cam2pix[None], (len(self.poses), 1, 1)))
        if self.params is not None:
            dist_params = [
                self.params['k1'], self.params['k2'], self.params['p1'],
                self.params['p2']
            ]
        else:
            dist_params = [0., 0., 0., 0.]
        dist_params = np.tile(np.array(dist_params),
                              len(self.poses)).reshape([len(self.poses), -1])
        self.dist_params = proc([dist_params])

    def export(self, data_dir):
        n = len(self.poses)
        poses_RUB = deepcopy(self.poses) # (N, 3, 4), already RUB, converted from COLMAP in func. process
        image_list = []
        suffs = ['*.png', '*.PNG', '*.jpg', '*.JPG']
        for suff in suffs:
            image_list += glob(pjoin(data_dir, 'images', suff))

        intrinsic = self.cam2pix # (N, 3, 3)
        
        dist_params = self.dist_params.reshape(-1, 4)
        bounds = np.array([0.1, 999]).reshape(1,2).repeat(n, axis=0) # [N, 2]

        cams_meta = np.concatenate([
            poses_RUB.reshape(n, -1),
            intrinsic.reshape(n, -1),
            dist_params.reshape(n, -1),
            bounds.reshape(n, -1)
        ],
        axis = 1)

        cams_meta = np.ascontiguousarray(cams_meta)

        np.save(os.path.join(data_dir, 'cams_meta_colmap.npy'), cams_meta)


def read_sparse_save_npy(data_dir):
    print("Parsing Colmap results to cams_meta_colmap.npy")
    dataset = Colamp_Dataset(data_dir)
    dataset.export(data_dir)


def align(data_dir, src_cams_meta="cams_meta_metashape.npy", dst_cams_meta="cams_meta_waymo.npy"):
    if src_cams_meta == "cams_meta_metashape.npy":
        print("Aligning Metashape's coordinates with Waymo's coordinates")

    elif src_cams_meta == "cams_meta_colmap.npy":
        print("Aligning Colmap's coordinates with Waymo's coordinates")
    
    cams_meta_data_source = np.load(os.path.join(data_dir, src_cams_meta))
    cams_meta_data_target = np.load(os.path.join(data_dir, dst_cams_meta))
    
    extrinsic_source = cams_meta_data_source[:, :12].reshape(-1, 3, 4)
    last_row = np.zeros((extrinsic_source.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    extrinsic_source = np.concatenate((extrinsic_source, last_row), axis=1)

    extrinsic_target = cams_meta_data_target[:, :12].reshape(-1, 3, 4)
    last_row = np.zeros((extrinsic_target.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    extrinsic_target = np.concatenate((extrinsic_target, last_row), axis=1)

    scale = np.linalg.norm(extrinsic_source[3, :3, -1] - extrinsic_source[0, :3, -1])  \
        / np.linalg.norm(extrinsic_target[3, :3, -1] - extrinsic_target[0, :3, -1])  # unit length scale

    rotate_0_target = extrinsic_target[0, :3, :3]
    rotate_0_source = extrinsic_source[0, :3, :3]

    # assume the first frame is aligned! two world coordinate are different
    rotate_source_world_to_target_world = rotate_0_target @ np.linalg.inv(
        rotate_0_source)
    rotate_source_world_to_target_world = rotate_source_world_to_target_world[None, ...]

    extrinsic_results = np.zeros_like(extrinsic_source)  # final output

    extrinsic_results[:, :3, :3] = rotate_source_world_to_target_world @ extrinsic_source[:, :3, :3]

    # delta translation between each frame and frame0 in metashape
    delta_translation_in_source_world = extrinsic_source[:, :3, -1:] - extrinsic_source[0:1, :3, -1:]

    delta_translation_in_target_world = (
        rotate_source_world_to_target_world @ delta_translation_in_source_world) / scale  # convert to waymo axis

    extrinsic_results[:, :3, -1:] = delta_translation_in_target_world + \
        extrinsic_target[0:1, :3, -1:]

    extrinsic_results[:, -1, -1] = 1

    # update source cam
    cams_meta_data_source[:, :12] = extrinsic_results[:,:3,:].reshape(-1, 12)

    data = np.ascontiguousarray(np.array(cams_meta_data_source).astype(np.float64))

    if src_cams_meta=="cams_meta_metashape.npy":
        print(f"\n{colored('[Imporant]', 'green', attrs=['bold'])} save to cams_meta.npy")
        np.save(os.path.join(data_dir, 'cams_meta.npy'), data)

    if src_cams_meta=="cams_meta_colmap.npy":
        # save in colmap/sparse_undistorted
        print(f"\n{colored('[Imporant]', 'green', attrs=['bold'])} Save to colmap/sparse_undistorted/cams_meta.npy")
        print(f"cams_meta.npy from metashape (in the root folder) will not be overwritten.")
        np.save(os.path.join(data_dir, 'colmap/sparse_undistorted/cams_meta.npy'), data)

        # convert point3D
        src_point3D_path = os.path.join(data_dir, 'colmap/sparse_undistorted/sparse/points3D.bin')
        dst_point3D_path = os.path.join(
            data_dir, 'colmap/sparse_undistorted/points3D_waymo.ply')

        points = load_colmap_sparse_points(src_point3D_path)
        points3D = points['xyz']
        points3D_colors = points['rgb']

        # delta_translation_in_source_world = points3D - extrinsic_source[0,:3,-1]
        delta_translation_in_source_world = points3D - \
            np.expand_dims(extrinsic_source[0, :3, -1], axis=0)

        # [N_points, 3, 1]
        delta_translation_in_source_world = delta_translation_in_source_world[..., np.newaxis]
        delta_translation_in_target_world = (
            rotate_source_world_to_target_world @ delta_translation_in_source_world) / scale
        
        # [1, 3, 1]
        translation_0_target = extrinsic_target[0:1, :3, -1:]
        points3D_in_target_world = delta_translation_in_target_world + translation_0_target

        # set up sfm points
        sfm_points = np.squeeze(points3D_in_target_world)
        sfm_colors = points3D_colors / 255.0

        # incorperate LiDAR into initialization
        lidar_open3d = o3d.io.read_point_cloud(os.path.join(data_dir, 'point_cloud/000_TOP.ply'))
        # we only keep x > 0 points
        lidar_points = np.array(lidar_open3d.points)
        lidar_colors = np.full(lidar_points.shape, 0.3) 
        mask = lidar_points[:, 0] > 0
        lidar_points = lidar_points[mask]
        lidar_colors = lidar_colors[mask]

        # save with open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([sfm_points, lidar_points], axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([sfm_colors, lidar_colors], axis=0))

        o3d.io.write_point_cloud(dst_point3D_path, pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', type=str)
    parser.add_argument('-c', '--calibration_tool', type=str)
    args, _ = parser.parse_known_args()
    if args.calibration_tool == 'metashape':
        read_xml_save_npy(args.datadir)  # this will generate cams_meta_metashape.npy
        align(args.datadir, 'cams_meta_metashape.npy')

    elif args.calibration_tool == 'colmap':
        read_sparse_save_npy(args.datadir) # this will generate cams_meta_colmap.npy
        align(args.datadir, 'cams_meta_colmap.npy')
