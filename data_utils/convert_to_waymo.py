import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from copy import deepcopy
from glob import glob
import click
import camera_utils
import cv2 as cv
from colmap_warpper.pycolmap.scene_manager import SceneManager
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

#Metashape
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
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)


def extrinsics_from_xml(xml_file, verbose=False):
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
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def read_xml(data_dir):
    print("Parsing Metashape results")
    intrinsic = intrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    extrinsic = extrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    extrinsic = np.stack(extrinsic[0],axis=0)

    poses = extrinsic
    poses = np.concatenate((-poses[:,:,1:2],poses[:,:,0:1],poses[:,:,2:3],poses[:,:,3:]),axis=-1)
    poses = poses[:,:3,:]
    poses_output = np.zeros((poses.shape[0],17))

    f = intrinsic[0][0,0]
    w = intrinsic[0][0,-1]*2
    h = intrinsic[0][1,-1]*2
    # import ipdb; ipdb.set_trace()
    hwf = np.array([h,w,f]).reshape(1,3,1).repeat(poses.shape[0],axis=0)

    bds = np.array([0.1,999]).reshape(1,-1).repeat(poses.shape[0],axis=0)
    poses = np.concatenate((poses,hwf),axis=-1)
    poses = poses.reshape(-1,15)
    poses = np.concatenate((poses,bds),axis=-1)
    np.save(os.path.join(data_dir, 'poses_bounds_metashape.npy'),poses)

#Colmap
# This implementation is from MipNeRF360
class NeRFSceneManager(SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def __init__(self, data_dir):
        # COLMAP
        if os.path.exists(pjoin(data_dir, 'sparse', '0')):
            sfm_dir = pjoin(data_dir, 'sparse', '0')
        # hloc
        else:
            sfm_dir = pjoin(data_dir, 'hloc_sfm')

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
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
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

        # calc near-far bounds
        self.bounds = np.zeros([self.n_images, 2], dtype=np.float32)
        name_to_ids = scene_manager.name_to_image_id
        points3D = scene_manager.points3D
        points3D_ids = scene_manager.point3D_ids
        point3D_id_to_images = scene_manager.point3D_id_to_images
        image_id_to_image_idx = np.zeros(self.n_images + 10, dtype=np.int32)

        print("num of idx:", len(image_id_to_image_idx), ";num of id:",
              len(name_to_ids),";num of in:",len(self.names))
              
        for image_name in self.names:
            image_id_to_image_idx[
                name_to_ids[image_name]] = sorted_image_names.index(image_name)

        vis_arr = []
        for pts_i in range(len(points3D)):
            cams = np.zeros([self.n_images], dtype=np.uint8)
            images_ids = point3D_id_to_images[points3D_ids[pts_i]]
            for image_info in images_ids:
                image_id = image_info[0]
                image_idx = image_id_to_image_idx[image_id]
                cams[image_idx] = 1
            vis_arr.append(cams)

        vis_arr = np.stack(vis_arr, 1)  # [n_images, n_pts ]

        for img_i in range(self.n_images):
            vis = vis_arr[img_i]
            pts = points3D[vis == 1]
            c2w = np.diag([1., 1., 1., 1.])
            c2w[:3, :4] = self.poses[img_i]
            w2c = np.linalg.inv(c2w)
            z_vals = (w2c[None, 2, :3] * pts).sum(-1) + w2c[None, 2, 3]
            depth = -z_vals
            near_depth, far_depth = np.percentile(depth, 1.), np.percentile(
                depth, 99.)
            near_depth = near_depth * .5
            far_depth = far_depth * 5.
            self.bounds[img_i, 0], self.bounds[img_i,
                                               1] = near_depth, far_depth

        # Move all to numpy
        def proc(x):
            return np.ascontiguousarray(np.array(x).astype(np.float64))

        self.poses = proc(self.poses)
        self.cam2pix = proc(
            np.tile(self.cam2pix[None], (len(self.poses), 1, 1)))
        self.bounds = proc(self.bounds)
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
        poses = deepcopy(self.poses)
        image_list = []
        suffs = ['*.png', '*.PNG', '*.jpg', '*.JPG']
        for suff in suffs:
            image_list += glob(pjoin(data_dir, 'images', suff))
        h, w, _ = cv.imread(image_list[0]).shape
        focal = (self.cam2pix[0, 0, 0] + self.cam2pix[0, 1, 1]) * .5

        # poses_ = torch::cat({ poses_.index({Slc(), Slc(), Slc(1, 2)}),
        # -poses_.index({Slc(), Slc(), Slc(0, 1)}),
        # poses_.index({Slc(), Slc(), Slc(2, None)})}, 2);
        poses = np.concatenate(
                [-poses[:, :, 1:2], poses[:, :, 0:1], poses[:, :, 2:]], 2)

        hwf = np.zeros([n, 3])
        hwf[:, 0] = h
        hwf[:, 1] = w
        hwf[:, 2] = focal
        bounds = self.bounds
        poses_hwf = np.concatenate([poses, hwf[:, :, None]], -1)
        data = np.concatenate(
            [poses_hwf.reshape([n, -1]),
                bounds.reshape([n, -1])], -1)
        data = np.ascontiguousarray(np.array(data).astype(np.float64))
        np.save(pjoin(data_dir, '{}.npy'.format('poses_bounds_colmap')), data)

def read_sparse(data_dir):
    print("Parsing Colmap results")
    dataset = Colamp_Dataset(data_dir)
    dataset.export(data_dir)


def align(data_dir, src_poses_bounds="poses_bounds_metashape.npy", dst_pose_bounds="poses_bounds_waymo.npy"):
    if src_poses_bounds == "poses_bounds_metashape.npy":
        print("Aligning Metashape's coordinates with Waymo's coordinates")
        poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_metashape.npy'))
    elif src_poses_bounds == "poses_bounds_colmap.npy":
        print("Aligning Colmap's coordinates with Waymo's coordinates")
        poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_colmap.npy'))

    poses_bounds = poses_bounds[:,:15].reshape(-1,3,5)

    extrinsic = poses_bounds[:,:,:4]
    last_row = np.zeros((extrinsic.shape[0],1,4))
    last_row[:,:,-1] = 1
    extrinsic_metashape = np.concatenate((extrinsic,last_row),axis=1)  #####shape [n, 4, 4] extrinsic from metashape

    hwf = poses_bounds[0,:,-1]

    intrinsic = np.array([[hwf[2],0,hwf[1]*0.5+2.37],
                        [0,hwf[2],hwf[0]*0.5-1.89],
                        [0,0,1] ])

    poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_waymo.npy'))

    poses_bounds = poses_bounds[:,:15].reshape(-1,3,5)

    extrinsic = poses_bounds[:,:,:4]
    last_row = np.zeros((extrinsic.shape[0],1,4))
    last_row[:,:,-1] = 1
    extrinsic_waymo = np.concatenate((extrinsic,last_row),axis=1)   #####shape [n, 4, 4] extrinsic from waymo

    scale = np.linalg.norm(extrinsic_metashape[1,:3,-1] - extrinsic_metashape[0,:3,-1])  \
        / np.linalg.norm(extrinsic_waymo[1,:3,-1] - extrinsic_waymo[0,:3,-1])   ## unit length scale

    rotate_0_waymo = extrinsic_waymo[0,:3,:3]
    rotate_0_metashape = extrinsic_metashape[0,:3,:3]

    rotate_metashape_to_waymo = rotate_0_waymo @ np.linalg.inv(rotate_0_metashape)  #### the rotation matrix convert metashape to waymo axis
    rotate_metashape_to_waymo = rotate_metashape_to_waymo[None,...]

    extrinsic_results = np.zeros_like(extrinsic_metashape)   #final output

    extrinsic_results[:,:3,:3] = rotate_metashape_to_waymo @ extrinsic_metashape[:,:3,:3]


    delta_translation_in_metashape = extrinsic_metashape[:,:3,-1:] - extrinsic_metashape[0:1,:3,-1:]  ###delta translation between each frame and frame0 in metashape

    delta_translation_in_metashape = delta_translation_in_metashape


    delta_translation_in_waymo = (rotate_metashape_to_waymo @ delta_translation_in_metashape)  / scale ####convert to waymo axis

    extrinsic_results[:,:3,-1:] = delta_translation_in_waymo + extrinsic_waymo[0:1,:3,-1:]

    extrinsic_results[:,-1,-1] = 1

    if src_poses_bounds == "poses_bounds_metashape.npy":
        poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_metashape.npy'))
    elif src_poses_bounds == "poses_bounds_colmap.npy":
        poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_colmap.npy'))

    poses_bounds_extrinsic_and_intrinsic = poses_bounds[:,:15].reshape(-1,3,5)
    poses_bounds_extrinsic_and_intrinsic[:,:,:-1] = extrinsic_results[:,:3,:]

    extrinsic_results_to_save = poses_bounds_extrinsic_and_intrinsic.reshape(-1,15)

    poses_bounds[:,:15] = extrinsic_results_to_save
    np.save(os.path.join(data_dir, 'poses_bounds.npy'), poses_bounds)       # LLFF

    print("Converting LLFF coordinates to NeRF coordinates")
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, 4]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    bounds = poses_bounds[:, 15: 17]
    n_poses = len(poses)
    intri = np.zeros([n_poses, 3, 3])
    intri[:, :3, :3] = np.eye(3)
    intri[:, 0, 0] = hwf[:, 2] 
    intri[:, 1, 1] = hwf[:, 2] 
    intri[:, 0, 2] = hwf[:, 1] * .5 
    intri[:, 1, 2] = hwf[:, 0] * .5

    data = np.concatenate([
        poses.reshape(n_poses, -1),
        intri.reshape(n_poses, -1),
        np.zeros([n_poses, 4]),
        bounds.reshape(n_poses, -1)
    ], -1)

    data = np.ascontiguousarray(np.array(data).astype(np.float64))
    np.save(os.path.join(data_dir, 'cams_meta.npy'), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', type = str)
    parser.add_argument('-c', '--calibration_tool', type = str)
    args,_ = parser.parse_known_args()
    if args.calibration_tool == 'metashape':
        read_xml(args.datadir)
        align(args.datadir,'poses_bounds_metashape.npy')
    elif args.calibration_tool == 'colmap':
        read_sparse(args.datadir)
        align(args.datadir,'poses_bounds_colmap.npy')
        

    