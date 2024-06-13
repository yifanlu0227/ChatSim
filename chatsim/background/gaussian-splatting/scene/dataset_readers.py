#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import imageio.v2 as imageio
import open3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array # c2w
    T: np.array # w2c
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array = None
    sky_mask: np.array = None
    normal: np.array = None
    depth: np.array = None
    exposure_scale: float = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, args):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # w2c -> c2w
        T = np.array(extr.tvec) # w2c

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        # For PINHOLE model, intr.params is [fx, fy, cx, cy]
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, K=intr.params)

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPlyOpen3D(path):
    open3d_data = open3d.io.read_point_cloud(path)
    positions = np.array(open3d_data.points) # Nx3
    colors = np.array(open3d_data.colors) # Nx3
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readChatsimSceneInfo(args):
    """
    This is modified for ChatSim, which use points3D_waymo.ply for initialization

    points3D_waymo.ply is from recalibration with COLMAP. See data_utils/README.md for details
    """
    path = args.source_path
    images = args.images

    cams_meta_file = os.path.join(path, "cams_meta.npy")
    ply_path = os.path.join(path, "points3D_waymo.ply")
    images_folder = os.path.join(path, "images")
    image_name_list = os.listdir(images_folder)
    image_file_list = [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
    image_name_list.sort()
    image_file_list.sort()
    cam_infos_unsorted = []

    cams_meta = np.load(cams_meta_file)
    for idx, cam_data in enumerate(cams_meta):
        image_path = image_file_list[idx]
        image_name = image_name_list[idx]
        image = Image.open(image_file_list[idx])
        H, W = image.size

        # from RUB to RDF (OpenCV convention)
        c2w_RUB = np.eye(4)
        c2w_RUB[:3,:] = cam_data[:12].reshape(3, 4)
        c2w_RDF = np.concatenate([c2w_RUB[:,0:1], -c2w_RUB[:,1:2], -c2w_RUB[:,2:3], c2w_RUB[:,3:4]], axis=1)
        c2w = c2w_RDF
        w2c = np.linalg.inv(c2w)

        camera_intrinsics = cam_data[12:21].reshape(3,3)
        R = c2w[:3, :3]
        T = w2c[:3, 3]
        K = np.array([camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2]])  # fx fy cx cy
        FoVx = 2 * np.arctan(W / (2 * camera_intrinsics[0, 0]))
        FoVy = 2 * np.arctan(H / (2 * camera_intrinsics[1, 1]))

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FoVy, FovX=FoVx, image=image,
                              image_path=image_path, image_name=image_name, width=W, height=H, K=K)

        if args.get('load_sky_mask', False):
            sky_mask_folder = args.sky_mask_folder
            sky_mask_path = image_path.replace(os.path.basename(images_folder), sky_mask_folder)
            try:
                sky_mask = Image.open(sky_mask_path)
            except:
                sky_mask = Image.open(sky_mask_path + ".png")
            sky_mask = np.array(sky_mask)
            cam_info = cam_info._replace(sky_mask=sky_mask)

        if args.get('load_normal', False):
            normal_folder = args.normal_folder
            normal_path = image_path.replace(os.path.basename(images_folder), normal_folder).replace(".png", ".exr")
            normal = Image.open(normal_path)
            normal = np.array(normal)
            cam_info = cam_info._replace(normal=normal)

        if args.get('load_depth', False):
            depth_folder = args.depth_folder
            depth_path = image_path.replace(os.path.basename(images_folder), depth_folder).replace(".png", ".exr")
            depth = imageio.imread(depth_path)
            cam_info = cam_info._replace(depth=depth)

        if args.get('load_exposure', False):
            exposure_folder = args.exposure_folder
            exposure_path = os.path.join(image_path.split("colmap/")[0],
                                         exposure_folder, 
                                         image_name + ".txt")
            with open(exposure_path, 'r') as f:
                exposure = float(f.read())

            exposure_scale = 1 + args.exposure_coefficient * exposure
            cam_info = cam_info._replace(exposure_scale=exposure_scale)

        cam_infos_unsorted.append(cam_info)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # points3D_waymo.ply is from recalibration with COLMAP, and stored with Open3D
    assert os.path.exists(ply_path), "Please run recalibration with colmap or download provided calibration files"

    try:
        pcd = fetchPlyOpen3D(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(args):
    path = args.source_path
    images = args.images

    try:
        cameras_extrinsic_file = os.path.join(path, f"sparse/{args.sparse_folder}", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"sparse/{args.sparse_folder}", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"sparse/{args.sparse_folder}", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"sparse/{args.sparse_folder}", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"sparse/{args.sparse_folder}", "points3D.ply")
    bin_path = os.path.join(path, f"sparse/{args.sparse_folder}", "points3D.bin")
    txt_path = os.path.join(path, f"sparse/{args.sparse_folder}", "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Chatsim": readChatsimSceneInfo
}