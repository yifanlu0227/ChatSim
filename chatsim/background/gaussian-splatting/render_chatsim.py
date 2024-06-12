"""
This file read camera extrinsics and intrisics from files (provided by chatsim),
and render the corresponding images.

'chatsim_extint.npz': 
    {'camera_extrinsics': camera_extrinsics, 'camera_intrinsics': camera_intrinsics,
        'H': scene.height, 'W': scene.width, 'N': scene.frames})
"""
import os
from os import makedirs
from scene import Scene
import numpy as np
import torch
from termcolor import colored
from utils.general_utils import safe_state
from argparse import ArgumentParser
from omegaconf import OmegaConf
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from utils.system_utils import searchForMaxIteration
import torchvision
from tqdm import tqdm

def create_view_cameras(camera_extrinsics, camera_intrinsics, H, W):
    """
    We will transform the camera extrinsics and intrinsics to scene.cameras.Camera objects.

    Note 1) camera extrinsics are RUB, but gaussians splatting requires COLMAP convention (RDF)
    Note 2) R is c2w, T is w2c. We need to inverse the camera_extrinsics to get T.

    Args:
        camera_extrinsics: [N_frames, 3, 4], c2w
        camera_intrinsics: [3, 3]
        H: height of the image
        W: width of the image
    """
    frames_num = camera_extrinsics.shape[0]
    # [N_frames, 3, 4], to COLMAP convention (RDF)
    camera_extrinsics = np.concatenate([camera_extrinsics[:,:,0:1], 
                                        -camera_extrinsics[:,:,1:2],
                                        -camera_extrinsics[:,:,2:3],
                                         camera_extrinsics[:,:,3:4]], axis=2)
    view_cameras = []
    for i in tqdm(range(frames_num)):
        # pad [3,4] to [4,4] as a homogeneous matrix
        c2w = np.eye(4)
        c2w[:3] = camera_extrinsics[i]
        w2c = np.linalg.inv(c2w)

        R = c2w[:3, :3]
        T = w2c[:3, 3]
        K = np.array([camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2]])  # fx fy cx cy
        FoVx = 2 * np.arctan(W / (2 * camera_intrinsics[0, 0]))
        FoVy = 2 * np.arctan(H / (2 * camera_intrinsics[1, 1]))
        image = torch.zeros((3, H, W), dtype=torch.float32)
        image_name = f"image_{i:03d}"
        uid = i

        camera = Camera(colmap_id=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image=image, gt_alpha_mask=None,
                        image_name=image_name, uid=uid, K=K)
        view_cameras.append(camera)

    return view_cameras

def render_set(args, view_cameras, iteration):
    with torch.no_grad():
        gaussians = GaussianModel(args)
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        gaussians.load_ply(os.path.join(args.model_path,
                                        "point_cloud",
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud.ply"))

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        model_path = args.model_path

        render_path = os.path.join(model_path, 'chatsim_novel_views')
        makedirs(render_path, exist_ok=True)

        for idx, view in enumerate(view_cameras):
            render_pkg = render(view, gaussians, args, background, exposure_scale=1.0 if args.get('load_exposure', False) else None)
            torchvision.utils.save_image(render_pkg['render'], os.path.join(render_path, '{0:03d}'.format(idx) + ".png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)

    args = parser.parse_args()
    iteration = args.iteration

    model_path = args.model_path
    model_path = model_path.split('chatsim/background/gaussian-splatting/')[-1]
    
    extint_file = os.path.join(model_path, 'chatsim_extint.npz')
    extint = np.load(extint_file)
    camera_extrinsics = extint['camera_extrinsics'] # [N_frames, 3, 4], c2w
    camera_intrinsics = extint['camera_intrinsics'] # [3, 3]
    H = extint['H']
    W = extint['W']

    print(f"camera_extrinsics: {camera_extrinsics.shape}")
    print(f"camera_intrinsics: {camera_intrinsics.shape}")
    print(f"H: {H}, W: {W}")
    view_cameras = create_view_cameras(camera_extrinsics, camera_intrinsics, H, W)

    args = OmegaConf.load(model_path + "/config.yaml")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render
    render_set(args, view_cameras, iteration=iteration)