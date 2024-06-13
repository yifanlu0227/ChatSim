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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from omegaconf import OmegaConf


def render_set(args, name, iteration, views, gaussians, background):
    model_path = args.model_path

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if args.render_depth:
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        makedirs(depth_path, exist_ok=True)
    
    if args.render_opacity:
        opacity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "opacity")
        makedirs(opacity_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, args, background, exposure_scale=view.exposure_scale)
        gt = view.original_image[0:3, :, :]

        # do not consider 2-level folder here.
        torchvision.utils.save_image(render_pkg['render'], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(args, iteration : int, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test:
             render_set(args, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, background)

        if not skip_train:
             render_set(args, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, background)

        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    print("Rendering " + args.model_path)
    iteration = args.iteration
    skip_train = args.skip_train
    skip_test = args.skip_test
    args = OmegaConf.load(args.model_path + "/config.yaml")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args, iteration, skip_train, skip_test)