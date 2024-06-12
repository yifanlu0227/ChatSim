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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from omegaconf import OmegaConf
from icecream import ic
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(args):
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")

    first_iter = 0
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians)
    gaussians.training_setup(args)
    if args.start_checkpoint:
        # change to one function within gaussians
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, args.iterations + 1):
        if args.gui:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, args.convert_SHs_python, args.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, args, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, args.source_path)
                    if do_training and ((iteration < int(args.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if args.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, args, bg, exposure_scale=viewpoint_cam.exposure_scale)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if args.render_depth:
            depth = render_pkg["depth"]
        if args.render_opacity:
            opacity = render_pkg["opacity"]

        # Loss
        loss_dict = {}

        gt_image = viewpoint_cam.original_image.cuda()

        loss_l1 = l1_loss(image, gt_image)
        loss_dict["l1_loss"] = loss_l1.item()

        loss_ssim = (1.0 - ssim(image, gt_image))
        loss_dict["ssim_loss"] = loss_ssim.item()

        loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if args.get('lambda_opacity', 0.0) > 0.0:
            sky_mask = viewpoint_cam.sky_mask.cuda()                # sky mask is 1 where sky is visible
            opacity_mask = ~ sky_mask                               # opacity mask is 1 where sky is not visible
            opacity_mask = opacity_mask.float().unsqueeze(0)        # [1, H, W]
            opacity = opacity.clamp(1e-6, 1.0 - 1e-6)               # avoid undefined

            # binary cross entropy loss
            loss_opacity = - (opacity_mask * torch.log(opacity) + (1 - opacity_mask) * torch.log(1 - opacity)).mean()
            loss += args.lambda_opacity * loss_opacity
            loss_dict["opacity_loss"] = loss_opacity.item()

        if args.get('lambda_depth', 0.0) > 0.0:
            depth_mask = depth_mask > 0                              
            # L1 loss for depth supervision
            loss_depth = (torch.abs(depth - viewpoint_cam.depth.to("cuda")) * depth_mask).mean()
            loss += args.lambda_depth * loss_depth
            loss_dict["depth_loss"] = loss_depth.item()


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix_dict = {"EMA Loss": f"{ema_loss_for_log:.{3}f}"}
                for key, value in loss_dict.items():
                    postfix_dict[key] = f"{value:.{3}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == args.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss_dict, iter_start.elapsed_time(iter_end), args.testing_iterations, scene, render, (args, background))
            if (iteration in args.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training_report(tb_writer, iteration, loss_dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for key, value in loss_dict.items():
            tb_writer.add_scalar(f'train/{key}', value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, exposure_scale=viewpoint.exposure_scale)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_config", type=str, default = "configs/default/train.yaml")
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    
    # save args to args.model_path with OmegaConf
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    OmegaConf.save(args, os.path.join(args.model_path, "config.yaml"))

    print(args)
    
    args.saving_iterations.append(args.iterations)
    args.testing_iterations.append(args.iterations)

    # Start GUI server, configure and run training
    if args.gui:
        network_gui.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args)

    # All done
    print("\nTraining complete.")

    # automatic rendering and calc metric
    print("Auto rendering ...")
    os.system(f"python render.py -m {args.model_path}")
    print("Auto evaluating ...")
    os.system(f"python metrics.py -m {args.model_path}")