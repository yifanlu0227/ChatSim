import numpy as np
from termcolor import colored
import imageio.v2 as imageio
from tqdm import tqdm
import os
from chatsim.agents.utils import check_and_mkdirs, transform_nerf2opencv_convention, generate_rays, blending_hdr_sky, srgb_gamma_correction
import cv2
import torch
import yaml
import shutil
import multiprocessing
import sys
import time
import subprocess

class ForegroundRenderingAgent:
    def __init__(self, config):
        self.config = config
        self.blender_dir = config["blender_dir"]
        self.blender_utils_dir = config["blender_utils_dir"]

        # skydome lighting
        self.skydome_hdri_dir = config['skydome_hdri_dir']
        self.skydome_hdri_idx = config['skydome_hdri_idx']
        self.use_surrounding_lighting = config['use_surrounding_lighting']

        # surrounding lighting, call mcnerf
        self.is_wide_angle = config["nerf_config"]['is_wide_angle']
        self.scene_name = config["nerf_config"]["scene_name"]
        self.f2nerf_dir = config["nerf_config"]["f2nerf_dir"]
        self.nerf_exp_name = config["nerf_config"]["nerf_exp_name"]
        self.f2nerf_config = config["nerf_config"]["f2nerf_config"]
        self.dataset_name = config["nerf_config"]['dataset_name'] # waymo_multi_view
        self.nerf_exp_dir = os.path.join(
            self.f2nerf_dir, "exp", self.scene_name, self.nerf_exp_name
        )
        
        nerf_output_foler_name = "wide_angle_novel_images" if self.is_wide_angle else "novel_images"
        self.nerf_novel_view_dir = os.path.join(self.nerf_exp_dir, nerf_output_foler_name)

        self.nerf_quiet_render = config["nerf_config"]["nerf_quiet_render"]

        # depth estimation
        self.estimate_depth = config['estimate_depth']
        if self.estimate_depth:
            from segment_anything import (SamAutomaticMaskGenerator, sam_model_registry)
            self.depth_est_method = config['depth_est']['method'] # currently SAM + LiDAR
            self.sam_checkpoint = config['depth_est']['SAM']['ckpt']
            self.sam_model_type = config['depth_est']['SAM']['model_type']
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint).cuda()
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def func_blender_add_cars(self, scene):
        """
        use blender to add cars for multiple frames. Static image is one frame.

        call self.blender_add_cars_single_frame in multi processing
        """
        check_and_mkdirs(os.path.join(scene.cache_dir, "blender_npz"))
        check_and_mkdirs(os.path.join(scene.cache_dir, "blender_output"))
        check_and_mkdirs(os.path.join(scene.cache_dir, "blender_yaml"))
        check_and_mkdirs(os.path.join(scene.cache_dir, "spatial_varying_hdri"))

        output_path = os.path.join(scene.cache_dir, "blender_output")

        if len(scene.added_cars_dict) > 0:
            # check added cars all static
            scene.check_added_car_static()

            # multiprocess rendering
            real_render_frames = 1 if scene.add_car_all_static else scene.frames
            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Start rendering {real_render_frames} images.")
            print(f"see the log in {os.path.join(scene.cache_dir, 'rendering_log')} if save_cache is enabled")

            # [SAM + LiDAR projection]
            # real_update_frames = scene.frames if scene.is_ego_motion else 1
            # for frame_id in range(real_update_frames):
            #     self.update_depth_single_frame(scene, frame_id=frame_id)

            # [Depth Estimation]
            # It works unstably. We can assume the depth of background is infinitely large
            background_depth_list = []
            if self.estimate_depth:
                real_update_frames = scene.frames if scene.is_ego_motion else 1

                if self.depth_est_method == 'SAM':
                    background_depth_list = self.update_depth_batch_SAM(scene, scene.current_images[:real_update_frames])
                else:
                    raise NotImplementedError

                print(f"{colored('[Depth Estimation]', 'cyan', attrs=['bold'])} Finish depth estimation {real_update_frames} images.")
            
            # prepare the config and npz files for rendering 
            print('preparing input files for blender rendering')
            for frame_id in tqdm(range(real_render_frames)):
                self.func_blender_add_cars_prepare_files_single_frame(scene, frame_id, background_depth_list)

            # parallel rendering process for all frames
            print(f'start rendering in parallel, process number is {scene.multi_process_num}.')
            print('This may take a few minutes. To speed up the foreground rendering, you can lower the `frames` number or render not-wide images.')
            print('If you find the results are incomplete or missing, that may due to OOM. You can reduce the multi_process_num in config yaml.')
            print('You can also check the log file for debugging with `save_cache` enabled in the yaml.')
            self.func_parallel_blender_rendering(scene)

            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Finish rendering {real_render_frames} images.")

            for frame_id in range(real_render_frames, scene.frames):
                assert real_render_frames == 1
                source_blender_output_folder = f"{output_path}/0"
                target_blender_output_folder = f"{output_path}/{frame_id}"
                shutil.copytree(source_blender_output_folder, target_blender_output_folder, dirs_exist_ok=True)

            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Copying Remaining {scene.frames - real_render_frames} images.")


            # image list to video
            video_frames = []
            for frame_id in range(scene.frames):
                video_frame_file = os.path.join(scene.cache_dir, "blender_output", str(frame_id), "RGB_composite.png")
                img = imageio.imread(video_frame_file)
                video_frames.append(img)
    
        else:
            video_frames = scene.current_inpainted_images

        scene.final_video_frames = video_frames

    def func_blender_add_cars_prepare_files_single_frame(self, scene, frame_id, background_depth_list):
        # save single frame's npz file for blender utils rendering
        np.savez(
            os.path.join(scene.cache_dir, "blender_npz", f"{frame_id}.npz"),
            H=scene.height,
            W=scene.width,
            focal=scene.focal,
            rgb=scene.current_inpainted_images[frame_id],
            depth=background_depth_list[frame_id] if len(background_depth_list) > 0 else 1000,
            extrinsic=transform_nerf2opencv_convention(
                scene.current_extrinsics[frame_id]
            )
        )

        # generate one frame (multiple cars)
        car_list_for_blender = []

        for car_name, car_info in scene.added_cars_dict.items():
            car_blender_file = car_info['blender_file']

            car_list_for_blender.append(
                {
                    "new_obj_name": car_name,
                    "blender_file": car_blender_file,
                    "insert_pos": [
                        car_info["motion"][frame_id, 0].tolist(),
                        car_info["motion"][frame_id, 1].tolist(),
                        0,
                    ],
                    "insert_rot": [0, 0, car_info["motion"][frame_id, 2].tolist()],
                    "model_obj_name": "Car",
                    **({"target_color": {
                        "material_key": "car_paint",
                        "color": [i / 255 for i in car_info["color"]] + [1],
                    }} if car_info["color"] != 'default' else {}) ,
                }
            )

        yaml_path = os.path.join(scene.cache_dir, "blender_yaml", f"{frame_id}.yaml")
        output_path = os.path.join(scene.cache_dir, "blender_output")

        skydome_hdri_path = os.path.join(self.skydome_hdri_dir, self.scene_name, f"{self.skydome_hdri_idx}.exr")
        final_hdri_path = skydome_hdri_path

        if self.use_surrounding_lighting:
            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Generating Spatial Varying HDRI.")
            
            # Since HDRI is a scene-level configuration, we currently only consider this case when car number is 1.
            # Spatial varying with multiple cars is also feasible, but need separate rendering for each car.
            # Put it as future work
            assert len(scene.added_cars_dict) == 1
            car_info = list(scene.added_cars_dict.values())[0]

            insert_x = car_info["motion"][frame_id, 0].tolist()
            insert_y = car_info["motion"][frame_id, 1].tolist()

            # generate ray_o, ray_d
            generate_rays(insert_x, insert_y, scene.ext_int_path, self.nerf_exp_dir)

            # nerf panorama rendering
            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir) # do not generate intermediate file at root directory
            print(f"{colored('[Mc-NeRF]', 'red', attrs=['bold'])} Generating Panorama.")
            render_command = f'python scripts/run.py \
                                    --config-name={self.f2nerf_config} dataset_name={self.dataset_name} \
                                    case_name={self.scene_name} \
                                    exp_name={self.nerf_exp_name} \
                                    mode=render_panorama_shutter \
                                    is_continue=true \
                                    +work_dir={os.getcwd()}'
            if self.nerf_quiet_render:
                render_command += ' > /dev/null 2>&1'
            os.system(render_command)
            os.chdir(current_dir)

            # an output example
            # . (self.nerf_exp_dir)
            # ├── last_trans.pt
            # ├── nerf_panorama.pt (in hdr space)
            # └── panorama
            #     └── 50000_000.png (visualization)

            nerf_last_trans_file = os.path.join(self.nerf_exp_dir, 'last_trans.pt')
            nerf_panorama_dir = os.path.join(self.nerf_exp_dir, 'panorama')
            nerf_panorama_pngs = os.listdir(nerf_panorama_dir)
            assert len(nerf_panorama_pngs) == 1

            nerf_panorama_pt_file = os.path.join(self.nerf_exp_dir, 'nerf_panorama.pt')

            arbitray_H = 128 # any number
            sky_mask = np.zeros((arbitray_H, arbitray_H*4, 3))

            # read last_trans and panorama
            nerf_env_panorama = torch.jit.load(nerf_panorama_pt_file).state_dict()['0'].cpu().numpy() 
            nerf_last_trans = torch.jit.load(nerf_last_trans_file).state_dict()['0'].cpu().numpy() # [H_, 4H_, 1]

            
            # HDRI blending and save
            pure_sky_hdri_path = skydome_hdri_path.replace('.exr', '_sky.exr')
            sky_dome_panorama = imageio.imread(pure_sky_hdri_path)

            

            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Merging HDRI")
            blending_panorama = blending_hdr_sky(nerf_env_panorama, sky_dome_panorama, nerf_last_trans, sky_mask) # [H, 4H, 3]

            # save intermediate results
            nerf_env_panorama_gamma_corrected = (srgb_gamma_correction(nerf_env_panorama)*255).astype(np.uint8)
            sky_dome_panorama_gamma_corrected = (srgb_gamma_correction(sky_dome_panorama)*255).astype(np.uint8)
            blending_hdr_sky_gamma_corrected = (srgb_gamma_correction(blending_panorama)*255).astype(np.uint8)

            final_hdri_path = os.path.join(scene.cache_dir, "spatial_varying_hdri", f"{frame_id}.exr")

            imageio.imwrite(final_hdri_path.replace('.exr', '_env.png'), nerf_env_panorama_gamma_corrected)
            imageio.imwrite(final_hdri_path.replace('.exr', '_sky.png'), sky_dome_panorama_gamma_corrected)
            imageio.imwrite(final_hdri_path.replace('.exr', '_blending.png'), blending_hdr_sky_gamma_corrected)

            sky_H, sky_W, _ = blending_panorama.shape
            blending_panorama_full = np.zeros((sky_H*2, sky_W, 3))
            blending_panorama_full[:sky_H] = blending_panorama

            imageio.imwrite(final_hdri_path, blending_panorama_full.astype(np.float32))
            print(f"{colored('[Blender]', 'magenta', attrs=['bold'])} Finish Merging HDRI")


        blender_dict = {
            "render_name": str(frame_id),
            "output_dir": output_path,
            "scene_file": os.path.join(
                scene.cache_dir, "blender_npz", f"{frame_id}.npz"
            ),
            "hdri_file": final_hdri_path,
            "render_downsample": 2,
            "cars": car_list_for_blender,
            "depth_and_occlusion": scene.depth_and_occlusion,
            "backup_hdri": scene.backup_hdri
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data=blender_dict, stream=f, allow_unicode=True)
        

    def func_compose_with_new_depth_single_frame(self, scene, frame_id):
        output_path = os.path.join(scene.cache_dir, "blender_output")
        background_image = imageio.imread(os.path.join(output_path, str(frame_id), "backup", "RGB.png"))
        depth_map = np.load(f"{output_path}/{frame_id}/depth/background_depth.npy")

        sys.path.append(
            os.path.join(
                self.blender_utils_dir,
                "postprocess",
            )
        )
        # compose function from blender_utils
        import compose

        compose.compose(
            os.path.join(output_path, str(frame_id)),  # rendered output dir
            background_image,  # background RGB. numpy.ndarray
            depth_map,  # background depth. numpy.ndarray
            2, # downsample_rate
        )

    def func_parallel_blender_rendering(self, scene):
        multi_process_num = scene.multi_process_num
        log_dir = os.path.join(scene.cache_dir, "rendering_log")
        check_and_mkdirs(os.path.join(scene.cache_dir, 'rendering_log'))

        frames = scene.frames

        segment_length = frames // multi_process_num

        processes = []

        for i in range(multi_process_num):
            start_frame = i * segment_length
            end_frame = (i + 1) * segment_length if i < multi_process_num - 1 else frames
            log_file = os.path.join(log_dir, f"{i}.txt")

            command = (
                f"{self.blender_dir} -b --python {self.blender_utils_dir}/main_multicar.py "
                f"-- {os.path.join(scene.cache_dir, 'blender_yaml')} -- {start_frame} -- {end_frame} > {log_file}"
            )

            with open(log_file, 'w') as f:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                processes.append(process)

        for process in processes:
            stdout, stderr = process.communicate()
            


    def get_sparse_depth_from_LiDAR(self, scene, frame_id):
        extrinsic_opencv = transform_nerf2opencv_convention(
                scene.current_extrinsics[frame_id]
            )

        pointcloud_world = np.concatenate((scene.pcd, np.ones((scene.pcd.shape[0], 1))), axis=1).T # [4, N_pcd]
        pointcloud_camera = (np.linalg.inv(extrinsic_opencv) @ pointcloud_world)[:3] # [3, N_pcd]
        pointcloud_image = (scene.intrinsics @ pointcloud_camera)[:2] / pointcloud_camera[2:3] # [2, N_pcd]

        z_positive = pointcloud_camera[2] > 0 # [N_pcd,]

        valid_points = (
            (pointcloud_image[0] > 0) & 
            (pointcloud_image[0] < scene.width) & 
            (pointcloud_image[1] > 0) & 
            (pointcloud_image[1] < scene.height) & 
            z_positive
        )

        pointcloud_image_valid = pointcloud_image[:, valid_points]
        valid_u_coord = pointcloud_image_valid[0].astype(np.int32)
        valid_v_coord = pointcloud_image_valid[1].astype(np.int32)

        sparse_depth_map = np.zeros((scene.height, scene.width))

        sparse_depth_map[valid_v_coord, valid_u_coord] = pointcloud_camera[2, valid_points]
        return sparse_depth_map


    def update_depth_batch_SAM(self, scene, image_list):
        """
        update depth batch use [SAM] + [LiDAR projection correction] to get instance-level depth

        Args:
            image_list : list of np.ndarray, len = 1 or scene.frames
                image is [H, W, 3] shape

        Returns:
            overlap_depth_list : list of np.array, len = 1 or scene.frames
                depth is [H, W] shape
        """

        real_update_frames = len(image_list)
        overlap_depth_list = []

        for frame_id in range(real_update_frames):
            output_path = os.path.join(scene.cache_dir, "blender_output")
            rendered_car_mask = imageio.imread(f"{output_path}/{frame_id}/mask/vehicle_and_shadow0001.exr")

            rendered_car_mask = cv2.resize(rendered_car_mask, (scene.current_inpainted_images[frame_id].shape[1],scene.current_inpainted_images[frame_id].shape[0]))
            
            rendered_car_mask = rendered_car_mask[..., 0] > (20/255)

            masks = self.mask_generator.generate(scene.current_inpainted_images[frame_id])
            num_masks = len(masks)
            import itertools
            mask_pairs = list(itertools.permutations(range(num_masks),2))

            valid_mask_idx = np.ones(num_masks,dtype=bool)
            for pair in mask_pairs:
                mask_1 = masks[pair[0]]
                mask_2 = masks[pair[1]]
                if (mask_1['segmentation'] & mask_2['segmentation']).sum() > 0:
                    if mask_1['area'] < mask_2['area']:
                        valid_mask_idx[pair[0]] = False
                    else:
                        valid_mask_idx[pair[1]] = False

            idx = np.where(valid_mask_idx==True)[0]
            masks = [masks[i] for i in idx]

            sparse_depth_map = self.get_sparse_depth_from_LiDAR(scene, frame_id)
            sparse_depth_mask = sparse_depth_map != 0

            overlap_depth = np.ones((scene.height, scene.width)) * 500
            
            for i in range(len(masks)):
                intersection_area = masks[i]["segmentation"] & rendered_car_mask
                if (intersection_area).sum() > 0:
                    intersection_area_with_depth = intersection_area & sparse_depth_mask
                    if (
                        intersection_area_with_depth.sum() > 0
                        and intersection_area_with_depth.sum() > 10
                    ):
                        
                        avg_depth = sparse_depth_map[intersection_area_with_depth].mean()
                        min_depth = sparse_depth_map[intersection_area_with_depth].min()
                        max_depth = sparse_depth_map[intersection_area_with_depth].max()
                        median_depth = np.median(sparse_depth_map[intersection_area_with_depth])

                        overlap_depth[intersection_area] = avg_depth

            overlap_depth_list.append(overlap_depth.astype(np.float32))

        return overlap_depth_list
        
