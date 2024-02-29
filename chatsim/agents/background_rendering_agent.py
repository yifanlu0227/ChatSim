import numpy as np
from termcolor import colored
import imageio.v2 as imageio
import os

class BackgroundRenderingAgent:
    def __init__(self, config):
        self.config = config

        self.is_wide_angle = config["nerf_config"]['is_wide_angle']
        self.scene_name = config["nerf_config"]["scene_name"]
        self.f2nerf_dir = config["nerf_config"]["f2nerf_dir"]
        self.nerf_exp_name = config["nerf_config"]["nerf_exp_name"]
        self.f2nerf_config = config["nerf_config"]["f2nerf_config"] # wanjinyou_big
        self.dataset_name = config["nerf_config"]['dataset_name'] # waymo_multi_view
        self.nerf_mode = config["nerf_config"]['rendering_mode']
        self.nerf_exp_dir = os.path.join(
            self.f2nerf_dir, "exp", self.scene_name, self.nerf_exp_name
        )
        self.nerf_data_dir = os.path.join(
            self.f2nerf_dir, "data", self.dataset_name, self.scene_name
        )
        
        nerf_output_foler_name = "wide_angle_novel_images" if self.is_wide_angle else "novel_images"
        self.nerf_novel_view_dir = os.path.join(self.nerf_exp_dir, nerf_output_foler_name)

        if self.is_wide_angle:
            assert 'wide' in self.nerf_mode
        else:
            assert 'wide' not in self.nerf_mode

    def func_render_background(self, scene):
        """
        Call the NeRF, store results in scene.current_images
        """
        # first update scene.is_ego_motion
        scene.is_ego_motion = not np.all(scene.current_extrinsics == scene.current_extrinsics[0])

        if scene.is_ego_motion:
            print(f"{colored('[Mc-NeRF]', 'red', attrs=['bold'])} is_ego_motion is True, rendering multiple frames")

            poses_render = scene.current_extrinsics[:, :3, :]
            np.save(os.path.join(self.nerf_data_dir, 'poses_render.npy'), poses_render)

            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir) # do not generate intermediate file at root directory
            os.system(f'python scripts/run.py \
                                --config-name={self.f2nerf_config} \
                                dataset_name={self.dataset_name} \
                                case_name={self.scene_name} \
                                exp_name={self.nerf_exp_name} \
                                mode={self.nerf_mode} \
                                is_continue=true \
                                +work_dir={os.getcwd()}')
            os.chdir(current_dir)
            
            scene.current_images = [] # to be updated
            img_path_list = os.listdir(self.nerf_novel_view_dir)
            img_path_list.sort(key=lambda x:int(x[:-4]))
            for img_path in img_path_list:
                scene.current_images.append(imageio.imread(os.path.join(self.nerf_novel_view_dir, img_path))[:, :scene.width])

        else:
            print(f"{colored('[Mc-NeRF]', 'red', attrs=['bold'])} is_ego_motion is False, rendering one frame")

            poses_render = scene.current_extrinsics[0:1, :3, :]
            np.save(os.path.join(self.nerf_data_dir, 'poses_render.npy'), poses_render)

            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir) # do not generate intermediate file at root directory
            os.system(f'python scripts/run.py \
                                --config-name={self.f2nerf_config} \
                                dataset_name={self.dataset_name} \
                                case_name={self.scene_name} \
                                exp_name={self.nerf_exp_name} \
                                mode={self.nerf_mode} \
                                is_continue=true \
                                +work_dir={os.getcwd()}')
            os.chdir(current_dir)

            novel_image = imageio.imread(os.path.join(self.nerf_novel_view_dir, '50000_000.png'))[:, :scene.width]  #wide angle
            scene.current_images = [novel_image] * scene.frames
