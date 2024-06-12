"""
This is a 3dgs version of our background rendering, which offers real-time performance
"""
import numpy as np
from termcolor import colored
import imageio.v2 as imageio
import os

class BackgroundRendering3DGSAgent:
    def __init__(self, config):
        self.config = config

        self.is_wide_angle = config["nerf_config"]['is_wide_angle']

        self.gs_dir = config["gs_config"]['gs_dir']
        self.model_folder = os.path.join(config["gs_config"]['gs_dir'],
                                         config["gs_config"]["output_folder"],
                                         config["gs_config"]["gs_model_name"])
        self.gs_novel_view_dir = os.path.join(self.model_folder, "chatsim_novel_views")

    def func_render_background(self, scene):
        """
        Call the NeRF, store results in scene.current_images
        """
        # first update scene.is_ego_motion
        scene.is_ego_motion = not np.all(scene.current_extrinsics == scene.current_extrinsics[0])

        if scene.is_ego_motion:
            print(f"{colored('[Background Gaussian Splatting]', 'red', attrs=['bold'])} is_ego_motion is True, rendering multiple frames")

            camera_extrinsics = scene.current_extrinsics[:, :3, :] # [N_frames, 3, 4]
            camera_intrinsics = scene.intrinsics # [3, 3]
        else:
            print(f"{colored('[Background Gaussian Splatting]', 'red', attrs=['bold'])} is_ego_motion is False, rendering one frame")
            camera_extrinsics = scene.current_extrinsics[0:1, :3, :] # [1, 3, 4]
            camera_intrinsics = scene.intrinsics # [3, 3]

        np.savez(os.path.join(self.model_folder, 'chatsim_extint.npz'), 
                    camera_extrinsics = camera_extrinsics, 
                    camera_intrinsics = camera_intrinsics,
                    H = scene.height, 
                    W = scene.width
                )

        # remove previous rendered images
        if os.path.exists(self.gs_novel_view_dir) and len(os.listdir(self.gs_novel_view_dir)) > 0:
            os.system(f"rm -r {self.gs_novel_view_dir}/*")

        current_dir = os.getcwd()
        os.chdir(self.gs_dir) # do not generate intermediate file at root directory
        render_command = f'python render_chatsim.py \
                            --model_path {self.model_folder}'

        os.system(render_command)
        os.chdir(current_dir)
        
        scene.current_images = [] # to be updated
        img_path_list = os.listdir(self.gs_novel_view_dir)
        img_path_list.sort(key=lambda x:int(x[:-4]))

        for img_path in img_path_list:
            scene.current_images.append(imageio.imread(os.path.join(self.gs_novel_view_dir, img_path))[:, :scene.width])

        if not scene.is_ego_motion:
            scene.current_images = scene.current_images * scene.frames
