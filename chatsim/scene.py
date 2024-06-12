import os
import pickle
import cv2
import imageio.v2 as imageio
import numpy as np
import torch.nn as nn
import open3d as o3d

from chatsim.agents.utils import (check_and_mkdirs, generate_vertices, get_attributes_for_one_car, get_outlines,
                    transform_nerf2opencv_convention, get_color, getColorList)
import datetime
import shutil

class Scene(nn.Module):
    def __init__(self, config):
        self.data_root = config['data_root']
        self.scene_name = config['scene_name']

        self.ext_int_path = os.path.join(self.data_root, self.scene_name, config['ext_int_file'])
        self.bbox_path = os.path.join(self.data_root, self.scene_name, config['bbox_file']) # first frame's bbox
        self.map_path = os.path.join(self.data_root, self.scene_name, config['map_file'])
        self.pcd_path = os.path.join(self.data_root, self.scene_name, config['pcd_file'])
        self.init_img_path = os.path.join(self.data_root, self.scene_name, config['init_img_file']) # first frame's image (wide/not-wide)
        
        with open(self.map_path, "rb") as f:
            self.map_data = pickle.load(f)

        self.is_wide_angle = config['is_wide_angle']
        self.fps = config.get('fps', 20)
        self.frames = config['frames']
        self.multi_process_num = config.get('multi_process_num', 1)
        self.if_backup = config.get('if_backup', True)
        self.if_with_depth = config.get('if_with_depth', False)

        """
        [static scene data] 
        """
        self.bbox_data = np.load(
            self.bbox_path, allow_pickle=True
        ).item()

        pcd = o3d.io.read_point_cloud(self.pcd_path)
        self.pcd = np.asarray(pcd.points)
        self.pcd = self.pcd[self.pcd[:,-1]>0.5]

        all_current_vertices = []
        for k in self.bbox_data.keys():
            current_vertices = generate_vertices(self.bbox_data[k])
            all_current_vertices.append(current_vertices)
        self.all_current_vertices = np.array(all_current_vertices)

        if self.all_current_vertices.shape[0] > 0:
            self.all_current_vertices_coord = np.mean(self.all_current_vertices,axis=1)[:,:2]
        else:
            self.all_current_vertices_coord = np.zeros((0,2))

        # read extrinsics from cams_meta.npy. NeRF (RUB) convention.
        extrinsics = np.load(self.ext_int_path)[:, :12].reshape(-1, 3, 4)
        extrinsics = extrinsics[:, :3, :4]

        self.nerf_motion_extrinsics = extrinsics # [N, 3, 4]

        # read intrinsics from cams_meta.npy 
        self.intrinsics = np.load(self.ext_int_path)[:, 12:21].reshape(-1, 3, 3)[0]
        self.focal = self.intrinsics[0, 0]
        self.height = 1280
        self.width = 1920

        if self.is_wide_angle:
            self.intrinsics[0, 2] += 1920 # shift the principal point to the right.
            self.width = 1920 * 3

        """
        [dynamic scene data], will be updated during parsing. 
        ---
        current_extrinsics : np.npdarray [N, 3, 4] 
            N=#frames, correspond to current_images. NeRF (RUB) convention

        current_images : list of np.ndarray [H, W, 3] with len=frames
            Show to users. NeRF's output: current_images

        current_inpainted_images: list of np.ndarray [H, W, 3] with len=frames
            Show to users. NeRF + inpaint's output: current_inpainted_images

        """
        self.is_ego_motion = False
        self.add_car_all_static = True # check every time before blender rendering

        self.current_extrinsics = self.nerf_motion_extrinsics[3:4]  # use the second frame because it is in the training set. Better visualization
        self.current_extrinsics = self.current_extrinsics.repeat(self.frames, axis=0)

        self.removed_cars = []  # keys of cars which are removed
        self.added_cars_dict = {} 
        self.added_cars_count = 0
        
        self.past_operations = []

        self.all_trajectories = []

        # use current time as cache
        current_time = datetime.datetime.now()
        short_scene_name = self.scene_name.lstrip('segment-')[:4]

        simulation_name = config['simulation_name'] # from main.py argparse
        self.logging_name = current_time.strftime(f"{short_scene_name}_{simulation_name}_%Y_%m_%d_%H_%M_%S")

        self.save_cache = config['save_cache']
        self.cache_dir = os.path.join(config["cache_dir"], self.logging_name)
        self.output_dir = config["output_dir"]

        check_and_mkdirs(self.cache_dir)
        check_and_mkdirs(self.output_dir)

    def setup_cars(self):
        """
        Call at the beginning of each interaction. 
        calculate the information of cars from original scene based on current extrinsic
        """
        # get the information of u, v, depth, mask of each car with current extrinsic
        original_cars_dict = {}
        name_to_bbox_car_id = {}
        bbox_car_id_to_name = {}
        
        mask_list = []
        mask_corners_list = []
        depth_list = []
        u_v_depth_list = []
        car_id_list = []

        for car_id in self.bbox_data.keys():
            extrinsic_for_project = transform_nerf2opencv_convention(
                self.current_extrinsics[0]
            )
            u_v_depth = get_attributes_for_one_car(
                self.bbox_data[car_id], extrinsic_for_project, self.intrinsics
            )
            if (
                u_v_depth["u"] < 0
                or u_v_depth["u"] > self.width
                or u_v_depth["v"] < 0
                or u_v_depth["v"] > self.height
            ):
                continue
            corners = generate_vertices(self.bbox_data[car_id])
            mask, mask_corners = get_outlines(
                corners,
                extrinsic_for_project,
                self.intrinsics,
                self.height,
                self.width,
            )

            mask_list.append(mask)
            mask_corners_list.append(mask_corners)
            depth_list.append(u_v_depth["depth"])
            u_v_depth_list.append(u_v_depth)
            car_id_list.append(car_id)

        # add color information
        color_dict = getColorList()
        for idx_in_list, car_id in enumerate(car_id_list):
            car_name = f"original_car_{car_id}"
            name_to_bbox_car_id[car_name] = car_id
            bbox_car_id_to_name[car_id] = car_name

            original_cars_dict[car_name] = u_v_depth_list[idx_in_list]
            current_mask_corner = mask_corners_list[idx_in_list]

            color = get_color(
                self.current_images[0][
                    current_mask_corner[0] + 50 : current_mask_corner[1] - 50,
                    current_mask_corner[2] + 50 : current_mask_corner[3] - 50,
                ]
            )  

            color_vector = (color_dict[color][0] + color_dict[color][1]) / 2
            color_vector = np.uint8(color_vector.reshape(1, 1, 3))
            original_cars_dict[car_name]["rgb"] = cv2.cvtColor(
                color_vector, cv2.COLOR_HSV2RGB
            )
            original_cars_dict[car_name]["x"] = self.bbox_data[car_id]["cx"]
            original_cars_dict[car_name]["y"] = self.bbox_data[car_id]["cy"]

        self.original_cars_dict = original_cars_dict
        self.name_to_bbox_car_id = name_to_bbox_car_id
        self.bbox_car_id_to_name = bbox_car_id_to_name

    def remove_car(self, car_name):
        """
        append car_id to self.removed_cars, inpaint them later.

        car_name
        """
        self.removed_cars.append(car_name)

    def add_car(self, added_car_info):
        """
        Add a single car to self.added_cars_dict dictionary.
        added_car_id is the number of cars added so far.
        """
        # added_car_dict: {'type': str, 'position': temporal [xyz], 'direction': temporal [theta], ...}
        added_car_info['need_placement_and_motion'] = True
        added_car_id = str(self.added_cars_count)
        car_name = f'added_car_{added_car_id}'

        self.added_cars_dict[car_name] = added_car_info
        self.added_cars_count += 1

        return car_name

    def check_added_car_static(self):
        """
        if all added cars are static, we only need to render one frame in blender
        """
        self.add_car_all_static = True
        for added_car_id, added_car_info in self.added_cars_dict.items():
            is_static = np.all(added_car_info['motion'] == added_car_info['motion'][0])
            self.add_car_all_static = self.add_car_all_static and is_static

    def clean_cache(self):
        folder_path = self.cache_dir
        shutil.rmtree(folder_path)