import bpy
from blender_utils.camera.set_camera import set_camera_params
from blender_utils.model.set_model import add_plane, add_model_params
from blender_utils.world.hdri import set_hdri
from blender_utils.render.render import render_scene, check_mkdir
from blender_utils.postprocess.compose import compose
from blender_utils.utils.common_utils import rm_all_in_blender

import numpy as np
import yaml
import sys
import os
import imageio.v2 as imageio
import shutil

def render(render_opt):
    render_downsample = render_opt.get('render_downsample', 1)
    motion_blur_degree = render_opt.get('motion_blur_degree', 4)

    hdri_file = render_opt['hdri_file']
    intrinsic = render_opt['intrinsic']
    cam2world = render_opt['cam2world']
    background_RGB = render_opt['background_RGB']  # [H,W,3]
    background_depth = render_opt['background_depth']

    set_camera_params(intrinsic, cam2world)
    model_obj_names = []

    car_list = render_opt['cars']
    for car_obj in car_list:
        add_model_params(car_obj)
        model_obj_names.append(car_obj['new_obj_name'])

    add_plane(227) # should be big enough to include all cars

    set_hdri(hdri_file, None)

    check_mkdir(render_opt['output_dir'])
    render_output_dir = os.path.join(render_opt['output_dir'], render_opt['render_name'])
    check_mkdir(render_output_dir)

    # back up hdri and rgb image
    render_output_backup_dir = os.path.join(render_output_dir, 'backup')
    check_mkdir(render_output_backup_dir)
    imageio.imsave(os.path.join(render_output_backup_dir, 'RGB.png'), background_RGB)
    shutil.copy(hdri_file, os.path.join(render_output_backup_dir, 'hdri.exr'))

    render_scene(render_output_dir, intrinsic, model_obj_names, render_downsample)
    compose(render_output_dir, background_RGB, background_depth, render_downsample, motion_blur_degree)

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    render_yaml = argv[0]

    with open(render_yaml, 'r') as file:
        render_opt = yaml.safe_load(file)
    
    bpy.ops.wm.read_homefile(app_template="")
    rm_all_in_blender()

    # read config from scene
    scene_data = render_opt['scene_file']
    data_dict = np.load(scene_data)

    H = data_dict['H'].tolist()
    W = data_dict['W'].tolist()
    focal = data_dict['focal'].tolist()
    render_opt['intrinsic'] = {"H":H, "W":W, "focal":focal}
    render_opt['cam2world'] = data_dict['extrinsic']
    render_opt['background_RGB'] = data_dict['rgb']
    render_opt['background_depth'] = data_dict['depth']

    render(render_opt)

if __name__=="__main__":
    main()