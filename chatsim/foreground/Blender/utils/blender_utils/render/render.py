"""
This file output the rendering results, including
1) vehicle 2) vehicle mask 3) depth map 4) vechile+plane 5) plane

Final rendering result = vehicle + shadow 

And composite Final rendering result with background with depth test.
"""
import os
import bpy
from collections import OrderedDict
import numpy as np
import pyquaternion
from blender_utils.utils.box_utils import create_bbx
from blender_utils.utils.common_utils import save_yaml



def check_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def set_render_params(render_H, render_W, render_downsample, sample_num=32, device='GPU'):
    # set Cycles
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Set the device and feature set
    bpy.context.scene.cycles.device = device

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    print("preferences.compute_device_type: ", \
          bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    for dev in bpy.context.preferences.addons["cycles"].preferences.devices:
        # dev['use'] = 1 # Using all devices, include GPU and CPU
        print(f"Use Device {dev['name']}: {dev['use']}")

    scene.cycles.samples = sample_num
    scene.render.resolution_x = render_W
    scene.render.resolution_y = render_H
    scene.render.resolution_percentage = 100 // render_downsample

    # transparent background
    scene.render.film_transparent = True

    # enable passes. bpy.context.view_layer or bpy.context.scene.view_layers["ViewLayer"]
    # combined + depth + shadow catcher
    bpy.context.view_layer.use_pass_combined = True
    bpy.context.view_layer.use_pass_z = True 
    bpy.context.view_layer.cycles.use_pass_shadow_catcher = True



def set_composite_node(output_dir, render_downsample, depth_and_occlusion):
    """
        setup composite node.
    """
    scene = bpy.context.scene
    # use nodes in Compositing Editor. Note it is not WORLD nodes in hdri.py
    scene.use_nodes = True 
    node_tree = scene.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()

    # Create Render Layers node
    render_node = tree_nodes.new('CompositorNodeRLayers')
    render_node.name = "Render_node"
    render_node.location = -300, 0

    # Create transform node (for upsampling or downsampling)
    transform_node_for_image = tree_nodes.new(type='CompositorNodeTransform')
    transform_node_for_image.location = 300, 400
    transform_node_for_image.filter_type = 'BILINEAR'
    transform_node_for_image.inputs['Scale'].default_value = 1 / render_downsample


    # Create Image node
    image_node = tree_nodes.new('CompositorNodeImage')
    image_node.name = "Image_node"
    image_node.location = 0, 400
    image_path = os.path.join(output_dir, 'backup', 'RGB.png')
    image_node.image = bpy.data.images.load(image_path)
    image_node.image.colorspace_settings.name = 'Filmic sRGB'

    # Create File Output node for RGB file.
    # RGB is the render vehicle + shadow over background (RGBA, no depth test)
    RGB_output_node = tree_nodes.new('CompositorNodeOutputFile')
    RGB_output_node.name = 'RGB_output_node'
    RGB_output_node.location = 1500, 200
    RGB_output_node.format.file_format = 'PNG'  # 设置输出格式
    RGB_output_node.format.color_mode = 'RGBA'
    RGB_folder = os.path.join(output_dir, 'RGB')
    RGB_output_node.base_path = RGB_folder  # 设置输出目录
    RGB_output_node.file_slots[0].path = "vehicle_and_shadow_over_background" # the filename will append 0001 as suffix
    check_mkdir(RGB_folder)

    # Create File Output node for depth file.
    # depth is vehicle + plane (RGB, with very large number 65534 at empty place)
    if depth_and_occlusion == True:
        depth_output_node = tree_nodes.new('CompositorNodeOutputFile')
        depth_output_node.name = 'Depth_output_node'
        depth_output_node.location = 1500, 0
        depth_output_node.format.file_format = 'OPEN_EXR'  # 设置输出格式
        depth_output_node.format.color_mode = 'RGBA' # if not go through alpha set, output will not have alpha channel.
        depth_folder = os.path.join(output_dir, 'depth')
        depth_output_node.base_path = depth_folder  # 设置输出目录
        depth_output_node.file_slots[0].path = "vehicle_and_plane"
        check_mkdir(depth_folder)

    # Create File Output node for mask file.
    # mask is vehicle + shadow (RGBA, with very large number 65534 at empty places)
    if depth_and_occlusion == True:
        mask_output_node = tree_nodes.new('CompositorNodeOutputFile')
        mask_output_node.name = "Mask_output_node"
        mask_output_node.location = 1500, -300
        mask_output_node.format.file_format = 'OPEN_EXR'  # 设置输出格式
        mask_output_node.format.color_mode = 'RGBA' # if not go through alpha set, output will not have alpha channel.
        mask_folder = os.path.join(output_dir, 'mask')
        mask_output_node.base_path = mask_folder  # 设置输出目录
        mask_output_node.file_slots[0].path = "vehicle_and_shadow"
        check_mkdir(mask_folder)

    # Create Multiply node
    multiply_node = tree_nodes.new('CompositorNodeMixRGB')
    multiply_node.name = "Multiply_node"
    multiply_node.blend_type = 'MULTIPLY'
    multiply_node.location = 600, 100

    # Create Alpha Over node
    alpha_over_node = tree_nodes.new('CompositorNodeAlphaOver')
    alpha_over_node.name = "Alpha_over_node"
    alpha_over_node.location = 900, 100

    # Create Invert node
    invert_node = tree_nodes.new('CompositorNodeInvert')
    invert_node.name = "Invert_node"
    invert_node.location = 300, -300

    # Create Set Alpha nodes
    set_alpha_node_1 = tree_nodes.new('CompositorNodeSetAlpha')
    set_alpha_node_1.name = "Set_alpha_node_1"
    set_alpha_node_1.location = 600, -300
    set_alpha_node_1.inputs[0].default_value = (1,1,1,1)

    set_alpha_node_2 = tree_nodes.new('CompositorNodeSetAlpha')
    set_alpha_node_2.name = "Set_alpha_node_2"
    set_alpha_node_2.location = 600, -500
    set_alpha_node_2.inputs[0].default_value = (1,1,1,1)

    # Create Add node
    add_node = tree_nodes.new('CompositorNodeMixRGB')
    add_node.name = "Add_node"
    add_node.blend_type = 'ADD'
    add_node.location = 900, -300
    add_node.use_clamp = True

    # Create Seperate RGBA node
    separate_rgba_node = tree_nodes.new(type='CompositorNodeSepRGBA')
    separate_rgba_node.name = "Seperate_RGBA"
    separate_rgba_node.location = 1200, -300

    node_tree.links.clear()
    links = node_tree.links


    # (optional) upsampling Render Layer output
    # links.new(render_node.outputs['Image'], transform_node_for_image.inputs['Image'])
    # links.new(render_node.outputs['Alpha'], transform_node_for_alpha.inputs['Image'])
    # links.new(render_node.outputs['Depth'], transform_node_for_depth.inputs['Image'])
    # links.new(render_node.outputs['Shadow Catcher'], transform_node_for_shadow_catcher.inputs['Image'])

    # depth 
    if depth_and_occlusion == True:
        links.new(render_node.outputs['Depth'], depth_output_node.inputs[0])

    # RGB over background
    links.new(render_node.outputs['Image'], alpha_over_node.inputs[2]) # the second Image socket of alpha_over_node
    links.new(render_node.outputs['Shadow Catcher'], multiply_node.inputs[1])  # the first Image socket of multiply_node

    links.new(image_node.outputs['Image'], transform_node_for_image.inputs['Image'])
    links.new(transform_node_for_image.outputs['Image'], multiply_node.inputs[2]) # the second Image socket of multiply_node
    links.new(multiply_node.outputs['Image'], alpha_over_node.inputs[1]) # the first Image socket of alpha_over_node
    links.new(alpha_over_node.outputs['Image'], RGB_output_node.inputs[0])

    # mask 
    if depth_and_occlusion == True:
        links.new(render_node.outputs['Alpha'], set_alpha_node_2.inputs['Alpha'])
        links.new(render_node.outputs['Shadow Catcher'], invert_node.inputs['Color']) 
        links.new(invert_node.outputs['Color'], set_alpha_node_1.inputs['Alpha'])
        links.new(set_alpha_node_1.outputs['Image'], add_node.inputs[1]) # the first Image socket of add
        links.new(set_alpha_node_2.outputs['Image'], add_node.inputs[2]) # the second Image socket of add
        links.new(add_node.outputs['Image'], separate_rgba_node.inputs['Image']) 
        links.new(separate_rgba_node.outputs['R'], mask_output_node.inputs["Image"]) # RGB are the same.

    return node_tree



def render_scene(output_dir, 
                 intrinsic,
                 model_obj_names,
                 render_downsample,
                 depth_and_occlusion):
    
    set_render_params(intrinsic['H'], intrinsic['W'], render_downsample)
    node_tree = set_composite_node(output_dir, render_downsample, depth_and_occlusion)
    
    # encode object's position, rotation, dimension to 8 corners
    objects_corners = dict()

    for model_obj_name in model_obj_names:
        model = bpy.data.objects[model_obj_name]
        # render vehicle + plane (has shadow)
        model.hide_render = False

        # calculate obj corners
        position = model.location
        model.rotation_mode = "QUATERNION"
        rotation_quat = model.rotation_quaternion # w x y z
        obj2world = pyquaternion.Quaternion(rotation_quat).transformation_matrix
        obj2world[:3, 3] = position

        dimension = model.dimensions
        corner_in_obj = create_bbx([dimension[0]/2, dimension[1]/2, dimension[2]/2])
        corner_in_obj[:, -1] += dimension[2]/2 # origin is at the bottom 
        corner_in_obj_homo = np.hstack([corner_in_obj, np.ones((corner_in_obj.shape[0], 1))])
        corner_in_world_homo = (obj2world @ corner_in_obj_homo.T).T
        corner_in_world = corner_in_world_homo[:,:3]

        objects_corners[model_obj_name] = {
            'world_8_corners': corner_in_world.tolist()
        }
    
    save_yaml(objects_corners, os.path.join(output_dir, 'label.yaml'))

    bpy.ops.render.render()
