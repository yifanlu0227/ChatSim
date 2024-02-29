"""
Set the hdri environment map

https://blender.stackexchange.com/questions/209584/using-python-to-add-an-hdri-to-world-node
"""

import bpy

def set_hdri(hdri_path, rotation=None):
    """
    Args:
        hdri_path: str
            path to hdri
        rotation: list of float
            [rotate_x, rotate_y, rotate_z] rotate the HDRI. (rad)
            rotate_z (pos) will rotate the skydome clockwise

            By default, the HDRI is set to x-positive view.
    """
    # Add a Environment Texture Node
    C = bpy.context
    scn = C.scene

    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')

    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(hdri_path) # Relative path
    node_environment.location = -300,0

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])


    # rotate the HDRI
    if rotation is not None:
        node_map = tree_nodes.new('ShaderNodeMapping')
        node_map.location = -500,0
        node_texcoor = tree_nodes.new('ShaderNodeTexCoord')
        node_texcoor.location = -700,0
        link = links.new(node_texcoor.outputs['Generated'], node_map.inputs['Vector'])
        link = links.new(node_map.outputs['Vector'], node_environment.inputs['Vector'])

        if isinstance(rotation, list):
            node_map.inputs['Rotation'].default_value = rotation # rad
        elif isinstance(rotation, str):
            if rotation == 'camera_view':
                camera_obj_name = "Camera"
                camera = bpy.data.objects[camera_obj_name]
                camera.rotation_mode = 'XYZ'
                camera_rot_z = camera.rotation_euler.z
                print(camera.rotation_euler)
                node_map.inputs['Rotation'].default_value[2] = -camera_rot_z
                camera.rotation_mode = 'QUATERNION' # turn back to quaternion
            else:
                raise 'This HDRI rotation is not implemented'
        else:
            raise 'This HDRI rotation is not implemented'