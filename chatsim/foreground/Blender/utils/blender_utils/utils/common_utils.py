import bpy
import yaml 

def rm_all_in_blender():

    # 清除所有默认的物体
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)        

    # 清除所有默认的材质
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)

    # 清除所有默认的网格数据
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # 清除所有默认的贴图
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # 清除所有默认的图片
    for image in bpy.data.images:
        bpy.data.images.remove(image)

def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)