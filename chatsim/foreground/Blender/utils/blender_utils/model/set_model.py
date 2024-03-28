"""
Modify the car paint color.
"""
import bpy
import os

def modify_car_color(model:bpy.types.Object, material_key, color):
    """
    Args:
        model: bpy_types.Objct
            car model
        material_key: str
            key name in model.material_slots. Refer to the car paint material.
        color: list of float
            target base color, [R,G,B,alpha] 
    """
    material = model.material_slots[material_key].material
    # Modifiy Metaillic, Specular, Roughness if needed
    # Suppose use Principled BSDF
    material.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color


def set_model_params(loc, rot, rot_mode="XYZ", model_obj_name="Car", target_color=None):
    """
    Args:
        loc: list
            [x, y, z]
        rot: list
            [angle1, angle2, angle3] (rad.)
        rot_mode: str
            Euler angle order
        model_obj_name: str
            name of the entire model. New obj name.
        target_color: dict (optinoal)
            {"material_key":.., "color": ...}
    """
    model = bpy.data.objects[model_obj_name]
    model.location = loc
    model.rotation_mode = rot_mode
    model.rotation_euler = rot

    if target_color is not None:
        modify_car_color(model, 
                         target_color['material_key'], 
                         target_color['color']
        )


def add_model_params(model_setting):
    """
    model_setting includes:
        - blender_file: path to object model file
        - insert_pos: list of len 3
        - insert_rot: list of len 3 
        - model_obj_name: object name within blender_file
        - new_obj_name: object name in this scene
        - target_color: optional .
    """
    blender_file = model_setting['blender_file']
    model_obj_name = model_setting['model_obj_name']
    new_obj_name = model_setting['new_obj_name']
    target_color = model_setting.get('target_color', None)

    # append object into the scene, use bpy.ops.wm.append. 
    # not work in non batch mode (wo -b)，会遇到上下文的bug
    # inner_path = 'Object'
    # bpy.ops.wm.append(
    #     filepath=os.path.join(blender_file, inner_path, model_obj_name),
    #     directory=os.path.join(blender_file, inner_path),
    #     filename=model_obj_name,
    # )

    # append object into the scene, use bpy.data.libraries.load
    # 这一步仅仅是将从外部.blend文件中加载的对象数据复制到了data_to 上下文，但尚未将这些对象添加到任何场景。
    # 相关材质和贴图也被自动load了，不用手动重复load
    with bpy.data.libraries.load(blender_file, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    # link to context is required! 这是将实际对象添加到场景的关键步骤。
    for obj in data_to.objects: 
        if obj.name == model_obj_name:
            bpy.context.collection.objects.link(obj)
            
    if model_obj_name in bpy.data.objects:
        imported_object = bpy.data.objects[model_obj_name]
        imported_object.name = new_obj_name
        print(f"rename {model_obj_name} to {new_obj_name}")

    # rename material
    for slot in imported_object.material_slots:
        material = slot.material
        if material:
            # 为每个材质添加前缀
            material.name = new_obj_name + "_" + material.name

    if target_color is not None:
        target_color['material_key'] = new_obj_name + "_" + target_color['material_key']


    set_model_params(model_setting['insert_pos'],
                     model_setting['insert_rot'], 
                     rot_mode="XYZ", 
                     model_obj_name=new_obj_name, 
                     target_color=target_color)


def add_plane(size):
    # we need create a plane for the model
    bpy.ops.mesh.primitive_plane_add(size=1)

    if hasattr(bpy.context, 'object'): # background mode
        plane = bpy.context.object
    else:   # interface mode
        plane = bpy.data.objects["Plane"]

    plane.scale = (size, size, 1)
    plane.name =  "plane"
    plane.is_shadow_catcher = True

    # new material for the plane
    material = bpy.data.materials.new(name="new_plane_material")
    plane.data.materials.append(material)

    # set material color for the plane
    material.use_nodes = True
    nodes = material.node_tree.nodes
    BSDF_node = nodes.get("Principled BSDF")

    if BSDF_node:
        # base color default dark. will not affect shadow color, but will affect the reflection on the car
        BSDF_node.inputs[0].default_value = (0.004, 0.005, 0.006, 1) 
        BSDF_node.inputs[9].default_value = 1  # roughness
        BSDF_node.inputs[21].default_value = 1  # alpha. Otherwise composition with background will lead to strong black.