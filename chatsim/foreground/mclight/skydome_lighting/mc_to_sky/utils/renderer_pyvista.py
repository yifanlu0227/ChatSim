import pyvista as pv
import numpy as np
import trimesh
import imageio.v2 as imageio
from icecream import ic
from mc_to_sky.utils.obj_utils import materialed_meshes
from mc_to_sky.utils.brdf_utils import rename_material_dict
from pyvista import examples

trimesh_scene = materialed_meshes("/home/yfl/Desktop/3d_assets/chevrolet_suv/chevrolet-suv-rigged_waymo_sunny1.obj")

# env_data = imageio.imread("/home/yfl/workspace/dataset_ln/HDR_ours/train/abandoned_parking_1k.exr")
env_data = np.ones((1024, 2048 ,3)).astype(np.float32)
env_texture = pv.Texture(env_data)
env_texture.SetColorModeToDirectScalars()
env_texture.SetMipmap(True)
env_texture.SetInterpolate(True)

pl = pv.Plotter()
pl.set_environment_texture(env_texture)

for materialed_mesh in trimesh_scene.scene_dump:
    mesh = pv.wrap(materialed_mesh).rotate_x(-90.0, inplace=True)
    material = materialed_mesh.visual.material
    
    if 'kd' in material.kwargs:
        material_dict = rename_material_dict(material.kwargs)
        ic(materialed_mesh.metadata['node'], material_dict['kd'])
        pl.add_mesh(mesh, pbr=True, 
                    color = material_dict['kd'],
                    specular=material_dict['ks'][0], 
                    roughness=material_dict['pr'], 
                    metallic=material_dict['pm'])
    else: # use texture
        material_dict = rename_material_dict(material.kwargs)
        pl.add_mesh(mesh, pbr=True, 
                    texture = np.array(material.image),
                    specular=material_dict['ks'][0], 
                    roughness=material_dict['pr'], 
                    metallic=material_dict['pm'])

pl.show()