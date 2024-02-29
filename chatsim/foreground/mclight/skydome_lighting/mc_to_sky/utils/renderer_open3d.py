from icecream import ic
import numpy as np
import open3d as o3d
from open3d.visualization import rendering
import trimesh
from mc_to_sky.utils.obj_utils import materialed_meshes
from mc_to_sky.utils.brdf_utils import rename_material_dict

trimesh_scene = materialed_meshes("/home/yfl/Desktop/3d_assets/chevrolet_suv/chevrolet-suv-rigged_waymo_sunny1.obj")

render = rendering.OffscreenRenderer(640*3, 480*3)
geometrys = []

for materialed_mesh in trimesh_scene.scene_dump:
    mesh_o3d = materialed_mesh.as_open3d
    mesh_o3d.compute_triangle_normals()
    ic(mesh_o3d.get_min_bound())
    ic(mesh_o3d.get_max_bound())
    material = materialed_mesh.visual.material
    material_dict = rename_material_dict(material.kwargs)
    
    material_o3d = rendering.MaterialRecord()
    material_o3d.shader = "defaultLit"

    if 'kd' not in material_dict:
        material_o3d.albedo_img = o3d.geometry.Image(np.array(material.image))
    else:
        material.kwargs['kd'] = material.kwargs['kd'] + [1.0]
        material_o3d.base_color = material.kwargs['kd']


    
    material_o3d.base_reflectance = material_dict['ks'][0]
    material_o3d.base_roughness = material_dict['pr']
    material_o3d.base_metallic = material_dict['pm']
    
    render.scene.add_geometry(materialed_mesh.metadata['name'], mesh_o3d, material_o3d)
    geometrys.append({"name":materialed_mesh.metadata['name'],
                      "geometry": mesh_o3d,
                      "material": material_o3d})

# o3d.visualization.draw(geometrys)


render.setup_camera(60.0, [40, 5, 0], [20, 1, 2], [0, 0, 1])

render.scene.scene.set_sun_light([0 , 0.0, -1], [1.0, 1.0, 1.0],
                                    750000)
# render.scene.scene.set_indirect_light("/home/yfl/Downloads/cmft_lin64/okretnica_pmrem")

render.scene.scene.enable_sun_light(True)
render.scene.scene.enable_indirect_light(True)
render.scene.show_axes(True)

img = render.render_to_image()
print("Saving image at test.png")
o3d.io.write_image("test.png", img, 9)