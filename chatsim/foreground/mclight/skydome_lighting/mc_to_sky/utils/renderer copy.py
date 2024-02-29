import numpy as np
import trimesh
from collections import OrderedDict
from mc_to_sky.utils.ray_utils import get_ray_directions, get_rays, random_sample_on_hemisphere, random_samples_on_hemisphere
from mc_to_sky.utils.brdf_utils import UE4BRDF, rename_material_dict
from mc_to_sky.utils.obj_utils import materialed_meshes
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction_torch, srgb_inv_gamma_correction_torch
from mc_to_sky.utils.time_utils import Timer
from envmap import EnvironmentMap
from math import floor
import imageio.v2 as imageio
import torch
from icecream import ic 
import multiprocessing
from copy import deepcopy

class subRender:
    def __init__(self, scene, inter_dict):
        self.scene = scene
        self.inter_dict = inter_dict
        self.num_samples = 5000
    
    def render(self, ids):
        color_list = []
        for i in ids:
            normal = self.inter_dict['intersection_normals'][i]
            idx_in_ray = self.inter_dict['index_ray'][i]
            idx_in_faces = self.inter_dict['index_tri'][i]
            wi = - self.inter_dict['ray_d'][idx_in_ray]
            material, face_local, uv_local = self.scene.get_material_from_face_idx_of_all(idx_in_faces)

            if 'kd' not in material.kwargs:
                material_image = material.image
                # no interpolate yet
                u, v = uv_local[0]
                u = u - floor(u)
                v = v - floor(v)
                width, height = material_image.size
                pixel_x = int(u * (width - 1))
                pixel_y = int(v * (height - 1))

                kd = material_image.getpixel((pixel_x, pixel_y)) # tuple
                if isinstance(kd, int): # grey
                    kd = [kd, kd, kd]

                material.kwargs['kd'] = [kd[0]/255, kd[1]/255, kd[2]/255]
            
            material_dict = rename_material_dict(material.kwargs)

            bdrf = UE4BRDF(base_color=material_dict['kd'], metallic=material_dict['pm'], roughness=material_dict['pr'], specular=material_dict['ks'])
            normal = torch.from_numpy(normal).cuda()
            normal = normal / normal.norm()
            light_dir = torch.from_numpy(wi).cuda()
            light_dir = light_dir / light_dir.norm()

            normal = normal.expand(self.num_samples, 3)
            light_dir = light_dir.expand(self.num_samples, 3)
            view_dir = random_samples_on_hemisphere(normal, self.num_samples)
            color = bdrf.evaluate_parallel(normal, light_dir, view_dir).mean(dim=0)
            color_list.append(color)

        return torch.stack(color_list)

def parallel_rendering(scene, inter_dict, ids):
    sub_render = subRender(scene, inter_dict)
    return sub_render.render(ids)

class Renderer:
    def __init__(self, obj_path):
        self.scene = materialed_meshes(obj_path)
        self.num_samples = 5000
        self.num_processes = 2
        
    
    def read_int(self):
        self.H = 1280 
        self.W = 1920 
        self.focal = 2083 

        self.buffer = torch.zeros(self.H*self.W, 3).to('cuda')

    
    def read_ext(self):
        self.c2w = np.array([
                [ 0.0123957 , -0.00906409, -0.99988209,  2.35675933],
                [-0.99987913,  0.00927219, -0.01247972, -0.01891149],
                [ 0.00938421,  0.99991593, -0.00894806,  2.11490003]]
            ).astype(np.float32)

        
    def read_env(self, envpath):
        """ envmap: viewing -Z
            Y
            | 
            |
            .------ X
           /       
          Z
        """
        self.env = EnvironmentMap(envpath, 'latlong')
    
    def IBL(self, light_dir):
        """
        transform light_dir in world coord to envmap coor. hand-crafted
        """
        light_dir_np = light_dir.cpu().numpy()
        light_dir_envmap = [-light_dir_np[1], light_dir_np[2], -light_dir_np[0]]
        uu, vv = self.env.world2pixel(light_dir_envmap[0], light_dir_envmap[1], light_dir_envmap[2])
        light_intensity = torch.tensor(self.env.data[vv, uu], device='cuda', dtype=torch.float32)
        return light_intensity

    def render(self):
        multiprocessing.set_start_method('spawn')

        # get rays
        timer = Timer()
        directions = get_ray_directions(self.H, self.W, self.focal)
        self.ray_o, self.ray_d = get_rays(directions, self.c2w)
        timer.print("generating rays")
        # find intersections of the ray with the mesh
        mesh_all = self.scene.get_all_meshes()
        intersections, index_ray, index_tri = mesh_all.ray.intersects_location(
            ray_origins=self.ray_o, 
            ray_directions=self.ray_d,
            multiple_hits=False,
        )
        intersection_normals = mesh_all.face_normals[index_tri].astype(np.float32)
        timer.print('ray-mesh intersection')

        num_hit = intersections.shape[0]
        print(f'number of intersection: {num_hit}')

        # add to dict
        self.inter_dict = OrderedDict()
        self.inter_dict['index_ray'] = index_ray
        self.inter_dict['index_tri'] = index_tri
        self.inter_dict['intersection_normals'] = intersection_normals
        self.inter_dict['ray_d'] = self.ray_d
        
        pool = multiprocessing.Pool(processes=self.num_processes)

        # 构造任务列表
        tasks = np.array_split(np.arange(num_hit), self.num_processes)

        # 使用多进程并行执行draw函数
        results = pool.starmap(parallel_rendering, 
                               [(deepcopy(self.scene), deepcopy(self.inter_dict), ids) for ids in tasks])

        pool.close()
        pool.join()

        colors = torch.cat(results, dim=0)
        self.buffer[index_ray] = colors

        timer.print("Rendering foreground")

        # # fill background with envmap
        # for i in range(self.H*self.W):
        #     if i in index_ray:
        #         continue
        #     ray_d = self.ray_d[i]
        #     ray_d = torch.from_numpy(ray_d).cuda()
        #     ray_d /= ray_d.norm()
        #     color = self.IBL(ray_d)
        #     self.buffer[i] = color
        # timer.print("Rendering background")


        self.renderd_image = self.buffer.reshape(self.H, self.W, 3)
        self.renderd_image = srgb_gamma_correction_torch(self.renderd_image)
        output = (self.renderd_image * 255).cpu().numpy().astype(np.uint8)
        imageio.imsave("/home/yfl/workspace/LDR_to_HDR/logs/rendered_result.png", output)


def main():
    renderer = Renderer("/home/yfl/Desktop/3d_assets/chevrolet_suv/chevrolet-suv-rigged_waymo_sunny1.obj")
    renderer.read_env("/home/yfl/workspace/dataset_ln/HDR_ours/train/abandoned_parking_1k.exr")
    renderer.read_ext()
    renderer.read_int()
    renderer.render()

if __name__ == "__main__":
    main()
