import numpy as np
import trimesh
from copy import deepcopy
from mc_to_sky.utils.ray_utils import get_ray_directions, get_rays, random_sample_on_hemisphere, random_samples_on_hemisphere
from mc_to_sky.utils.brdf_utils import UE4BRDF, rename_material_dict
from mc_to_sky.utils.obj_utils import materialed_meshes
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction_torch, srgb_inv_gamma_correction
from mc_to_sky.utils.time_utils import Timer
from envmap import EnvironmentMap, rotation_matrix
from math import floor
import imageio.v2 as imageio
import torch
from icecream import ic 


class Renderer:
    def __init__(self, obj_path):
        self.scene = materialed_meshes(obj_path)
        self.num_samples = 5000
        
    
    def read_int(self):
        self.H = 1280  // 3
        self.W = 1920  // 3
        self.focal = 2083 // 3

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
        # self.env.rotate(rotation_matrix(azimuth=np.pi, elevation=0))
        data = np.ones((2048, 4096, 3))
        self.env = EnvironmentMap(data, 'latlong')
    
    def IBL(self, light_dir):
        """
        Args:
            light_dir: [num_sample, 3], torch.tensor

        Returns:
            light_intensity:  [num_sample, 3], torch.tensor
        """
        def world2latlong(x, y, z):
            """Get the (u, v) coordinates of the point defined by (x, y, z) for
            a latitude-longitude map."""
            u = 1 + (1 / np.pi) * torch.arctan2(x, -z)
            v = (1 / np.pi) * torch.arccos(y)
            # because we want [0,1] interval
            u = u / 2
            return u, v 

        light_dir_envmap = [-light_dir[:,1], light_dir[:,2], -light_dir[:,0]]
        uu, vv = world2latlong(light_dir_envmap[0], light_dir_envmap[1], light_dir_envmap[2]) # [num_samples, ]
        uu = np.floor(uu.cpu().numpy() * self.env.data.shape[1] % self.env.data.shape[1]).astype(int)
        vv = np.floor(vv.cpu().numpy() * self.env.data.shape[0] % self.env.data.shape[0]).astype(int)

        light_intensity = self.env.data[vv, uu] # [num_samples, 3]
        light_intensity = torch.from_numpy(light_intensity).to(light_dir)

        return light_intensity

    def render_hdri(self):
        self.buffer = self.IBL(torch.from_numpy(self.ray_d))

    def render(self):
        # get rays
        timer = Timer()
        directions = get_ray_directions(self.H, self.W, self.focal)
        self.ray_o, self.ray_d = get_rays(directions, self.c2w)
        timer.print("generating rays")

        # Rendering world environment
        self.render_hdri()
        timer.print("HDRI background rendering")

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

        for i in range(num_hit):
            # timer.print("start each pixel")
            intersection_p = intersections[i]
            normal = intersection_normals[i]
            idx_in_ray = index_ray[i]
            idx_in_faces = index_tri[i]
            wo = - self.ray_d[idx_in_ray]
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

            # Pure specular material
            if material_dict['pr'] == 0:
                wi = 2 * np.dot(wo, normal) * normal - wo
                wi = torch.from_numpy(wi).cuda().reshape(1,3)
                colors = self.IBL(wi)
            else:
                color = torch.zeros(3).cuda()
                normal = torch.from_numpy(normal).cuda()
                normal = normal / normal.norm()
                normal = normal.expand(self.num_samples, 3) # num_samples, 3
                view_dir = torch.from_numpy(wo).cuda()
                view_dir = view_dir.expand(self.num_samples, 3)
                
                light_dir = random_samples_on_hemisphere(normal, self.num_samples)
                brdfs = bdrf.evaluate_parallel(normal, light_dir, view_dir) # num_samples, 3
                light_intensity = self.IBL(light_dir) # num_samples, 3
                n_dot_l = torch.einsum('ij,ij->i', normal, light_dir).unsqueeze(-1).expand(-1, 3) # num_samples, 3

                colors = light_intensity * brdfs * n_dot_l / (0.5/np.pi)
            self.buffer[idx_in_ray] = colors.mean(0)

        timer.print("Rendering foreground")

        self.renderd_image = self.buffer.reshape(self.H, self.W, 3)
        self.renderd_image = srgb_gamma_correction_torch(self.renderd_image)
        output = (self.renderd_image * 255).cpu().numpy().astype(np.uint8)
        imageio.imsave("/home/yfl/workspace/LDR_to_HDR/logs/rendered_result_pos2_whitehdr_kloppenheim_05_1k.png", output)


def main():
    renderer = Renderer("/home/yfl/Desktop/3d_assets/chevrolet_suv2/chevrolet.obj")
    # renderer = Renderer("/home/yfl/Desktop/3d_assets/glass/carpaint.obj")
    # renderer.read_env("/home/yfl/workspace/dataset_ln/HDR_ours/train/abandoned_parking_1k.exr")
    renderer.read_env("/home/yfl/workspace/dataset_ln/HDR_ours/train/kloppenheim_05_1k.exr")
    renderer.read_ext()
    renderer.read_int()
    renderer.render()

if __name__ == "__main__":
    main()
