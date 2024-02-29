import torch
import numpy as np
from collections import OrderedDict
from mc_to_sky.utils.hdr_utils import srgb_inv_gamma_correction_torch
from icecream import ic

class UE4BRDF:
    def __init__(self, base_color, metallic, roughness, specular):
        self.base_color = torch.tensor(base_color, dtype=torch.float64).cuda()
        self.base_color = srgb_inv_gamma_correction_torch(self.base_color)
        self.metallic = torch.tensor(metallic, dtype=torch.float64).cuda()

        self.roughness = torch.tensor(roughness, dtype=torch.float64).cuda().clamp(min=1e-4)
        # for metallic materials, reflectance should calculate from the base color
        self.specular = torch.tensor(specular, dtype=torch.float64).cuda() # Fresnel reflectance at normal incidence for dielectric surfaces. 
        self.c_diffuse = (1 - self.metallic) * self.base_color
        self.c_specular = (1 - self.metallic) * self.specular + self.metallic * self.base_color


    def lambertian_diffuse(self):
        return self.c_diffuse / np.pi

    def fresnel_schlick(self, h_dot_v): # F term
        # [3] + [3] * [100, 1]
        # return self.c_specular + (1 - self.c_specular) * (torch.pow(2, -5.55473 * h_dot_v - 6.98316) * h_dot_v).repeat_interleave(3, dim=-1)
        return self.c_specular + (1 - self.c_specular) * torch.pow(1-h_dot_v, 5).repeat_interleave(3, dim=-1)

    def smith_g1(self, n_dot_v):
        k = pow(self.roughness + 1, 2) / 8
        return n_dot_v / (n_dot_v * (1 - k) + k)
    
    def geometry(self, n_dot_v, n_dot_l): # G term
        return self.smith_g1(n_dot_v)*self.smith_g1(n_dot_l)

    # def normal_distribution(self, n, h): # D term
    #     n_dot_h = torch.clamp(torch.einsum('ij,ij->i', n, h).unsqueeze(-1), min = 1e-5)
    #     alpha = pow(self.roughness, 2).clamp(min=2e-4) # avoid the denominator to be 0, when alpha is 0 and n_dot_h is 1
    #     d = alpha**2 / (np.pi * (n_dot_h**2 * (alpha**2 - 1) + 1)**2)
    #     return d
    
    def normal_distribution(self, n, h): # D term
        roughness = pow(self.roughness, 2) # scalar
        n_cross_h = torch.cross(n, h, dim=1) # [n_samples, 3]

        n_dot_h = torch.einsum('ij,ij->i', n, h).unsqueeze(-1) # [n_samples, 1]
        a = n_dot_h * roughness # [n, samples, 1]
        k = roughness / (torch.einsum("ij,ij->i", n_cross_h, n_cross_h).unsqueeze(-1) + a * a) # [n_samples, 1]
        d = k * k * (1 / np.pi) # [n_samples, 1]

        return d


    def evaluate(self, normal, light_dir, view_dir):
        n_dot_l = torch.clamp(torch.dot(normal, light_dir), min=0.00001)
        n_dot_v = torch.clamp(torch.dot(normal, view_dir), min=0.00001)
        half_dir = (light_dir + view_dir) / torch.norm(light_dir + view_dir)
        h_dot_v = torch.clamp(torch.dot(half_dir, view_dir), min=0.00001)
        n_dot_h = torch.clamp(torch.dot(normal, half_dir), min=0.00001)

        diffuse_term = self.lambertian_diffuse()
        specular_term = (self.fresnel_schlick(h_dot_v) * self.geometry(n_dot_l, n_dot_v) * self.normal_distribution(normal, half_dir)) / (4 * n_dot_l * n_dot_v)

        return diffuse_term + specular_term
    
    def evaluate_parallel(self, normal, light_dir, view_dir):
        """
        Args:
            normal: [num_samples, 3]
            light_dir: [num_samples, 3]
            view_dir: [num_samples]
        """
        normal = normal.double()
        light_dir = light_dir.double()
        view_dir = view_dir.double()
        num_samples = normal.shape[0]
        n_dot_l = torch.clamp(torch.einsum('ij,ij->i', normal, light_dir).unsqueeze(-1), min = 1e-5) # [num_samples, 1]
        n_dot_v = torch.clamp(torch.einsum('ij,ij->i', normal, view_dir).unsqueeze(-1), min = 1e-5)
        half_dir = (light_dir + view_dir) / torch.norm(light_dir + view_dir, p=2, dim=1, keepdim=True)
        h_dot_v = torch.clamp(torch.einsum('ij,ij->i', half_dir, view_dir).unsqueeze(-1), min = 1e-5)
        # n_dot_h = torch.clamp(torch.einsum('ij,ij->i', normal, half_dir).unsqueeze(-1), min = 1e-5)

        diffuse_term = self.lambertian_diffuse().expand(num_samples, 3)
        specular_term = (self.fresnel_schlick(h_dot_v) * self.geometry(n_dot_l, n_dot_v) * self.normal_distribution(normal, half_dir)) / (4 * n_dot_l * n_dot_v)
        brdf = diffuse_term + specular_term
        return brdf.float()



def rename_material_dict(old_dict):
    new_dict = OrderedDict()
    new_dict['ks'] = old_dict['ks']
    if 'kd' in old_dict:
        new_dict['kd'] = old_dict['kd']

    for key in ['pm', 'pr']:
        if isinstance(old_dict[key], list):
            new_dict[key] = eval(old_dict[key][0])
        else:
            new_dict[key] = eval(old_dict[key])


    return new_dict