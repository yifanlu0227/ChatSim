# https://github.com/kwea123/nerf_pl/blob/master/datasets/ray_utils.py
import torch
import numpy as np
from kornia import create_meshgrid
import torch.nn.functional as F

def random_sample_on_hemisphere(normal):
    random_direction = torch.rand(3).to('cuda')
    random_direction = random_direction / random_direction.norm()

    # Calculate the dot product between random direction and the normal vector
    dot_product = torch.dot(random_direction, normal)
    if dot_product < 0:
        random_direction = - random_direction
        
    return random_direction

def random_samples_on_hemisphere(normal, num_samples):
    random_direction = torch.rand(num_samples, 3).to('cuda')
    random_direction = random_direction / random_direction.norm(p=2, dim=1, keepdim=True) # [n_samples, 3]

    # Calculate the dot product between random direction and the normal vector
    dot_product = torch.einsum("ij,ij->i", random_direction, normal).unsqueeze(-1)
    random_direction = torch.where(dot_product < 0, - random_direction, random_direction)

    return random_direction

def get_ray_directions(H, W, focal, use_np=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1).to(torch.float32) # (H, W, 3)
    
    if use_np:
        directions = directions.numpy()

    return directions


def get_rays(directions, c2w, use_np=True):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    if use_np:
        directions = torch.from_numpy(directions)
        c2w = torch.from_numpy(c2w)

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)

    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    if use_np:
        rays_d = rays_d.numpy()
        rays_o = rays_o.numpy()

    return rays_o, rays_d
