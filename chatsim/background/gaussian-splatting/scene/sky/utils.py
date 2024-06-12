import torch
import cv2
import math
import numpy as np

def get_ray_directions(H, W, FoVx, FoVy, c2w):
    """
    Get ray directions for all pixels in the camera coordinate system. Suppose opencv convention

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        FoVx (float): FoV in the x direction. radians
        FoVy (float): FoV in the y direction. radians
        c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).

    Returns:
        ray_directions (torch.Tensor): Ray directions in the world coordinate system of shape (H, W, 3).
    """
    c2w = c2w.cuda()

    # Create a grid of pixel coordinates
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                          torch.arange(H, dtype=torch.float32),
                          indexing='xy')
    
    directions = torch.stack([(i - W * 0.5) / (W * 0.5) * math.tan(FoVx * 0.5),
                              (j - H * 0.5) / (H * 0.5) * math.tan(FoVy * 0.5),
                              torch.ones_like(i)], dim=-1).cuda()  # (H, W, 3)
    directions = directions.unsqueeze(-1) # (H, W, 3, 1)

    # Convert ray directions from camera to world coordinate system
    ray_directions = torch.einsum('ij,hwjk->hwik', c2w[:3, :3], directions).squeeze(-1)  # (H, W, 3)

    # normalize the ray_directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    return ray_directions

def convert_cubemap(img, view_name):
    """
    Args:
        img : np.ndarray
            [res, res, 3]
        view_name : str
            one of ['front', 'back', 'left', 'right', 'top', 'bottom']

    if view name is in ['front', 'back', 'left', 'right'], horizontal flip the image!
    if view name is in ['top', 'bottom'], vertical flip the image!
    """
    if view_name in ['front', 'back', 'left', 'right']:
        img = np.flip(img, axis=1)
    elif view_name in ['top', 'bottom']:
        img = np.flip(img, axis=0)
    return img

def convert_cubemap_torch(img, view_name):
    """
    Args:
        img : torch.Tensor
            [res, res, 3]
        view_name : str
            one of ['front', 'back', 'left', 'right', 'top', 'bottom']

    if view name is in ['front', 'back', 'left', 'right'], horizontal flip the image!
    if view name is in ['top', 'bottom'], vertical flip the image!
    """
    if view_name in ['front', 'back', 'left', 'right']:
        img = torch.flip(img, [1])
    elif view_name in ['top', 'bottom']:
        img = torch.flip(img, [0])
    return img


def flatten(cubemap):
    """
    flatten the cube map for visualization
    Args:
        cubemap : torch.tensor
            shape [6, N, N, C], C usually 3
    Returns:
        cubemap_flatten : torch.tensor
            shape [3*N, 4*N, C]
    """
    view_names = ['right', 'left', 'top', 'bottom', 'back', 'front']
    # flip cubemap for visualization
    cubemap_vis = torch.clone(cubemap).detach()
    for i, view_name in enumerate(view_names):
        cubemap_vis[i] = convert_cubemap_torch(cubemap_vis[i], view_name)
    _, N, N, C = cubemap_vis.shape

    # order ['right', 'left', 'top', 'bottom', 'back', 'front']
    cubemap_flatten = torch.zeros(N*3, N*4, C).to(cubemap)
    cubemap_flatten[:N, N:2*N] = cubemap_vis[2] # top
    cubemap_flatten[N:2*N, :N] = cubemap_vis[1] # left
    cubemap_flatten[N:2*N, N:2*N] = cubemap_vis[5] # front
    cubemap_flatten[N:2*N, 2*N:3*N] = cubemap_vis[0] # right
    cubemap_flatten[N:2*N, 3*N:4*N] = cubemap_vis[4] # back
    cubemap_flatten[2*N:3*N, N:2*N] = cubemap_vis[3] # bottom

    return cubemap_flatten