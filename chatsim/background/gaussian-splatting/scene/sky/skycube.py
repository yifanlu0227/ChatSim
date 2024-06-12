# https://github.com/fudan-zvg/PVG/blob/main/scene/envlight.py

import torch
from scene.sky.utils import get_ray_directions

class SkyCube(torch.nn.Module):
    def __init__(self, sky_model_args):
        super().__init__()
        resolution = sky_model_args.resolution

        # transform the world coordinate!
        #       z                   y
        #       |  x                |
        #       |/                  |
        # y ----o   (waymo)   ==>   o----- x  (opengl)
        #                          /
        #                       z /  
        self.waymo_to_opengl = torch.tensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=torch.float32, device="cuda")
        
        # 'front', 'back', 'left', 'right' are stored horizontally flipped!
        # 'top', 'bottom' are stored vertically flipped!
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def capture(self):
        return self.base
    
    def train_params(self):
        return [self.base]
        
    def restore(self, model_args):
        self.base = model_args      
        
    def _forward(self, l):
        import nvdiffrast.torch as dr

        l = (l.reshape(-1, 3) @ self.waymo_to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light
    
    def forward(self, viewpoint_camera):
        c2w = torch.linalg.inv(viewpoint_camera.world_view_transform.transpose(0, 1))
        ray_d_world = get_ray_directions(viewpoint_camera.image_height, 
                                         viewpoint_camera.image_width, 
                                         viewpoint_camera.FoVx, 
                                         viewpoint_camera.FoVy, 
                                         c2w).cuda()  # [H, W, 3]
        
        skymap = self._forward(ray_d_world) # [H, W, 3]
        skymap = skymap.permute(2, 0, 1)  # [3, H, W]

        return skymap