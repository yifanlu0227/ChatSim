import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import omegaconf
import math
from utils.graphics_utils import OETF


def gsplat_render(viewpoint_camera, pc : GaussianModel, args: omegaconf.dictconfig.DictConfig, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, exposure_scale = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    if viewpoint_camera.K is not None:
        focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K
        K = torch.tensor([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1.0]
        ]).to(pc.get_xyz)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ]
        ).to(pc.get_xyz)

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    render_colors, render_alphas, info = rasterization(
        means=means3D,    # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,    # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=None,
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+ED',
    )
    # [1, H, W, 4] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    # [1, H, W, 4] -> [1, H, W]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    # [1, H, W, 1] -> [1, H, W]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)

    if exposure_scale is not None:
        rendered_image *= exposure_scale
        rendered_image = OETF(rendered_image)

    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_pkg =  {"render_3dgs": rendered_image,
                    "viewspace_points": info["means2d"],
                    "visibility_filter" : radii > 0,
                    "radii": radii}

    if args.render_depth:
        return_pkg["depth"] = rendered_depth

    if args.render_opacity:
        return_pkg["opacity"] = rendered_alphas  # [1, H, W]

    if args.render_sky:
        # can be implemented by sky box / sky HDRI / sky MLP
        sky_bg = pc.get_sky_bg(viewpoint_camera)
        return_pkg["sky_bg"] = sky_bg # expect [3, H, W]

        # blend sky with rendered image
        if args.blend_sky:
            assert args.render_opacity
            rendered_image = rendered_image + (1 - rendered_alphas) * sky_bg

    return_pkg['render'] = rendered_image

    return return_pkg