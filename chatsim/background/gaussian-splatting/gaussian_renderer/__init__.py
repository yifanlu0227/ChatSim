from gaussian_renderer.gsplat_renderer import gsplat_render
from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc, args, bg_color, scaling_modifier=1.0, override_color=None, exposure_scale=None):
    return gsplat_render(viewpoint_camera, pc, args, bg_color, scaling_modifier, override_color, exposure_scale)

__all__ = ["render"]