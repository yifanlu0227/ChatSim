# different modules to predict the latent vector.

from mc_to_sky.model.sub_module.residual import build_layer
from torch.nn import MultiheadAttention
import torch
import torch.nn as nn
from mc_to_sky.model.sub_module import build_module
import importlib
from envmap import EnvironmentMap, rotation_matrix
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

def build_latent_predictor(args):
    model_name = args['name']
    model_lib = importlib.import_module("mc_to_sky.model.sub_module.skypred_modules")
    model_cls = None
    target_model_name = model_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model_cls = cls
    
    return model_cls(args)

class NaiveSingleView(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)

        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])
        self.latent_mlp = build_module(args['latent_mlp'])
        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])


    def forward(self, x):
        """
        x: B, N_view, C, H, W
        """
        x = x[:, self.center_view, ...]
        
        x = self.img_encoder(x).permute(0,2,3,1) # [N, H, W, C]
        x_flatten = x.flatten(1)
        
        deep_vector = self.shared_mlp(x_flatten)

        latent_vector = self.latent_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(deep_vector)
        peak_int_vector = self.peak_int_mlp(deep_vector)

        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True) # normalize
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)

        return peak_vector, latent_vector

class CatMultiView(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.sort_index = np.argsort(self.view_dis_deg)
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)

        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])
        self.latent_mlp = build_module(args['latent_mlp'])
        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])


    def forward(self, x):
        """
        x: B, N_view, C, H, W, suppose center view is the first 
        """
        B, N_view, C, H, W = x.shape
        x = x[:, self.sort_index, ...]
        x = x.permute(0, 2, 3, 1, 4).flatten(3) # B, C, H, N_view * W

        x = self.img_encoder(x).permute(0,2,3,1) # [N, H, W, C]
        x_flatten = x.flatten(1)
        
        deep_vector = self.shared_mlp(x_flatten)

        latent_vector = self.latent_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(deep_vector)
        peak_int_vector = self.peak_int_mlp(deep_vector)

        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True) # normalize
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)

        return peak_vector, latent_vector

    
class AvgMultiView(nn.Module):
    """
    Avg is not suitable.

    use self.attention to fuse latent vector
    use max to fuse peak intensity
    use avg to fuse peak direction
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)

        self.crop_H = view_args['camera_H'] // view_args['downsample_for_crop']
        self.crop_W = view_args['camera_W'] // view_args['downsample_for_crop']
        self.camera_vfov = np.degrees(np.arctan2(view_args['camera_H']/2, view_args['focal'])) * 2
        self.aspect_ratio = view_args['camera_W'] / view_args['camera_H']

        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])

        self.latent_mlp = build_module(args['latent_mlp'])
        int_channel = args['latent_mlp']['args']['layer_channels'][-1]
        self.att = ScaledDotProductAttention(int_channel)


        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])


    def forward(self, x):
        """
        x: B, N_view, C, H, W
        """
        B, N_view, _, _, _ = x.shape

        x = x.flatten(0,1) # B * N_view, C, H, W
        x = self.img_encoder(x).permute(0,2,3,1) # [B * N_view, H, W, C]

        x_flatten = x.flatten(1) # [B * N_view, H * W * C]
        
        deep_vector = self.shared_mlp(x_flatten) # [B * N_view, 256]
        deep_vector = deep_vector.view(B, N_view, -1) # [B, N_view, 256]

        # latent vector
        latent_vector = self.latent_mlp(deep_vector) # [B, N_view, 256]
        latent_vector = self.att(latent_vector[:, self.center_view: self.center_view+1], 
                                 latent_vector,
                                 latent_vector)

        # peak dir
        peak_dir_vector = self.peak_dir_mlp(deep_vector) # [B, N_view, 3]
        for i in range(self.view_num):  # rotate peak dir to center
            azimuth_rad_i = self.view_dis_rad[i]
            rotation_mat_i = rotation_matrix(azimuth = azimuth_rad_i, elevation = 0) # in data loader, pos azimuth rotation applied to peak dir 
            inv_rotation_mat_i = rotation_matrix(azimuth = - azimuth_rad_i, elevation = 0) # so we should use neg azimuth rotation to rotate it back to ego view
            inv_rotation_mat_i = torch.from_numpy(inv_rotation_mat_i).to(x.device).float()
            
            peak_dir_vector[:, i] = (inv_rotation_mat_i @ peak_dir_vector[:, i].T).T 

            # peak_dir_vector, [B, N_view, 3]

        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=-1, keepdim=True) # normalize, [B, N_view, 3]
        peak_dir_vector_sum = peak_dir_vector.mean(dim=1)  # [B, 3]
        peak_dir_vector_avg = peak_dir_vector_sum / peak_dir_vector_sum.norm(dim=-1, keepdim=True)

        # peak int
        peak_int_vector = self.peak_int_mlp(deep_vector).mean(dim=1)

        peak_vector = torch.cat([peak_dir_vector_avg, peak_int_vector], dim=-1)

        return peak_vector, latent_vector
