"""
Peak Direction and Intensity sky model (without peak residual connection.)
"""


import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from envmap import EnvironmentMap
from torch import nn
from torch.optim.lr_scheduler import StepLR
from functools import reduce

from mc_to_sky.loss import build_loss
from mc_to_sky.model.sub_module import build_module
from mc_to_sky.utils.hdr_utils import (srgb_gamma_correction_torch,
                                        srgb_inv_gamma_correction_torch)


class SkyModel(pl.LightningModule):
    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        downsample = hypes['dataset']['downsample']
        self.sky_H = hypes['dataset']['image_H'] // downsample // 2 
        self.sky_W = hypes['dataset']['image_W'] // downsample

        self.teacher_prob = hypes['model']['teacher_prob']
        self.env_template = EnvironmentMap(self.sky_H, 'skylatlong')
        world_coord = self.env_template.worldCoordinates() # tuple, (x, y, z, valid)
        self.pos_encoding = torch.from_numpy(np.stack([world_coord[0], world_coord[1], world_coord[2]], axis=-1)) # [H, W, 3]
        self.pos_encoding = self.pos_encoding.to('cuda')
        self.input_inv_gamma = hypes['model']['input_inv_gamma']
        self.input_add_pe = hypes['model']['input_add_pe']
        self.encoder_outdim = hypes['model']['ldr_encoder']['args']['layer_channels'][-1]
        self.feat_down = reduce(lambda x, y: x*y, hypes['model']['ldr_encoder']['args']['strides'])
        self.save_hyperparameters()

        # module related
        self.ldr_encoder = build_module(hypes['model']['ldr_encoder'])
        self.shared_mlp = build_module(hypes['model']['shared_mlp'])

        self.latent_mlp = build_module(hypes['model']['latent_mlp'])
        self.latent_mlp_recon = build_module(hypes['model']['latent_mlp_recon'])

        self.peak_dir_mlp = build_module(hypes['model']['peak_dir_mlp'])
        self.peak_int_mlp = build_module(hypes['model']['peak_int_mlp'])

        self.ldr_decoder = nn.Sequential(
                                build_module(hypes['model']['ldr_decoder']),
                                nn.Sigmoid()
                            )   
        self.hdr_decoder = build_module(hypes['model']['hdr_decoder'])

        # loss related
        self.ldr_recon_loss = build_loss(hypes['loss']['ldr_recon_loss'])
        self.hdr_recon_loss = build_loss(hypes['loss']['hdr_recon_loss'])
        self.peak_int_loss = build_loss(hypes['loss']['peak_int_loss'])
        self.peak_dir_loss = build_loss(hypes['loss']['peak_dir_loss'])

        self.fix_modules = hypes['model'].get('fix_modules', [])
        self.on_train_epoch_start()


    def encode_forward(self, x):
        """
        Encode LDR panorama to sky vector: 
            1) peak dir 
            2) peak int 
            3) latent vector
            where 1) and 2) can cat together
        
        deep vector -> shared vector -->    latent vector      --> recon deep vector 
                                     |
                                     .->  peak int/dir vector

        Should we add explicit inv gamma to the input?
        """
        if self.input_inv_gamma:
            x = srgb_inv_gamma_correction_torch(x)
        if self.input_add_pe:
            x = x + self.pos_encoding.permute(2, 0, 1)

        deep_feature = self.ldr_encoder(x)
        deep_vector = deep_feature.permute(0,2,3,1).flatten(1) # N, 4096

        shared_vector = self.shared_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(shared_vector)
        peak_int_vector = self.peak_int_mlp(shared_vector)
        latent_vector = self.latent_mlp(shared_vector)

        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True) # normalize

        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)

        return peak_vector, latent_vector


    def decode_forward(self, latent_vector, peak_vector, peak_vector_gt):
        use_gt_peak = False
        if self.training and np.random.rand() < self.teacher_prob:
            use_gt_peak = True
            peak_vector = peak_vector_gt

        B = peak_vector.shape[0]
        peak_dir_encoding, peak_int_encoding = self.build_peak_map(peak_vector)
        decoder_input = torch.cat([peak_dir_encoding, peak_int_encoding, self.pos_encoding.expand(B,-1,-1,-1)], dim=-1)
        decoder_input = decoder_input.permute(0,3,1,2) # [B, 7, H, W]

        recon_deep_vector = self.latent_mlp_recon(latent_vector) # [N, 64] -> [N, 4096]
        recon_deep_feature = recon_deep_vector.view(B, 
                                                    self.sky_H//self.feat_down, 
                                                    self.sky_W//self.feat_down, 
                                                    self.encoder_outdim
                             ).permute(0,3,1,2)
        
        # ldr_recon
        ldr_skypano_recon = self.ldr_decoder(recon_deep_feature)

        # hdr_recon
        hdr_skypano_recon = self.hdr_decoder(decoder_input, recon_deep_feature)

        return hdr_skypano_recon, ldr_skypano_recon, use_gt_peak


    def build_peak_map(self, peak_vector):
        """
        Args:
            peak_vector: [B, 6]
                3 for peak dir, 3 for peak intensity

        Returns:
            peak encoding map: [B, 4, H, W]
                1 for peak dir using spherical gaussian lobe, 3 for peak intensity
        """
        dir_vector = peak_vector[...,:3] # should be normalized
        int_vector = peak_vector[...,3:] # B, 3
        
        dir_vector_expand = dir_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1) # B, H, W, C
        peak_dir_encoding = torch.exp(100*(
                torch.einsum('nhwc,nhwc->nhw', dir_vector_expand, self.pos_encoding.expand(dir_vector_expand.shape)) - 1
            )).unsqueeze(-1) # B, H, W, 1

        sun_mask = torch.gt(peak_dir_encoding, 0.9).expand(-1,-1,-1,3) # [B, H, W, 3]
        int_vector_expand = int_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1) # [B, H, W, 3]
        peak_int_encoding = torch.where(sun_mask, int_vector_expand, 0)

        return peak_dir_encoding, peak_int_encoding

    def on_train_epoch_start(self):
        print(f"Module fixed in training: {self.fix_modules}.")
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").eval()

    def training_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch

        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_gt)

        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[...,:3], peak_vector_gt[...,:3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[...,3:], peak_vector_gt[...,3:])

        loss = hdr_recon_loss + peak_dir_loss + peak_int_loss + ldr_recon_loss

        self.log('train_loss', loss)
        self.log('hdr_recon_loss', hdr_recon_loss)
        self.log('ldr_recon_loss', ldr_recon_loss)
        self.log('peak_dir_loss', peak_dir_loss)
        self.log('peak_int_loss', peak_int_loss)

        log_info = f"|| loss: {loss:.3f} || hdr_recon_loss: {hdr_recon_loss:.3f}  || ldr_recon_loss: {ldr_recon_loss:.3f} || peak_dir_loss: {peak_dir_loss:.3f} " + \
                   f"|| peak_int_loss: {peak_int_loss:.3f}"

        print(log_info)

        return loss

    
    def validation_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch

        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred)

        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        # peak_dir_loss = self.peak_dir_loss(peak_vector_pred[...,:3], peak_vector_gt[...,:3])
        # peak_int_loss = self.peak_int_loss(peak_vector_pred[...,3:], peak_vector_gt[...,3:])

        loss = hdr_recon_loss # + peak_dir_loss + peak_int_loss + ldr_recon_loss

        self.log('val_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    
    def predict_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch

        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred) # both peak_vector_pred
        
        ### test rotating the peak dir vector
        # hdr_skypano_pred_list = []
        # for deg in range(0, 360, 45):
        #     azimuth_rad = np.radians(deg)
        #     rotation_mat = torch.from_numpy(rotation_matrix(azimuth = azimuth_rad, elevation = 0)).cuda().float()
        #     peak_vector_pred_rot = peak_vector_pred.clone()[0] # [1, 6]
        #     peak_vector_pred_rot[:3] = (rotation_mat @ peak_vector_pred_rot[:3].reshape(3,1)).flatten()
        #     peak_vector_pred_rot = peak_vector_pred_rot.view(1, -1)
        #     hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred_rot, peak_vector_pred_rot) # both peak_vector_pred
        #     hdr_skypano_pred_list.append(hdr_skypano_pred)
        # hdr_skypano_pred = torch.cat(hdr_skypano_pred_list) # [N, C, H, W]
        # hdr_skypano_pred = hdr_skypano_pred.permute(0,2,3,1).flatten(0,1).permute(2,0,1).unsqueeze(0)

        print(f'{batch_idx:0>3} \n \
                 HDRI Peak Intensity:\t\t {hdr_skypano_pred[0].flatten(1,2).max(dim=-1)[0]} \n \
                 Peak Intensity Vector:\t {peak_vector_pred[0][3:]} \n \
                 Ground Truth Peak Intensity:\t {peak_vector_gt[0][3:]}')

        return_dict = {
            'ldr_skypano_input': ldr_skypano.permute(0,2,3,1),
            'ldr_skypano_pred': ldr_skypano_recon.permute(0,2,3,1),
            'hdr_skypano_gt': hdr_skypano_gt.permute(0,2,3,1),
            'hdr_skypano_pred': hdr_skypano_pred.permute(0,2,3,1),
            'batch_idx': batch_idx
        }

        return return_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypes['lr_schedule']['init_lr'])
        lr_scheduler = StepLR(optimizer=optimizer, step_size=self.hypes['lr_schedule']['decay_per_epoch'], gamma=self.hypes['lr_schedule']['decay_rate'])
        return [optimizer], [lr_scheduler]

