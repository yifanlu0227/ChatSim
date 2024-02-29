"""
Wang 22' ECCV

Single image to HDR panorama estimation
"""
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from envmap import EnvironmentMap
from icecream import ic
from torch import nn
from torch.optim.lr_scheduler import StepLR

from mc_to_sky.loss.loss import L1AngularLoss, L1Loss, LogEncodedL2Loss
from mc_to_sky.model.skymodel import SkyModel
from mc_to_sky.model.skymodel_enhanced import SkyModelEnhanced
from mc_to_sky.model.sub_module.residual import build_layer
from mc_to_sky.model.sub_module.skypred_modules import build_latent_predictor
from mc_to_sky.model.sub_module.unet import UNet
from mc_to_sky.utils.hdr_utils import srgb_gamma_correction_torch
from mc_to_sky.model.sub_module import build_module
from mc_to_sky.loss import build_loss


class SkyPred(pl.LightningModule):
    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        self.save_hyperparameters()

        self.latent_predictor = build_latent_predictor(hypes['model']['latent_predictor'])

        sky_model_core_method = hypes['model']['sky_model']['core_method']
        sky_model_core_method_ckpt_path = hypes['model']['sky_model']['ckpt_path']
        
        if sky_model_core_method == "sky_model_enhanced":
            self.sky_model = SkyModelEnhanced.load_from_checkpoint(sky_model_core_method_ckpt_path)
        elif sky_model_core_method == "sky_model":
            self.sky_model = SkyModel.load_from_checkpoint(sky_model_core_method_ckpt_path)

        self.ldr_recon_loss = build_loss(hypes['loss']['ldr_recon_loss'])
        self.hdr_recon_loss = build_loss(hypes['loss']['hdr_recon_loss'])
        self.peak_int_loss = build_loss(hypes['loss']['peak_int_loss'])
        self.peak_dir_loss = build_loss(hypes['loss']['peak_dir_loss'])
        self.latent_loss = build_loss(hypes['loss']['latent_loss'])

        self.fix_modules = hypes['model'].get('fix_modules', [])
        self.on_train_epoch_start()

        
    def decode_forward(self, latent_vector, peak_vector):
        return self.sky_model.decode_forward(latent_vector, peak_vector, peak_vector)

    def on_train_epoch_start(self):
        print(f"Module fixed in training: {self.fix_modules}.")
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").eval()

    def training_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        
        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)

        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_envmap_tensor, mask_envmap_tensor) 
        ldr_recon_loss = self.ldr_recon_loss(srgb_gamma_correction_torch(hdr_skypano_pred), ldr_envmap_tensor, mask_envmap_tensor) 
        latent_loss = self.latent_loss(latent_vector_pred, latent_vector_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[...,:3], peak_vector_gt[...,:3]) 
        peak_int_loss = self.peak_int_loss(peak_vector_pred[...,3:], peak_vector_gt[...,3:]) 

        loss = hdr_recon_loss + ldr_recon_loss + latent_loss + peak_dir_loss + peak_int_loss

        self.log('train_loss', loss)
        self.log('hdr_recon_loss', hdr_recon_loss)
        self.log('ldr_recon_loss', ldr_recon_loss)
        self.log('latent_loss', latent_loss)
        self.log('peak_dir_loss', peak_dir_loss)
        self.log('peak_int_loss', peak_int_loss)

        print(f"|| loss: {loss:.3f} || hdr_recon_loss: {hdr_recon_loss:.3f} || ldr_recon_loss: {ldr_recon_loss:.3f} " + \
              f"|| latent_loss: {latent_loss:.3f} || peak_dir_loss: {peak_dir_loss:.3f} || peak_int_loss: {peak_int_loss:.3f}")

        return loss
    
    def validation_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        mask_envmap_tensor = torch.gt(mask_envmap_tensor, 0.8) # trans to bool

        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)

        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_envmap_tensor, mask_envmap_tensor) 
        ldr_recon_loss = self.ldr_recon_loss(srgb_gamma_correction_torch(hdr_skypano_pred), ldr_envmap_tensor, mask_envmap_tensor)
        latent_loss = self.latent_loss(latent_vector_pred, latent_vector_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[...,:3], peak_vector_gt[...,:3]) 
        peak_int_loss = self.peak_int_loss(peak_vector_pred[...,3:], peak_vector_gt[...,3:]) 

        loss = hdr_recon_loss + ldr_recon_loss

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        mask_envmap_tensor = torch.gt(mask_envmap_tensor, 0.8) # trans to bool

        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)


        return_dict = {
            "hdr_skypano_pred": hdr_skypano_pred.permute(0,2,3,1),
            'ldr_skypano_pred': srgb_gamma_correction_torch(hdr_skypano_pred).permute(0,2,3,1),
            "hdr_skypano_gt": hdr_envmap_tensor.permute(0,2,3,1),
            "ldr_skypano_input": ldr_envmap_tensor.permute(0,2,3,1),
            "mask_env": mask_envmap_tensor.permute(0,2,3,1),
            "image_crops": img_crops_tensor.permute(0,1,3,4,2), # N, N_view, C, H, W -> N, N_view, H, W, C
            'batch_idx': batch_idx
        }

        return return_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypes['lr_schedule']['init_lr'])
        lr_scheduler = StepLR(optimizer=optimizer, step_size=self.hypes['lr_schedule']['decay_per_epoch'], gamma=self.hypes['lr_schedule']['decay_rate'])
        return [optimizer], [lr_scheduler]
