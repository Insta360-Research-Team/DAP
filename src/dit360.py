import copy
import math
import torch
import random
import numpy as np
from torch import nn
import lightning as L
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, cast, Tuple, Any
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

from src.pipeline import DiT360Pipeline
from src.yaw_rotate import equirectangular_rotate_yaw
from src.cube_map import cube_map_from_equirectangular


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * \
        vae.config.scaling_factor
    return pixel_latents


def get_sigmas(noise_scheduler_copy, timesteps, device, dtype, n_dim=4):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device=device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class DiT360(L.LightningModule):

    def __init__(self, args, lora_config, ckpt=None):
        super().__init__()

        self.save_hyperparameters()

        args = self.hparams.args

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae")
        self.flux_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.vae.requires_grad_(False)
        self.flux_transformer.requires_grad_(False)

        self.flux_transformer.add_adapter(lora_config)

        ##############################################
        # only load weights for flux_transformer
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.flux_transformer.load_state_dict(state_dict, strict=True)
        ##############################################

        for param in self.flux_transformer.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=torch.float32)

        self.flux_transformer.train()

        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.weighting_scheme = args.weighting_scheme
        self.logit_mean = args.logit_mean
        self.logit_std = args.logit_std
        self.mode_scale = args.mode_scale


    def on_fit_start(self):
        self.vae = self.vae.to(dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        pixel_latents = encode_images(
            batch["pixel_values"], self.vae).to(self.dtype)
        bsz = pixel_latents.shape[0]
        noise = torch.randn_like(
            pixel_latents, device=self.device, dtype=self.dtype)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )

        indices = (
            u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(
            device=self.device)

        # Add noise according to flow matching.
        sigmas = get_sigmas(
            self.noise_scheduler_copy,
            timesteps,
            device=self.device,
            dtype=self.dtype,
            n_dim=pixel_latents.ndim
        )
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

        # pack the latents.
        packed_noisy_model_input = DiT360Pipeline._pack_latents(
            noisy_model_input,
            batch_size=bsz,
            num_channels_latents=noisy_model_input.shape[1],
            height=noisy_model_input.shape[2],
            width=noisy_model_input.shape[3],
        )

        ##############################################################
        padding_n = self.hparams.args.padding_n

        packed_height = noisy_model_input.shape[2] // 2
        packed_width = noisy_model_input.shape[3] // 2
        dim = packed_noisy_model_input.shape[-1]

        if padding_n > 0:
            packed_noisy_model_input = packed_noisy_model_input.reshape(bsz, packed_height, packed_width, dim)
            first_col = packed_noisy_model_input[:, :, 0:padding_n, :]
            last_col = packed_noisy_model_input[:, :, -padding_n:, :]

            packed_noisy_model_input = torch.cat([last_col, packed_noisy_model_input, first_col], dim=2)
            packed_noisy_model_input = packed_noisy_model_input.reshape(bsz, -1, dim)

        latent_image_ids = DiT360Pipeline._prepare_latent_image_ids(
            bsz,
            packed_height,
            packed_width,
            self.device,
            self.dtype,
        )
        
        if padding_n > 0:
            latent_image_ids = latent_image_ids.reshape(packed_height, packed_width, 3)
            first_col_image_ids = latent_image_ids[:, 0:padding_n, :]
            last_col_image_ids = latent_image_ids[:, -padding_n:, :]

            latent_image_ids = torch.cat([last_col_image_ids, latent_image_ids, first_col_image_ids], dim=1)
            latent_image_ids = latent_image_ids.reshape(-1, 3)
        ##############################################################

        if self.flux_transformer.config.guidance_embeds:
            guidance_vec = torch.full(
                (bsz,),
                self.hparams.args.guidance_scale,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            guidance_vec = None

        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        text_ids = batch["text_ids"]

        model_pred = self.flux_transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        ######################################################
        if padding_n > 0:
            model_pred = model_pred.reshape(bsz, packed_height, packed_width + 2 * padding_n, -1)
            model_pred = model_pred[:, :, padding_n:-padding_n, :]
            model_pred = model_pred.reshape(bsz, packed_height * packed_width, -1)
        ######################################################

        model_pred = DiT360Pipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[2] * self.vae_scale_factor,
            width=noisy_model_input.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas)

        target = noise - pixel_latents 

        ############################################################################################
        # main loss
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )

        loss_v = loss.mean()

        # yaw loss
        angles = [60, 180, 300]
        a = random.choice(angles)

        target_yaw = equirectangular_rotate_yaw(
            noise.clone(), a) - equirectangular_rotate_yaw(pixel_latents.clone(), a)
        model_pred_yaw = equirectangular_rotate_yaw(model_pred.clone(), a)

        loss_yaw = torch.mean(
            (weighting.float() * (model_pred_yaw.float() - target_yaw.float())
             ** 2).reshape(target_yaw.shape[0], -1), 1,
        )
        loss_yaw = loss_yaw.mean()

        # cube loss
        target_cube = cube_map_from_equirectangular(
            noise.clone()) - cube_map_from_equirectangular(pixel_latents.clone())
        model_pred_cube = cube_map_from_equirectangular(model_pred.clone())

        loss_cube = torch.mean(
            (weighting.unsqueeze(1).float() * (model_pred_cube.float() -
             target_cube.float()) ** 2).reshape(target_cube.shape[0], -1), 1,
        )
        loss_cube = loss_cube.mean()
        ######################################################

        if self.current_epoch < 2:
            loss = loss_v
        else:
            loss = loss_v + self.hparams.args.lambda_yaw * loss_yaw + self.hparams.args.lambda_cube * loss_cube

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_v", loss_v, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_yaw", loss_yaw, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_cube", loss_cube, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        args = self.hparams.args
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.flux_transformer.parameters()))
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay
        )

        num_training_steps = args.max_steps
        num_warmup_steps = int(0.05 * num_training_steps)
        
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
