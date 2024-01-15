# Adpated from https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

from tqdm import tqdm
from typing import Union
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from diffusers import DDIMScheduler


class NullInversion:
    
    def prev_step(self, 
                model_output: Union[torch.FloatTensor, np.ndarray], 
                timestep: int, 
                sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.ddim_scheduler.config.num_train_timesteps // self.ddim_scheduler.num_inference_steps
        alpha_prod_t = self.ddim_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.ddim_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, 
                model_output: Union[torch.FloatTensor, np.ndarray], 
                timestep: int, 
                sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.ddim_scheduler.config.num_train_timesteps // self.ddim_scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.ddim_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.ddim_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, 
                            latents, 
                            t,
                            context,
                            grounding_input,
                            inpainting_extra_input):
        bsz = latents.shape[0]
        input = dict(
            x = latents.type(torch.float16),
            timesteps = torch.full((bsz,), t, device=latents.device, dtype=latents.dtype),
            context = context.type(torch.float16),
            grounding_input = grounding_input,
            inpainting_extra_input = inpainting_extra_input,
        )

        noise_pred = self.unet(
                input,
        )

        return noise_pred

    def get_noise_pred(
        self, 
        latents, 
        t, 
        is_forward=True, 
        context=None,
        grounding_input=None,
        inpainting_extra_input=None):
        latents_input_cond = latents.clone()
        latents_input_uncond = latents.clone()
        if context is None:
            context = self.context
        context_uncond, context_cond = context.chunk(2)    
        guidance_scale = 1 if is_forward else self.guidance_scale
        
        bsz = latents_input_cond.shape[0]
        input_cond = dict(
            x = latents_input_cond.type(torch.float16),
            timesteps = torch.full((bsz,), t, device=latents.device, dtype=latents.dtype),
            context = context_cond.type(torch.float16),
            grounding_input = grounding_input,         
            inpainting_extra_input = inpainting_extra_input,   
        )
        noise_prediction_text = self.unet(
            input_cond,
        )

        input_uncond = dict(
            x = latents_input_uncond.type(torch.float16),
            timesteps = torch.full((bsz,), t, device=latents.device, dtype=latents.dtype),
            context = context_uncond.type(torch.float16),
            grounding_input = grounding_input,         
            inpainting_extra_input = inpainting_extra_input,   
        )
        noise_pred_uncond = self.unet(
            input_uncond,
        )

        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(
        self, 
        prompt: str, 
        batch_size,
        text_encoder):
        uncond_embeddings = text_encoder.encode(batch_size*[""])
        text_embeddings = text_encoder.encode(batch_size*[prompt])
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(
        self, 
        latent, 
        grounding_input, 
        inpainting_extra_input):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        cond = cond_embeddings if self.null_inv_with_prompt else uncond_embeddings
        all_latent = [latent.clone()]
        latent = latent.clone().detach()
        for i in range(self.num_inv_steps):
            t = self.ddim_scheduler.timesteps[ len( self.ddim_scheduler.timesteps ) - i - 1 ]
            noise_pred = self.get_noise_pred_single(latent, t, cond, grounding_input, inpainting_extra_input)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(
        self, 
        latent, 
        grounding_input, 
        inpainting_extra_input):
        ddim_latents = self.ddim_loop(latent, grounding_input, inpainting_extra_input)
        return ddim_latents

    def null_optimization(
        self, 
        latents, 
        null_inner_steps, 
        epsilon, 
        null_base_lr=1e-2,
        grounding_input=None, 
        inpainting_extra_input=None):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=null_inner_steps * self.num_inv_steps)
        for i in range(self.num_inv_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=null_base_lr * (1. - i / 100.), eps=1e-4)
            latent_prev = latents[len(latents) - i - 2]
            t = self.ddim_scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, grounding_input, inpainting_extra_input)
            for j in range(null_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, grounding_input, inpainting_extra_input)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(uncond_embeddings, 1.0)
                optimizer.step()
                assert not torch.isnan(uncond_embeddings.abs().mean())
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, null_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, grounding_input, inpainting_extra_input)
        bar.close()
        return uncond_embeddings_list
    
    def invert(
        self, 
        latents: torch.Tensor, 
        prompt: str,
        grounding_input=None,
        inpainting_extra_input=None,
        text_encoder=None, 
        null_inner_steps=1, 
        early_stop_epsilon=1e-5, 
        verbose=True, 
        null_base_lr=1e-2,
        batch_size=1,
        nti=True,
        ):
        self.init_prompt(prompt, batch_size, text_encoder)
        if verbose:
            print("DDIM inversion...")
        ddim_latents = self.ddim_inversion(
            latents, grounding_input, inpainting_extra_input)

        if not nti:
            return ddim_latents[-1], None
            
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            ddim_latents, null_inner_steps, early_stop_epsilon,
            null_base_lr=null_base_lr, grounding_input=grounding_input, inpainting_extra_input=inpainting_extra_input)

        return ddim_latents[-1], uncond_embeddings
        
    
    def __init__(
        self, 
        unet, 
        num_inv_steps, 
        guidance_scale, 
        null_inv_with_prompt,
        ):
        self.null_inv_with_prompt = null_inv_with_prompt
        self.guidance_scale = guidance_scale
        self.unet = unet
        self.ddim_scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
        )
        self.ddim_scheduler.set_timesteps(num_inv_steps)
        self.num_inv_steps = num_inv_steps
        self.prompt = None
        self.context = None
