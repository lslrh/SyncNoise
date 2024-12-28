# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
InstructPix2Pix module

File copied from https://github.com/ayaanzhaque/instruct-nerf2nerf/tree/main/in2n
"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import os
import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float

import torch.nn.functional as F

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/pips')
from syncnoise.pips.nets.pips import Pips
from syncnoise.match_utils import gather_and_avg_by_traj, pixel_replace_by_traj, match_by_pips, init_latents, visualize_match
import random 
from PIL import Image

CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        # EulerAncestralDiscreteScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "/home/notebook/code/personal/S9049601/threestudio/pretrained_models/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "/home/notebook/data/group/liruihuang/in2n/model/instruct-pix2pix"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(
        self, 
        device: Union[torch.device, str], 
        num_train_timesteps: int = 1000, 
        ip2p_use_full_precision = False,
        cluster_thresh = [0., 0.3],
        cluster_ratio: float = 1.0,
        drop_ratio: float = 0.,
    ) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        # pipe.scheduler = DDIMScheduler.from_pretrained(SD_SOURCE, subfolder="scheduler")
        # pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae

        self.pips_model = Pips(stride=4).cuda()
        # self.pips_model.load_state_dict(torch.load(os.path.expanduser(cfg.pips_model_path))['model_state_dict'])
        pips_model_path = os.path.expanduser(os.path.join(torch.hub.get_dir(), 'pips', f'model-000200000.pth'))
        if not os.path.exists(pips_model_path):
            print(f'PIPS model not found, downloading...')
            import urllib.request
            os.makedirs(os.path.dirname(pips_model_path), exist_ok=True)
            urllib.request.urlretrieve('https://huggingface.co/aharley/pips/resolve/main/model-000200000.pth', pips_model_path)
        self.pips_model.load_state_dict(torch.load(pips_model_path)['model_state_dict'])

        self.cluster_thresh = cluster_thresh
        self.cluster_ratio = cluster_ratio
        self.drop_ratio = drop_ratio
        def cluster_func(x, y, percent):
            if percent < self.cluster_thresh[0] or percent > self.cluster_thresh[1]:
                return x
            new = x + (y-x)*self.cluster_ratio
            keep_mask = (torch.rand_like(x[[0]]) < self.drop_ratio).expand_as(x)
            new[keep_mask] = x[keep_mask]
            return new
        self.cluster_func = cluster_func
        self.noise = None

        CONSOLE.print("InstructPix2Pix loaded!")

    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        mask,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        ori_h, ori_w = image.shape[2:]
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(ori_h//8, ori_w//8), mode='nearest').squeeze(0)
        soft_mask = torch.where(mask.bool(), 1.0, 0.3)

        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)
        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)
        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, torch.stack([self.scheduler.timesteps[0]]*len(latents), dim=0))  # type: ignore
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)  * (soft_mask.unsqueeze(0)) 
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # for i in range(len(decoded_img)):
        #     raw_image = np.uint8(decoded_img[i].cpu().permute(1,2,0).numpy()*255)
        #     plt.imshow(raw_image)
        #     plt.savefig(f'./decode_image/face/batman/decode.png')
        # import pdb; pdb.set_trace()
        return decoded_img

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1
        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    
    @torch.no_grad()
    def refine_img(
        self,
        text_embeddings,
        image,
        image_cond,
        mask,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
        ):
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        self.scheduler.set_timesteps(diffusion_steps)

        # prepare image and image_cond latents
        latents_list = []
        index = 0

        bl = 4
        for i in range(len(image)//bl):
            latents = self.imgs_to_latent(image[index:index+bl])
            latents_list.append(latents)
            index +=bl
        latents = torch.cat(latents_list, 0)
        image_cond_latents = torch.stack([self.prepare_image_latents(image_cond[[k]]) for k in range(len(image))])

        # get traj
        ori_h, ori_w = image.shape[2:]
        mask = F.interpolate(mask[:,0].unsqueeze(0).float(), size=(ori_h//8, ori_w//8), mode='nearest').squeeze(0)
        soft_mask = torch.where(mask.bool(), 1.0, 0.2)

        # add noise
        if self.noise is None:
            self.noise = torch.randn_like(latents)
        # noise = init_latents(latents.shape, pixel_coord_flat_list, invalid_mask_list, latents.device)[0]

        latents = self.scheduler.add_noise(latents, self.noise, torch.stack([self.scheduler.timesteps[0]]*len(latents), dim=0))  # type: ignore
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py

        for i, t in enumerate(self.scheduler.timesteps):
            noise_pred_all = []
            for k in range(len(latents)):
                # pred noise
                latent_model_input = torch.cat([latents[[k]]] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents[k]], dim=1)
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred_all.append(
                    noise_pred_uncond 
                    + guidance_scale * (noise_pred_text - noise_pred_image) * (soft_mask[k].unsqueeze(0).unsqueeze(0)) 
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )
            noise_pred = torch.cat(noise_pred_all, dim=0)
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample                            

        # decode latents to get edited image
        with torch.no_grad():
            # decoded_img = self.latents_to_img(latents)
            decoded_img_list = []
            index = 0
            for i in range(len(latents)//bl):
                decoded_img = self.latents_to_img(latents[index:index+bl])
                decoded_img_list.append(decoded_img)
                index +=bl
            decoded_img = torch.cat(decoded_img_list, 0)
        return decoded_img

    @torch.no_grad()
    def refine_single_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        mask,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        ori_h, ori_w = image.shape[2:]
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(ori_h//8, ori_w//8), mode='nearest').squeeze(0)

        self.scheduler.config.num_train_timesteps = max_step
        self.scheduler.set_timesteps(diffusion_steps)
        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)
        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, torch.stack([self.scheduler.timesteps[0]]*len(latents), dim=0))  # type: ignore
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            if t <= 5:
                soft_mask = torch.where(mask.bool(), 1.0, 0.2)
            else:
                soft_mask = torch.where(mask.bool(), 1.0, 0.5)
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)  * (soft_mask.unsqueeze(0)) 
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)
        return decoded_img

    @torch.no_grad()
    def edit_image_by_cor_batch(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        pixel_coord_list,
        invalid_mask_list,
        dense_coord_list,
        dense_invalid_mask_list,
        mask,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        # self.scheduler.config.num_train_timesteps = T.item()
        
        # self.scheduler.config.num_train_timesteps = max_step
        self.scheduler.set_timesteps(diffusion_steps)

        # prepare image and image_cond latents
        latents_list = []
        index = 0

        bl = 4
        for i in range(len(image)//bl):
            latents = self.imgs_to_latent(image[index:index+bl])
            latents_list.append(latents)
            index +=bl
        latents = torch.cat(latents_list, 0)

        image_cond_latents = torch.stack([self.prepare_image_latents(image_cond[[k]]) for k in range(len(image))])
        # get traj
        ori_h, ori_w = image.shape[2:]
        # image_8batch = torch.cat([image]+[image[[-1]]]*(8-len(image)), dim=0)[None]
        pips_height = ori_h//8*8
        pips_width = ori_w//8*8

        pixel_coord_flat_list = []
        for pixel_coord in pixel_coord_list:
            pixel_coord_flat = (pixel_coord[...,1]//8*pips_width//8 + pixel_coord[...,0]//8).long()
            pixel_coord_flat[pixel_coord_flat<0] = 0
            pixel_coord_flat[pixel_coord_flat>pips_height//8*pips_width//8-1] = pips_height//8*pips_width//8-1
            pixel_coord_flat_list.append(pixel_coord_flat)
        
        dense_coord_flat_list = []
        for dense_coord in dense_coord_list:
            dense_coord_flat = (dense_coord[...,1].round()*pips_width + dense_coord[...,0].round()).long()
            dense_coord_flat[dense_coord_flat<0] = 0
            dense_coord_flat[dense_coord_flat>pips_height*pips_width-1] = pips_height*pips_width-1
            dense_coord_flat_list.append(dense_coord_flat)

        mask = F.interpolate(mask.unsqueeze(0).float(), size=(pips_height, pips_width), mode='nearest').squeeze(0)
        latent_mask = F.interpolate(mask.unsqueeze(0).float(), size=(ori_h//8, ori_w//8), mode='nearest').squeeze(0)
        soft_mask = torch.where(latent_mask.bool(), 1.0, 0.1)

        # add noise
        if self.noise is None:
            self.noise = torch.randn_like(latents)
        # noise = init_latents(latents.shape, pixel_coord_flat_list, invalid_mask_list, latents.device)[0]

        latents = self.scheduler.add_noise(latents, self.noise, torch.stack([self.scheduler.timesteps[0]]*len(latents), dim=0))  # type: ignore
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py

        for i, t in enumerate(self.scheduler.timesteps):
            noise_pred_all = []
            for k in range(len(latents)):
                # pred noise
                latent_model_input = torch.cat([latents[[k]]] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents[k]], dim=1)
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred_all.append(
                    noise_pred_uncond 
                    + guidance_scale * (noise_pred_text - noise_pred_image) * (soft_mask[k].unsqueeze(0).unsqueeze(0)) 
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )
            noise_pred = torch.cat(noise_pred_all, dim=0)
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # reg according to traj
            progress = float(i)/diffusion_steps

            for i in range(len(pixel_coord_flat_list)):
                pixel_coord_flat = torch.roll(pixel_coord_flat_list[i], shifts=i, dims=0)
                invalid_mask = torch.roll(invalid_mask_list[i], shifts=i, dims=0)
                latents = gather_and_avg_by_traj(latents, pixel_coord_flat, invalid_mask, i, lambda x,y: self.cluster_func(x,y,progress))
        # del image
        # del image_cond  
        # del image_cond_latents
        # del pixel_coord_list                                

        # decode latents to get edited image
        with torch.no_grad():
            # decoded_img = self.latents_to_img(latents)
            decoded_img_list = []
            index = 0
            for i in range(len(latents)//bl):
                decoded_img = self.latents_to_img(latents[index:index+bl])
                decoded_img_list.append(decoded_img)
                index +=bl
            decoded_img = torch.cat(decoded_img_list, 0)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # for i in range(len(decoded_img)):
        #     raw_image = np.uint8(decoded_img[i].cpu().permute(1,2,0).numpy()*255)
        #     image = Image.fromarray(raw_image)
        #     image.save(f"./decode_image/fangzhou/Egyptian_noise_align/frame_{i}.png")
        # import pdb; pdb.set_trace()

        # wait_list = [0,11,19,21,23,42,44,50,64,76] #panda
        # wait_list = [0,1,7,9,14,16,17,22,24,28,35,39,43,47,52,56,65,72,74,77,79]  #face-Einstein
        # wait_list = [0,1,7,9,17,25,26,30,32,41,43,52]   #face-clown
        # wait_list = [0,8,12,14,15,16,22,23,25,36,45,50,51]   #face-batman  iron man wait_list = [45]

        edited_list = []
        wait_list = [11,14,17,28,32]
        for index in wait_list:
            mask[index] = 0
        mask = mask.unsqueeze(1)

        while len(wait_list)!=0:
            index = wait_list.pop(0)
            decoded_img[index] = self.refine_single_image(text_embeddings, 
                                                decoded_img[index].unsqueeze(0)*(1-mask[index]), 
                                                decoded_img[index].unsqueeze(0), 
                                                mask[index,0], 
                                                guidance_scale, 
                                                image_guidance_scale,
                                                diffusion_steps,
                                                lower_bound,
                                                upper_bound)
            dense_coord_flat = torch.roll(dense_coord_flat_list[index], shifts=index, dims=0)
            dense_invalid_mask = torch.roll(dense_invalid_mask_list[index], shifts=index, dims=0)
            decoded_img, mask, wait_list, edited_list = pixel_replace_by_traj(decoded_img, mask, dense_coord_flat, dense_invalid_mask, wait_list, edited_list, index=index, use_cluster=False)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # for i in range(len(decoded_img)):
        #     raw_image = np.uint8(decoded_img[i].cpu().permute(1,2,0).numpy()*255)
        #     # raw_image = np.uint8(mask[i].cpu().permute(1,2,0).numpy()*255)
        #     image = Image.fromarray(raw_image)
        #     image.save(f"./decode_image_avg/fangzhou/Egyptian_2/frame_{i}.png")
        # import pdb; pdb.set_trace()
        return decoded_img
