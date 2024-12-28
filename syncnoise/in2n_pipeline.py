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
InstructPix2Pix Pipeline and trainer

File copied from https://github.com/ayaanzhaque/instruct-nerf2nerf/tree/main/in2n
"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.viewer_elements import ViewerNumber, ViewerText

from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat
from PIL import Image
import numpy as np

from syncnoise.in2n_datamanager import (
    InstructNeRF2NeRFDataManagerConfig,
)
from syncnoise.ip2p import InstructPix2Pix


@dataclass
class InstructNeRF2NeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFPipeline)
    """target class to instantiate"""
    datamanager: InstructNeRF2NeRFDataManagerConfig = InstructNeRF2NeRFDataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    edit_rate: int = 10
    """how many NeRF steps before image edit"""
    syncnoise_edit_rate: int = 500
    """how many NeRF steps before image edit for syncnoise"""
    use_syncnoise_steps: int = 500
    """how many NeRF steps before stop using syncnoise"""
    edit_count: int = 1
    """how many images to edit per NeRF step"""
    syncnoise_edit_count: int = 80
    """how many images to edit per NeRF step for syncnoise"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.6
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.8
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = True
    """Whether to use full precision for InstructPix2Pix"""

class InstructNeRF2NeRFPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: InstructNeRF2NeRFPipelineConfig

    def __init__(
        self,
        config: InstructNeRF2NeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

        # viewer elements
        self.prompt_box = ViewerText(name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback)
        self.guidance_scale_box = ViewerNumber(name="Text Guidance Scale", default_value=self.config.guidance_scale, cb_hook=self.guidance_scale_callback)
        self.image_guidance_scale_box = ViewerNumber(name="Image Guidance Scale", default_value=self.config.image_guidance_scale, cb_hook=self.image_guidance_scale_callback)

    def guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for guidance scale slider"""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for text guidance scale slider"""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Callback for prompt box, change prompt in config and update text embedding"""
        self.config.prompt = handle.value
        
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        # use the first step as the editing start step
        if getattr(self, 'starting_step', None) is None:
            self.starting_step = step

        ray_bundle, batch = self.datamanager.next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        # train on updated images
        use_syncnoise = step < (self.starting_step+self.config.use_syncnoise_steps)

        # only sample edited images
        if ((use_syncnoise and ((step+1) % self.config.syncnoise_edit_rate != 0))
            or
            (not use_syncnoise and ((step+1) % self.config.edit_rate != 0))):
            self.datamanager.only_sample_updated = True
        else:
            self.datamanager.only_sample_updated = False

        # all images are updated
        if torch.all(self.datamanager.image_batch_updated):
            # use_syncnoise = False
            self.datamanager.only_sample_updated = False

        # edit an image every ``edit_rate`` steps with in2n
        if not use_syncnoise and (step % self.config.edit_rate == 0):
            self.edit_and_update()

        # edit an image every ``edit_rate`` steps with syncnoise, or no images edited
        if ((use_syncnoise and (step % self.config.syncnoise_edit_rate == 0))
            or not torch.any(self.datamanager.image_batch_updated)):
            self.edit_correspond_and_update()

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def pixel_to_world(self, pixel_coord, fx, fy, cx, cy, camera_to_world):
        R = camera_to_world[:,:3].t()
        T = -torch.matmul(R, camera_to_world[:,3].unsqueeze(1))
        world_to_camera = torch.cat((R,T), dim=-1)
        depth_map = pixel_coord[..., 2]

        U = (pixel_coord[..., 0])
        V = (pixel_coord[..., 1])

        x_c = (U - cx) * depth_map / fx
        y_c = (V - cy) * depth_map / fy
        z_c = depth_map   
        
        camera_coords = torch.stack((x_c, y_c, z_c, torch.ones_like(x_c)), dim=-1)
        world_coords = torch.matmul(camera_coords, world_to_camera.transpose(0, 1))
        
        return world_coords[..., :3]

    def world_to_pixel(self, world_coord, fx, fy, cx, cy, camera_to_world):
        world_coord_flat = world_coord.reshape(-1,3)
        world_coord_flat = torch.cat((world_coord_flat, torch.ones(len(world_coord_flat), 1).to(world_coord_flat.device)), dim=1)
        camera_coord_flat = torch.matmul(world_coord_flat, camera_to_world.t())

        U = fx * camera_coord_flat[:,0] / camera_coord_flat[:,2] + cx
        V = fy * camera_coord_flat[:,1] / camera_coord_flat[:,2] + cy

        Depth = camera_coord_flat[:,2]
        return torch.stack((U, V, Depth), dim=-1)

    def edit_correspond_and_update(self):
        original_image_list = []
        rendered_image_list = []
        depth_map_list = []
        current_index_list = []
        mask_list = []
        spots = []

        for i in range(self.config.syncnoise_edit_count):
            # iterate through "spot in dataset"
            current_spot = next(self.train_indices_order)

            # get original image from dataset
            original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
            # generate current index in datamanger
            current_index = self.datamanager.image_batch["image_idx"][current_spot]

            # generate mask
            file_path = "./data/fangzhou-small/masks/frame_" + str((current_index+1).cpu().numpy()).zfill(5) + ".png"
            mask = Image.open(file_path)
            mask = torch.tensor(np.array(mask, dtype="uint8")/255.)>0.5

            # get current camera, include camera transforms from original optimizer
            camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
            
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

            # get current render of nerf
            original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

            # image = Image.fromarray(rendered_image[0].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            # image.save(f"./renders/fangzhou-small/roman/frame_{i}.png")

            def project_pts_on_img(img_points, raw_img, index, max_distance=34, thickness=-1):
                import matplotlib.pyplot as plt
                import cv2
                img = raw_img.copy()
                
                cmap = plt.cm.get_cmap('hsv', 256)
                cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
                for i in range(img_points.shape[0]):
                    depth = img_points[i, 2]
                    color = cmap[np.clip(int(max_distance * 10 / depth), -255, 255), :]
                    cv2.circle(
                        img,
                        center=(int(np.round(img_points[i, 0])),
                                int(np.round(img_points[i, 1]))),
                        radius=1,
                        color=tuple(color),
                        thickness=thickness,
                    )
                plt.imshow(img)
                plt.savefig(f'./depth/depth_{index}.png')

            # pixel to world transformation
            depth_map = camera_outputs["depth"].cpu()/self.datamanager.train_dataparser_outputs.dataparser_scale

            # delete to free up memory
            del camera_outputs
            del current_camera
            del current_ray_bundle
            del camera_transforms
            torch.cuda.empty_cache()
            original_image_list.append(original_image)
            rendered_image_list.append(rendered_image)
            current_index_list.append(current_index+1)
            depth_map_list.append(depth_map)
            mask_list.append(mask)
            spots.append(current_spot)

        pixel_coord_all = []
        invalid_mask_all = []
        dense_coord_all = []
        dense_invalid_mask_all = []
        for i in range(len(current_index_list)):
            pixel_coord_list = []
            invalid_mask_list = []
            dense_coord_list = []
            dense_invalid_mask_list = []
            for j, current_index in enumerate(current_index_list):
                # generate camera2world matrix
                im_data = self.datamanager.im_id_to_image[current_index.item()]
                intrinsics = self.datamanager.cam_id_to_camera[1].params
                
                rotation = torch.tensor(qvec2rotmat(im_data.qvec))
                T = torch.tensor(im_data.tvec).view(-1,1)
                c2w = torch.cat((rotation, T), dim=-1)

                depth_map = depth_map_list[j]
                if j==0:
                    height, width, _ = depth_map.shape
                    x = torch.arange(0, height).float()
                    y = torch.arange(0, width).float()
                    xx, yy = torch.meshgrid(x, y)
                    pixel_coord = torch.stack((yy, xx), dim=-1)
                    dense_coord = torch.cat((pixel_coord, depth_map), dim=2).reshape(-1,3)
                    mask = mask_list[j].reshape(-1,)
                    dense_coord = dense_coord[mask]

                    pixel_coord = torch.cat((pixel_coord, depth_map), dim=2)[::16, ::16, :][1:, 1:].reshape(-1,3)
                    mask = mask_list[j][::16, ::16][1:, 1:].reshape(-1,)
                    pixel_coord = pixel_coord[mask]

                    # raw_image = np.uint8(original_image_list[j][0].cpu().permute(1,2,0).numpy()*255)
                    # project_pts_on_img(dense_coord.cpu().numpy(), raw_image, j)

                    world_coord = self.pixel_to_world(pixel_coord, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    dense_world_coord = self.pixel_to_world(dense_coord, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    
                    c2w_pre = c2w
                    invalid_mask = torch.zeros(len(world_coord))
                    dense_invalid_mask = torch.zeros(len(dense_world_coord))
                else:
                    pixel_coord = self.world_to_pixel(world_coord, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    invalid_mask = (pixel_coord[:,0]>=width) | (pixel_coord[:,0]<0) | (pixel_coord[:,1]>=height) | (pixel_coord[:,1]<0.1) | (pixel_coord[:,2]<0)
                    pixel_coord_2d = pixel_coord[:,:2].int()
                    pixel_coord_2d[invalid_mask] = 0

                    mask = mask_list[j][pixel_coord_2d[:,1], pixel_coord_2d[:,0]]
                    depth = depth_map.squeeze(2)[pixel_coord_2d[:,1], pixel_coord_2d[:,0]]

                    pixel_coord_back = torch.cat((pixel_coord_2d, depth.unsqueeze(1)), dim=1)
                    world_coord_back = self.pixel_to_world(pixel_coord_back, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    pixel_coord_back = self.world_to_pixel(world_coord_back, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w_pre.float())
                    pix_dist = torch.sum((pixel_coord_back[:,:2] - pixel_coord_list[0][:,:2])**2, dim=1).sqrt()
                    invalid_mask = invalid_mask | (torch.abs(depth - pixel_coord[:,2])>0.8) | (~mask) | (pix_dist>8)

                    dense_coord = self.world_to_pixel(dense_world_coord, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    dense_invalid_mask = (dense_coord[:,0]>=width) | (dense_coord[:,0]<0) | (dense_coord[:,1]>=height) | (dense_coord[:,1]<0.0) | (dense_coord[:,2]<0)
                    dense_coord_2d = dense_coord[:,:2].int()
                    dense_coord_2d[dense_invalid_mask] = 0
                    
                    mask = mask_list[j][dense_coord_2d[:,1], dense_coord_2d[:,0]]
                    depth = depth_map.squeeze(2)[dense_coord_2d[:,1], dense_coord_2d[:,0]]

                    dense_coord_back = torch.cat((dense_coord_2d, depth.unsqueeze(1)), dim=1)
                    world_coord_back = self.pixel_to_world(dense_coord_back, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w.float())
                    dense_coord_back = self.world_to_pixel(world_coord_back, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], c2w_pre.float())
                    pix_dist = torch.sum((dense_coord_back[:,:2]-dense_coord_list[0][:,:2])**2, dim=1).sqrt()
                    dense_invalid_mask = dense_invalid_mask | (torch.abs(depth - dense_coord[:,2])>0.8) | (~mask) #| (pix_dist>3)
                    
                    # dense_coord[dense_invalid_mask, 2] = 5
                    # raw_image = np.uint8(original_image_list[j][0].cpu().permute(1,2,0).numpy()*255)
                    # project_pts_on_img((dense_coord).cpu().numpy(), raw_image, j)

                pixel_coord_list.append(pixel_coord)
                invalid_mask_list.append(invalid_mask)
                dense_coord_list.append(dense_coord)
                dense_invalid_mask_list.append(dense_invalid_mask)

            pixel_coord_all.append(torch.stack(pixel_coord_list, 0).to(self.ip2p_device))
            invalid_mask_all.append(torch.stack(invalid_mask_list, 0).to(self.ip2p_device))
            dense_coord_all.append(torch.stack(dense_coord_list, 0).to(self.ip2p_device))
            dense_invalid_mask_all.append(torch.stack(dense_invalid_mask_list, 0).to(self.ip2p_device))

            current_index_list.insert(len(current_index_list), current_index_list[0])
            current_index_list.remove(current_index_list[0])

            original_image_list.insert(len(original_image_list), original_image_list[0])
            original_image_list.remove(original_image_list[0])

            depth_map_list.insert(len(depth_map_list), depth_map_list[0])
            depth_map_list.remove(depth_map_list[0])

            mask_list.insert(len(mask_list), mask_list[0])
            mask_list.remove(mask_list[0])

        edited_image = self.ip2p.edit_image_by_cor_batch(
            self.text_embedding.to(self.ip2p_device),
            torch.cat(rendered_image_list, 0).to(self.ip2p_device),
            torch.cat(original_image_list, 0).to(self.ip2p_device),
            pixel_coord_all,
            invalid_mask_all,
            dense_coord_all,
            dense_invalid_mask_all,
            torch.stack(mask_list, 0).to(self.ip2p_device),
            guidance_scale=self.config.guidance_scale*2,
            image_guidance_scale=self.config.image_guidance_scale,
            diffusion_steps=self.config.diffusion_steps,
            lower_bound=self.config.lower_bound,
            upper_bound=self.config.upper_bound,
        )

        # resize to original image size (often not necessary)
        if (edited_image.size() != rendered_image.size()):
            edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

        # write edited image to dataloader
        self.datamanager.image_batch["image"][spots] = edited_image.permute(0,2,3,1).to(self.datamanager.image_batch["image"])
        self.datamanager.image_batch_updated[spots] = True

    def edit_and_update(self):
        for i in range(self.config.edit_count):

            # iterate through "spot in dataset"
            current_spot = next(self.train_indices_order)
            
            # get original image from dataset
            original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
            # generate current index in datamanger
            current_index = self.datamanager.image_batch["image_idx"][current_spot]
    
            file_path = "./data/fangzhou-small/masks/frame_" + str((current_index+1).cpu().numpy()).zfill(5) + ".png"
            mask = Image.open(file_path)
            mask = torch.tensor(np.array(mask, dtype="uint8")/255.)>0.5

            # get current camera, include camera transforms from original optimizer
            camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

            # get current render of nerf
            original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

            # delete to free up memory
            del camera_outputs
            del current_camera
            del current_ray_bundle
            del camera_transforms
            torch.cuda.empty_cache()

            edited_image = self.ip2p.edit_image(
                        self.text_embedding.to(self.ip2p_device),
                        rendered_image.to(self.ip2p_device),
                        original_image.to(self.ip2p_device),
                        mask.to(self.ip2p_device),
                        guidance_scale=self.config.guidance_scale,
                        image_guidance_scale=self.config.image_guidance_scale,
                        diffusion_steps=self.config.diffusion_steps,
                        lower_bound=self.config.lower_bound,
                        upper_bound=self.config.upper_bound,
                    )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

            # write edited image to dataloader
            self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)
            self.datamanager.image_batch_updated[current_spot] = True

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
