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
Model for InstructNeRF2NeRF

File copied from https://github.com/ayaanzhaque/instruct-nerf2nerf/tree/main/in2n
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import models, transforms
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class InstructNeRF2NeRFModelConfig(NerfactoModelConfig):
    """Configuration for the InstructNeRF2NeRFModel."""
    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFModel)
    use_lpips: bool = False
    """Whether to use LPIPS loss"""
    use_style: bool = False
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""
    style_loss_mult: float = 0.1
    """Multiplier for LPIPS loss."""



class InstructNeRF2NeRFModel(NerfactoModel):
    """Model for InstructNeRF2NeRF."""

    config: InstructNeRF2NeRFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to()
        self.vgg16.eval()
        self.vgg16 = self.vgg16.features
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_gram(self, imgs, target_layers=(1, 6, 11, 18, 25)): 
        x = self.normalize(imgs)
        gram_mats = []
        for i, layer in enumerate(self.vgg16[:max(target_layers)+1]):
            x = layer(x)
            if i in target_layers:
                gram_mats.append(torch.einsum('nchw,ndhw->ncd', x, x) / (x.shape[-1] * x.shape[-2]))
        return gram_mats

    def get_style_loss(self, pred, gt):
        gt_gram = self.get_gram(gt)
        pred_gram = self.get_gram(pred)
        return sum([torch.mean((gt_gram[i] - pred_gram[i])**2) for i in range(len(gt_gram))])

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)

        if self.config.use_style:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2))
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2))
            loss_dict["style_loss"] = self.config.style_loss_mult * self.get_style_loss(out_patches, gt_patches)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict
