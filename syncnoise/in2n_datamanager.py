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
Instruct-NeRF2NeRF Datamanager.

File copied from https://github.com/ayaanzhaque/instruct-nerf2nerf/tree/main/in2n
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

CONSOLE = Console(width=120)

@dataclass
class InstructNeRF2NeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the InstructNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class InstructNeRF2NeRFDataManager(VanillaDataManager):
    """Data manager for InstructNeRF2NeRF."""

    config: InstructNeRF2NeRFDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device),)

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)
        self.image_batch['image'] = self.image_batch['image'][..., :3]
        self.image_batch_updated = torch.zeros(self.image_batch['image'].shape[0], dtype=torch.bool)
        self.only_sample_updated = False

        from pathlib import PosixPath
        from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary
        self.cam_id_to_camera = read_cameras_binary(PosixPath("./data/fangzhou-small/colmap/sparse/0/cameras.bin"))
        self.im_id_to_image = read_images_binary(PosixPath("./data/fangzhou-small/colmap/sparse/0/images.bin"))

        # rearrange
        new_idx = torch.argsort(self.image_batch['image_idx'])
        self.image_batch['image'] = self.image_batch['image'][new_idx.to(self.image_batch['image'].device)]
        self.image_batch['image_idx'] = self.image_batch['image_idx'][new_idx]

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch['image'] = self.image_batch['image'].clone()
        self.original_image_batch['image_idx'] = self.image_batch['image_idx'].clone()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None
        if self.only_sample_updated and torch.any(self.image_batch_updated):
            batch = self.train_pixel_sampler.sample({
                k: v[self.image_batch_updated.to(v.device)]
                for k, v in self.image_batch.items()})
        else:
            batch = self.train_pixel_sampler.sample(self.image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        return ray_bundle, batch
