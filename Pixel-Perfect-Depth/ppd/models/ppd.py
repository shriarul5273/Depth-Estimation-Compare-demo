from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from huggingface_hub import hf_hub_download
from ppd.utils.timesteps import Timesteps
from ppd.utils.schedule import LinearSchedule
from ppd.utils.sampler import EulerSampler
from ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd.models.dit import DiT

class PixelPerfectDepth(nn.Module):
    def __init__(
        self,
        semantics_pth='checkpoints/depth_anything_v2_vitl.pth',
        sampling_steps=10,

    ):
        super(PixelPerfectDepth, self).__init__()

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = DEVICE

        self.semantics_encoder = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024]
        )
        semantics_pth = hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Large", 
            filename="depth_anything_v2_vitl.pth", 
            repo_type="model")
        self.semantics_encoder.load_state_dict(torch.load(semantics_pth, map_location='cpu'), strict=False)
        self.semantics_encoder = self.semantics_encoder.to(self.device).eval()
        self.dit = DiT()

        self.sampling_steps = sampling_steps

        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
            device=self.device,
        )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
        )
    
    @torch.no_grad()
    def infer_image(self, image, sampling_steps=None, use_fp16: bool = True):
        h, w = image.shape[:2]
        resize_image = resize_keep_aspect(image)
        image = image2tensor(resize_image)
        image = image.to(self.device)

        if sampling_steps is not None and sampling_steps != self.sampling_steps:
            self.sampling_steps = sampling_steps
            self.sampling_timesteps = Timesteps(
                T=self.schedule.T,
                steps=self.sampling_steps,
                device=self.device,
            )
            self.sampler = EulerSampler(
                schedule=self.schedule,
                timesteps=self.sampling_timesteps,
                prediction_type='velocity'
            )

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            depth = self.forward_test(image)
        # depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)[0, 0]

        return depth.squeeze().cpu().numpy(), resize_image
    
    @torch.no_grad()
    def forward_test(self, image):

        semantics = self.semantics_prompt(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
        
        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)

        return latent + 0.5


    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.semantics_encoder(image)
        return semantics
