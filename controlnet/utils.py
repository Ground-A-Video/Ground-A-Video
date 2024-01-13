import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
import cv2
import PIL
from PIL import Image
from diffusers.utils import PIL_INTERPOLATION

import torch
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode, write_jpeg
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image

from annotator.zoe import ZoeDetector
from annotator.util import HWC3


def prepare_magnitude_map(optical_flows, save_dir=None):
    outs = []

    for optical_flow in optical_flows:
        max_norm = torch.sum(optical_flow**2, dim=0).sqrt().max()
        epsilon = torch.finfo((optical_flow).dtype).eps
        normalized_flow = optical_flow / (max_norm + epsilon)
        magnitude_map = torch.sqrt(normalized_flow[0]**2 + normalized_flow[1]**2)
        outs.append(magnitude_map.unsqueeze(0))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i,out in enumerate(outs):
            Image.fromarray(
                (out[0].cpu()*255).numpy().astype(np.uint8)
            ).save(f"{save_dir}/{i+1}.jpg")

    return outs


def infer_optical_flow(model, frameA, frameB, height=512, width=512, device='cuda', save_dir=None, i=0):

    input_frame_1 = Image.open(frameA).convert('RGB')
    input_frame_2= Image.open(frameB).convert('RGB')

    input_frame_1 = F.to_tensor(input_frame_1)
    input_frame_2 = F.to_tensor(input_frame_2)

    img1_batch = torch.stack([input_frame_1])
    img2_batch = torch.stack([input_frame_2])
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[512, 512])
        img2_batch = F.resize(img2_batch, size=[512, 512])
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flow = list_of_flows[-1][0] 

    original_flow = predicted_flow.clone()
    flow_img = flow_to_image(predicted_flow).to("cpu")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        write_jpeg((flow_img).type(torch.uint8), f"{save_dir}/{i}.jpg")

    return original_flow


def prepare_optical_flow(input_image_path, device, save_dir=None):
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
    model = model.eval()
    outs = []

    input_image_paths = [os.path.join(input_image_path, file) for file in sorted(os.listdir(input_image_path))]

    for i in tqdm(range(1,len(input_image_paths)), desc=f"Optical Flow Estimation x {len(input_image_paths)-1}"):
        original_flow = infer_optical_flow(
            model, input_image_paths[i-1], input_image_paths[i], height=512, width=512, device=device, save_dir=save_dir, i=i
        )
        outs.append(original_flow)
    
    return outs


def prepare_depth(input_images, device, dtype, save_dir=None):
    input_images = [rearrange(x.squeeze(0), "c h w -> h w c") for x in input_images]
    input_images = [np.array(x).astype(np.uint8) for x in input_images]
    outs = []
    
    apply_model = ZoeDetector()
    for image in tqdm(input_images, desc=f"Depth Estimation x {len(input_images)}"):
        image = HWC3(image)
        depth_map = apply_model(image)
        depth_map = HWC3(depth_map)
        depth_map = Image.fromarray(depth_map)
        outs.append(depth_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        outs[0].save(
            os.path.join(save_dir, "depth.gif"), 
            save_all=True, 
            append_images=outs[1:], 
            optimize=False, 
            loop=0, 
            duration=250
        )
        for i,img in enumerate(outs):
            img.save(os.path.join(save_dir,f"{i}.png"), "png")
    
    outs = prepare_cond(outs, device, dtype)

    return outs


def prepare_cond(
    images, device, dtype, width=512, height=512, batch_size=1, num_images_per_prompt=1, do_classifier_free_guidance=False
):  # 'do_classifier_free_guidance': even if we are doing cfg, just set it as False
    assert isinstance(images, list)
    
    outs = []
    for image in images:
        out = prepare_image(image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance)
        outs.append(out)

    outs = torch.stack(outs,dim=0)
    outs = rearrange(outs,"f b c h w -> b c f h w")
    
    return outs


def prepare_image(
    image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance:
        image = torch.cat([image] * 2)

    return image