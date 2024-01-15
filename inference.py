import os 
import numpy as np
import random
from PIL import Image
from einops import rearrange, repeat
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator

import torch 
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import clip
from transformers import CLIPProcessor, CLIPModel
from diffusers import DDIMScheduler

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from controlnet.controlnet_3d import ControlNetModel3D
from controlnet.utils import prepare_depth, prepare_optical_flow, prepare_magnitude_map
from null_inv import NullInversion


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import CrossFrameGatedAttentionDense
    for module in model.modules():
        if type(module) == CrossFrameGatedAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def draw_masks_from_boxes(boxes, size, randomize_fg_mask=False, random_add_bg_mask=False):
    "boxes should be the output from dataset, which is a batch of bounding boxes"

    image_masks = [] 
    for box in boxes: # This is batch dimension

        image_mask = torch.ones(size,size)
        for bx in box:
            x0,y0,x1,y1 = bx*size
            x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
            obj_width = x1-x0
            obj_height = y1-y0
            if randomize_fg_mask and (random.uniform(0,1)<0.5) and (obj_height>=4) and (obj_width>=4):
                obj_mask = get_a_fg_mask(obj_height, obj_width)
                image_mask[y0:y1,x0:x1] = image_mask[y0:y1,x0:x1] * obj_mask # put obj mask into the inpainting mask 
            else:
                image_mask[y0:y1,x0:x1] = 0  # box itself is mask for the obj
        

        # So far we already drew all masks for obj, add bg mask if needed
        if random_add_bg_mask and (random.uniform(0,1)<0.5):
            bg_mask = get_a_bg_mask(size)
            image_mask *= bg_mask

        image_masks.append(image_mask)
    return torch.stack(image_masks).unsqueeze(1)


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask


def prepare_batch_video(meta, batch=1, max_objs=30, clip_length=8, device='cuda', dtype=torch.float16):
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    for i in tqdm(range(clip_length), desc="preparing groundings for video"):
        out = prepare_batch(meta, 1, max_objs, ith_clip=i, model=model, processor=processor)
        if i==0:
            boxes_container = deepcopy(out["boxes"])
            masks_container = deepcopy(out["masks"])
            text_masks_container = deepcopy(out["text_masks"])
            image_masks_container = deepcopy(out["image_masks"])
            text_embeddings_container = deepcopy(out["text_embeddings"])
            image_embeddings_container = deepcopy(out["image_embeddings"])
        else:
            boxes_container = torch.cat([boxes_container, deepcopy(out["boxes"])], dim=0)
            masks_container = torch.cat([masks_container, deepcopy(out["masks"])], dim=0)
            text_masks_container = torch.cat([text_masks_container, deepcopy(out["text_masks"])], dim=0)
            image_masks_container = torch.cat([image_masks_container, deepcopy(out["image_masks"])], dim=0)
            text_embeddings_container = torch.cat([text_embeddings_container, deepcopy(out["text_embeddings"])], dim=0)
            image_embeddings_container = torch.cat([image_embeddings_container, deepcopy(out["image_embeddings"])], dim=0)

    out = {
        "boxes" : boxes_container.to(device, dtype=dtype),
        "masks" : masks_container.to(device, dtype=dtype),
        "text_masks" : text_masks_container.to(device, dtype=dtype),
        "image_masks" : image_masks_container.to(device, dtype=dtype),
        "text_embeddings"  : text_embeddings_container.to(device, dtype=dtype),
        "image_embeddings" : image_embeddings_container.to(device, dtype=dtype)
    }

    return out


@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30, ith_clip=None, model=None, processor=None):
    phrases, images = meta.get("phrases"), meta.get("images")
    if images == None:
        images = [None]*len(phrases[ith_clip])
    else:
        images = images[ith_clip]
    if phrases == None:
        phrases = [None]*len(images[ith_clip])
    else:
        phrases = phrases[ith_clip]

    '''
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)
    '''

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'][ith_clip], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),    # not sure what text_mask is, so leave it intact
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ), # same
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return out


def save_images_as_gif(images, save_path, optimize=False, loop=0, duration=250):
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )


def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    autoencoder = instantiate_from_config(config['autoencoder']).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).eval()
    diffusion = instantiate_from_config(config['diffusion'])

    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return autoencoder, text_encoder, diffusion, config


def load_unet3d(pretrained_unet_path):
    # instantiate from `config.json`
    unet3d_config = OmegaConf.load(os.path.join(pretrained_unet_path,"config.json"))
    unet3d = instantiate_from_config(unet3d_config).eval()
    
    # load weights
    state_dict = torch.load(os.path.join(pretrained_unet_path,"diffusion_pytorch_model.bin"), map_location="cpu")
    unet3d.load_2d_state_dict(state_dict=state_dict)

    return unet3d


def load_unet2d(pretrained_unet_path):
    unet_config = OmegaConf.load(os.path.join(pretrained_unet_path,"config.json"))
    unet_config["target"] = "ldm.modules.diffusionmodules.openaimodel.UNetModel"
    unet2d = instantiate_from_config(unet_config).eval()

    state_dict = torch.load(os.path.join(pretrained_unet_path,"diffusion_pytorch_model.bin"), map_location="cpu")
    unet2d.load_2d_state_dict(state_dict=state_dict)

    return unet2d


@torch.no_grad()
def run(meta, config, starting_noise=None):
    args = OmegaConf.create(config)

    # prepare accelerator
    accelerator = Accelerator(mixed_precision = args.mixed_precision)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # ==================================================== #
    # GLGIEN modules: "gligen/gligen-inpainting-text-box"  #
    # UNet(only): "ground-a-video/unet3d_ckpts"            #
    # ControlNet: "lllyasviel/control_v11f1p_sd15_depth"   #
    # ==================================================== #
    autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])  # HuggingFace: "gligen/gligen-inpainting-text-box/diffusion_pytorch_model.bin"
    model = load_unet3d("unet3d_ckpts")
    unet2d = load_unet2d("unet3d_ckpts")                     # HuggingFace: "ground-a-video/unet3d_ckpts"
    controlnet = ControlNetModel3D.from_2d_model("control_v11f1p_sd15_depth").eval()

    autoencoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    diffusion.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)
    unet2d.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)


    # prepare models to handle groundings
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #
    batch = prepare_batch_video(meta=meta, batch=args.batch_size, clip_length=args.clip_length, device=accelerator.device, dtype=weight_dtype)
    meta_gt = meta.copy()
    meta_gt["phrases"] = meta_gt["source_phrases"]
    batch_gt = prepare_batch_video(meta=meta_gt, batch=args.batch_size, clip_length=args.clip_length, device=accelerator.device, dtype=weight_dtype)


    # encode prompts
    if args.a_prompt:
        meta["prompt"] = meta["prompt"] + ", " + args.a_prompt
    context = text_encoder.encode([meta["prompt"]]*args.batch_size )
    uc = text_encoder.encode(args.batch_size*[""])
    if args.negative_prompt is not None:
        uc = text_encoder.encode(args.batch_size*[args.negative_prompt])


    # load ddim sampler
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    sampler = DDIMSampler(diffusion, model, controlnet, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale, dtype=weight_dtype)
    steps = args.denoising_steps


    # ============================ #
    # Input Conditions Preparation #
    # ============================ #
    inpainting_mask = z0 = None
    inpainting_extra_input = None
    if "input_images_path" in meta:

        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).to(accelerator.device, dtype=weight_dtype)
        
        # prepare input images, depth, optical flow
        input_image = [
            F.pil_to_tensor(Image.open(os.path.join(meta["input_images_path"], file)).convert("RGB").resize((512,512))).float().unsqueeze(0)
             for file in sorted(os.listdir(meta["input_images_path"]))
        ]
        conditions_save_dir = f"{args.folder}/input_conditions"
        depth_maps = prepare_depth(input_image, accelerator.device, weight_dtype, save_dir=f"{conditions_save_dir}/depth_maps")
        optical_flows = prepare_optical_flow(meta["input_images_path"], accelerator.device, save_dir=f"{conditions_save_dir}/optical_flow")

        # x_0 -> z_0
        input_image = torch.cat(input_image, dim=0)
        input_image = (input_image / 255 - 0.5) / 0.5
        input_image = input_image.to(accelerator.device, dtype=weight_dtype)
        z0 = autoencoder.encode( input_image )
        
        # masking
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
        z0 = rearrange(z0, "(b f) c h w -> b c f h w", f=args.clip_length)
        inpainting_mask = rearrange(inpainting_mask, "(b f) c h w -> b c f h w", f=args.clip_length)
        inpainting_extra_input = rearrange(inpainting_extra_input, "(b f) c h w -> b c f h w", f=args.clip_length).to(accelerator.device, dtype=weight_dtype)


    # prepare groundings
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    grounding_input_gt = grounding_tokenizer_input.prepare(batch_gt)


    # =================== #
    # Per-Frame Inversion #
    #    using T2I UNet   #
    # =================== #
    grounding_tokenizer_input_2d = instantiate_from_config(config['grounding_tokenizer_input'])
    unet2d.grounding_tokenizer_input = grounding_tokenizer_input_2d
    null_inversion = NullInversion(
        unet=unet2d,
        num_inv_steps=args.ddim_inv_steps,
        guidance_scale=args.guidance_scale,
        null_inv_with_prompt=False,
    )
    ddim_inv_latent_list, uncond_embeddings_list = [], []
    
    with torch.enable_grad():
        for i in tqdm(range(args.clip_length), desc="Per-Frame Inversion"):

            # prepare 2D inputs
            z0_ith = z0[:, :, i, :, :].clone()
            batch_ith = dict(
                boxes = batch_gt["boxes"][i:i+1],
                masks = batch_gt["masks"][i:i+1],
                text_embeddings = batch_gt["text_embeddings"][i:i+1]
            )
            grounding_input_ith = grounding_tokenizer_input_2d.prepare(batch_ith)
            inpainting_extra_input_ith= inpainting_extra_input[:, :, i, :, :].clone() if inpainting_extra_input is not None else None

            ddim_inv_latent, uncond_embeddings = null_inversion.invert(
                latents=z0_ith,
                prompt=meta["source_prompt"],
                grounding_input=grounding_input_ith,
                inpainting_extra_input=inpainting_extra_input_ith,
                text_encoder=text_encoder,
                null_base_lr=1e-2,
                batch_size=args.batch_size,
                verbose=False,
            )
            '''
            [Shape of Outputs]
                ddim_inv_latent: Tensor of shape (1, 4, 64, 64)
                uncond_embeddings: List [ (1,77,768), (1,77,768), ..., (1,77,768) ] where num_elements = ddim_inv_steps
            '''
            ddim_inv_latent_list.append(ddim_inv_latent)
            uncond_embeddings_list.append(uncond_embeddings)
    
    # aggregate inverted latents
    ddim_inv_aggregated = torch.cat(ddim_inv_latent_list, dim=0)
    ddim_inv_aggregated = rearrange(ddim_inv_aggregated, "(b f) c h w -> b c f h w", f=args.clip_length)
    starting_noise = ddim_inv_aggregated

    # aggregate uncondtional embeddings
    uncond_embeddings_aggregated = uncond_embeddings_list[0]
    for i in range(1, args.clip_length):
        ith_frame_uncond_embeddings_list = uncond_embeddings_list[i]
        for j in range(args.ddim_inv_steps):
            uncond_embeddings_aggregated[j] = torch.cat([uncond_embeddings_aggregated[j], ith_frame_uncond_embeddings_list[j]], dim=0)
    #uc = uncond_embeddings_aggregated

    del null_inversion
    unet2d.cpu()
    del unet2d

    # ===================== #
    #  Optical Flow guided  #
    # inverted z0 smoothing #
    # ===================== #
    magnitude_maps = prepare_magnitude_map(optical_flows, save_dir=f"{conditions_save_dir}/magnitude_maps")
    magnitude_maps = torch.stack(magnitude_maps, dim=0)
    magnitude_maps = torch.nn.functional.interpolate(
        magnitude_maps,
        size=(starting_noise.shape[-2], starting_noise.shape[-1]),
        mode='bilinear'
    )
    magnitude_maps = magnitude_maps[:, 0, :, :]  
    image_residual_mask = (magnitude_maps > args.flow_smooth_threshold).type(weight_dtype)
    image_residual_mask = repeat(image_residual_mask, '(b f) h w -> b f h w', b=args.batch_size)
    image_residual_mask = repeat(image_residual_mask, 'b f h w -> b c f h w', c=starting_noise.shape[1])

    for n_frame in range(1, args.clip_length):
        starting_noise[:,:, n_frame, :, :] \
            = (starting_noise[:,:, n_frame, :, :] - starting_noise[:,:, n_frame-1, :, :]) * image_residual_mask[:,:, n_frame-1, :, :] \
            + starting_noise[:,:, n_frame-1, :, :]

        
    # ========================= #
    # Reverse Diffusion Process #
    # ========================= #
    input = dict(
        x=starting_noise, 
        timesteps=None, 
        context=context, 
        grounding_input=grounding_input,
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,
        controlnet_cond=depth_maps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    shape = (args.batch_size, model.in_channels, args.clip_length, model.image_size, model.image_size)

    generated_samples = sampler.sample(
        S=steps, 
        shape=shape, 
        input=input,  
        uc=uc, 
        uncond_embeddings=uncond_embeddings_aggregated,
        guidance_scale=args.guidance_scale, 
        mask=inpainting_mask, 
        x0=z0
    )

    # z_0 -> x_0
    generated_samples = rearrange(generated_samples, "b c f h w -> (b f) c h w") 
    generated_samples = autoencoder.decode(generated_samples)
    generated_samples = rearrange(generated_samples,"(b f) c h w -> b c f h w", f=args.clip_length)

    # save outputs #
    output_folder = f"{args.folder}/outputs"
    os.makedirs(output_folder, exist_ok=True)

    start = len(os.listdir(output_folder))
    image_list =[]
    generated_samples = rearrange(generated_samples, "b c f h w -> (b f) c h w") 
    for i, sample in enumerate(generated_samples):
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        image_list.append(sample)
        sample.save(f"{output_folder}/{i}.png")
    save_images_as_gif(image_list, os.path.join(output_folder, "samples.gif"))
    

