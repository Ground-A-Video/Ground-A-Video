import argparse
import yaml
import os
from inference import run


def load_meta_from_yaml(file_path):
    with open(file_path, "r") as f:
        meta= yaml.safe_load(f)
    return meta


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="video_configs/rabbit_watermelon.yaml")
    parser.add_argument("--folder", type=str, default="outputs/rabbit_watermelon", help="Path to output folder")
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--denoising_steps", type=int, default=50)
    parser.add_argument("--ddim_inv_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=12.5)
    parser.add_argument("--negative_prompt", type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
    parser.add_argument("--a_prompt", type=str, default='best quality, extremely detailed')
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--flow_smooth_threshold", type=float, default=0.2)
    parser.add_argument("--nti", type=bool, default=False)
    
    args = vars(parser.parse_args())
    meta = load_meta_from_yaml(args["config"])

    # save config file as yaml
    os.makedirs(args["folder"], exist_ok=True)
    with open(os.path.join(args["folder"], "config.yaml"), 'w') as f:
        yaml.dump(args, f)
    
    # Ground-A-Video inference
    run(meta, args, starting_noise=None)
