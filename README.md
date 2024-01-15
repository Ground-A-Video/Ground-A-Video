# Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models

This repository contains the official pytorch implementation of [Ground-A-Video](#).
<br/> <br/>
[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://ground-a-video.github.io/)

## Abstract
<b><font color="red">Ground A Video</font> is the first groundings-driven video editing framework, specially designed for [Multi-Attribute Video Editing](#).<br>
<font color="red">Ground A Video</font> is the first framework to intergrate spatially-continuous and spatially-discrete conditions.<br>
<font color="red">Ground A Video</font> does not neglect edits, confuse edits, but does preserve non-target regions.<br>
:o: Pretrained Stable Diffusion |
:o: Optical Flow, Depth Map, Groundings |
:x: Any training</b>

<details><summary>Full abstract</summary>


> We introduce a novel groundings guided video-to-video translation framework called Ground-A-Video. Recent endeavors in video editing have showcased promising results in single-attribute editing or style transfer tasks, either by training T2V models on text-video data or adopting training-free methods. However, when confronted with the complexities of multi-attribute editing scenarios, they exhibit shortcomings such as omitting or overlooking intended attribute changes, modifying the wrong elements of the input video, and failing to preserve regions of the input video that should remain intact. Ground-A-Video attains temporally consistent multi-attribute editing of input videos in a training-free manner without aforementioned shortcomings. Central to our method is the introduction of cross-frame gated attention which incorporates groundings information into the latent representations in a temporally consistent fashion, along with Modulated Cross-Attention and optical flow guided inverted latents smoothing. Extensive experiments and applications demonstrate that Ground-A-Video's zero-shot capacity outperforms other baseline methods in terms of edit-accuracy and frame consistency.
</details>

## News
* [11/11/2023] The paper is currently under review process. We plan to make the code public once the process is done, since there could be not minor modifications.
  <br> (Apologies for the late release, but please stay tuned!)

## Teaser
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Video Groundings</b></td>
  <td style="text-align:center;"><b>Depth Map</b></td>
  <td style="text-align:center;"><b>Optical Flow</b></td>
  <td style="text-align:center;"><b>Output Video</b></td>
</tr>

<tr>
  <td width=20% style="text-align:center;color:gray;">"A <ins>man</ins> is walking a <ins>dog</ins> on the <ins>road</ins>."</td>
  <td width=20% style="text-align:center;">man, dog, road</td>
  <td width=20% style="text-align:center;color:gray;">by ZoeDepth</td>
  <td width=20% style="text-align:center;">by RAFT-large</td>
  <td width=20% style="text-align:center;color:gray;">"<ins>Iron Man</ins> is walking a <ins>sheep</ins> on the <ins>lake</ins>."</td>
</tr>

<tr>
  <td style colspan="1"><img src="assets/dog_walking/input.gif"></td>
  <td style colspan="1"><img src="assets/dog_walking/grounding.gif"></td>
  <td style colspan="1"><img src="assets/dog_walking/depth.gif"></td>  
  <td style colspan="1"><img src="assets/dog_walking/flow.gif"></td> 
  <td style colspan="1"><img src="assets/dog_walking/output.gif"></td>  
</tr>

<tr>
  <td width=20% style="text-align:center;color:gray;">"A <ins>rabbit</ins> is eating a <ins>watermelon</ins> on the <ins>table</ins>."</td>
  <td width=20% style="text-align:center;">rabbit, watermelon, table</td>
  <td width=20% style="text-align:center;color:gray;">by ZoeDepth</td>
  <td width=20% style="text-align:center;">by RAFT-large</td>
  <td width=20% style="text-align:center;color:gray;">"A <ins>squirrel</ins> is eating an <ins>orange</ins> on the <ins>grass</ins>, <ins>under the aurora</ins>."</td>
</tr>

<tr>
  <td style colspan="1"><img src="assets/rabbit_watermelon/input.gif"></td>
  <td style colspan="1"><img src="assets/rabbit_watermelon/grounding.gif"></td>
  <td style colspan="1"><img src="assets/rabbit_watermelon/depth.gif"></td>  
  <td style colspan="1"><img src="assets/rabbit_watermelon/flow.gif"></td> 
  <td style colspan="1"><img src="assets/rabbit_watermelon/output.gif"></td>  
</tr>

</table>


## Setup

### Requirements

```shell
git clone https://github.com/Ground-A-Video/Ground-A-Video.git
cd Ground-A-Video

conda create -n groundvideo python=3.8
conda activate groundvideo
pip install -r requirements.txt
```

### Weights
<strong>Important: Ensure that you download the model weights before executing the scripts</strong>

```shell
git lfs install
git clone https://huggingface.co/gligen/gligen-inpainting-text-box
git clone https://huggingface.co/ground-a-video/unet3d_ckpts
git clone https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth
```

These commands will place the pretrained GLIGEN weights at:  
- `Ground-A-Video/gligen-inpainting-text-box/diffusion_pytorch_model.bin`  
- `Ground-A-Video/unet3d_ckpts/diffusion_pytorch_model.bin`  
- `Ground-A-Video/unet3d_ckpts/config.json`
- `Ground-A-Video/control_v11f1p_sd15_depth/diffusion_pytorch_model.bin`  
- `Ground-A-Video/control_v11f1p_sd15_depth/config.json`
  
Alternatively, you can manually download the weights using the web interface from the following links:
- [GLIGEN](https://huggingface.co/gligen/gligen-inpainting-text-box/tree/main)  
- [Ground A Video](https://huggingface.co/ground-a-video/unet3d_ckpts/tree/main)
- [ControlNet](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/tree/main)  

### Data
The input video frames should be stored in `video_images` , organized by each video's name.  
Pre-computed groundings, including bounding box coordinates and corresponding text annotations, for each video are available in configuration files located at `video_configs/{video_name}.yaml`


## Usage

### Inference
Ground-A-Video is designed to be a training-free framework. To run the inference script, use the following command:

```bash
python main.py --config configs/rabbit_watermelon.yaml --folder outputs/rabbit_watermelon
```
#### Arguments
- `--config`: Specifies the path to the configuration file. Modify the config files under `video_configs` as needed
- `--folder`: Designates the directory where output videos will be saved
- `--clip_length`: Sets the number of input video frames. Default is 8.
- `--denoising_steps`: Defines the number of denoising steps. Default is 50.
- `--ddim_inv_steps`: Determines the number of steps for per-frame DDIM inversion and Null-text Optimization. Default is 20.
- `--guidance_scale`: Sets the CFG scale. Default is 12.5.
- `--flow_smooth_threshold`: Threshold for optical flow guided smoothing. Default is 0.2.
- `--controlnet_conditioning_scale`: Sets the conditioning scale for ControlNet. Default is 1.0.
- `--nti`: Whether to perfrom Null-text Optimization after DDIM Inversion. Default is False.  
  (If your CUDA Version is 11.4, then you can set is as True. If your CUDA Version is 12.2 or higher, set it as False: The codes are implemented using fp16 dtypes but in 12.2 higher CUDA version, the gradient backpropagation incurs errors/)

## More Results
<table class="center">
  <tr>
    <td style="text-align:center;"><b>Input Videos</b></td>
    <td style="text-align:center;" colspan="1"><b>Output Videos</b></td>
  </tr>
  <tr>
    <td><img src="https://ground-a-video.github.io/static/gifs/cat_flower/input.gif" width="384" height="384"></td>
    <td><img src="https://ground-a-video.github.io/static/gifs/cat_flower/three3.gif" width="384" height="384"></td>
  </tr>
  <tr>
    <td><img src="https://ground-a-video.github.io/static/gifs/swan/input.gif" width="384" height="384"></td>
    <td><img src="https://ground-a-video.github.io/static/gifs/swan/blue_snowy_lagoon.gif" width="384" height="384"></td>
  </tr>
  <tr>
    <td><img src="https://ground-a-video.github.io/static/gifs/bird_forest/input.gif" width="384" height="384"></td>
    <td><img src="https://ground-a-video.github.io/static/gifs/bird_forest/output2.gif" width="384" height="384"></td>
  </tr>
  <tr>
    <td><img src="https://ground-a-video.github.io/static/gifs/surfing/input.gif" width="384" height="384"></td>
    <td><img src="https://ground-a-video.github.io/static/gifs/surfing/output2.gif" width="384" height="384"></td>
  </tr>
  <tr>
    <td><img src="https://ground-a-video.github.io/static/gifs/skiing/input.gif" width="384" height="384"></td>
    <td><img src="https://ground-a-video.github.io/static/gifs/skiing/output2.gif" width="384" height="384"></td>
  </tr>
</table>


## Citation
If you like our work, please cite our paper.

```bibtex

```

## Shoutouts
* Ground-A-Video builds upon huge open-source projects:<br>
  [diffusers](https://github.com/huggingface/diffusers), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion),
  <b>[GLIGEN](https://github.com/gligen/GLIGEN)</b>, [ControlNet](https://github.com/lllyasviel/ControlNet), [GLIP](https://github.com/microsoft/GLIP), [RAFT](https://github.com/princeton-vl/RAFT).
  <br>Thank you for open-sourcing!<br>
* Evaluation of Ground-A-Video was made possible thanks to open-sourced SOTA baselines:<br>
  [Tune-A-Video](https://github.com/showlab/Tune-A-Video), [Control-A-Video](https://github.com/Weifeng-Chen/control-a-video), [ControlVideo](https://github.com/YBYBZhang/ControlVideo) and RunwayML's web-based product [Gen-1](https://research.runwayml.com/gen1)<br>
