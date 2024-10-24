import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
import requests
import warnings
import functools
from omegaconf import OmegaConf
from torchvision import transforms

warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi")

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.extras import load_model_from_config
import clip
from huggingface_hub import hf_hub_download

# Load Image Mixer model
mixer_ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-full.ckpt")
mixer_config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

# Load Super Resolution model
sr_config = hf_hub_download(repo_id="lambdalabs/stable-diffusion-super-res", filename="sd-superres-config.yaml")
sr_ckpt = hf_hub_download(repo_id="lambdalabs/stable-diffusion-super-res", filename="sd-superres-pruned.ckpt")
decoder_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.ckpt")

device = "cuda:0"

# Initialize Image Mixer model
mixer_model = load_model_from_config(mixer_config, mixer_ckpt, device=device, verbose=False)
mixer_model = mixer_model.to(device).float()

# Initialize Super Resolution model
sr_config = OmegaConf.load(sr_config)
sr_model = load_model_from_config(sr_config, sr_ckpt, device=device)
decoder = torch.load(decoder_path, map_location='cpu')["state_dict"]
sr_model.first_stage_model.load_state_dict(decoder, strict=False)
sr_model.half()

# Load CLIP model
clip_model, preprocess = clip.load("ViT-L/14", device=device)

n_inputs = 2
torch.cuda.empty_cache()

def make_unc(model, n_samples, all_conds):
    uc_tmp = model.get_unconditional_conditioning(n_samples, [""])
    uc = dict()
    for k in all_conds:
        if k == "c_crossattn":
            assert isinstance(all_conds[k], list) and len(all_conds[k]) == 1
            uc[k] = [uc_tmp]
        elif k == "c_adm":
            assert isinstance(all_conds[k], torch.Tensor)
            uc[k] = torch.ones_like(all_conds[k]) * model.low_scale_model.max_noise_level
        elif isinstance(all_conds[k], list):
            uc[k] = [all_conds[k][i] for i in range(len(all_conds[k]))]
        else:
            uc[k] = all_conds[k]
    return uc

@torch.no_grad()
def super_resolve_image(image, target_res=2048, steps=50, prompt="high quality high resolution uhd 4k image"):
    # Convert PIL image to tensor
    input_im = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    # Calculate intermediate size
    current_size = min(image.size)
    intermediate_size = min(current_size * 2, target_res)
    
    input_im = transforms.Resize((intermediate_size, intermediate_size))(input_im)
    input_im = input_im * 2 - 1

    sampler = PLMSSampler(sr_model)
    
    with autocast("cuda"):
        c = sr_model.get_learned_conditioning([prompt])
        shape = [4, intermediate_size // 8, intermediate_size // 8]
        
        x_low = input_im.tile(1, 1, 1, 1)
        x_low = x_low.to(memory_format=torch.contiguous_format).half()
        
        if hasattr(sr_model, 'get_first_stage_encoding'):
            zx = sr_model.get_first_stage_encoding(sr_model.encode_first_stage(x_low))
            all_conds = {"c_concat": [zx], "c_crossattn": [c]}
            
            # Add noise level conditioning for PLMS sampler
            noise_level = torch.Tensor([sr_model.low_scale_model.max_noise_level]).to(device)
            all_conds["c_adm"] = noise_level
        else:
            zx = sr_model.low_scale_model.model.encode(x_low).sample()
            zx = zx * sr_model.low_scale_model.scale_factor
            noise_level = torch.Tensor([sr_model.low_scale_model.max_noise_level]).to(device)
            all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}
        
        # Create unconditional conditioning
        uc = make_unc(sr_model, 1, all_conds)
        
        samples_ddim, _ = sampler.sample(
            S=steps,
            conditioning=all_conds,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=uc,
            eta=0.0,
        )
        
        x_samples_ddim = sr_model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
        return Image.fromarray(x_sample.astype(np.uint8))

# Original Image Mixer functions with modifications
@functools.lru_cache()
def get_url_im(t):
    user_agent = {'User-agent': 'gradio-app'}
    response = requests.get(t, headers=user_agent)
    return Image.open(BytesIO(response.content))

@torch.no_grad()
def get_im_c(im_path, clip_model):
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()

@torch.no_grad()
def get_txt_c(txt, clip_model):
    text = clip.tokenize([txt,]).to(device)
    return clip_model.encode_text(text)

def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims

@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast", ddim_steps=50):
    ddim_eta = 0.0
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            x_T=start_code
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)

def run(*args):
    inps = []
    for i in range(0, len(args)-4, n_inputs):
        inps.append(args[i:i+n_inputs])

    scale, n_samples, seed, steps = args[-4:]
    h = w = 640

    sampler = DDIMSampler(mixer_model)
    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h//8, w//8, device=device)
    conds = []

    for b, t, im, s in zip(*inps):
        if b == "Image":
            this_cond = s*get_im_c(im, clip_model)
        elif b == "Text/URL":
            if t.startswith("http"):
                im = get_url_im(t)
                this_cond = s*get_im_c(im, clip_model)
            else:
                this_cond = s*get_txt_c(t, clip_model)
        else:
            this_cond = torch.zeros((1, 768), device=device)
        conds.append(this_cond)
    
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    try:
        # Generate initial images
        initial_images = sample(sampler, mixer_model, conds, 0*conds, scale, start_code, h=h, w=w, ddim_steps=steps)
        
        # Super resolve each image
        final_images = []
        for img in initial_images:
            # Apply super resolution once with larger step count
            img = super_resolve_image(img, steps=50)
            final_images.append(img)
        
        return final_images
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return None

# Gradio interface setup remains the same as in original script
import gradio as gr
from functools import partial
from itertools import chain

def change_visible(txt1, im1, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[txt1] = gr.update(visible=False)
    elif val == "Text/URL":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=True)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=False)
    return outputs

with gr.Blocks(title="Image Mixer with Super Resolution", css=".gr-box {border-color: #8136e2}") as demo:
    gr.Markdown("""
    # Image Mixer with Super Resolution
    _Created by [Justin Pinkney](https://www.justinpinkney.com) at [Lambda Labs](https://lambdalabs.com/). Enhanced with Super Resolution._
    """)

    btns = []
    txts = []
    ims = []
    strengths = []

    with gr.Row():
        with gr.Column():
            for i in range(n_inputs):
                with gr.Box():
                    with gr.Column():
                        btn1 = gr.Radio(
                            choices=["Image", "Text/URL", "Nothing"],
                            label=f"Input {i} type",
                            interactive=True,
                            value="Nothing",
                        )
                        txt1 = gr.Textbox(
                            label="Text or Image URL", visible=False, interactive=True
                        )
                        im1 = gr.Image(
                            label="Image",
                            interactive=True,
                            visible=False,
                            type="pil",
                        )
                        strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=4,
                            step=0.05,
                            value=1,
                            interactive=True,
                        )

                        fn = partial(change_visible, txt1, im1)
                        btn1.change(fn=fn, inputs=[btn1], outputs=[txt1, im1])

                        btns.append(btn1)
                        txts.append(txt1)
                        ims.append(im1)
                        strengths.append(strength)

        with gr.Column():
            with gr.Box():
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="CFG scale",
                        value=3.5,
                        minimum=1,
                        maximum=20,
                        step=0.5,
                    )
                    n_samples = gr.Slider(
                        label="Num samples", value=1, minimum=1, maximum=2, step=1
                    )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", value=0, minimum=0, maximum=5000, step=1
                    )
                    steps = gr.Slider(
                        label="Steps", value=40, minimum=10, maximum=100, step=5
                    )
            with gr.Row():
                submit = gr.Button("Generate Image")

            output = gr.Gallery().style(grid=[1, 2], height="auto")

    inps = list(chain(btns, txts, ims, strengths))
    inps.extend([cfg_scale, n_samples, seed, steps])
    submit.click(fn=run, inputs=inps, outputs=[output])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)