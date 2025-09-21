from io import BytesIO
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
import requests
import warnings
import functools

warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi") # Filter out specific warning

# Fix for Pydantic schema generation error with Starlette request objects
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

# Import the patch before importing gradio
try:
    import pydantic_patch
except ImportError:
    print("Pydantic patch not found, continuing without it")

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.extras import load_model_from_config, load_training_dir
import clip

from PIL import Image

from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-full.ckpt")
config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

device = "cuda:0"
model = load_model_from_config(config, ckpt, device=device, verbose=False)
model = model.to(device).float()

clip_model, preprocess = clip.load("ViT-L/14", device=device)

n_inputs = 2

torch.cuda.empty_cache()

@functools.lru_cache()
def get_url_im(t):
    user_agent = {'User-agent': 'gradio-app'}
    response = requests.get(t, headers=user_agent)
    return Image.open(BytesIO(response.content))

@torch.no_grad()
def get_im_c(im_path, clip_model):
    # im = Image.open(im_path).convert("RGB")
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()

@torch.no_grad()
def get_txt_c(txt, clip_model):
    text = clip.tokenize([txt,]).to(device)
    return clip_model.encode_text(text)

def get_txt_diff(txt1, txt2, clip_model):
    return get_txt_c(txt1, clip_model) - get_txt_c(txt2, clip_model)

def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims

@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast",ddim_steps=50):
    ddim_eta=0.0
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)

def run(*args):

    inps = []
    for i in range(0, len(args)-4, n_inputs):
        inps.append(args[i:i+n_inputs])

    scale, n_samples, seed, steps = args[-4:]
    h = w = 640

    sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)

    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h//8, w//8, device=device)
    conds = []

    for b, t, im, s in zip(*inps):
        if b == "Immagine":
            this_cond = s*get_im_c(im, clip_model)
        elif b == "Testo/URL":
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

    ims = sample(sampler, model, conds, 0*conds, scale, start_code, h=h, w=w, ddim_steps=steps)
    # return make_row(ims)
    return ims


# Import gradio after the patch is in place
import gradio as gr
from functools import partial
from itertools import chain

def change_visible(txt1, im1, val):
    outputs = {}
    if val == "Immagine":
        outputs[im1] = gr.update(visible=True)
        outputs[txt1] = gr.update(visible=False)
    elif val == "Testo/URL":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=True)
    elif val == "Niente":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=False)
    return outputs


css = """
.gr-group {border: 1px solid #8136e2; border-radius: 10px; padding: 15px}
.gr-form {border-color: #8136e2}
"""

with gr.Blocks(title="Mixer di Immagini", css=css) as demo:

    gr.Markdown("""
    # Mixer di Immagini
    _Creato da [Justin Pinkney](https://www.justinpinkney.com) a [Lambda Labs](https://lambdalabs.com/). Rinnovato da @3clyp50_
    """)

    btns = []
    txts = []
    ims = []
    strengths = []

    with gr.Row():
        with gr.Column():  # Input column
            for i in range(n_inputs):
                with gr.Group():
                    with gr.Column():
                        btn1 = gr.Radio(
                            choices=["Immagine", "Testo/URL", "Niente"],
                            label=f"Tipo di input {i}",
                            interactive=True,
                            value="Niente",
                        )
                        txt1 = gr.Textbox(
                            label="Testo o URL Immagine", visible=False, interactive=True
                        )
                        im1 = gr.Image(
                            label="Immagine",
                            interactive=True,
                            visible=False,
                            type="pil",
                        )
                        strength = gr.Slider(
                            label="Importanza",
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

        with gr.Column():  # Settings and Gallery column
            with gr.Group():  # Settings Group
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="Scala CFG",
                        value=3.5,
                        minimum=1,
                        maximum=20,
                        step=0.5,
                    )
                    n_samples = gr.Slider(
                        label="Numero campioni", value=1, minimum=1, maximum=2, step=1
                    )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", value=0, minimum=0, maximum=5000, step=1
                    )
                    steps = gr.Slider(
                        label="Passi", value=40, minimum=10, maximum=100, step=5
                    )
            with gr.Row():  # Submit button row
                submit = gr.Button("Genera immagine")

            # Updated Gallery component for Gradio 4.x
            output = gr.Gallery(
                label="Immagini Generate",
                columns=[1, 2],
                height=640,
                object_fit="contain"
            )

    inps = list(chain(btns, txts, ims, strengths))
    inps.extend([cfg_scale, n_samples, seed, steps])
    
    # Set concurrency limit directly on the event listener
    submit.click(
        fn=run, 
        inputs=inps, 
        outputs=[output],
        concurrency_limit=1  # Limits to 1 concurrent execution
    )

# Launch the app with max_threads parameter
demo.launch(
    server_name="0.0.0.0", 
    server_port=7860, 
    show_api=False, 
    share=True,
    max_threads=4  # Controls the total number of threads used for processing
)
