"""Image Mixer - Gradio Web UI

Generates images from mixed text + image inputs using Stable Diffusion
with CLIP-based conditioning. Uses the lambdalabs/image-mixer model.

UI language: Italian (by design)
"""
import logging
import functools
import warnings
from io import BytesIO
from contextlib import nullcontext
from itertools import chain
from functools import partial

import torch
import numpy as np
import requests
from PIL import Image
from einops import rearrange
from huggingface_hub import hf_hub_download
import clip
import gradio as gr

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.extras import load_model_from_config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("image_mixer")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
DEVICE = "cuda"

logger.info("Downloading model from HuggingFace...")
ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-full.ckpt")
config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

logger.info("Loading diffusion model...")
model = load_model_from_config(config, ckpt, device=DEVICE, verbose=False)
model = model.to(DEVICE).float()

logger.info("Loading CLIP model (ViT-L/14)...")
clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

N_INPUTS = 2

torch.cuda.empty_cache()
logger.info("Models loaded successfully.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def get_url_im(url: str) -> Image.Image:
    """Fetch an image from a URL with caching."""
    user_agent = {"User-agent": "gradio-app"}
    response = requests.get(url, headers=user_agent, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


@torch.no_grad()
def get_im_c(im_path, clip_model):
    """Get CLIP image embedding."""
    prompts = preprocess(im_path).to(DEVICE).unsqueeze(0)
    return clip_model.encode_image(prompts).float()


@torch.no_grad()
def get_txt_c(txt: str, clip_model):
    """Get CLIP text embedding."""
    text = clip.tokenize([txt]).to(DEVICE)
    return clip_model.encode_text(text)


def to_im_list(x_samples_ddim: torch.Tensor) -> list[Image.Image]:
    """Convert model output tensor to list of PIL images."""
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims


@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast", ddim_steps=50):
    """Run DDIM sampling with optional autocast."""
    ddim_eta = 0.0
    precision_scope = (
        functools.partial(torch.amp.autocast, device_type="cuda")
        if precision == "autocast"
        else nullcontext
    )
    with precision_scope():
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
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)


def run(*args):
    """Main generation function called by Gradio."""
    inps = []
    for i in range(0, len(args) - 4, N_INPUTS):
        inps.append(args[i : i + N_INPUTS])

    scale, n_samples, seed, steps = args[-4:]
    n_samples = int(n_samples)
    seed = int(seed)
    steps = int(steps)
    h = w = 640

    sampler = DDIMSampler(model)

    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h // 8, w // 8, device=DEVICE)
    conds = []

    for b, t, im, s in zip(*inps):
        if b == "Immagine":
            this_cond = s * get_im_c(im, clip_model)
        elif b == "Testo/URL":
            if t.startswith("http"):
                im = get_url_im(t)
                this_cond = s * get_im_c(im, clip_model)
            else:
                this_cond = s * get_txt_c(t, clip_model)
        else:
            this_cond = torch.zeros((1, 768), device=DEVICE)
        conds.append(this_cond)

    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    logger.info(f"Generating {n_samples} image(s) | seed={seed} | steps={steps} | scale={scale}")
    ims = sample(sampler, model, conds, 0 * conds, scale, start_code, h=h, w=w, ddim_steps=steps)
    return ims


# ---------------------------------------------------------------------------
# Gradio UI (Italian)
# ---------------------------------------------------------------------------
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
            for i in range(N_INPUTS):
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
            with gr.Group():
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
            with gr.Row():
                submit = gr.Button("Genera immagine")

            output = gr.Gallery(
                label="Immagini Generate",
                columns=[1, 2],
                height=640,
                object_fit="contain",
            )

    inps = list(chain(btns, txts, ims, strengths))
    inps.extend([cfg_scale, n_samples, seed, steps])

    submit.click(
        fn=run,
        inputs=inps,
        outputs=[output],
        concurrency_limit=1,
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=True,
        max_threads=4,
    )
