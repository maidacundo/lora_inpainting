from typing import List
from diffusers import StableDiffusionInpaintPipeline
from safetensors import safe_open
import torch

def load_textual_inversion(
    text_inversion_path: str,
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
):
    pipe = StableDiffusionInpaintPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=None,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.load_textual_inversion(
        pretrained_model_name_or_path=text_inversion_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler
    del pipe
    return tokenizer, text_encoder, vae, unet, scheduler

def replace_textual_inversion(
    text_inversion_path: str,
    new_tokens: List[str],
    new_tokens_ids: List[int],
    tokenizer,
    text_encoder,
):  
    with safe_open(text_inversion_path, framework="pt", device=0) as f:
        for k in f.keys():
            if k in new_tokens:
                idx_update = new_tokens_ids[new_tokens.index(k)]
                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight[idx_update] = f.get_tensor(k)
                print(f"Updated {k} with {idx_update}")
    return tokenizer, text_encoder