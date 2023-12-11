from typing import List, Optional
import re

from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
)
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)

import torch

def get_models(
    pretrained_model_name_or_path: str,
    pretrained_vae_name_or_path: str,
    new_tokens: Optional[List[str]] = None,
    initializer_tokens: Optional[List[str]] = None,
    device: str = "cuda",
    load_from_safetensor=False,
) -> [CLIPTextModel, AutoencoderKL, UNet2DConditionModel, CLIPTokenizer, DDIMScheduler]:
    if load_from_safetensor:

        print('loading VAE...')
        vae = AutoencoderKL.from_single_file(
            pretrained_vae_name_or_path,
        )
        print('loading model...')
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            pretrained_model_name_or_path,
            vae=vae,
        )
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        )

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        )

        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

    
    
    # Add placeholder tokens to tokenizer
    new_token_ids = []
    if new_tokens:
        print('initial tokens:', len(tokenizer))
        for token, init_tok in zip(new_tokens, initializer_tokens):
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )

            new_token_id = tokenizer.convert_tokens_to_ids(token)

            new_token_ids.append(new_token_id)

            # Load models and create wrapper for stable diffusion

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            if init_tok.startswith("<rand"):
                # <rand-"sigma">, e.g. <rand-0.5>
                sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

                token_embeds[new_token_id] = (
                    torch.randn_like(token_embeds[0]) * sigma_val
                )
                print(
                    f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[new_token_id].mean().item():.3f} +- {token_embeds[new_tokens].std().item():.3f}"
                )
                print(f"Norm : {token_embeds[new_token_id].norm():.4f}")

            elif init_tok == "<zero>":
                token_embeds[new_token_id] = torch.zeros_like(token_embeds[0])
            else:
                token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")

                initializer_token_id = token_ids[0]
                token_embeds[new_token_id] = token_embeds[initializer_token_id]
            print(f"Initialized {token} with {init_tok}")
        
        print('new tokens:', len(tokenizer))
    
    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        scheduler,
        new_token_ids,
    )