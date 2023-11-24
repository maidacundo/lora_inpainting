from typing import List
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from sympy import li
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from .data import ImageDataset, ImageDataloader

def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
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


    """
    # Add placeholder tokens to tokenizer
    placeholder_token_ids = []
    if placeholder_tokens:
        for token, init_tok in zip(placeholder_tokens, initializer_tokens):
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )

            placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

            placeholder_token_ids.append(placeholder_token_id)

            # Load models and create wrapper for stable diffusion

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            if init_tok.startswith("<rand"):
                # <rand-"sigma">, e.g. <rand-0.5>
                sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

                token_embeds[placeholder_token_id] = (
                    torch.randn_like(token_embeds[0]) * sigma_val
                )
                print(
                    f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
                )
                print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

            elif init_tok == "<zero>":
                token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
            else:
                token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")

                initializer_token_id = token_ids[0]
                token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    """

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        scheduler,
    )


class DinoScorer(nn.Module):
    def __init__(
            self, dino_model: str, 
            ref_images: List[str],
            ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.model = AutoModel.from_pretrained(dino_model)
        
        self.model.eval()
        self.model.to('cuda')
        ref_images_dataset = ImageDataset(
            images=[Image.open(img) for img in ref_images]
        )
        self.ref_images_embedding = self.get_images_embedding(ref_images_dataset)

    def get_images_embedding(self, images_dataset):
        data_loader = ImageDataloader(images_dataset, self.processor)
        embeddings = []
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(**batch['inputs'].to('cuda'))
                embeddings.append(outputs.last_hidden_state[:, 0])
        return torch.cat(embeddings)

    def forward(self, images):
        if isinstance(images[0], str):
            images_dataset = ImageDataset(
                images=[Image.open(img) for img in images]
            )
        elif isinstance(images[0], Image.Image):
            images_dataset = ImageDataset(images=images)
        else:
            raise ValueError('images must be a list of str or Image.Image')

        embeddings = self.get_images_embedding(images_dataset)

        # compute all-pairs cosine similarity between ref images and generated images
        # shape: (num_ref_images, num_generated_images)
        sim = F.cosine_similarity(embeddings.unsqueeze(0), self.ref_images_embedding.unsqueeze(1), dim=-1)

        # compute the mean of cosine similarity
        return sim.mean().cpu().numpy()
        
        

    