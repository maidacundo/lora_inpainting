from matplotlib.pyplot import step
import torch
from diffusers import (
    StableDiffusionInpaintPipeline
)
from typing import List
import wandb

def evaluate_pipe(
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        eval_dataset:torch.utils.data.Dataset,
        config,
    ):
    g_cuda = torch.Generator(device=config.device).manual_seed(config.seed)

    with torch.no_grad():
        pipe = StableDiffusionInpaintPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                )
        for prompt in config.eval.prompts:
            
            if config.prompt.global_caption:
                prompt = prompt + ', ' + config.prompt.global_caption
            images_log = {}
            
            for strength in config.eval.strengths:
                generations = []
                for example in eval_dataset:

                    image = example["instance_images"]
                    mask_image = example["instance_masks"]

                    generated_image = pipe(
                        prompt=prompt,
                        negative_prompt=config.prompt.negative_caption,
                        image=image,
                        mask_image=mask_image,
                        generator=g_cuda,
                        num_inference_steps=20,
                        height=image.shape[1],
                        width=image.shape[2],
                        strength=strength,
                        ).images[0]
                    
                    image = wandb.Image(generated_image)
                    generations.append(image)

            images_log[prompt] = generations
    return images_log