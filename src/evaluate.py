from matplotlib.pyplot import step
from numpy import dtype
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    logging,
)
from typing import List
import wandb
import gc
from pipeline_attention_inpainting import StableDiffusionAttentionStoreInpaintPipeline

logging.set_verbosity_error()

def evaluate_pipe(
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        eval_dataset:torch.utils.data.Dataset,
        config,
        attn_res=(32,32),
    ):
    g_cuda = torch.Generator(device=config.device).manual_seed(config.seed)

    with torch.no_grad():
        pipe = StableDiffusionAttentionStoreInpaintPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None, 
                )
        pipe.set_progress_bar_config(disable=True)

        images_log = {}
        for prompt in config.eval.prompts:
            if config.prompt.global_caption:
                prompt += ', ' + config.prompt.global_caption
            
            generations = []
            attention_maps = []
            
            for strength in config.eval.strengths:
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
                        attn_res=attn_res,
                        ).images[0]
                    
                    attention_plot = wandb.Image(pipe.save_plot_attention(prompt))
                    image = wandb.Image(generated_image)
                    generations.append(image)
                    attention_maps.append(attention_plot)

            images_log[prompt] = generations
            images_log['ATTN MAP: ' + prompt] = attention_maps

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images_log