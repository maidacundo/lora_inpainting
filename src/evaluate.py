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

from .config import Config
from .pipeline_attention_inpainting import StableDiffusionAttentionStoreInpaintPipeline

logging.set_verbosity_error()

def evaluate_pipe(
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        eval_dataset: torch.utils.data.Dataset,
        config: Config,
        attn_res=(32,32),
    ):
    g_cuda = torch.Generator(device=config.device).manual_seed(config.seed)

    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():

        if config.eval.log_attention_maps:

            pipe = StableDiffusionAttentionStoreInpaintPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        scheduler=noise_scheduler,
                        safety_checker=None,
                        feature_extractor=None, 
                    )
        else:
            pipe = StableDiffusionInpaintPipeline(
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
            if config.eval.log_attention_maps:
                attention_maps = []
            
            for strength in config.eval.strengths:
                for example in eval_dataset:

                    image = example["instance_images"]
                    mask_image = example["instance_masks"]

                    # add the arg attn_res if config.eval.log_attention_maps is True
                    kwargs = {
                        "prompt": prompt,
                        "negative_prompt": config.prompt.negative_caption,
                        "image": image,
                        "mask_image": mask_image,
                        "height": image.shape[1],
                        "width": image.shape[2],
                    }
                    if config.eval.log_attention_maps:
                        kwargs['attn_res'] = attn_res

                    generated_image = pipe(
                        generator=g_cuda,
                        num_inference_steps=20,
                        strength=strength,
                        num_images_per_prompt=config.eval.num_images_per_prompt,
                        **kwargs,
                        ).images[0]
                    
                    
                    image = wandb.Image(generated_image)
                    generations.append(image)

                    if config.eval.log_attention_maps:
                        attention_plot = wandb.Image(pipe.save_plot_attention(prompt))
                        attention_maps.append(attention_plot)
                        pipe.attention_store.reset()
                        del pipe.attention_store
                        gc.collect()
                        torch.cuda.empty_cache()

            images_log[prompt] = generations
            if config.eval.log_attention_maps:
                images_log['ATTN MAP: ' + prompt] = attention_maps

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images_log