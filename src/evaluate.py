from matplotlib.pyplot import step
from numpy import dtype
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    logging,
)
from typing import List, Optional
import wandb
import gc

from .metrics import DinoScorer, FIDScorer
from .config import Config
from .pipeline_attention_inpainting import StableDiffusionAttentionStoreInpaintPipeline

logging.set_verbosity_error()

def evaluate_pipe(
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        dataset: torch.utils.data.Dataset,
        config: Config,
        dino_scorer: Optional[DinoScorer] = None,
        fid_scorer: Optional[FIDScorer] = None,
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

        evaluation_logs = {}

        if config.eval.compute_dino_score:
            gen_imgs = []

        for prompt in config.eval.prompts:
            if config.prompt.global_caption:
                prompt += ', ' + config.prompt.global_caption
            
            gen_imgs_to_log = []
            if config.eval.log_attention_maps:
                attention_maps = []
            
            for strength in config.eval.strengths:
                for example in dataset:

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

                    generated_images = pipe(
                        generator=g_cuda,
                        num_inference_steps=20,
                        strength=strength,
                        num_images_per_prompt=config.eval.num_images_per_prompt,
                        **kwargs,
                        ).images
                    
                    gen_imgs_to_log.append(wandb.Image(generated_images[0]))

                    if config.eval.log_attention_maps:
                        attention_plot = wandb.Image(pipe.save_plot_attention(prompt))
                        attention_maps.append(attention_plot)
                        pipe.attention_store.reset()
                        del pipe.attention_store
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    if dino_scorer or fid_scorer:
                        for img in generated_images:
                            gen_imgs.append(img)

            # log the generated images for each prompt  
            evaluation_logs[prompt] = gen_imgs_to_log

            # log the attention maps for each prompt
            if config.eval.log_attention_maps:
                evaluation_logs['ATTN MAP: ' + prompt] = attention_maps

        # compute the DINO score for each prompt
        if dino_scorer:
            dino_score = dino_scorer(gen_imgs)
            evaluation_logs['DINO Score'] = dino_score
            print('DINO Score:', dino_score)

        # compute the FID score for each prompt
        if fid_scorer:
            fid_score = fid_scorer(gen_imgs)
            evaluation_logs['FID Score'] = fid_score
            print('FID Score:', fid_score)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return evaluation_logs