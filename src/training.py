import os
import itertools
import wandb
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.autograd import Variable

from diffusers import StableDiffusionInpaintPipeline, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from peft import LoraConfig, LoraModel
from controlnet_aux import MLSDdetector

from .utils import set_random_seed, get_label_mapping, print_trainable_parameters, save_loras
from .data import InpaintLoraDataset, InpaintingDataLoader, download_roboflow_dataset
from .model import get_models
from .evaluate import evaluate_pipe
from .config import Config
from .losses import SSIM_loss, MS_SSIM_loss
from .model import DinoScorer

logging.set_verbosity_error()

def train(config: Config):
    download_roboflow_dataset(config)
    set_random_seed(config.seed)

    text_encoder, vae, unet, tokenizer, noise_scheduler = get_models(
        config.model.model_path,
        config.model.vae_path,
        device=config.device,
        load_from_safetensor=True,
    )

    label_mapping = get_label_mapping(os.path.join(config.dataset.data_root, "data.yaml"))

    train_dataset = InpaintLoraDataset(
        instance_data_root= os.path.join(config.dataset.data_root, "train"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=config.dataset.image_size,
        normalize=config.dataset.normalize_images,
        scaling_pixels=config.dataset.scaling_pixels,
        do_classifier_free_guidance=config.dataset.do_classifier_free_guidance,
    )

    valid_dataset = InpaintLoraDataset(
        instance_data_root=os.path.join(config.dataset.data_root, "valid"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=config.dataset.image_size,
        normalize=False,
        augmentation=False,
        is_val=True,
    )

    train_dataloader = InpaintingDataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        tokenizer=tokenizer,
        device=config.device,
    )
    valid_dataloader = InpaintingDataLoader(
        valid_dataset,
        batch_size=config.train.eval_batch_size,
        tokenizer=tokenizer,
        device=config.device,
    )

    # Freeze all weights
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if config.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.train.use_xformers and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if config.log_wandb:
        wandb.init(
            project=config.wandb.project_name, 
            entity=config.wandb.entity_name,
            config=config,
            id=config.wandb.run_name,
            tags=config.wandb.tags,
        )
        wandb.watch(unet)
        wandb.watch(text_encoder)

    if config.train.train_unet:
        unet_peft = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.unet_target_modules,
            lora_dropout=0.1,
            bias='none',
        )

        unet = LoraModel(unet, unet_peft, config.lora.unet_adapter_name)
        print_trainable_parameters(unet, "unet")

        params_to_optimize = [
            {
                "params": itertools.chain(unet.parameters()),
                "lr": config.train.unet_lr,
            },
        ]

    if config.train.train_text_encoder:
        text_encoder_peft = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.text_encoder_target_modules,
            lora_dropout=0.1,
            bias='none',
        )

        text_encoder = LoraModel(text_encoder, text_encoder_peft, config.lora.text_encoder_adapter_name)
        print_trainable_parameters(text_encoder, "text_encoder")

        params_to_optimize += [
            {
                "params": itertools.chain(text_encoder.parameters()),
                "lr": config.train.text_encoder_lr,
            }
        ]

    optimizer_lora = optim.AdamW(
        params_to_optimize,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    if config.train.train_unet:
        unet.train()
        
    if config.train.train_text_encoder:
        text_encoder.train()

    lr_scheduler_lora = get_scheduler(
        config.train.scheduler_type,
        optimizer=optimizer_lora,
        num_warmup_steps=config.train.scheduler_warmup_steps,
        num_training_steps=config.train.total_steps,
        num_cycles=config.train.scheduler_num_cycles,
    )

    if not os.path.exists(config.train.checkpoint_folder):
        os.makedirs(config.train.checkpoint_folder)
        print(f"Directory '{config.train.checkpoint_folder}' created.")
    else:
        print(f"Directory '{config.train.checkpoint_folder}' already exists.")

    loss_sum = 0.0
    progress_bar = tqdm(range(config.train.total_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    if config.train.text_encoder_train_ratio < 1.0:
        text_encoder_steps = math.ceil(config.train.total_steps * config.train.text_encoder_train_ratio)
    else:
        text_encoder_steps = config.train.total_steps
    
    mse = torch.nn.MSELoss(reduction='mean')
    ssim = SSIM_loss(data_range=1.0, size_average=True, channel=4)
    ms_ssim = MS_SSIM_loss(data_range=1.0, size_average=True, channel=4)

    if config.train.criterion == 'mse':
        criterion = mse
        print('Using MSE loss')
    elif config.train.criterion == 'ssim':
        criterion = ssim
        print('Using SSIM loss')
    elif config.train.criterion == 'ms_ssim':
        criterion = ms_ssim
        print('Using MS-SSIM loss')
    elif config.train.criterion == 'mse+ssim':
        def criterion(pred, target, alpha=0.3):
            return alpha * mse(pred, target) + (1-alpha) * ssim(pred, target)
        print('Using MSE + SSIM loss')

    if config.eval.compute_dino_score:
        dino_scorer = DinoScorer('facebook/dino-vits16', valid_dataset.imgs)

    for epoch in range(math.ceil(config.train.total_steps / len(train_dataloader))):
        unet.train()
        text_encoder.train()
        for batch in train_dataloader:
            optimizer_lora.zero_grad()
            model_pred, target = forward_step(
                batch,
                unet,
                vae,
                text_encoder,
                noise_scheduler,
                t_mutliplier=config.train.t_mutliplier,
                mixed_precision=True,
                mask_temperature=config.train.mask_temperature,
                loss_on_latent=config.train.loss_on_latent,
            )
            train_loss = criterion(model_pred.float(), target.float())
            train_loss.backward()
            loss_sum += train_loss.detach().item()

            if config.train.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(unet.parameters(), text_encoder.parameters()), 
                    max_norm=config.train.clip_gradients_max_norm
                )
            optimizer_lora.step()
            lr_scheduler_lora.step()
            progress_bar.update(1)
            logs = {
                config.train.criterion: train_loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            global_step += 1
            if text_encoder_steps == global_step:
                text_encoder.train(False)
                text_encoder.requires_grad_(False)
                unet.train(True)
                print('-'*50)
                print("Text encoder training finished")
                print_trainable_parameters(text_encoder, "text_encoder")
                print('-'*50)

        if config.log_wandb:
            logs = {
                "train_"+config.train.criterion: loss_sum / len(train_dataloader),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
        loss_sum = 0.0

        if epoch % config.train.eval_every_n_epochs == 0:
            
            unet.eval()
            text_encoder.eval()


            mse_loss = 0.0
            ssim_loss = 0.0
            ms_ssim_loss = 0.0

            # set the number of timesteps to 20 for evaluation
            num_train_timesteps = noise_scheduler.config.num_train_timesteps
            noise_scheduler.config.num_train_timesteps = 20

            for _ in range(config.eval.eval_epochs):
                for batch in valid_dataloader:
                    with torch.no_grad():
                        model_pred, target = forward_step(
                                                batch,
                                                unet,
                                                vae,
                                                text_encoder,
                                                noise_scheduler,
                                                t_mutliplier=config.train.t_mutliplier,
                                                mixed_precision=True,
                                                mask_temperature=config.train.mask_temperature,
                                                loss_on_latent=False,
                                            )
                        mse_loss += mse(model_pred.float(), target.float()).detach().item()
                        ssim_loss += ssim(model_pred.float(), target.float()).detach().item()
                        ms_ssim_loss += ms_ssim(model_pred.float(), target.float()).detach().item()


            # reset the number of timesteps
            noise_scheduler.config.num_train_timesteps = num_train_timesteps

            logs['val_mse'] = mse_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            logs['val_ssim'] = ssim_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            logs['val_ms_ssim'] = ms_ssim_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            
            loss_sum = 0.0
            if config.log_wandb:
                evaluation_logs = evaluate_pipe(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    valid_dataset=valid_dataset,
                    config=config,
                    dino_scorer=dino_scorer if config.eval.compute_dino_score else None,
                )
                wandb.log(evaluation_logs, step=global_step)
            save_path = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_lora_{global_step}.safetensors')
            save_loras(
                unet if config.train.train_unet else None, 
                text_encoder if config.train.train_text_encoder else None, 
                save_path, 
                config
            )
        wandb.log(logs, step=global_step)
    wandb.finish()

def forward_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    vae_scale_factor=8,
    loss_on_latent=False,
):
    weight_dtype = torch.float32

    # encode the image
    latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
        ).latent_dist.sample()


    # encode the masked image
    masked_image_latents = vae.encode(
                batch["masked_image_values"].to(dtype=weight_dtype).to(unet.device)
            ).latent_dist.sample()
    masked_image_latents = masked_image_latents * vae.config.scaling_factor
    latents = latents * vae.config.scaling_factor

    # scale the mask
    mask = F.interpolate(
                batch["mask"].to(dtype=weight_dtype).to(unet.device),
                scale_factor=1 / 8,
            )

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    latent_model_input = torch.cat(
            [noisy_latents, mask, masked_image_latents], dim=1
        )

    if mixed_precision:
        with torch.cuda.amp.autocast():

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
    else:
        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(text_encoder.device)
        )[0]

        model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:

        mask = (
            batch["mask"]
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0], 1, model_pred.shape[2] * vae_scale_factor, model_pred.shape[3] * vae_scale_factor
            )
        )
        # resize to match model_pred
        mask = F.interpolate(
            mask.float(),
            size=model_pred.shape[-2:],
            mode="nearest",
        )

        mask = mask.pow(mask_temperature)

        mask = mask / mask.max()

        model_pred = model_pred * mask

        target = target * mask

    if loss_on_latent:
        target = scheduler.add_noise(latents, target, timesteps)
        model_pred = scheduler.add_noise(latents, model_pred, timesteps)


    return model_pred, target


