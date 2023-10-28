import os
import itertools
import wandb
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from .utils import set_random_seed, get_label_mapping
from .data import InpaintLoraDataset, InpaintingDataLoader, download_roboflow_dataset
from .model import get_models
from .lora import inject_trainable_lora, UNET_DEFAULT_TARGET_REPLACE, TEXT_ENCODER_DEFAULT_TARGET_REPLACE, save_all
from .evaluate import evaluate_pipe
from .config import Config


def train(config: Config):
    download_roboflow_dataset(config)
    set_random_seed(config.seed)

    text_encoder, vae, unet, tokenizer, noise_scheduler, placeholder_token_ids = get_models(
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
        resize=True,
        normalize=config.dataset.normalize_images,
        scaling_pixels=config.dataset.scaling_pixels,
    )

    valid_dataset = InpaintLoraDataset(
        instance_data_root=os.path.join(config.dataset.data_root, "valid"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=config.dataset.image_size,
        resize=True,
        normalize=False,
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
        unet_lora_params, unet_lora_params_names = inject_trainable_lora(
            unet,
            r=config.lora.rank,
            target_replace_module=UNET_DEFAULT_TARGET_REPLACE,
            dropout_p=config.train.lora_dropout_p,
            scale=config.train.lora_scale,
        )

        params_to_optimize = [
            {
                "params": itertools.chain(*unet_lora_params), 
                "lr": config.train.unet_lr
            },
        ]

    if config.train.train_text_encoder:
        text_encoder_lora_params, text_encoder_lora_params_names = inject_trainable_lora(
            text_encoder,
            r=config.lora.rank,
            target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
            dropout_p=config.lora.dropout_p,
            scale=config.lora.scale,
        )

        params_to_optimize += [
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": config.train.text_encoder_lr,
            }
        ]

    optimizer_lora = optim.AdamW(
        params_to_optimize,
        lr=config.train.learning_rate,
        weight_decay=config.train.adam_weight_decay,
        eps=config.traing.adam_epsilon,
    )

    if config.train.train_unet:
        unet.train()
        
    if config.train.train_text_encoder:
        text_encoder.train()

    unet_params_num = sum(p.numel() for p in unet.parameters() if p.requires_grad) 
    text_encoder_params_num = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    
    print('Unet LoRA params:', unet_params_num)
    print('CLIP LoRA params:', text_encoder_params_num)

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

    for epoch in range(math.ceil(config.train.total_steps / len(train_dataloader))):
        for batch in train_dataloader:
            optimizer_lora.zero_grad()
            loss_lora = loss_step(
                batch,
                unet,
                vae,
                text_encoder,
                noise_scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=config.train.mask_temperature,
            )
            loss_lora.backward()
            loss_sum += loss_lora.detach().item()

            if config.train.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(unet.parameters(), text_encoder.parameters()), 
                    max_norm=config.train.clip_gradients_max_norm
                )
            optimizer_lora.step()
            lr_scheduler_lora.step()
            progress_bar.update(1)
            logs = {
                "loss_lora": loss_lora.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

        if config.log_wandb:
            logs = {
                "loss_lora": loss_sum / len(train_dataloader),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            loss_sum = 0.0

        if epoch % config.train.eval_every_n_epochs == 0:

            loss_sum = 0.0
            # evaluate the unet
            unet.eval()
            text_encoder.eval()
            for _ in range(config.eval.eval_epochs):
                for batch in valid_dataloader:
                    with torch.no_grad():
                        val_loss = loss_step(
                            batch,
                            unet,
                            vae,
                            text_encoder,
                            noise_scheduler,
                            t_mutliplier=0.8,
                            mixed_precision=True,
                            mask_temperature=config.train.mask_temperature,
                        )
                        loss_sum += val_loss.detach().item()

            logs['val_loss'] = loss_sum / (len(valid_dataloader) * config.eval.eval_epochs)
            loss_sum = 0.0
            if config.log_wandb:
                images_log = evaluate_pipe(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    noise_scheduler,
                    valid_dataset,
                    config,
                )
                wandb.log(images_log, step=global_step)
            save_path = os.path.join(config.train.checkpoint_folder, f'lora_{epoch}.safetensors')
            save_all(
                unet,
                text_encoder,
                save_path,
                placeholder_token_ids=None,
                placeholder_tokens=None,
                save_lora=True,
                save_ti=False,
                target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                target_replace_module_unet=UNET_DEFAULT_TARGET_REPLACE,
                safe_form=True,
            )
        wandb.log(logs, step=global_step)
    wandb.finish()

def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    vae_scale_factor=8,
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

    loss = (
        F.mse_loss(model_pred.float(), target.float(), reduction="none")
        .mean([1, 2, 3])
        .mean()
    )

    return loss

