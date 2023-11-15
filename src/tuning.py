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
from .config import Config, LoraConfig
from .losses import SSIM_loss, MS_SSIM_loss
from training import loss_step

import optuna
from functools import partial 

def objective(trial, config: Config, text_encoder, tokenizer, vae, unet, noise_scheduler):

    set_random_seed(config.seed)

    label_mapping = get_label_mapping(os.path.join(config.dataset.data_root, "data.yaml"))

    train_dataset = InpaintLoraDataset(
        instance_data_root= os.path.join(config.dataset.data_root, "train"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=config.dataset.image_size,
        normalize=config.dataset.normalize_images,
        scaling_pixels=config.dataset.scaling_pixels,
    )

    valid_dataset = InpaintLoraDataset(
        instance_data_root=os.path.join(config.dataset.data_root, "valid"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=config.dataset.image_size,
        normalize=False,
        augmentation=False,
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

    unet_target_modules_suggestions = [
        ["ff.net.0.proj"],
        ["to_q", "to_v", "ff.net.0.proj"],
        ["to_q", "to_v", "proj_in", "conv1", "conv2"],
        ["to_q", "to_v", "to_k", "to_out.0", "ff.net.0.proj"],
        ["to_q", "to_v", "ff.net.0.proj", "proj_in", "conv1", "conv2"],
    ]

    text_encoder_target_modules_suggestions = [
        ["q_proj", "v_proj"],
        ["q_proj", "v_proj", "k_proj", "out_proj"],
        ["q_proj", "v_proj", "mlp.fc1"],
        ["q_proj", "v_proj", "mlp.fc1", "mlp.fc2"],
        ["mlp.fc1", "mlp.fc2"],
    ]
    # LoraConfig
    lora_config = LoraConfig(
        rank=trial.suggest_categorical('rank', [8, 16, 32]),
        alpha=trial.suggest_categorical('alpha', [1, 8, 16, 32]),
        unet_target_modules=trial.suggest_categorical('unet_target_modules', unet_target_modules_suggestions),
        text_encoder_target_modules=trial.suggest_categorical('text_encoder_target_modules', text_encoder_target_modules_suggestions),
    )

    config.train.train_unet = trial.suggest_categorical('train_unet', [True, False])
    config.train.train_text_encoder = trial.suggest_categorical('train_text_encoder', [True, False])
    config.train.text_encoder_train_ratio = trial.suggest_uniform('text_encoder_train_ratio', 0.0, 1.0)
    config.lora = lora_config
    config.wandb.run_name = f"lora_{lora_config.rank}_{lora_config.alpha}_{str(lora_config.unet_target_modules)}_{str(lora_config.text_encoder_target_modules)}"
    
    if config.log_wandb:
        wandb.init(
            project=config.wandb.project_name, 
            entity=config.wandb.entity_name,
            config=config,
            id=config.wandb.run_name,
            tags=config.wandb.tags,
        )

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

    loss_sum = 0.0
    progress_bar = tqdm(range(config.train.total_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    if config.train.text_encoder_train_ratio < 1.0:
        text_encoder_steps = math.ceil(config.train.total_steps * config.train.text_encoder_train_ratio)
    else:
        text_encoder_steps = config.train.total_steps

    if config.train.criterion == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif config.train.criterion == 'ssim':
        criterion = SSIM_loss(data_range=1.0, size_average=True, channel=4)
    elif config.train.criterion == 'ms_ssim':
        criterion = MS_SSIM_loss(data_range=1.0, size_average=True, channel=4)

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
                criterion=criterion,
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
                "loss": loss_lora.detach().item(),
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
                "loss_lora": loss_sum / len(train_dataloader),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            loss_sum = 0.0

        if epoch % config.train.eval_every_n_epochs == 0:

            loss_sum = 0.0
            # evaluate the unet
            unet.eval()
            text_encoder.eval()

            # set the number of timesteps to 20 for evaluation
            num_train_timesteps = noise_scheduler.config.num_train_timesteps
            noise_scheduler.config.num_train_timesteps = 20

            for _ in range(config.eval.eval_epochs):
                for batch in valid_dataloader:
                    with torch.no_grad():
                        val_loss = loss_step(
                            batch,
                            unet,
                            vae,
                            text_encoder,
                            noise_scheduler,
                            t_mutliplier=1,
                            mixed_precision=True,
                            mask_temperature=config.train.mask_temperature,
                            criterion=criterion,
                        )
                        loss_sum += val_loss.detach().item()

            # reset the number of timesteps
            noise_scheduler.config.num_train_timesteps = num_train_timesteps
            
            val_loss = loss_sum / (len(valid_dataloader) * config.eval.eval_epochs)
            logs['val_loss'] = val_loss
            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            loss_sum = 0.0
            
        wandb.log(logs, step=global_step)
    wandb.finish()
    # at the end of the training the model is unloaded
    unet.unload()
    text_encoder.unload()

def optimize(config, n_trials=100):
    study = optuna.create_study(
        study_name="study",
        direction='minimize',
    )

    download_roboflow_dataset(config)

    text_encoder, vae, unet, tokenizer, noise_scheduler = get_models(
        config.model.model_path,
        config.model.vae_path,
        device=config.device,
        load_from_safetensor=True,
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

    config.train.total_steps = 200
    config.train.scheduler_warmup_steps = 0

    objective = partial(objective, 
                        config=config, 
                        text_encoder=text_encoder, 
                        tokenizer=tokenizer, 
                        vae=vae, 
                        unet=unet, 
                        noise_scheduler=noise_scheduler)
    
    study.optimize(objective, n_trials=100, gc_after_trial=True)