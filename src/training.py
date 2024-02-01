import os
import itertools
from pyexpat import model
import wandb
import math
from tqdm import tqdm
import gc

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from peft import LoraConfig, LoraModel

from .timesteps_scheduler import TimestepScheduler

from .utils import set_random_seed, get_label_mapping, print_trainable_parameters, save_loras, save_textual_inversion
from .ti_utils import replace_textual_inversion, load_textual_inversion
from .data import InpaintLoraDataset, InpaintingDataLoader, download_roboflow_dataset
from .model import get_models
from .evaluate import evaluate_pipe
from .config import Config
from .losses import SSIM_loss, MS_SSIM_loss, MLSD_Perceptual_loss
from .metrics import DinoScorer, FIDScorer

logging.set_verbosity_error()

def train(config: Config):
    download_roboflow_dataset(config)
    set_random_seed(config.seed)

    text_encoder, vae, unet, tokenizer, noise_scheduler, new_token_ids = get_models(
        config.model.model_path,
        config.model.vae_path,
        device=config.device,
        load_from_safetensor=True,
        new_tokens=config.train.new_tokens,
        initializer_tokens=config.train.initializer_tokens,
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

    if not os.path.exists(os.path.join(config.dataset.data_root, "valid")):
        raise ValueError("Valid dataset not found. Please create a test dataset.")

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

    if not os.path.exists(os.path.join(config.dataset.data_root, "test")):
        raise ValueError("Test dataset not found. Please create a test dataset.")

    test_dataset = InpaintLoraDataset(
        instance_data_root=os.path.join(config.dataset.data_root, "test"),
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        global_caption=config.prompt.global_caption,
        size=512,
        normalize=False,
        augmentation=False,
        is_val=True,
    )

    train_dataloader = InpaintingDataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        tokenizer=tokenizer,
        device=config.device,
        shuffle=True,
    )
    valid_dataloader = InpaintingDataLoader(
        valid_dataset,
        batch_size=config.train.eval_batch_size,
        tokenizer=tokenizer,
        device=config.device,
        shuffle=False,
    )

    dino_scorer = None
    if config.eval.compute_dino_score:
        dino_scorer = DinoScorer('facebook/dino-vits16', valid_dataset.imgs)
    
    fid_scorer = None
    if config.eval.compute_fid_score:
        fid_scorer = FIDScorer(valid_dataset.imgs)

    timesteps_scheduler = None
    if config.train.use_timestep_scheduler:
        timesteps_scheduler = TimestepScheduler(
            change_every_n_steps=config.train.timestep_scheduler_change_every_n_steps,
            fixed_bounds_idx=config.train.timestep_scheduler_fixed_bounds,
        )

    # perform the textual inversion.
    if len(new_token_ids) > 0:

        if config.train.load_textual_embeddings:
            raise ValueError("Cannot load textual embeddings when new tokens are added.")

        print("New tokens added to tokenizer:")
        for token_id in new_token_ids:
            print(f"{tokenizer.decode([token_id])}")

        train_inversion(
            new_token_ids,
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            train_dataloader,
            valid_dataloader,
            test_dataset,
            dino_scorer,
            fid_scorer,
            timesteps_scheduler,
            config,
        )
        print("-" * 50)

        while True:
            selected_checkpoint = input("Select checkpoint to load: (type 'exit' to skip and continue with last checkpoint)")
            if selected_checkpoint == 'exit':
                break
            selected_checkpoint = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_ti_{selected_checkpoint}.safetensors')
            if os.path.exists(selected_checkpoint):
                tokenizer, text_encoder = replace_textual_inversion(
                    selected_checkpoint,
                    config.train.new_tokens,
                    new_token_ids,
                    tokenizer,
                    text_encoder,
                )
                print(len)
                print("Textual inversion loaded successfully.")
                print("Using checkpoint:", selected_checkpoint)
                break
            print("Checkpoint not found. Please try again.")
        
        print("-" * 50)

    if config.train.load_textual_embeddings and len(new_token_ids) == 0:
        tokenizer, text_encoder, vae, unet, noise_scheduler = load_textual_inversion(
            config.train.load_textual_embeddings,
            tokenizer,
            text_encoder,
            vae,
            unet,
            noise_scheduler,
        )
    
    train_lora(
        unet,
        vae,
        text_encoder,
        tokenizer,
        noise_scheduler,
        train_dataloader,
        valid_dataloader,
        test_dataset,
        dino_scorer,
        fid_scorer,
        timesteps_scheduler,
        config,
    )

def train_inversion(
    new_token_ids,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    train_dataloader,
    valid_dataloader,
    test_dataset,
    dino_scorer,
    fid_scorer,
    timesteps_scheduler,
    config: Config,
):  
    print("Performing textual inversion...")
    original_embeds = text_encoder.get_input_embeddings().weight.data.clone()

    index_no_updates = torch.arange(len(tokenizer)) != -1 # set all to True
    for token_id in new_token_ids:
        index_no_updates[token_id] = False # set new tokens to False

    index_updates = ~index_no_updates
    
    # Freeze all weights
    unet.requires_grad_(False)
    vae.requires_grad_(False)

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
        run_name = 'ti_' + config.wandb.run_name if config.wandb.run_name else None
        wandb.init(
            project=config.wandb.project_name, 
            entity=config.wandb.entity_name,
            config=config,
            id=run_name,
            tags='inversion',
        )

    params_to_optimize = text_encoder.get_input_embeddings().parameters()

    print_trainable_parameters(text_encoder, "text_encoder")

    optimizer_inversion = optim.AdamW(
        params_to_optimize,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    if config.train.train_unet:
        unet.train()
        
    if config.train.train_text_encoder:
        text_encoder.train()

    lr_scheduler_inversion = get_scheduler(
        config.train.scheduler_type,
        optimizer=optimizer_inversion,
        num_warmup_steps=config.train.scheduler_warmup_steps,
        num_training_steps=config.train.ti_total_steps,
        num_cycles=config.train.scheduler_num_cycles,
    )

    if not os.path.exists(config.train.checkpoint_folder):
        os.makedirs(config.train.checkpoint_folder)
        print(f"Directory '{config.train.checkpoint_folder}' created.")
    else:
        print(f"Directory '{config.train.checkpoint_folder}' already exists.")

    loss_sum = 0.0
    progress_bar = tqdm(range(config.train.ti_total_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    mse = torch.nn.MSELoss(reduction='mean')
    ssim = SSIM_loss(data_range=1.0, size_average=True, channel=4, win_size=config.train.ssim_win_size)
    ms_ssim = MS_SSIM_loss(data_range=1.0, size_average=True, channel=4, win_size=3)

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
    elif config.train.criterion == 'mlsd':
        criterion = MLSD_Perceptual_loss()
        noise_scheduler.set_timesteps(20, device="cuda")
        print('Using MLSD loss')
    else:
        raise ValueError(f'Unknown loss {config.train.criterion}, it must be either mse, ssim, ms_ssim or mse+ssim')

    for epoch in range(math.ceil(config.train.ti_total_steps / len(train_dataloader))):
        unet.train()
        text_encoder.train()
        for batch in train_dataloader:
            optimizer_inversion.zero_grad()
            model_pred, target, noisy_latents, latents, timesteps = forward_step(
                batch,
                unet,
                vae,
                text_encoder,
                noise_scheduler,
                t_mutliplier=config.train.t_mutliplier,
                mask_temperature=config.train.mask_temperature,
                loss_on_latent=config.train.loss_on_latent,
                timesteps_scheduler=timesteps_scheduler,
            )
            train_loss = criterion(model_pred.float(), target.float())
            train_loss.backward()
            loss_sum += train_loss.detach().item()

            if config.train.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(unet.parameters(), text_encoder.parameters()), 
                    max_norm=config.train.clip_gradients_max_norm
                )
            optimizer_inversion.step()
            lr_scheduler_inversion.step()

            # normalize the embeddings
            with torch.no_grad():
                pre_norm = (
                    text_encoder.get_input_embeddings()
                    .weight[index_updates, :]
                    .norm(dim=-1, keepdim=True)
                )

                lambda_ = min(1.0, 100 * lr_scheduler_inversion.get_last_lr()[0])
                text_encoder.get_input_embeddings().weight[index_updates] = F.normalize(
                    text_encoder.get_input_embeddings().weight[index_updates, :],
                    dim=-1,) * (pre_norm + lambda_ * (0.4 - pre_norm)
                )
                
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = original_embeds[index_no_updates]

            progress_bar.update(1)
            logs = {
                config.train.criterion: train_loss.detach().item(),
                "lr": lr_scheduler_inversion.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        if config.log_wandb:
            logs = {
                "train_"+config.train.criterion: loss_sum / len(train_dataloader),
                "lr": lr_scheduler_inversion.get_last_lr()[0],
            }
        loss_sum = 0.0

        if epoch % config.train.eval_every_n_epochs == 0:
            
            unet.eval()
            text_encoder.eval()

            mse_loss = 0.0
            ssim_loss = 0.0
            ms_ssim_loss = 0.0

            for _ in range(config.eval.eval_epochs):
                for batch in valid_dataloader:
                    with torch.no_grad():
                        model_pred, target, _, _, _ = forward_step(
                                                batch,
                                                unet,
                                                vae,
                                                text_encoder,
                                                noise_scheduler,
                                                t_mutliplier=config.train.t_mutliplier,
                                                mixed_precision=False,
                                                mask_temperature=config.train.mask_temperature,
                                                loss_on_latent=False,
                                            )
                        mse_loss += mse(model_pred.float(), target.float()).detach().item()
                        ssim_loss += ssim(model_pred.float(), target.float()).detach().item()
                        ms_ssim_loss += ms_ssim(model_pred.float(), target.float()).detach().item()

            logs['val_mse'] = mse_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            logs['val_ssim'] = ssim_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            logs['val_ms_ssim'] = ms_ssim_loss / (len(valid_dataloader) * config.eval.eval_epochs)
            
            if config.log_wandb:
                evaluation_logs = evaluate_pipe(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    dataset=test_dataset,
                    config=config,
                    dino_scorer=dino_scorer,
                    fid_scorer=fid_scorer,
                )
                wandb.log(evaluation_logs, step=global_step)
            save_path = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_ti_{global_step}.safetensors')
            
            save_textual_inversion(
                config.train.new_tokens,
                new_token_ids, 
                text_encoder,
                save_path,
            )

            unet.train()
            text_encoder.train()
            
        wandb.log(logs, step=global_step)
    del optimizer_inversion
    del lr_scheduler_inversion
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()

    save_path = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_ti_{global_step}.safetensors')
    save_textual_inversion(
        config.train.new_tokens,
        new_token_ids, 
        text_encoder,
        save_path,
    )

def train_lora(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    train_dataloader,
    valid_dataloader,
    test_dataset,
    dino_scorer,
    fid_scorer,
    timesteps_scheduler,
    config: Config,
):
    print("Training LoRA...")
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
        run_name = 'lora_' + config.wandb.run_name if config.wandb.run_name else None
        wandb.init(
            project=config.wandb.project_name, 
            entity=config.wandb.entity_name,
            config=config,
            id=run_name,
            tags=config.wandb.tags,
        )
        wandb.watch(unet)
        wandb.watch(text_encoder)

    params_to_optimize = []

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

        params_to_optimize += [
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
        num_training_steps=config.train.lora_total_steps,
        num_cycles=config.train.scheduler_num_cycles,
    )

    if not os.path.exists(config.train.checkpoint_folder):
        os.makedirs(config.train.checkpoint_folder)
        print(f"Directory '{config.train.checkpoint_folder}' created.")
    else:
        print(f"Directory '{config.train.checkpoint_folder}' already exists.")

    loss_sum = 0.0
    progress_bar = tqdm(range(config.train.lora_total_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    if config.train.text_encoder_train_ratio < 1.0:
        text_encoder_steps = math.ceil(config.train.lora_total_steps * config.train.text_encoder_train_ratio)
    else:
        text_encoder_steps = config.train.lora_total_steps
    
    mse = torch.nn.MSELoss(reduction='mean')
    ssim = SSIM_loss(data_range=1.0, size_average=True, channel=4, win_size=config.train.ssim_win_size)
    ms_ssim = MS_SSIM_loss(data_range=1.0, size_average=True, channel=4, win_size=3)

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
    elif config.train.criterion == 'mlsd+mse':
        criterion = MLSD_Perceptual_loss()
        noise_scheduler.set_timesteps(20, device="cuda")
        print('Using MLSD loss')
    else:
        raise ValueError(f'Unknown loss {config.train.criterion}, it must be either mse, ssim, ms_ssim or mse+ssim')

    for epoch in range(math.ceil(config.train.lora_total_steps / len(train_dataloader))):
        unet.train()
        text_encoder.train()
        for batch in train_dataloader:
            optimizer_lora.zero_grad()
            model_pred, target, noisy_latents, latents, timesteps = forward_step(
                batch,
                unet,
                vae,
                text_encoder,
                noise_scheduler,
                t_mutliplier=config.train.t_mutliplier,
                mask_temperature=config.train.mask_temperature,
                loss_on_latent=config.train.loss_on_latent,
                timesteps_scheduler=timesteps_scheduler,
            )
            if config.train.criterion == 'mlsd+mse':
                step_latents = []
                for i, t in enumerate(timesteps):
                    step_latents.append(noise_scheduler.step(model_pred[i], t, noisy_latents[i], return_dict=False)[0])
                alpha = 0.3
                train_loss = (alpha) * criterion(torch.stack(step_latents), latents) + (1-alpha) * mse(model_pred.float(), target.float())
            else:
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

            for _ in range(config.eval.eval_epochs):
                for batch in valid_dataloader:
                    with torch.no_grad():
                        model_pred, target, _, _, _ = forward_step(
                                                    batch,
                                                    unet,
                                                    vae,
                                                    text_encoder,
                                                    noise_scheduler,
                                                    t_mutliplier=config.train.t_mutliplier,
                                                    mixed_precision=False,
                                                    mask_temperature=config.train.mask_temperature,
                                                    loss_on_latent=False,
                                                )
                        mse_loss += mse(model_pred.float(), target.float()).detach().item()
                        ssim_loss += ssim(model_pred.float(), target.float()).detach().item()
                        ms_ssim_loss += ms_ssim(model_pred.float(), target.float()).detach().item()

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
                    dataset=test_dataset,
                    config=config,
                    dino_scorer=dino_scorer,
                    fid_scorer=fid_scorer,
                )
                wandb.log(evaluation_logs, step=global_step)
            save_path = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_lora_{global_step}.safetensors')
            save_loras(
                unet if config.train.train_unet else None, 
                text_encoder if config.train.train_text_encoder else None, 
                save_path, 
                config
            )

            unet.train()
            text_encoder.train()
        
        wandb.log(logs, step=global_step)
    wandb.finish()

    save_path = os.path.join(config.train.checkpoint_folder, f'{config.wandb.project_name}_lora_{global_step}.safetensors')
    save_loras(
        unet if config.train.train_unet else None,
        text_encoder if config.train.train_text_encoder else None,
        save_path,
        config
    )

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
    timesteps_scheduler=None,
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

    if timesteps_scheduler is not None:
        timesteps_bounds = timesteps_scheduler.get_timesteps_bounds()
        timesteps = torch.randint(
            timesteps_bounds[1],
            timesteps_bounds[0],
            (bsz,),
            device=latents.device,
        )
    else:
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

        mask = (mask + 0.01).pow(mask_temperature)

        mask = mask / mask.max()

        model_pred = model_pred * mask

        target = target * mask

    if loss_on_latent:
        target = scheduler.add_noise(latents, target, timesteps)
        model_pred = scheduler.add_noise(latents, model_pred, timesteps)

    return model_pred, target, noisy_latents, latents, timesteps
