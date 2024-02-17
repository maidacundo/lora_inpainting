from ast import arg, parse
from huggingface_hub import hf_hub_download
from src.config import DatasetConfig, PromptConfig, ModelConfig, LoraConfig, TrainConfig, WandbConfig, EvaluationConfig, Config
import argparse
from src.training import train

def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument(
        "--lora-injection",
        type=str,
        default="self-attention",
        help="Type of LoRA injection.",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="kvist_windows",
        help="kvist or sommerhus.",
    )

    parser.add_argument(
        "--textual-inversion",
        type=str,
        default=None,
        help="Textual Inversion path.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="mse",
        help="Loss criterion.",
    )

    args = parser.parse_args()
    return args

def main(args):

    inpainting_path = hf_hub_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", filename="Realistic_Vision_V5.1-inpainting.safetensors")
    vae_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.safetensors")

    model_config = ModelConfig(
        model_path=inpainting_path,
        vae_path=vae_path,
    )

    if args.dataset == "kvist_windows":
        project_name='kvist_windows'
        dataset_version=11
    
    elif args.dataset == "sommerhus":
        project_name='wood_facade-2'
        dataset_version=7

    elif args.dataset == "7er_stol":
        project_name='7er_stol_2'
        dataset_version=5

    
    dataset_config = DatasetConfig(
        project_name=project_name,
        dataset_version=dataset_version,
        roboflow_workspace='arked',
        image_size=512,
        normalize_images=True,
        scaling_pixels=25,
        data_root='lora_data_' + args.dataset,
    )

    run_name = 'easier-injection-' + args.lora_injection 
    project_name=args.dataset

    if args.textual_inversion is not None:
        run_name = run_name + "-pivotal-tuning"

    if args.lora_rank is not None:
        project_name = project_name + "_ranks"
        run_name = run_name + "_rank_" + str(args.lora_rank)

    wandb_config = WandbConfig(
        project_name=project_name,
        entity_name='maidacundo',
        run_name=run_name,
    )

    if args.dataset == "kvist_windows":
        prompts = ['kvist windows']
    elif args.dataset == "sommerhus":
        if args.textual_inversion is not None:
            prompts = ['sommerhus black wood facade']
        else:
            prompts = ['black wood facade']
    elif args.dataset == "7er_stol":
        prompts = ['7er chair']

    if args.dataset == "sommerhus":
        strengths = [1]
    else:
        strengths = [0.99]

    eval_config=EvaluationConfig(
        prompts=prompts,
        strengths=strengths
    )

    unet_target_modules = []
    text_encoder_target_modules = []

    if args.lora_injection == "self-attention":
        unet_target_modules = ["attn1.to_q", "attn1.to_v"]
    if args.lora_injection == "cross-attention":
        unet_target_modules = ["attn2.to_q", "attn2.to_v"]
    if args.lora_injection == "self-attention+geglu":
        unet_target_modules = ["attn1.to_q", "attn1.to_v", "ff.net.0.proj"]
    if args.lora_injection == "cross-attention+geglu":
        unet_target_modules = ["attn2.to_q", "attn2.to_v", "ff.net.0.proj"]
    if args.lora_injection == "attention-all":
        unet_target_modules = ["to_q", "to_v", "ff.net.0.proj"]
    if args.lora_injection == "geglu-resnet":
        unet_target_modules = ["ff.net.0.proj", "conv1", "conv2"]
    if args.lora_injection == "geglu":
        unet_target_modules = ["ff.net.0.proj"]
    if args.lora_injection == "geglu-all":
        unet_target_modules = ["ff.net.0.proj", "ff.net.2"]
    if args.lora_injection == "resnet-block":
        unet_target_modules = ["conv1", "conv2"]
    if args.lora_injection == "resnet-conv1":
        unet_target_modules = ["conv1"]
    if args.lora_injection == "resnet-conv2":
        unet_target_modules = ["conv2"]
    if args.lora_injection == "resnet-proj_in":
        unet_target_modules = ["proj_in"]
    if args.lora_injection == "text-encoder":
        text_encoder_target_modules = ["q_proj", "v_proj"]    
    if args.lora_injection == "text-encoder+geglu":
        text_encoder_target_modules = ["q_proj", "v_proj"]
        unet_target_modules =  ["ff.net.0.proj"]

    rank = args.lora_rank if args.lora_rank is not None else 8

    lora_config=LoraConfig(
        rank=rank,
        alpha=16,
        dropout_p=0.0,
        unet_target_modules=unet_target_modules,
        text_encoder_target_modules=text_encoder_target_modules,
    )

    if args.textual_inversion is not None:
        if args.dataset == "sommerhus":
            dataset_config.trigger_word = 'sommerhus'

    train_config=TrainConfig(
        checkpoint_folder=wandb_config.project_name + "_" + args.lora_injection + "_" + str(rank) + "_checkpoints" ,
        train_batch_size=2,
        train_unet=len(unet_target_modules) > 0,
        train_text_encoder=len(text_encoder_target_modules) > 0,
        unet_lr=3e-4,
        text_encoder_lr=3e-4,
        learning_rate=1e-3,
        scheduler_num_cycles=1,
        lora_total_steps=1000,
        scheduler_warmup_steps=100,
        criterion=args.criterion,
        timestep_snr_gamma=5.0,
        load_textual_embeddings=args.textual_inversion,
    )

    config = Config(
        dataset=dataset_config,
        model=model_config,
        wandb=wandb_config,
        eval=eval_config,
        train=train_config,
        lora=lora_config,
    )

    train(config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
