from huggingface_hub import hf_hub_download
from src.config import DatasetConfig, ModelConfig, LoraConfig, TrainConfig, WandbConfig, EvaluationConfig, Config
import argparse
from src.training import train

def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument(
        "--injection",
        type=str,
        default="self-attention",
        help="Type of LoRA injection.",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank of LoRA injection.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="mse",
        help="Criterion.",
    )

    args = parser.parse_args()
    return args

def main(args):

    inpainting_path = hf_hub_download(repo_id="webui/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.safetensors")
    vae_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.safetensors")

    model_config = ModelConfig(
        model_path=inpainting_path,
        vae_path=vae_path,
    )

    project_name = 'scandinavian-kitchen'
    dataset_version = 2

    dataset_config = DatasetConfig(
        project_name=project_name,
        dataset_version=dataset_version,
        roboflow_workspace='arked',
        image_size=512,
        normalize_images=True,
        scaling_pixels=0,
        data_root='data_' + args.dataset,
    )

    run_name = args.injection + "_" + str(args.lora_rank) + "_" + args.criterion
    project_name = "kitchen_testing"

    wandb_config = WandbConfig(
        project_name=project_name,
        entity_name='maidacundo',
        run_name=run_name,
    )

    prompts = ['scandinavian kitchen']
    strengths = [1]

    eval_config=EvaluationConfig(
        prompts=prompts,
        strengths=strengths
    )

    unet_target_modules = []
    text_encoder_target_modules = []

    if args.injection == "unet-all":
        unet_target_modules = ["to_q", "to_v", "to_k", "to_out.0", "ff.net.0.proj"]

    if args.injection == "small-all":
        unet_target_modules = ["to_q", "to_v", "ff.net.0.proj"]
        text_encoder_target_modules = ["q_proj", "v_proj"]

    if args.injection == "all":
        unet_target_modules = ["to_q", "to_v", "to_k", "to_out.0", "ff.net.0.proj"]
        text_encoder_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    lora_config=LoraConfig(
        rank=args.lora_rank,
        alpha=8,
        unet_target_modules=unet_target_modules,
        text_encoder_target_modules=text_encoder_target_modules,
    )

    train_config=TrainConfig(
        checkpoint_folder=run_name,
        train_batch_size=4,
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
        load_textual_embeddings=None,
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