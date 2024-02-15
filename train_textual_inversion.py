from ast import arg, parse
from huggingface_hub import hf_hub_download
from src.config import DatasetConfig, PromptConfig, ModelConfig, LoraConfig, TrainConfig, WandbConfig, EvaluationConfig, Config
import argparse
from src.training import train

def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="kvist_windows",
        help="kvist or sommerhus.",
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

    dataset_config = DatasetConfig(
        project_name=project_name,
        dataset_version=dataset_version,
        roboflow_workspace='arked',
        image_size=512,
        normalize_images=True,
        scaling_pixels=25,
        data_root='data_' + args.dataset,
    )

    wandb_config = WandbConfig(
        project_name=args.dataset + "_textual_inversion",
        entity_name='maidacundo',
    )

    if args.dataset == "kvist_windows":
        prompts = ['kvist_windows']
    elif args.dataset == "sommerhus":
        prompts = ['sommerhus']

    if args.dataset == "kvist_windows":
        strengths = [0.99]
    elif args.dataset == "sommerhus":
        strengths = [1]

    eval_config=EvaluationConfig(
        prompts=prompts,
        strengths=strengths
    )

    if args.dataset == "kvist_windows":
        new_tokens = ["kvist_windows"]
        initializer_tokens = ["<rand-0.5>"]
    elif args.dataset == "sommerhus":
        new_tokens = ["sommerhus"]
        initializer_tokens = ["<rand-0.5>"]

    train_config=TrainConfig(
        checkpoint_folder=wandb_config.project_name + "_checkpoints",
        train_batch_size=2,
        train_unet=False,
        train_text_encoder=False,
        unet_lr=3e-4,
        text_encoder_lr=3e-4,
        learning_rate=1e-4,
        scheduler_num_cycles=2,
        ti_total_steps=1000,
        scheduler_type='linear',
        scheduler_warmup_steps=0,
        criterion='mse',
        timestep_snr_gamma=5.0,
        new_tokens=new_tokens,
        initializer_tokens=initializer_tokens,
    )

    config = Config(
        dataset=dataset_config,
        model=model_config,
        wandb=wandb_config,
        eval=eval_config,
        train=train_config,
    )

    train(config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
