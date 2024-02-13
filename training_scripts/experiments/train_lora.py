from ast import parse
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
        "--dataset",
        type=str,
        default="kvist_windows",
        help="kvist or sommerhus.",
    )

    parser.add_argument(
        "--roboflow_api_key",
        type=str,
        default="HNXIsW3WwnidNDQZHexX",
        help="Roboflow API Key.",
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

    dataset_config = DatasetConfig(
        roboflow_api_key='HNXIsW3WwnidNDQZHexX',
        roboflow_workspace='arked',
        image_size=512,
        normalize_images=True,
        scaling_pixels=25,
    )

    if args.dataset == "kvist_windows":
        dataset_config.project_name='kvist_windows',
        dataset_config.dataset_version=7,
    
    elif args.dataset == "sommerhus":
        dataset_config.project_name='sommerhus',
        dataset_config.dataset_version=3,
    
    run_name = '-injection-' + args.lora_injection 

    wandb_config = WandbConfig(
        project_name=args.dataset,
        entity_name='maidacundo',
        run_name=run_name,
    )

    if args.dataset == "kvist_windows":
        prompts = ['kvist windows']
    elif args.dataset == "sommerhus":
        prompts = ['sommerhus']

    eval_config=EvaluationConfig(
        prompts=prompts,
    )

    unet_target_modules = []
    text_encoder_target_modules = []

    if args.lora_injection == "self-attention":
        unet_target_modules = ["attn1.to_q", "attn1.to_v"]
    if args.lora_injection == "cross-attention":
        unet_target_modules = ["attn2.to_q", "attn2.to_v"]
    if args.lora_injection == "geglu":
        unet_target_modules = ["ff.net.0.proj"]

    if args.lora_injection == "resnet-block":
        unet_target_modules = ["proj_in", "conv1", "conv2"]

    if args.lora_injection == "text-encoder":
        text_encoder_target_modules = ["q_proj", "v_proj"]

    lora_config=LoraConfig(
        rank=8,
        alpha=16,
        dropout_p=0.1,
        unet_target_modules=unet_target_modules,
        text_encoder_target_modules=text_encoder_target_modules,
    )

    train_config=TrainConfig(
        checkpoint_folder=wandb_config.project_name + "_" + wandb_config.project_name + "_checkpoints",
        train_batch_size=2,
        train_unet=len(unet_target_modules) > 0,
        train_text_encoder=len(text_encoder_target_modules) > 0,
        unet_lr=3e-4,
        text_encoder_lr=5e-5,
        learning_rate=1e-3,
        scheduler_num_cycles=2,
        lora_total_steps=1000,
        scheduler_warmup_steps=100,
        criterion='mse',
        timestep_snr_gamma=5.0,
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
    # parse arguments and run main
    main()
