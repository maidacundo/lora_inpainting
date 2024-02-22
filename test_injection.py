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

    parser.add_argument(
        "--injection",
        type=str,
        default="self-attention",
        help="Type of LoRA injection.",
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
        dataset_version=12
    
    elif args.dataset == "sommerhus":
        project_name='wood_facade-2'
        dataset_version=7

    elif args.dataset == "7er_stol":
        project_name='7er_stol_2'
        dataset_version=5

    elif args.dataset == "denisen":
        project_name='denisen'
        dataset_version=8
    
    dataset_config = DatasetConfig(
        project_name=project_name,
        dataset_version=dataset_version,
        roboflow_workspace='arked',
        image_size=512,
        normalize_images=True,
        scaling_pixels=25,
        data_root='lora_data_' + args.dataset,
    )

    run_name = args.lora_injection 
    project_name = args.dataset + "_injections"

    wandb_config = WandbConfig(
        project_name=project_name,
        entity_name='maidacundo',
        run_name=run_name,
    )

    if args.dataset == "kvist_windows":
        prompts = ['kvist windows']

    elif args.dataset == "sommerhus":
        prompts = ['sommerhus black wood facade']

    elif args.dataset == "7er_stol":
        prompts = ['7er chair']

    elif args.dataset == "dinesen":
        prompts = ['dinesen floor']

    if args.dataset == "sommerhus" or args.dataset == "dinesen":
        strengths = [1]
    else:
        strengths = [0.99]

    eval_config=EvaluationConfig(
        prompts=prompts,
        strengths=strengths
    )

    unet_target_modules = []
    text_encoder_target_modules = []

    if args.lora_injection == "self-attn":
        unet_target_modules = ["attn1.to_q", "attn1.to_v", "attn1.to_k", "attn1.to_out.0"]

    if args.lora_injection == "cross-attn":
        unet_target_modules = ["attn2.to_q", "attn2.to_v", "attn2.to_k", "attn2.to_out.0"]

    if args.lora_injection == "geglu":
        unet_target_modules = ["ff.net.0.proj"]

    if args.lora_injection == "resnet":
        unet_target_modules = ["conv1", "conv2"]

    if args.lora_injection == "text-encoder":
        text_encoder_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    lora_config=LoraConfig(
        rank=8,
        alpha=1,
        dropout_p=0.0,
        unet_target_modules=unet_target_modules,
        text_encoder_target_modules=text_encoder_target_modules,
    )

    train_config=TrainConfig(
        checkpoint_folder=args.dataset + '_' + wandb_config.run_name,
        train_batch_size=2,
        train_unet=True,
        train_text_encoder=False,
        unet_lr=3e-4,
        text_encoder_lr=3e-4,
        learning_rate=1e-3,
        scheduler_num_cycles=1,
        lora_total_steps=1000,
        scheduler_warmup_steps=100,
        criterion="mse",
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
