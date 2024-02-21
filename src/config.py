from dataclasses import dataclass, field

from typing import Optional

@dataclass
class DatasetConfig:
    roboflow_workspace: str
    project_name: str
    dataset_version: int
    data_root: str = 'data'
    image_size: int = 512
    normalize_images: bool = True
    scaling_pixels: int = 25
    do_classifier_free_guidance: bool = True
    trigger_word: str = None # to trigger Pivotal Tuning if text inversion is used

@dataclass
class PromptConfig:
    global_caption: str = None
    negative_caption: str = None

@dataclass
class ModelConfig:
    model_path: str = ''  # Fill in the path
    vae_path: str = ''  # Fill in the path

@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 32.0 
    dropout_p: float = 0.0 # TODO testing the dropout in the lora 110*
    unet_adapter_name: str = 'lora_unet'
    text_encoder_adapter_name: str = 'lora_te' 
    unet_target_modules: list = field(default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0", "ff.net.0.proj"]) #, "proj_in", "conv1", "conv2"]
    text_encoder_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"])
    # "to_q", "to_v", "to_k", "to_out.0" are the names of the modules in attention layers
    # "proj_in" is the name of the linear before the attention
    # "attn1.to_q", "attn1.to_v" add lora in the self attention layer
    # "attn2.to_q", "attn2.to_v" add lora in the cross attention layer
    # "ff.net.0.proj" is the name of the linear in the GEGLU activation
    
    # "conv1", "conv2" are the names of the modules in the resnet block

    # "q_proj", "v_proj", "k_proj", "out_proj" are the names of the modules in the text encoder attention layers
    # "mlp.fc1", "mlp.fc2" are the names of the modules in the text encoder mlp that produce the embeddings
    output_format: str = 'kohya_ss' # can be either 'kohya_ss' or 'peft', 'kohya_ss' is compatible with A1111

@dataclass
class TrainConfig:
    train_batch_size: int = 4
    eval_batch_size: int = 1
    train_unet: bool = True
    train_text_encoder: bool = True
    text_encoder_train_ratio: float = 1
    unet_lr: float = 1e-4
    unet_geglu_lr: float = 7e-4
    text_encoder_lr: float = 1e-4
    mask_temperature: float = 1.0
    criterion: str = 'mse'
    criterion_alpha: float = 0.1 # alpha for the weighted sum of the losses
    ssim_win_size: int = 11
    eval_every_n_epochs: int = 5
    mixed_precision: str = 'no'
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    use_scheduler: bool = True
    scheduler_type: str = 'cosine_with_restarts'
    scheduler_num_cycles: int = 1
    scheduler_warmup_steps: int = 500
    lora_total_steps: int = 2000
    ti_total_steps: int = 500
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    clip_gradients: bool = True
    clip_gradients_max_norm: float = 1.0
    use_xformers: bool = False
    checkpoint_folder: str = 'checkpoints'
    t_mutliplier: int = 0.8
    loss_on_latent: bool = False
    new_tokens: list = field(default_factory=list)
    initializer_tokens: list = field(default_factory=list)
    load_textual_embeddings: str = None
    use_timestep_scheduler: bool = False
    timestep_scheduler_change_every_n_steps: int = 100
    # timestep_scheduler_fixed_bounds are two integers that represent the bounds of the timestep scheduler
    timestep_scheduler_fixed_bounds: list = field(default_factory=lambda: [800, 200])
    timestep_snr_gamma: Optional[int] = None # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. if setted, criterion will be mse
    
@dataclass
class EvaluationConfig:
    prompts: list = field(default_factory=list)
    strengths: list = field(default_factory=lambda: [0.99])
    eval_epochs: int = 1
    log_attention_maps: bool = False
    num_images_per_prompt: int = 1 # the number of images to generate for each prompt during evaluation (used to compute FID and DINO scores)
    compute_dino_score: bool = True
    compute_fid_score: bool = True

@dataclass
class WandbConfig:
    project_name: str
    entity_name: str = 'arked'
    run_name: str = None
    tags: list = field(default_factory=lambda: ["lora"])

@dataclass
class Config:
    dataset: DatasetConfig
    wandb: WandbConfig
    seed: int = 2808
    logdir: str = "logs"
    device: str = 'cuda'
    prompt: PromptConfig = PromptConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    lora: LoraConfig = LoraConfig()
    eval: EvaluationConfig = EvaluationConfig()
    log_wandb: bool = True