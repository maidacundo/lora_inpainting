from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    roboflow_api_key: str
    roboflow_workspace: str
    project_name: str
    dataset_version: int
    data_root: str = 'data'
    image_size: int = 512
    max_image_size: int = 512
    normalize_images: bool = False # TODO check if this is needed and if it is done in the right way (for now it is set to false)
    scaling_pixels: int = 25
    do_classifier_free_guidance: bool = False

@dataclass
class PromptConfig:
    global_caption: str = None
    negative_caption: str = 'ugly, blurry, poor quality'

@dataclass
class ModelConfig:
    model_path: str = ''  # Fill in the path
    vae_path: str = ''  # Fill in the path

@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 32.0 
    dropout_p: float = 0.1
    unet_adapter_name: str = 'lora_unet'
    text_encoder_adapter_name: str = 'lora_te' 
    unet_target_modules: list = field(default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0", "ff.net.0.proj"]) #, "proj_in", "conv1", "conv2"]
    text_encoder_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj", "mlp.fc1", "mlp.fc2"])
    # "to_q", "to_v", "to_k", "to_out.0" are the names of the modules in attention layers
    # "ff.net.0.proj" is the name of the linear in the GEGLU activation
    # "proj_in", "conv1", "conv2" are the names of the modules in the resnet block
    # TODO understand better the role of each module and if it is needed to add more

    # "q_proj", "v_proj", "k_proj", "out_proj" are the names of the modules in the text encoder attention layers
    # "mlp.fc1", "mlp.fc2" are the names of the modules in the text encoder mlp that produce the embeddings
    output_format: str = 'peft' # can be either 'kohya_ss' or 'peft', 'kohya_ss' is compatible with A1111

@dataclass
class TrainConfig:
    train_batch_size: int = 2
    eval_batch_size: int = 1
    train_unet: bool = True
    train_text_encoder: bool = True
    text_encoder_train_ratio = 1
    unet_lr: float = 1e-4
    text_encoder_lr: float = 1e-4
    mask_temperature: float = 1.0
    criterion: str = 'mse'
    eval_every_n_epochs: int = 5
    num_checkpoint_limit: int = 5 # TODO use this and delete the previous checkpoints (if needed, not sure its a good feature)
    mixed_precision: str = 'no'
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    use_scheduler: bool = True
    scheduler_type: str = 'cosine_with_restarts'
    scheduler_num_cycles: int = 4
    scheduler_warmup_steps: int = 500
    total_steps: int = 4000
    optimizer: str = 'adamw' # TODO can be both adamw or lion optimizers (in order to test lion convergence and styling capabilities)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    clip_gradients: bool = True
    clip_gradients_max_norm: float = 1.0
    use_xformers: bool = False
    checkpoint_folder: str = 'checkpoints'
    t_mutliplier: int = 1 # changed from 0.8 to 1 to test the effect of the timestep on the diffusion
    loss_on_latent: bool = False

@dataclass
class EvaluationConfig:
    num_eval_steps: int = 20
    prompts: list = field(default_factory=list)
    strengths: list = field(default_factory=lambda: [1])
    eval_epochs: int = 10
    log_attention_maps: bool = False
    num_images_per_prompt: int = 8 # the number of images to generate for each prompt during evaluation (used to compute FID)
    compute_dino_score: bool = True

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
