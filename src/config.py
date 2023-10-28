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
    normalize_images: bool = True
    scaling_pixels: int = 4

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
    scale: float = 1.0
    dropout_p: float = 0.1

@dataclass
class TrainConfig:
    train_batch_size: int = 2
    eval_batch_size: int = 1
    train_unet: bool = True
    train_text_encoder: bool = True
    text_encoder_train_ratio = 0.75 # TODO stop the training of the text encoder before the end of the unet one
    unet_lr: float = 2e-4
    text_encoder_lr: float = 2e-4
    mask_temperature: float = 1.0
    num_train_timesteps: int = 50  # TODO add the new scheduler
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
    weight_decay: float = 1e-4
    clip_gradients: bool = True
    clip_gradients_max_norm: float = 1.0
    use_xformers: bool = False
    checkpoint_folder: str = 'checkpoints'

@dataclass
class EvaluationConfig:
    num_eval_steps: int = 20
    use_validation: bool = True
    prompts: list = field(default_factory=list)
    strengths: list = field(default_factory=lambda: [1])
    eval_epochs: int = 10

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