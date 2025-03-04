import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "microsoft/Phi-3.5-mini-instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    torch_dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True

@dataclass
class TrainingConfig:
    algorithm: str = "grpo"  # choices: ['grpo', 'reinforce-lite', 'monte-carlo']
    dataset_name: str = "gsm8k"
    batch_size: int = 1
    num_steps: int = 5000
    learning_rate: float = 1e-6
    gradient_clip_value: float = 1.0
    max_seq_length: int = 1024
    max_new_tokens: int = 1024

@dataclass
class AlgorithmConfig:
    # GRPO specific
    group_size: int = 10
    entropy_coef: float = 0.001
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1

@dataclass
class QFlowConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    algorithm: AlgorithmConfig = AlgorithmConfig()
    lora: LoRAConfig = LoRAConfig()

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        config = cls()
        
        # Update model config
        if hasattr(args, 'model_name'):
            config.model.name = args.model_name
            
        # Update training config
        if hasattr(args, 'algorithm'):
            config.training.algorithm = args.algorithm
        if hasattr(args, 'dataset_name'):
            config.training.dataset_name = args.dataset_name
        if hasattr(args, 'batch_size'):
            config.training.batch_size = args.batch_size
        if hasattr(args, 'num_steps'):
            config.training.num_steps = args.num_steps
        if hasattr(args, 'learning_rate'):
            config.training.learning_rate = args.learning_rate
            
        # Update algorithm config
        if hasattr(args, 'group_size'):
            config.algorithm.group_size = args.group_size
        if hasattr(args, 'entropy_coef'):
            config.algorithm.entropy_coef = args.entropy_coef
            
        return config

def get_default_config() -> QFlowConfig:
    """Get default configuration."""
    return QFlowConfig()