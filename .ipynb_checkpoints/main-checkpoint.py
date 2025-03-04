import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.cuda.amp as amp
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import logging
import sys
import argparse

# Import local modules
from qflow.trainers.grpo import GRPOTrainer
from qflow.trainers.reinforce_lite import ReinforceLiteTrainer
# from qflow.trainers.monte_carlo import MonteCarloTrainer
from qflow.data_processor import DatasetProcessor
from qflow.config import QFlowConfig, get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    config = get_default_config()
    parser = argparse.ArgumentParser(description='Q-Flow Training Application')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default=config.model.name,
                       help='Name or path of the model to use')
    
    # Training arguments
    parser.add_argument('--algorithm', type=str, default=config.training.algorithm,
                       choices=['grpo', 'reinforce-lite', 'monte-carlo'],
                       help='Training algorithm to use')
    parser.add_argument('--dataset_name', type=str, default=config.training.dataset_name,
                       help='Name of the dataset to use')
    parser.add_argument('--num_steps', type=int, default=config.training.num_steps,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=config.training.batch_size,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=config.training.learning_rate,
                       help='Learning rate for training')
    
    # Algorithm arguments
    parser.add_argument('--group_size', type=int, default=config.algorithm.group_size,
                       help='Group size for GRPO/Reinforce-Lite')
    parser.add_argument('--entropy_coef', type=float, default=config.algorithm.entropy_coef,
                       help='Entropy coefficient')
    parser.add_argument('--temperature', type=float, default=config.algorithm.temperature,
                       help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=config.algorithm.top_p,
                       help='Top-p for nucleus sampling')
    
    return parser.parse_args()

def get_trainer(config: QFlowConfig, policy_model, tokenizer, optimizer, device):
    """Factory function to create the appropriate trainer based on algorithm choice."""
    trainer_map = {
        'grpo': GRPOTrainer,
        'reinforce-lite': ReinforceLiteTrainer,
        # 'monte-carlo': MonteCarloTrainer
    }
    
    trainer_cls = trainer_map.get(config.training.algorithm)
    if trainer_cls is None:
        raise ValueError(f"Unknown algorithm: {config.training.algorithm}")
    
    trainer = trainer_cls(
        policy_model=policy_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device
    )
    
    # Set common trainer attributes
    trainer.max_seq_length = config.training.max_seq_length
    trainer.max_new_tokens = config.training.max_new_tokens
    trainer.entropy_coef = config.algorithm.entropy_coef
    
    # Set algorithm-specific attributes
    if hasattr(trainer, 'group_size'):
        trainer.group_size = config.algorithm.group_size
    if hasattr(trainer, 'temperature'):
        trainer.temperature = config.algorithm.temperature
    if hasattr(trainer, 'top_p'):
        trainer.top_p = config.algorithm.top_p
    
    return trainer

def main():
    """Main training function."""
    args = parse_arguments()
    config = QFlowConfig.from_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Training with algorithm: {config.training.algorithm}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{config.training.algorithm}_{timestamp}')
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    # Initialize dataset processor
    data_processor = DatasetProcessor(tokenizer)

    bnb_config = None
    if config.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.model.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="auto"
        )
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map="auto",
        use_cache=False,
        torch_dtype=config.model.torch_dtype,
        attn_implementation="flash_attention_2" if config.model.use_flash_attention else None
    )
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.training.learning_rate)
    
    # Initialize trainer
    trainer = get_trainer(config, policy_model, tokenizer, optimizer, device)
    trainer.format_prompt = data_processor.format_prompt
    trainer.get_reward = data_processor.get_reward_function(config.training.dataset_name)
    
    # Prepare training data
    prompts = data_processor.prepare_training_data(
        dataset_name=config.training.dataset_name,
        num_steps=config.training.num_steps,
        batch_size=config.training.batch_size
    )
    
    # Log all configurations
    for category in ['model', 'training', 'algorithm', 'lora']:
        config_obj = getattr(config, category)
        for key, value in config_obj.__dict__.items():
            writer.add_text(f'config/{category}/{key}', str(value))
    
    save_dir = f"checkpoints/{config.training.algorithm}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    window_size = 100
    recent_rewards = []
    
    try:
        for step in tqdm(range(config.training.num_steps)):
            batch_start = step * config.training.batch_size
            batch_end = (step + 1) * config.training.batch_size
            batch = prompts[batch_start:batch_end]
            
            loss, reward, policy_loss, entropy_loss, sample_response, response_lengths = trainer.train_step(
                batch=batch,
                step=step,
                save_dir=save_dir
            )
            
            recent_rewards.append(reward)
            if len(recent_rewards) > window_size:
                recent_rewards.pop(0)
            success_rate = sum(recent_rewards) / len(recent_rewards)
            
            if step % 5 == 0:
                with open(os.path.join(save_dir, "training_log.txt"), "a") as f:
                    successes = sum(1 for r in recent_rewards if r > 0)
                    total = len(recent_rewards)
                    f.write(f"\nStep {step}:\n")
                    f.write(f"Success Rate: {success_rate:.3f} ({successes}/{total} correct)\n")
                    f.write(f"Average Response Length: {sum(response_lengths) / len(response_lengths):.1f} words\n")
                    f.write(f"Sample Response:\n{sample_response}\n")
                    f.write("="*80 + "\n")
            
            optimizer.zero_grad()
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    config.training.gradient_clip_value
                )
                
                valid_gradients = True
                for param in policy_model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                
                if valid_gradients:
                    optimizer.step()
                else:
                    logger.warning("Skipping step due to invalid gradients")
            
            writer.add_scalar('Training/total_loss', loss.item(), step)
            writer.add_scalar('Training/policy_loss', policy_loss, step)
            writer.add_scalar('Training/entropy_loss', entropy_loss, step)
            writer.add_scalar('Training/reward', reward, step)
            writer.add_scalar('Training/success_rate', success_rate, step)
            writer.add_scalar('Training/avg_response_length', sum(response_lengths) / len(response_lengths), step)
            
            if step % 5 == 0:
                logger.info(f"Step {step} | Loss: {loss:.3f} | Reward: {reward:.3f} | "
                          f"Success Rate: {success_rate:.3f} | "
                          f"Avg Length: {sum(response_lengths) / len(response_lengths):.1f}")
                writer.add_text('samples/model_output', sample_response, step)
            
            if step % 100 == 0 or step == config.training.num_steps - 1:
                trainer.save_checkpoint(step, save_dir)
                
                torch.save({
                    'step': step,
                    'loss': loss.item(),
                    'reward': reward,
                    'policy_loss': policy_loss,
                    'entropy_loss': entropy_loss,
                    'success_rate': success_rate,
                    'model_config': policy_model.config,
                    'qflow_config': config
                }, os.path.join(save_dir, f"checkpoint-{step}/training_state.pt"))
                
                logger.info(f"Saved checkpoint at step {step}")
                
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        writer.close()
        
    logger.info("Training completed!")
    
    final_path = os.path.join(save_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    
    trainer.save_checkpoint("final", final_path)
    torch.save(optimizer.state_dict(), os.path.join(final_path, "optimizer.pt"))
    
    torch.save({
        'total_steps': config.training.num_steps,
        'final_loss': loss.item(),
        'final_reward': reward,
        'final_policy_loss': policy_loss,
        'final_entropy_loss': entropy_loss,
        'final_success_rate': success_rate,
        'model_config': policy_model.config,
        'qflow_config': config
    }, os.path.join(final_path, "final_training_state.pt"))
    
    logger.info(f"Final model and training state saved to {final_path}")

if __name__ == "__main__":
    main()