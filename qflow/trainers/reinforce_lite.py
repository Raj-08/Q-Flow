from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .algo import BaseTrainer
from ..data_processor import DatasetProcessor

class ReinforceLiteTrainer(BaseTrainer):
    def __init__(self, policy_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, optimizer: torch.optim.Optimizer, device: str, dataset_name: str = "gsm8k"):
        super().__init__(policy_model, tokenizer, optimizer, device)
        self.policy_model = policy_model
        self.max_seq_length = 1024
        self.max_new_tokens = 1024
        self.entropy_coef = 0.001
        self.group_size = 10
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Initialize DatasetProcessor for reward calculation
        self.dataset_processor = DatasetProcessor(tokenizer)
        self.dataset_name = dataset_name
        self.get_reward = self.dataset_processor.get_reward_function(dataset_name)
        self.format_prompt = self.dataset_processor.format_prompt

    def train_step(self, batch: List[Tuple[str, str]], step: int, save_dir: str):
        self.policy_model.train()
        prompts, targets = zip(*batch)
        batch_size = len(prompts)
        evaluated_group = 0

        all_logprobs = []
        all_rewards = []
        all_responses = []
        all_lengths = []

        for group_idx in range(self.group_size):
            formatted_prompts = [self.format_prompt(p) for p in prompts]
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)

            generate_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "pad_token_id": self.tokenizer.pad_token_id,
                "return_dict_in_generate": True,
            }

            if group_idx == evaluated_group:
                # Generate sequence first
                generated = self.policy_model.generate(**generate_kwargs)
                generated_ids = generated.sequences
                
                # Forward pass to get gradients
                outputs = self.policy_model(
                    generated_ids,
                    attention_mask=(generated_ids != self.tokenizer.pad_token_id).long()
                )
                
                # Calculate logprobs for generated tokens
                prompt_length = inputs.input_ids.shape[1]
                response_length = generated_ids.shape[1] - prompt_length
                
                if response_length > 0:
                    # Get logits for response tokens only
                    logits = outputs.logits[:, prompt_length-1:-1, :]  # (batch, response_len, vocab)
                    response_tokens = generated_ids[:, prompt_length:]  # (batch, response_len)
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_log_probs = torch.gather(
                        log_probs, 
                        -1, 
                        response_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    sequence_log_probs = token_log_probs.sum(dim=1)
                else:
                    sequence_log_probs = torch.zeros(batch_size, device=self.device)
            else:
                with torch.no_grad():
                    generated = self.policy_model.generate(**generate_kwargs)
                sequence_log_probs = torch.zeros(batch_size, device=self.device)

            # Decode responses
            responses = self.tokenizer.batch_decode(
                generated.sequences[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            rewards = torch.tensor([
                self.get_reward(resp, tgt) for resp, tgt in zip(responses, targets)
            ], device=self.device)

            all_responses.extend(responses)
            all_rewards.append(rewards)
            all_logprobs.append(sequence_log_probs)
            all_lengths.extend([len(r.split()) for r in responses])

        # Advantage calculation
        rewards_tensor = torch.stack(all_rewards)
        logprobs_tensor = torch.stack(all_logprobs)

        evaluated_rewards = rewards_tensor[evaluated_group]
        others_rewards = torch.cat([
            rewards_tensor[:evaluated_group], 
            rewards_tensor[evaluated_group+1:]
        ], dim=0)
        
        baseline = others_rewards.mean(dim=0)
        advantages = (evaluated_rewards - baseline) / (others_rewards.std(dim=0) + 1e-8)
        advantages = torch.clamp(advantages, -2.0, 2.0)

        policy_loss = -(logprobs_tensor[evaluated_group] * advantages.detach()).mean()

        return (
            policy_loss,
            rewards_tensor.mean().item(),
            policy_loss.item(),
            0.0,  # KL divergence (not used in this implementation)
            all_responses[0],
            all_lengths
        ) 