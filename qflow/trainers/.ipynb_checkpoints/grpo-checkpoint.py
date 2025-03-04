from qflow.trainers.algo import BaseTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List,Tuple
import torch
import os
class GRPOTrainer(BaseTrainer):
    def __init__(self, policy_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, optimizer: torch.optim.Optimizer, device: str):
        super().__init__(policy_model, tokenizer, optimizer, device)
        # Initialize old_policy_model as a copy of the current policy
        self.old_policy_model = self._clone_policy_model(policy_model)
        self.old_policy_model.eval()  # Old model remains frozen
        self.policy_model=policy_model
    def _clone_policy_model(self, model):
        # Helper function to create a deep copy of the model
        clone = AutoModelForCausalLM.from_pretrained(model.config._name_or_path).to(self.device)
        clone.load_state_dict(model.state_dict())
        return clone

    def _update_old_policy(self):
        # Update old policy to match current policy
        self.old_policy_model = self._clone_policy_model(self.policy_model)
        self.old_policy_model.eval()

    def _compute_advantages(self, rewards):
        # Helper function to compute advantages with proper handling of edge cases
        baseline = rewards.mean()
        if rewards.shape[0] <= 1:  # If batch size is 1 or less
            return torch.zeros_like(rewards)
        
        # Calculate std with a minimum value to prevent division by zero
        std = rewards.std(unbiased=False)  # Use biased std estimation for small batches
        if std < 1e-6:  # If all rewards are the same or very close
            return torch.zeros_like(rewards)
        
        advantages = (rewards - baseline) / (std + 1e-6)
        return torch.clamp(advantages, -2.0, 2.0)

    def train_step(self, batch: List[Tuple[str, str]], step: int, save_dir: str):
        self.policy_model.train()
        prompts, targets = zip(*batch)
        batch_size = len(prompts)

        formatted_prompts = [self.format_prompt(p) for p in prompts]
        inputs = self.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_length
        ).to(self.device)

        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
        }

        generated = self.policy_model.generate(**generate_kwargs)
        generated_ids = generated.sequences
        
        # Forward pass for current policy
        outputs = self.policy_model(generated_ids, attention_mask=(generated_ids != self.tokenizer.pad_token_id).long())
        logits = outputs.logits[:, inputs.input_ids.shape[1]-1:-1, :]
        response_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        log_probs = torch.log_softmax(logits, dim=-1)
        current_log_probs = torch.gather(log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        # Compute rewards and advantages
        responses = self.tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        rewards = torch.tensor([self.get_reward(resp, tgt) for resp, tgt in zip(responses, targets)], device=self.device)
        advantages = self._compute_advantages(rewards)

        # Only use importance sampling ratio after 10 steps
        if step >= 10:
            # Forward pass for old policy
            with torch.no_grad():
                old_outputs = self.old_policy_model(generated_ids, attention_mask=(generated_ids != self.tokenizer.pad_token_id).long())
                old_logits = old_outputs.logits[:, inputs.input_ids.shape[1]-1:-1, :]
                old_log_probs = torch.log_softmax(old_logits, dim=-1)
                old_log_probs = torch.gather(old_log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)
            
            # Compute importance sampling ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.8, 1.2)  # Clip the ratio to prevent extreme values
            # Compute loss with ratio
            policy_loss = -(ratio * advantages.detach()).mean()
        else:
            # Compute loss without ratio for first 10 steps
            policy_loss = -(current_log_probs * advantages.detach()).mean()

        # Update old policy model after computing the loss
        self._update_old_policy()

        return policy_loss, rewards.mean().item(), policy_loss.item(), 0.0, responses[0], [len(r.split()) for r in responses]
