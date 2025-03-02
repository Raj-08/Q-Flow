class BaseTrainer(ABC):
    def __init__(self, model, tokenizer, optimizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
    
    @abstractmethod
    def train_step(self, batch):
        pass
    
    def save_checkpoint(self, step, save_dir):
        checkpoint_path = os.path.join(save_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved at step {step}")

class MonteCarloTrainer(BaseTrainer):
    def train_step(self, batch):
        self.model.train()
        prompts, targets = zip(*batch)
        
        all_rewards, all_log_probs, all_responses = [], [], []
        
        for prompt, target in zip(prompts, targets):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.device)
            
            log_probs_per_prompt, rewards_per_prompt, responses_per_prompt = [], [], []
            for _ in range(N_MC_SAMPLES):
                generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)
                response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                reward = 1.0 if target in response else -1.0  # Simplified reward function
                
                log_probs_per_prompt.append(torch.tensor(reward, device=self.device))
                rewards_per_prompt.append(reward)
                responses_per_prompt.append(response)
            
            expected_reward = sum(rewards_per_prompt) / N_MC_SAMPLES
            policy_loss = -(sum(log_probs_per_prompt) * expected_reward)
            
            all_rewards.append(expected_reward)
            all_log_probs.append(sum(log_probs_per_prompt))
            all_responses.append(responses_per_prompt[0])

        final_policy_loss = sum(all_log_probs) * sum(all_rewards) / len(all_rewards)
        self.optimizer.zero_grad()
        final_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VALUE)
        self.optimizer.step()
        return final_policy_loss.item(), sum(all_rewards) / len(all_rewards), all_responses[0]


