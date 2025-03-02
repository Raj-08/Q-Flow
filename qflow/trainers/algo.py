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


class ReinforceLiteTrainer(BaseTrainer):
    def __init__(self, policy_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str):
        super().__init__(policy_model, tokenizer, device)
    
    def train_step(self, batch: List[Tuple[str, str]], step: int, save_dir: str):
        self.policy_model.train()
        prompts, targets = zip(*batch)
        batch_size = len(prompts)
        evaluated_group = 0

        all_logprobs, all_rewards, all_responses, all_lengths = [], [], [], []

        for group_idx in range(self.GROUP_SIZE):
            formatted_prompts = [self.format_prompt(p) for p in prompts]
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_SEQ_LENGTH
            ).to(self.device)

            generate_kwargs = {
                **inputs,
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id,
                "return_dict_in_generate": True,
            }

            if group_idx == evaluated_group:
                generated = self.policy_model.generate(**generate_kwargs)
                generated_ids = generated.sequences
                outputs = self.policy_model(
                    generated_ids,
                    attention_mask=(generated_ids != self.tokenizer.pad_token_id).long()
                )
                prompt_length = inputs.input_ids.shape[1]
                response_length = generated_ids.shape[1] - prompt_length

                if response_length > 0:
                    logits = outputs.logits[:, prompt_length-1:-1, :]
                    response_tokens = generated_ids[:, prompt_length:]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_log_probs = torch.gather(log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1)
                    sequence_log_probs = token_log_probs.sum(dim=1)
                else:
                    sequence_log_probs = torch.zeros(batch_size, device=self.device)
            else:
                with torch.no_grad():
                    generated = self.policy_model.generate(**generate_kwargs)
                sequence_log_probs = torch.zeros(batch_size, device=self.device)

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
            0.0,
            all_responses[0],
            all_lengths
        )

