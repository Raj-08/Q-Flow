
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
