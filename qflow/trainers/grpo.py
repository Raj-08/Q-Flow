class GRPOTrainer(BaseTrainer):
    def __init__(self, policy_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str):
        super().__init__(policy_model, tokenizer, device)
        self.old_policy_model = AutoModelForCausalLM.from_pretrained(policy_model.config._name_or_path).to(device)
        self.old_policy_model.eval()  # Old model remains frozen

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

        # Forward pass for old policy
        with torch.no_grad():
            old_outputs = self.old_policy_model(generated_ids, attention_mask=(generated_ids != self.tokenizer.pad_token_id).long())
            old_logits = old_outputs.logits[:, inputs.input_ids.shape[1]-1:-1, :]
            old_log_probs = torch.log_softmax(old_logits, dim=-1)
            old_log_probs = torch.gather(old_log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        
        # Compute importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Compute rewards and advantages
        responses = self.tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        rewards = torch.tensor([self.get_reward(resp, tgt) for resp, tgt in zip(responses, targets)], device=self.device)
        baseline = rewards.mean()
        advantages = (rewards - baseline) / (rewards.std() + 1e-8)
        advantages = torch.clamp(advantages, -2.0, 2.0)

        # Compute loss
        policy_loss = -(ratio * advantages.detach()).mean()

        return policy_loss, rewards.mean().item(), policy_loss.item(), 0.0, responses[0], [len(r.split()) for r in responses]
