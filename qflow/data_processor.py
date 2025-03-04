import re
from typing import List, Tuple, Dict, Optional, Callable
from datasets import load_dataset
from transformers import AutoTokenizer

class DatasetProcessor:
    """Handles dataset loading, parsing, and reward calculation for different datasets."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.dataset_processors = {
            'gsm8k': self._process_gsm8k,
            # Add more dataset processors here as needed
        }
        self.reward_functions = {
            'gsm8k': self._get_numerical_reward,
            # Add more reward functions here as needed
        }
    
    def format_prompt(self, question: str) -> str:
        """Format a question using the specified chat template."""
        prompt_template = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            },
            {
                "role": "user",
                "content": f"{question}\nShow your work in <think> </think> tags and return the final numerical answer in <answer> </answer> tags."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]
        return self.tokenizer.apply_chat_template(prompt_template, tokenize=False, continue_final_message=True)

    def load_dataset(self, dataset_name: str, split: str = "train") -> List[Tuple[str, str]]:
        """Load and process a dataset."""
        if dataset_name not in self.dataset_processors:
            raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(self.dataset_processors.keys())}")
        
        return self.dataset_processors[dataset_name](split)
    
    def get_reward_function(self, dataset_name: str) -> Callable:
        """Get the reward function for a specific dataset."""
        if dataset_name not in self.reward_functions:
            raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(self.reward_functions.keys())}")
        
        return self.reward_functions[dataset_name]
    
    def _process_gsm8k(self, split: str) -> List[Tuple[str, str]]:
        """Process the GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main")[split]
        return [
            (ex["question"], ex["answer"].split("####")[1].strip())
            for ex in dataset if "####" in ex["answer"]
        ]
    
    def _get_numerical_reward(self, completion: str, target: str) -> float:
        """Calculate reward for numerical answers by extracting numbers between <answer> and </answer> tags."""
        reward = -1.0
        try:
            completion = completion.strip().lower()
            start_tag = "<answer>"
            end_tag = "</answer>"
            
            start_indices = [i for i in range(len(completion)) if completion.startswith(start_tag, i)]
            end_indices = [i for i in range(len(completion)) if completion.startswith(end_tag, i)]
            
            if start_indices or end_indices:
                all_numbers = []
                
                if start_indices:
                    pre_text = completion[:start_indices[0]]
                    pre_numbers = re.findall(r'\b\d+\.?\d*\b', pre_text)
                    all_numbers.extend(pre_numbers)
                
                if end_indices:
                    post_text = completion[end_indices[-1] + len(end_tag):]
                    post_numbers = re.findall(r'\b\d+\.?\d*\b', post_text)
                    all_numbers.extend(post_numbers)
                
                for start_idx in start_indices:
                    substring_after_start = completion[start_idx + len(start_tag):]
                    end_idx = substring_after_start.find(end_tag)
                    if end_idx != -1:
                        answer = substring_after_start[:end_idx].strip()
                        between_numbers = re.findall(r'\b\d+\.?\d*\b', answer)
                        all_numbers.extend(between_numbers)
                
                for num_str in all_numbers:
                    try:
                        if '.' in num_str:
                            generated_num = float(num_str)
                            target_num = float(str(target).strip())
                            if abs(generated_num - target_num) < 1e-6:
                                reward = 1.0
                                break
                        else:
                            generated_num = int(num_str)
                            target_num = float(str(target).strip())
                            if abs(generated_num - target_num) < 1e-6:
                                reward = 1.0
                                break
                    except ValueError:
                        continue
                    
        except Exception as e:
            pass
        
        return reward
    
    def prepare_training_data(self, dataset_name: str, num_steps: int, batch_size: int, split: str = "train") -> List[Tuple[str, str]]:
        """Prepare training data with the required number of samples."""
        full_dataset = self.load_dataset(dataset_name, split)
        needed_samples = num_steps * batch_size
        num_cycles = (needed_samples + len(full_dataset) - 1) // len(full_dataset)
        return (full_dataset * num_cycles)[:needed_samples] 