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

