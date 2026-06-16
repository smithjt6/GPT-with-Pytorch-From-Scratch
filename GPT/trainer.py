# This is a simple Trainer for the GPT Model. It handles the training loop, validation, and plotting of loss curves. It also includes gradient clipping to prevent exploding gradients, which can be an issue when training large models like GPT.
# The Trainer class can get quite complex, especially when you start adding features like learning rate scheduling, mixed precision training, or distributed training. 
# But this is a good starting point for training a GPT model on a small dataset and gets the basics down.

# PyTorch does have a tutorial online for this as well, however, I found youtube to have better explanations.
# Another thing - the trainer uses something called a DataLoader, which is a PyTorch utility that helps with batching and shuffling the data.
# You can create a DataLoader from a dataset, but I'd suggest reading the tutorial ipynb for a better more in-depth explanation. 


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import GPTConfig

class Trainer:
    '''Trainer class for GPT model'''
    def __init__(self, model: torch.nn.Module, config: GPTConfig, device: torch.device, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader):
        self.model = model
        self.config = config
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, plot_loss: bool = True) -> None:
        train_loss, val_loss = [], []
        train_iter = iter(self.train_data)
        val_iter = iter(self.val_data) if self.val_data is not None else None

        for step in range(self.config.num_steps):
            # Get a batch of training data
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_data)
                inputs, targets = next(train_iter)

            loss = self._train_step(inputs, targets)
            train_loss.append(loss)

            # Validate every 100 steps
            if step % 100 == 0 and val_iter is not None:
                try:
                    val_inputs, val_targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_data)
                    val_inputs, val_targets = next(val_iter)
                val_loss.append(self._val_step(val_inputs, val_targets))
                print(f"Step {step}: Train Loss = {loss:.4f}, Val Loss = {val_loss[-1]:.4f}")
            
        if plot_loss:
            plt.plot(train_loss, label='Train Loss')
            if self.val_data is not None:
                plt.plot(range(0, self.config.num_steps, 100), val_loss, label='Val Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
    
    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(inputs)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
        self.optimizer.step()
        return loss.item()
    
    def _val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss.item()

