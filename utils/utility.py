# Utility functions that are used across the project.

# device_setup is a comprehensive function that sets random seeds for reproducibility, configures CUDA settings for optimal performance, and determines the computation device (GPU or CPU). 
# get_activation is a simple function that returns the corresponding activation function based on the provided name. It supports common activation functions like ReLU, GELU, Tanh, and Sigmoid.
# The InitStrategy enum provides different weight initialization strategies (Orthogonal, Kaiming, Xavier, and Default) and a static method to apply the selected initialization to a given module.

import torch
import torch.nn as nn
import numpy as np

from enum import Enum

def device_setup(seed, deterministic = True, benchmark = True):
    """
    Comprehensive device setup function to ensure reproducibility and optimal performance.
    Sets random seeds for torch and numpy, configures CUDA settings, and determines the computation device (GPU or CPU).

    Parameters:
    seed (int): The seed value for random number generators.
    deterministic (bool): If True, sets CUDA to be deterministic for reproducibility. Default is True.
    benchmark (bool): If True, enables cuDNN benchmarking for performance (recommended). Default is True.

    Returns:
    device (torch.device): The computation device (GPU or CPU).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Note: benchmark=True provides better performance even with deterministic=True
        # Deterministic only affects specific operations, not cuDNN conv selection
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for computation.")
        
    return device

def get_activation(name: str):
    '''Returns the activation function corresponding to the given name'''
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU() # GELU is the activation function used in the original GPT paper, and is generally considered to be better than ReLU for transformer models.
                         # Furthermore, gelu in the original GPT 2 used approximate = 'tanh' because ti was a slow implementation of the erf.
                         # Relevant link: https://arxiv.org/abs/1606.08415
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


# InitStrategy is a bit less talked about (at least in my experience) but weight initialization can have a significant impact on training stability and convergence speed.
# Generally, GPT models initialize their weights using a truncated normal distribution with a mean of 0 and a standard deviation of 0.02.
# This class was used in other projects by me, but I thought it would be useful to include it here as well for anyone who wants to experiment with different initialization strategies and see how it affects convergence. 

class InitStrategy(Enum):
    ORTHOGONAL = 'orthogonal'
    KAIMING = 'kaiming'
    XAVIER = 'xavier'
    GPT_DEFAULT = 'gpt_default'
    DEFAULT = 'default'

    @staticmethod
    def apply_init_weights(module: nn.Module, strategy: "InitStrategy", gain: float = 1.0)-> None:   
        '''Applies the specified weight initialization strategy to the given module.'''
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if strategy == InitStrategy.ORTHOGONAL:
                nn.init.orthogonal_(module.weight, gain=gain)
            elif strategy == InitStrategy.KAIMING:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif strategy == InitStrategy.XAVIER:
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif strategy == InitStrategy.GPT_DEFAULT:
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02) #Interesting stackexchange article on it: https://stats.stackexchange.com/questions/637798/why-the-standard-deviation-of-the-bert-weight-initialization-is-0-02-by-default
            elif strategy == InitStrategy.DEFAULT:
                pass
            else:  # DEFAULT
                print("No weight initialization selected, using default PyTorch initialization for", module)
                pass  # Use PyTorch's default initialization

            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def print_strategies(self):
        print("Available initialization strategies:")
        for strategy in InitStrategy:
            print(f"- {strategy.value}")


