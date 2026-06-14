# This contains the positional embedding layer, which is used to add positional information to the input tokens.
# Now, the original authors in Attention is all you need, used a fixed positional encoder with sine and cosine. 
# Nowadays, RoPE (Rotary Positional Embeddings) is much more efficient, and allows for higher context lengths. 
# ROPE paper: https://arxiv.org/abs/2104.09864 (RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al, 2021)

#ROPE is available in the torch.tune library: https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
# In the future I may add a by scratch implementation of ROPE, but for now I will use the one from torch.tune.

import torch
import torch.nn as nn
import math

# Usually embeddings are cached as well, so as not to recompute them every time. 
# This is done by registering the positional embedding as a buffer, which is a tensor that is not a parameter, but is still part of the module and will be saved and loaded with the model.

class OriginalEmbedding(nn.Module):
    '''Original positional embedding layer, using sine and cosine functions.'''
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Add positional encoding to input of shape (batch, seq_len, d_model).'''
        return x + self.pe[:, :x.size(1), :]