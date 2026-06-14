# Now this is where the magic happens, we will implement the attention mechanism here. 

# First we need to note what exactly attention is, why is it important and how does it work. 
# The best place to start is here: https://arxiv.org/abs/1706.03762 (Attention is all you need, Vaswani et al, 2017)
# Take a look at Figure 1 in the paper, it shows the transformer block, and within it, we see Masked Multi-Head Attention
# This is what we will be creating in this file. attention_block.py will contain the entire transformer block itself.

# So let us start!

import torch
import torch.nn as nn
import torch.nn.functional as F

from GPT.config import GPTConfig
from torchtune.modules import RotaryPositionalEmbeddings
from pos_embd import OriginalEmbedding

class Attention(nn.Module):
    '''Basic implementation of the attention mechanism.'''
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        if config.n_embd % config.num_heads != 0:
            raise ValueError(f"Embedding dimension {config.n_embd} must be divisible by the number of heads {config.num_heads}.")
        
        self.head_dim =config.n_embd//config.num_heads
        self.proj = nn.Linear(config.n_embd, config.n_embd * 3) # this takes input and turns it into  q,k,v
        self.attn = nn.Linear(config.n_embd, config.n_embd) # this takes the output from attention and projects it back into the original dimensions
        
        if config.use_from_scratch:
            # Sinusoidal: applied to x (full d_model) before QKV projection
            self.pos_embd = OriginalEmbedding(d_model=config.n_embd, max_len=config.block_size)
        else:
            # RoPE: applied per-head to Q and K after projection (operates on head_dim)
            self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.block_size)

        self.bias: torch.Tensor
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # This is the causal mask that prevents the model from attending to future tokens. We register it as a buffer so that it is not updated during training and is moved to the correct device when the model is moved.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Computes attention for the input x of shape (batch, seq_len, d_model).'''

        B, T, C = x.size() # batch size, sequence length, embedding dimension

        # Sinusoidal PE is additive to token representations — applied here, before any projection.
        # This means the model learns Q/K/V on already-position-aware tokens.
        if self.config.use_from_scratch:
            x = self.pos_embd(x) # (B, T, n_embd)

        q, k, v = self.proj(x).chunk(3, dim=-1) # (B, T, n_embd) -> (B, T, n_embd) for q, k, v

        q = q.view(B, T, self.config.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.config.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = v.view(B, T, self.config.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        if self.config.use_from_scratch:
            attn = (q @ k.transpose(-2, -1)) * (1/self.head_dim**0.5) # (B, num_heads, T, T)
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, num_heads, T, T)
            # This can seem confusing, but what is happening is look at our buffer in shape (B, num_heads, block_size/query legnth, values)
            # We want to mask the future tokens (since it could cheat anmd just know the answer), so we mask each element ahead of the others. 
            # So for the first token, we mask all tokens ahead of it, for the second token, we mask all tokens ahead of it, and so on.
            # So if block_size = 1024, and the current sequence is 8, we slice out (1,1,8,8) from the prebuilt buffer (1,1,1024,1024)
            # Pytorch broadcasting then expands the (1,1,8,8) to (B, num_heads, 8, 8) and applies the mask to the attention scores.
            # thus, the result after masked fill is a matrix where the upper triangle is -inf and lower diagon/triangle is unchanged
            attn = torch.softmax(attn, dim=-1) # (B, num_heads, T, T)
            # now we multiply the attn score against the values
            attn = attn @ v # (B, num_heads, T, head_dim)
        else:
            #Check Equation 1 in Attention is all you need
            q = self.rope(q)
            k = self.rope(k)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout) # (B, num_heads, T, head_dim)
            
        # Now, we take the components we found, (attn) and we need to project it back into the original dimensions
        # This projection is what is fed into the next layer (MLP/LN)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_embd)
        attn = self.attn(attn) # (B, T, n_embd)

        return attn