# Now we combine everythig into the neat little transformer image in figure one of the Attention is all you need paper. 
# NOTE: We are only constructing the decoder! Why? 
# It is because we are generatign text with contextual information! The encoder/decoder approach is useful for task like translation.
# Encoders can also be used to generate embeddings/representations of text, but we leave that for antoher time. 

#TODO: Add in more information and explain step-by-step how this is working in reference to the original paper

import torch

from GPT.config import GPTConfig
from mlp import MLP
from attention import Attention
from layernorm import LayerNorm

class AttentionBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        config = GPTConfig() # we can also pass in a custom config if we want to change the default values.
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(d_model=config.n_embd, d_hidden = 4*config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x)) # attention with residual connection
        x = x + self.mlp(self.ln2(x)) # mlp with residual connection
        return x