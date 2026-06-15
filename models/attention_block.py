# Now we combine everythig into the neat little transformer image in figure one of the Attention is all you need paper. 
# NOTE: We are only constructing the decoder! Why? 
# It is because we are generatign text with contextual information! The encoder/decoder approach is useful for task like translation.
# Encoders can also be used to generate embeddings/representations of text, but we leave that for antoher time. 

#TODO: Add in more information and explain step-by-step how this is working in reference to the original paper

import torch
import torch.nn as nn

from GPT.config import GPTConfig
from mlp import MLP
from attention import Attention
from layernorm import LayerNorm

class AttentionBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(self.config.n_embd) if config.use_from_scratch else nn.LayerNorm(self.config.n_embd)
        self.attn = Attention(self.config)
        self.ln2 = LayerNorm(self.config.n_embd) if config.use_from_scratch else nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(d_model=self.config.n_embd, d_hidden = 4*self.config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is PRE-NORM (used by GPT-2 and most modern transformers)
        # LN --> sublayer --> residual
        #
        # The original "Attention Is All You Need" paper used POST-NORM (LN after residual)
        #
        # Pre-norm is more stable for deep networks because the residual stream is
        # normalised BEFORE each sublayer sees it. Post-norm is a headache to work with and GPT-2 shows it performs better to use pre-norm.
        #
        # Pre-norm means the final block's output is never post-normalised,
        # which is why GPT.py adds a standalone ln_f after all the blocks. (see the class GPT)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x