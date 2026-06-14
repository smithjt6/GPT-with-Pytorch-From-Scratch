
#TODO: docstrinf explanation 

from dataclasses import dataclass

@dataclass
class GPTConfig:
    '''Configuration for the attention mechanism.'''
    block_size: int = 1024 # The maximum token input length, this is how many tokens the model can attend to.
    num_heads: int = 8 # The number of attention heads, this is how many parallel attention mechanisms we will have.
    n_embd: int = 512 # The dimension of the embedding, this is the size of the input and output vectors for each head.
    dropout: float = 0.1 # The dropout rate, this is used to prevent overfitting.
    use_from_scratch: bool = True # Wether to use pytorch functions or the from scratch implementations. This is fun just to see the differences in training time.