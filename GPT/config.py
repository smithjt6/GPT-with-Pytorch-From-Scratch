
#TODO: docstrinf explanation 

from dataclasses import dataclass

@dataclass
class GPTConfig:
    '''Configuration for the GPT model and training.'''
    block_size: int = 1024              # The maximum token input length, this is how many tokens the model can attend to.
    vocab_size: int = 50257             # The size of the vocabulary, this is the number of unique tokens in the dataset. 50257 is the size of the GPT-2 vocabulary.
    n_layer: int = 12                   # The number of transformer blocks, this is how many times we will apply the attention mechanism and the MLP. This is also known as the depth of the model.
    num_heads: int = 8                  # The number of attention heads, this is how many parallel attention mechanisms we will have.
    n_embd: int = 512                   # The dimension of the embedding, this is the size of the input and output vectors for each head.
    dropout: float = 0.1                # The dropout rate, this is used to prevent overfitting.
    use_from_scratch: bool = True       # Whether to use pytorch functions or the from scratch implementations. This is fun just to see the differences in training time.
    num_steps: int = 1000               # The number of training steps, this is how many times we will update the model's weights during training.
    learning_rate: float = 3e-4         # The learning rate, this is how much we will update the model's weights during training.
    batch_size: int = 64                # The batch size, this is how many samples we will use to calculate the loss and update the model's weights during training.
    weight_decay: float = 0.01          # The weight decay, this is used to prevent overfitting by adding a penalty to the loss function for large weights.