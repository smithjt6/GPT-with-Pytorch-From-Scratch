# Multilayer Perceptron (MLP) are foundaational building blocks for virtually all of deep learning.
# If you feel uncomfortable with MLPs, I highly recommend you read: "Dive into Deep Learning" https://d2l.ai/chapter_multilayer-perceptrons/index.html and review the basics of MLPs. 

# Note that class MLP is a hardcoded MLP while BaseMLP is a flexible MLP I wrote for a few different projects (RL, ML, and another GPT implementation).
# It was inspired by the MLP in scikit-learn, where you could provide a list of integers for hidden layer dims. I just like torch for torch.compile() and other functionalities.

# Now, MLPs in a transformer block are relatively small (strictly speaking - a single hidden layer), which is perhaps a bit surprising. 
# GPTs often use [4 * number of embedding] as the hidden dimension for the MLP. 

import torch.nn as nn

from utils.utility import get_activation
from typing import List, Optional

class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, use_bias=False): #Note: We set bias to false. I am actually not sure why or the original paper asserting that disabling bias is faster/more stable.
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=use_bias) # This is our first linear layer 
        self.act1 = nn.GELU() # We use GELU activation function, which is the activation function used in the original GPT paper. Relevant link: https://stackoverflow.com/questions/57532679/why-gelu-activation-function-is-used-instead-of-relu-in-bert
                              # GELU paper: https://arxiv.org/abs/1606.08415
        self.fc2 = nn.Linear(d_hidden, d_model, bias=use_bias) # This is our second linear layer that projects back to the original dimension.

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


# ====================================================
# More customizable/abstract MLP class  
# ====================================================

class BaseMLP(nn.Module):
    '''Basic MLP Block that is customizable'''
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str,
                 dropout: float,
                 output_activation: Optional[str] = None,
                 use_bias: bool = False
                 ):
        super().__init__()

        layers = []
        current_dim = input_dim

        # The idea here is to take the list and look, do we need dropout? Do we need activation/which activation? Do we need bias? We can easily customize the MLP by changing the arguments.

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim, bias=use_bias), # The extend here is simply appending layers to the list.
                get_activation(activation),
            ])
            if dropout > 0.0: # Checks to see if we want a dropout layer. If dropout is 0, then we skip it.
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
 
        layers.append(nn.Linear(current_dim, output_dim, bias=use_bias))
        if output_activation:   #Checks if we want an activation function for the output layer. If output_activation is None, then we skip it. Activation functions at output are often used in RL
            layers.append(get_activation(output_activation)) 
        self.network = nn.Sequential(*layers) # Sequenctial is a container module that processes the input through the layers in the order they are added. 
                                              # The * operator is used to unpack the list of layers into individual arguments for nn.Sequential.
    
    def forward(self, x):
        return self.network(x)