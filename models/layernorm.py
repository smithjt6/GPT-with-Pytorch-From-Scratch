# this is a from scratch implementation of LayerNorm, which is used in the transformer architecture.
# Citation from Arxiv paper: "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (https://arxiv.org/abs/1607.06450)
# Torch documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html

# Layernorm essentially does the following:
#   1. It computes the mean and variance of the input across the last dimension (features).
#   2. It normalizes the input using these statistics.
#   3. It applies a learnable scale (gamma) and shift (beta) to allow the model to learn an optimal representation.
#   4. It is far easier to use since it is independant of batch size (consider sentences of differing lengths), and it is easier to use in distributed training since it doesn't require communication across devices.

# Layer norm then really just normalizes data across feature dims instead of batches (original batch norm). 
# We will use it in transformer architecture for fun but this implementation is a little slow and I'd suggest the PyTorch version.

import torch

class LayerNorm(torch.nn.Module):
    ''' Simple LayerNorm Implementation'''
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(d_model))  # scale parameter
        self.beta = torch.nn.Parameter(torch.zeros(d_model))  # shift parameter

    def forward(self, x):

        # First, we compute the mean (written as E[x] in torch docs and mu in the paper) - we keepdim=True to maintain the original shape of the input for broadcasting purposes. 
        mean = x.mean(dim=-1, keepdim=True)

        # Secondly, we calculate the variance (Var[x] in torch docs and sigma^2 in the paper). 
        # We set unbiased=False to get the population variance instead of the sample variance, which is more appropriate for normalization.
        # Note that unbiased=False means we divide by N instead of N-1 when calculating the variance, which is what we want for normalization purposes.
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # finally, we normalize the input using the computed mean and variance, and then apply the learnable scale (gamma) and shift (beta) parameters.
        x_normalized = (x - mean) / torch.sqrt(var + self.eps) 
        return self.gamma * x_normalized + self.beta  # scale and shift the normalized input --> This is what we want! 