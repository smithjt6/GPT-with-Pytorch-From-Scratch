# Now we put together the pieces to create the GPT model. To see the configurations see GPT/config.py
#TODO: Add in docs/explanation for the GPT functions. Add in the configuratiions for training

import os
import torch
import torch.nn as nn

from models.attention_block import AttentionBlock
from models.pos_embd import OriginalEmbedding
from models.layernorm import LayerNorm
from config import GPTConfig
from dataclasses import asdict

class GPT(nn.Module):
    '''A basic from scratch implementation of the GPT architecture.'''
    def __init__(self, config: GPTConfig, device = torch.device):
        super().__init__()

        self.config = config
        self.save_path = config.save_path
        self.device = device

        # In "Attention Is All You Need" (Vaswani et al., 2017) and GPT-2/3, absolute positional
        # embeddings are added ONCE — right after the token embedding, before any transformer block.

        # Placing PE inside an AttentionBlock would re-inject position at every layer, which:
        # 1. overwrites features learned by the previous block (if PE replaces values)
        # 2. accumulates positional noise that destabilises deep networks (if PE adds to values).

        # RoPE, however, does NOT live here —-> it is applied to Q and K inside each Attention layer, which is
        # mathematically correct because it only rotates the query/key projections and never touches
        # the residual stream that flows between blocks.
        if config.use_from_scratch:
            self.pos_embd = OriginalEmbedding(d_model=config.n_embd, max_len=config.block_size)

        # The transformer blocks - the original Attention is all you need used 6 transformer blocks stacked ontop of each other
        # GPT-2 (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) used four different models (see Table 2)
        # We will edit the config to work for our small dataset sizes but this is it. Simply a loop (you could also use a module dictionary too but this is a bit neater)
        self.blocks = nn.Sequential(*[AttentionBlock(self.config) for _ in range(self.config.n_layer)])

        # GPT-2 added a LayerNorm after the last transformer block, right before the language-model
        # head. Each AttentionBlock already uses pre-norm (LN before attention and MLP), but that
        # means the *output* of the final block is un-normalised residual stream values — potentially
        # at wildly different scales. ln_f brings everything back to a consistent scale so the linear
        # head can form clean logits. Without it, the head would have to absorb those scale differences
        # into its weights, making training harder.
        self.ln_f = LayerNorm(self.config.n_embd) if config.use_from_scratch else nn.LayerNorm(self.config.n_embd)

        # The final linear projection layer that takes the output and projects it to the size of the vocabulary.
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the GPT model.'''
        # Apply sinusoidal PE here — once, before the stack of transformer blocks.
        # After this point every block sees position-aware token representations, just like the paper.
        if self.config.use_from_scratch:
            x = self.pos_embd(x)

        x = self.blocks(x)
        x = self.ln_f(x)   # normalise residual stream before the logit projection
        x = self.head(x)
        return x

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 end_token_id: int | None = None) -> torch.Tensor:
        '''Autoregressively sample max_new_tokens tokens, stopping early at end_token_id.'''
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # crop to block_size
            logits = self(idx_cond)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if end_token_id is not None and next_token.item() == end_token_id:
                break
        return idx

    def save(self, path: str) -> None:
        '''Save model weights and config to a .pt file.'''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)
        print(f"Model saved --> {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device) -> "GPT":
        '''Load a model saved with save(). Returns a GPT instance ready for inference or further training.'''
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = GPTConfig(**checkpoint["config"])
        model  = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"Model loaded <-- {path}  ({sum(p.numel() for p in model.parameters()):,} params)")
        return model
    
    def list_parameters(self):
        print("Model parameters:")
        for name, param in self.named_parameters():
            print(f"  {name}: {param.shape}")