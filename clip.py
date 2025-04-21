import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embeddin = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeross(n_tokens, n_embd))

    def forward(self, tokens):
        # (bs, seq_len) -> (bs, seq_len, Dim)
        x = self.token_embeddin(tokens)
        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):
    
    def __init__(self, n_heads: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_heads, n_embd)
        self.layernorm = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd , n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (bs, seq_len, n_embd)
        residue = x
        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask = True)

        x += residue
        
        residue = x

        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702*x) #QuickGELU activation function

        x = self.linear_2(x)

        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self,):
        self.embedding =  CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12) # CLIPLayer(n_heads, embedd_size)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (bs, seq_len) -> (bs, seq_len, Dim)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        # (bs, seq_len, Dim)
        output = self.layernorm(state)
        return output

        



