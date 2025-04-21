import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias= True, out_proj_bias= True ):
        super().__init__()

        # in_proj is W_q, W_k, W_v
        # out_proj is W_out
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias= in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed ,bias= True )
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads 

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (batch_sze, seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embedd = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)
        # (bs, seq_len, dim) -> (bs, seq_len, dim*3) -> 3 tensors of shape (bs, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (bs, seq_len, dim) -> (bs, seq_len, h, dim/h) -> (bs, h, seq_len, dim/h)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1,2)

        # (bs, h, seq_len, seq_len )
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triange (above the principal diagonal) is made up of 1 
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim= -1)
        output = weight @ v
        #(bs, h, seq_len, dim/h) -> (bs, seq_len, h, dim / h)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias= False):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = n_heads // d_embed

    def forward(self, x: torch.Tensor, context: torch.Tensor):

        # x: (bs, seqlen_q, Dim_q)
        # context: (bs, seqlen_kv, Dim_kv) or (bs, seq_len, embedd_size) = (bs, 77, 768)

        input_shape= x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head )

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        q = q.view(interim_shape).transpose(1, 2) #( bs, n_heads, seq_len, d_head)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight , dim = -1)

        output =  weight @ v
        output = output.transpose(1, 2).contigous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output






