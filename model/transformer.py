"""
    Inspired by Aleksa Gordic's implementation at https://github.com/gordicaleksa/pytorch-original-transformer
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module: nn.Module, num_of_deep_copies: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class PositionalEmbedding(nn.Module):
    """Implementation of Positional Encoding function."""
    def __init__(self, 
                 p_drop: float,
                 max_len: int,
                 d_model: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        self.src_embed = nn.Embedding(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                            * (-math.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)  # parameter is saved but not trained
        
    def forward(self, x: torch.Tensor):
        x = self.src_embed(x) + nn.Parameter(self.pe, requires_grad=False)     
        return self.dropout(x)
    

# class Attention(nn.Module):
#     """Implementation of Scaled Dot-Product Attention module."""
#     def __init__(self, 
#                  d_model: int, 
#                  d_k: int,
#                  ) -> None:
#         super().__init__()
#         self.d_k = d_k
#         self.qkv_nets = get_clones(nn.Linear(d_model, d_k), 3)
        
#     def forward(self, 
#                 query: torch.Tensor, 
#                 key: torch.Tensor, 
#                 value: torch.Tensor
#                 ):
#         """
#         Args:
#             query (torch.Tensor): query tensor, shape (B, max_len, d_k)
#             key (torch.Tensor): key tensor, shape (B, max_len, d_k)
#             value (torch.Tensor): value tensor, shape (B, max_len, d_k)

#         Returns:
#             torch.Tensor: output tensor, shape (B, max_len, d_k)
#         """
#         query, key, value = [net(x) for net, x in zip(self.qkv_nets, (query, key, value))]
#         scaled_dot = torch.matmul(query, key.transpose(-2, -1)).divide(math.sqrt(self.d_k))
#         out = F.softmax(scaled_dot, dim=-1)
#         out = torch.matmul(out, value)
#         return out

def attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor
    ):
    """Perform Scaled Dot-Product Attention Function
    
    Args:
        d_k: dimension of projected vectors
        query (torch.Tensor): query tensor, shape (B, max_len, n_heads*d_k)
        key (torch.Tensor): key tensor, shape (B, max_len, n_heads*d_k)
        value (torch.Tensor): value tensor, shape (B, max_len, n_heads*d_k)

    Returns:
        torch.Tensor: output tensor, shape (B, max_len, n_heads*d_k)
    """
    d_k  = query.size(-1)
    scaled_dot = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k))
    p_attn = F.softmax(scaled_dot, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_k: int, 
                 d_v: int,
                 n_heads: int,
                 p_drop: int
                 ) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.linears = get_clones(nn.Linear(d_model, n_heads*d_k), 4)
        self.dropout = nn.Dropout(p_drop)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor
                ):
        """
        Args:
            query (torch.Tensor): query tensor, shape (B, max_len, d_model)
            key (torch.Tensor): key tensor, shape (B, max_len, d_model)
            value (torch.Tensor): value tensor, shape (B, max_len, d_model)

        Returns:
            torch.Tensor: output tensor, shape (B, max_len, d_model)
        """
        n_batches = x.size(0)
        
        # Linearly project query, key, and value to different heads
        query, key, value = [net(x).view(n_batches, -1, self.n_heads, self.d_k) 
                             for net, x in zip(self.linears[:-1], (query, key, value))]
        
        # Perform attention function in parallel
        x, self.attn = attention(query, key, value)
        
        # Project the final time
        x = self.fc(x.view(n_batches, -1, self.n_heads * self.d_k))
        
        return x

class LayerNorm(nn.Module):
    def __init__(self,
                 features: int,
                 eps: float = 1e-6
                 ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        pass
    

class Encoder(nn.Module):
    """Encoder block for Transformer module."""
    def __init__(self, 
                 n_heads: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 d_inner: int
                 ) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        pass
    

class Decoder(nn.Module):
    """Decoder Block for Transformer module."""
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        pass
    

class Transformer(nn.Module):
    """
    Transformer Module in the paper "Attention is all you need". 
    Link: https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, 
                 n_heads: int,
                 d_model: int,
                 d_k: int,
                 d_inner: int
                 ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass
    

if __name__ == "__main__":
    pass