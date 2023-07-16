
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module: nn.Module, num_of_deep_copies: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class Embedding(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        
    def forward(self, x: torch.Tensor):
        pass
        

class Attention(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 embedding_dim: int,
                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.qkv_nets = get_clones(nn.Linear(input_dim, embedding_dim), 3)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.TensorType
                ):
        query, key, value = [net(x) for net, x in zip(self.qkv_nets, (query, key, value))]
        scaled_dot = torch.matmul(query, key.transpose(-2, -1)).divide(math.sqrt(self.embedding_dim))
        out = F.softmax(scaled_dot, dim=-1)
        out = torch.matmul(out, value)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 embedding_dim: int, 
                 n_heads: int) -> None:
        super().__init__()
        self.heads = get_clones(Attention(input_dim, embedding_dim), n_heads)


class Encoder(nn.Module):
    """Encoder block for Transformer module."""
    def __init__(self, 
                 n_heads: int,
                 model_dim: int,
                 embedding_dim: int,
                 inner_dim: int) -> None:
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
                 model_dim: int,
                 embedding_dim: int,
                 inner_dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass
    

if __name__ == "__main__":
    pass