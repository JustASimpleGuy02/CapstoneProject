"""
    Inspired by https://github.com/gordicaleksa/pytorch-original-transformer 
    and https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module: nn.Module, num_of_deep_copies: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class Embedding(nn.Module):
    """Implementation of Positional Encoding function."""
    def __init__(self, 
                 vocab: int,
                 d_model: int,
                ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    """Implementation of Positional Encoding function."""
    def __init__(self, 
                 max_len: int,
                 d_model: int,
                 p_drop: float,
                ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                            * (-math.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)  # parameter is saved but not trained
        
    def forward(self, x: torch.Tensor):
        x = x + nn.Parameter(self.pe, requires_grad=False)     
        return self.dropout(x)
    

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
    def __init__(self, d_model: int, d_inner: int, p_drop: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_inner)
        self.linear_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.linear_2(F.relu(x))
        return self.dropout(x)
    

class SublayerConnection(nn.Module):
    def __init__(self, size: int, p_drop: int) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p_drop)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        x = self.dropout(self.norm(x + sublayer(x)))
        return x
    

class EncoderLayer(nn.Module):
    """An Encoder Layer for Transformer module."""
    def __init__(self,
                 size: int,
                 multi_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout: float
                ) -> None:
        super().__init__()
        self.multi_attn = multi_attn
        self.ffn = feed_forward
        self.sublayers = get_clones(SublayerConnection(size, dropout), 2)
        
        
    def forward(self, x: torch.Tensor):
        x = self.sublayers[0](x, self.multi_attn(x, x, x))
        x = self.sublayers[1](x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """An Encoder Layer for Transformer module."""
    def __init__(self,
        size: int,
        multi_attn: nn.Module,
        src_attn: nn.Module,
    ) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        return x


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, 
        layer: nn.Module, 
        N: int
    ) -> None:
        super().__init__()
        self.layers = get_clones(layer, N)
        
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder composed of N decoder layers with mask"""
    def __init__(
        self,
        layer,
        N: int
    ) -> None:
        super().__init__()
        self.layers = get_clones(layer, N)
        
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
    )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
    

if __name__ == "__main__":
    pass