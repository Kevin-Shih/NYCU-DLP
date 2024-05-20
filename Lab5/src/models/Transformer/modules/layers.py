import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = dim  #768
        self.perhead_dim = dim//num_heads #48
        self.attn_drop = attn_drop

        self.linear_in = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.attn_drop)

    def forward(self, x: Tensor):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        bs, token_len, _ = x.size()
        qkv:Tensor = self.linear_in(x)
        # divide to 16H and swap H & T then chunk it into q,k,v
        q, k, v = qkv.reshape(bs, token_len, self.num_heads, 3 * self.perhead_dim)\
                        .permute(0, 2, 1, 3).chunk(3, dim=-1) #-> 32B, 16H, 256T(seq), 48dim
        dk = q.size()[-1]
        # see 32B, 16H as batch like, compute at once
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk) # QK^T/sqrt(dk) 
        attention = F.softmax(scores, dim= -1)
        attention = self.dropout(attention)
        # swap H & T then merge all heads
        out = torch.matmul(attention, v)\
                .permute(0, 2, 1, 3).reshape(bs, token_len, self.embed_dim) #-> 32B, 256T(seq), 768dim

        out = self.linear_out(out)
        return out

    
class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    