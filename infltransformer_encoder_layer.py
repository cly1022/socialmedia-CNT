
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from multihead_attention import MultiHeadAttention

class InflTransformerEncoderLayer(nn.Module):
    def __init__(
                self,
                embedding_dim: int,
                ffn_embedding_dim: int,
                attention_head_num: int,
                attention_dropout: float,
                ffn_dropout: float,
                pre_layernorm: bool,
                activate_fn: str = "relu",
                export: bool = False
                ) -> None:
        super(InflTransformerEncoderLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.attention_head_num = attention_head_num
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.attention_dropout_module = nn.Dropout(attention_dropout)
        self.pre_layernorm = pre_layernorm
        
        self.self_atten = MultiHeadAttention(embedding_dim, attention_head_num, embedding_dim, embedding_dim, embedding_dim, attention_dropout)
               
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        
    def forward(
                self,
                x: torch.Tensor,
                self_attn_bias: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None,
                self_attn_padding_mask: Optional[torch.Tensor] = None
    ):
        residual = x
        x, atten = self.self_atten(
            query = x,
            key = x,
            value = x,
            atten_bias = self_attn_bias,
            atten_mask = self_attn_mask,
            key_padding_mask = self_attn_padding_mask
        )
        x = self.attention_dropout_module(x)
        x = residual + x
    
        
        residual = x
        
        x = F.relu(self.fc1(x))
        x = self.ffn_dropout(x)
        x = self.fc2(x)
        return x, atten
        
        
        
        

