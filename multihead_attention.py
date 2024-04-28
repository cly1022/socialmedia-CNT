import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        head_num,
        k_dim=None,
        q_dim=None,
        v_dim=None,
        attention_dropout=0.0,
        bias=True,
        self_attention=True,
        ):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.k_dim = k_dim 
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.qkv_same_dim = self.k_dim == embedding_dim and self.v_dim == self.embedding_dim
        self.self_attention = self_attention
        assert self.self_attention, "只支持self-attention"
        assert self.k_dim == self.q_dim == self.v_dim, ("self-attention 要求query, key, value 的shape一致")
        
        self.head_num = head_num
        self.head_dim = self.embedding_dim // self.head_num
        assert (self.head_num * self.head_dim == self.embedding_dim), ("embedding_dim 必须整除 head_num")
        
        self.dropout = nn.Dropout(self.attention_dropout)
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        self.k_proj = nn.Linear(self.embedding_dim, self.k_dim, bias=bias)
        self.v_proj = nn.Linear(self.embedding_dim, self.v_dim, bias=bias)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        
        self.reset_parameters()
        
    
    
    def forward(
                self,
                query: Optional[Tensor],
                key: Optional[Tensor],
                value: Optional[Tensor],
                atten_bias: Optional[Tensor] = None,
                atten_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                ) ->Tuple[Tensor, Optional[Tensor]]:
        batch_size, nodes_num, embedding_dim = query.size()
        assert embedding_dim == self.embedding_dim, f"query dim{embedding_dim} != {self.embedding_dim}"
        
        assert list(key.size()) == [batch_size, nodes_num, embedding_dim], ("key与query不一致")
        assert list(value.size()) == [batch_size, nodes_num, embedding_dim], ("value与query不一致")
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = (
            # tgt_len, bsz * self.num_heads, self.head_dim
            q.contiguous()
            .view(nodes_num, batch_size * self.head_num, self.head_dim)
            .transpose(0,1)
        )
        k = (
            k.contiguous()
            .view(nodes_num, batch_size * self.head_num, self.head_dim)
            .transpose(0,1)
        )
        v = (
            v.contiguous()
            .view(nodes_num, batch_size * self.head_num, self.head_dim)
            .transpose(0,1)
        )
        if key_padding_mask is not None:
         assert list(key_padding_mask.size()) == [batch_size, nodes_num, nodes_num], "key_padding_mask形状不正确"
        
        atten_weights = torch.matmul(q, k.transpose(-1, -2))
        assert list(atten_weights.size()) == [batch_size * self.head_num, nodes_num, nodes_num]

        if atten_bias is not None:
            atten_weights += atten_bias.repeat(self.head_num, 1, 1)
        
        if atten_mask is not None:
            atten_weights += atten_mask.repeat(self.head_num, 1, 1)
        
        if key_padding_mask is not None:
            atten_weights = atten_weights.view(batch_size, self.head_num, nodes_num, nodes_num)
            atten_weights = atten_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float(1e-06))
            atten_weights = atten_weights.view(batch_size * self.head_num, nodes_num, nodes_num)
            
        atten_weights_float = F.softmax(atten_weights, dim=-1)
        atten_weights = atten_weights_float.type_as(atten_weights)
        attn_probs = self.dropout(atten_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn_probs.size()) == [batch_size * self.head_num, nodes_num, nodes_num], ("计算结果有问题")
        
        attn = attn.transpose(0, 1).contiguous().view(nodes_num, batch_size, self.embedding_dim).transpose(0,1)
        return attn, atten_weights
        

            
        
     
        
    def reset_parameters(self):
        # ELN xavier_uniform_均匀分布，_就地操作 || xavier_normal_正态分布
        # ELN这里是对学习的权重w进行操作
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))

            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
        
        nn.init.xavier_uniform_(self.out_proj.weight)

        # ELN 初始化常量一般函数：constant, ones, zeros, eye
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)   
        
    