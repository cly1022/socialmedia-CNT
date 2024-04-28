'''
Descripttion: 
version: 
@School: 兰州理工大学
Author: 瞿继焘
Date: 2023-07-19 09:42:04
LastEditors: 瞿继焘
LastEditTime: 2024-04-19 20:09:14
'''
import torch 
import torch.nn as nn
from infltransformer_encoder_layer import InflTransformerEncoderLayer

from layer_drop import LayerDropModuleList

class InflTransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        layer_num,
        attention_ffn_embedding_dim,
        attention_head_num,
        attention_dropout,
        attention_ffn_dropout,
        attention_pre_layernorm,
        encoder_dropout,
        layer_dropout,
        
        ) -> None:
        super(InflTransformerEncoder, self).__init__()
        self.embedidng_dim = embedding_dim
        self.layer_num = layer_num
        self.dropout_module = nn.Dropout(encoder_dropout)
        if layer_dropout > 0.0:
            self.layers = LayerDropModuleList(p=layer_dropout)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            InflTransformerEncoderLayer(embedding_dim=embedding_dim, ffn_embedding_dim=attention_ffn_embedding_dim, attention_head_num=attention_head_num, attention_dropout=attention_dropout,  ffn_dropout=attention_ffn_dropout, pre_layernorm=attention_pre_layernorm)
            for  _ in range(layer_num)
        ])
        
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.fc1 = nn.Linear(self.embedidng_dim, ffn_embedding_dim,)
        self.fc2 = nn.Linear(ffn_embedding_dim, 1)
        
    def forward(self, spatial_feature):
        
        x = spatial_feature
        for index, layer in enumerate(self.layers):
            x, _ = layer(x)
        
        re = self.fc1(x)
        self.encoder_dropout(re)
        output = self.fc2(re)
        return output
        
