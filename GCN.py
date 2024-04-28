import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 两层经典GCN模型
class GCN2(nn.Module):
    def __init__(self, in_channel, ff_d, out_channel,  export: bool = False, dropout=0) -> None:
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(in_channels=in_channel, out_channels=ff_d)
        self.conv2 = GCNConv(in_channels=ff_d, out_channels=out_channel)
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, x, edge_index):
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.leaky_relu(hid)
        hid = self.dropout(hid)
        output = self.conv2(x=hid, edge_index=edge_index)
        return output