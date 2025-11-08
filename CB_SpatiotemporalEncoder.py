import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from CB_TemporalCNN import TemporalCNN

class SpatiotemporalEncoder(nn.Module):
    def __init__(self, cnn_in_channels, cnn_out_channels, cnn_kernel_size,
                 gat_hidden_channels, gat_out_channels, gat_heads, dropout,
                 cnn_negative_slope=0.01):
        super().__init__()

        self.temporal_cnn = TemporalCNN(
            in_channels=cnn_in_channels,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            negative_slope=cnn_negative_slope
        )

        self.gat_layer_1 = GATv2Conv(
            in_channels=cnn_out_channels,
            out_channels=gat_hidden_channels,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True  
        )

        gat_in_channels_2 = gat_hidden_channels * gat_heads

        self.gat_layer_2 = GATv2Conv(
            in_channels=gat_in_channels_2,
            out_channels=gat_out_channels,
            heads=1,  
            concat=False,
            dropout=dropout,
            add_self_loops=True
        )

        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(gat_in_channels_2)
        self.norm2 = nn.LayerNorm(gat_out_channels)

    def forward(self, x, edge_index):
        
        h_cnn = self.temporal_cnn(x)
        
    
        h_gat1 = self.gat_layer_1(h_cnn, edge_index)
        h_gat1 = self.norm1(h_gat1) 
        h_gat1 = F.elu(h_gat1)
        h_gat1 = self.dropout(h_gat1)
        
        
        h_gat2 = self.gat_layer_2(h_gat1, edge_index)
        h_gat2 = self.norm2(h_gat2) 
        h_gat2 = F.elu(h_gat2)  
        
        return h_gat2