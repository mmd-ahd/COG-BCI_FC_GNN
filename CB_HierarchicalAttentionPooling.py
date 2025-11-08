import torch
import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation
from CB_TemporalAttentionPooling import TemporalAttentionPooling

class HierarchicalAttentionPooling(nn.Module):
    def __init__(self, input_dim, spatial_gate_hidden_dim=128, temporal_attention_hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim

        spatial_gate_nn = nn.Sequential(
            nn.Linear(input_dim, spatial_gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(spatial_gate_hidden_dim, 1)
        )
        self.spatial_pool = AttentionalAggregation(gate_nn=spatial_gate_nn)

        self.temporal_pool = TemporalAttentionPooling(
            input_dim=input_dim,
            hidden_dim=temporal_attention_hidden_dim
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        seq_len = x.shape[2]

        pooled_over_nodes_list = []
        batch_index = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        x_permuted = x.permute(2, 0, 1, 3)

        for t in range(seq_len):
            nodes_t = x_permuted[t]
            nodes_t_reshape = nodes_t.reshape(batch_size * num_nodes, self.input_dim)
            pooled_t = self.spatial_pool(nodes_t_reshape, index=batch_index)
            pooled_over_nodes_list.append(pooled_t)

        sequence_summary = torch.stack(pooled_over_nodes_list, dim=1)
        final_epoch_embedding, temporal_weights = self.temporal_pool(sequence_summary)

        return final_epoch_embedding, temporal_weights