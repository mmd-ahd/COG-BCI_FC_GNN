import torch
import torch.nn as nn

from CB_SpatiotemporalEncoder import SpatiotemporalEncoder
from CB_TemporalTransformerEncoder import TemporalTransformerEncoder
from CB_HierarchicalAttentionPooling import HierarchicalAttentionPooling
from CB_MLPClassifier import MLPClassifier

class DCGATT(nn.Module):
    def __init__(self, num_nodes, seq_len, num_classes, cnn_params, gat_params, transformer_params, pooling_params, mlp_params):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.spatiotemporal_encoder = SpatiotemporalEncoder(
            cnn_in_channels=cnn_params['in_channels'],
            cnn_out_channels=cnn_params['out_channels'],
            cnn_kernel_size=cnn_params['kernel_size'],
            gat_hidden_channels=gat_params['hidden_channels'],
            gat_out_channels=gat_params['out_channels'],
            gat_heads=gat_params['heads'],
            dropout=gat_params['dropout'],
            cnn_negative_slope=cnn_params.get('negative_slope', 0.01)
        )

        transformer_input_dim = gat_params['out_channels']

        self.temporal_transformer = TemporalTransformerEncoder(
            input_dim=transformer_input_dim,
            num_layers=transformer_params['num_layers'],
            num_heads=transformer_params['num_heads'],
            hidden_dim=transformer_params['hidden_dim'],
            dropout=transformer_params['dropout'],
            max_seq_len=seq_len + 4
        )

        pooling_input_dim = transformer_input_dim

        self.hierarchical_pooling = HierarchicalAttentionPooling(
            input_dim=pooling_input_dim,
            spatial_gate_hidden_dim=pooling_params.get('spatial_gate_hidden_dim', 128),
            temporal_attention_hidden_dim=pooling_params.get('temporal_attention_hidden_dim', 128)
        )

        mlp_input_dim = pooling_input_dim
        self.output_norm = nn.LayerNorm(mlp_input_dim)

        self.mlp_classifier = MLPClassifier(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_params['hidden_dim'],
            output_dim=num_classes,
            dropout=mlp_params['dropout']
        )

    def forward(self, x_sequence, edge_index_sequence):
        batch_size = x_sequence[0].shape[0] // self.num_nodes

        all_time_step_embeddings = []
        for t in range(self.seq_len):
            x_t = x_sequence[t]
            edge_index_t = edge_index_sequence[t]

            h_t = self.spatiotemporal_encoder(x_t, edge_index_t)
            all_time_step_embeddings.append(h_t)

        stacked_embeddings = torch.stack(all_time_step_embeddings, dim=1)

        feature_dim = stacked_embeddings.shape[-1]
        transformer_input = stacked_embeddings.view(batch_size, self.num_nodes, self.seq_len, feature_dim)

        transformer_output = self.temporal_transformer(transformer_input)

        epoch_embedding, temporal_weights = self.hierarchical_pooling(transformer_output)

        epoch_embedding_norm = self.output_norm(epoch_embedding)

        logits = self.mlp_classifier(epoch_embedding_norm)

        return logits


if __name__ == "__main__":
    num_nodes = 63
    seq_len = 26
    num_classes = 3

    cnn_params = {
        'in_channels': 1,
        'out_channels': 32,
        'kernel_size': 10,
        'negative_slope': 0.01
    }
    gat_params = {
        'hidden_channels': 128,
        'out_channels': 128,
        'heads': 8,
        'dropout': 0.2
    }
    transformer_params = {
        'num_layers': 2,
        'num_heads': 8,
        'hidden_dim': 512,
        'dropout': 0.2
    }
    pooling_params = {
        'spatial_gate_hidden_dim': 128,
        'temporal_attention_hidden_dim': 128
    }
    mlp_params = {
        'hidden_dim': 64,
        'dropout': 0.5
    }

    model = DCGATT(
        num_nodes=num_nodes,
        seq_len=seq_len,
        num_classes=num_classes,
        cnn_params=cnn_params,
        gat_params=gat_params,
        transformer_params=transformer_params,
        pooling_params=pooling_params,
        mlp_params=mlp_params
    )

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
import torch
import torch.nn as nn

from CB_SpatiotemporalEncoder import SpatiotemporalEncoder
from CB_TemporalTransformerEncoder import TemporalTransformerEncoder
from CB_HierarchicalAttentionPooling import HierarchicalAttentionPooling
from CB_MLPClassifier import MLPClassifier

class DCGATT(nn.Module):
    def __init__(self, num_nodes, seq_len, num_classes, cnn_params, gat_params, transformer_params, pooling_params, mlp_params):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.spatiotemporal_encoder = SpatiotemporalEncoder(
            cnn_in_channels=cnn_params['in_channels'],
            cnn_out_channels=cnn_params['out_channels'],
            cnn_kernel_size=cnn_params['kernel_size'],
            gat_hidden_channels=gat_params['hidden_channels'],
            gat_out_channels=gat_params['out_channels'],
            gat_heads=gat_params['heads'],
            dropout=gat_params['dropout'],
            cnn_negative_slope=cnn_params.get('negative_slope', 0.01)
        )

        transformer_input_dim = gat_params['out_channels']

        self.temporal_transformer = TemporalTransformerEncoder(
            input_dim=transformer_input_dim,
            num_layers=transformer_params['num_layers'],
            num_heads=transformer_params['num_heads'],
            hidden_dim=transformer_params['hidden_dim'],
            dropout=transformer_params['dropout'],
            max_seq_len=seq_len + 4
        )

        pooling_input_dim = transformer_input_dim

        self.hierarchical_pooling = HierarchicalAttentionPooling(
            input_dim=pooling_input_dim,
            spatial_gate_hidden_dim=pooling_params.get('spatial_gate_hidden_dim', 128),
            temporal_attention_hidden_dim=pooling_params.get('temporal_attention_hidden_dim', 128)
        )

        mlp_input_dim = pooling_input_dim
        self.output_norm = nn.LayerNorm(mlp_input_dim)

        self.mlp_classifier = MLPClassifier(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_params['hidden_dim'],
            output_dim=num_classes,
            dropout=mlp_params['dropout']
        )

    def forward(self, x_sequence, edge_index_sequence):
        batch_size = x_sequence[0].shape[0] // self.num_nodes

        all_time_step_embeddings = []
        for t in range(self.seq_len):
            x_t = x_sequence[t]
            edge_index_t = edge_index_sequence[t]

            h_t = self.spatiotemporal_encoder(x_t, edge_index_t)
            all_time_step_embeddings.append(h_t)

        stacked_embeddings = torch.stack(all_time_step_embeddings, dim=1)

        feature_dim = stacked_embeddings.shape[-1]
        transformer_input = stacked_embeddings.view(batch_size, self.num_nodes, self.seq_len, feature_dim)

        transformer_output = self.temporal_transformer(transformer_input)

        epoch_embedding, temporal_weights = self.hierarchical_pooling(transformer_output)

        epoch_embedding_norm = self.output_norm(epoch_embedding)

        logits = self.mlp_classifier(epoch_embedding_norm)

        return logits

if __name__ == "__main__":
    num_nodes = 62
    seq_len = 26
    num_classes = 3

    cnn_params = {
        'in_channels': 1,
        'out_channels': 64,
        'kernel_size': 5,
        'negative_slope': 0.01
    }
    gat_params = {
        'hidden_channels': 128,
        'out_channels': 128,
        'heads': 8,
        'dropout': 0.2
    }
    transformer_params = {
        'num_layers': 4,
        'num_heads': 8,
        'hidden_dim': 512,
        'dropout': 0.2
    }
    pooling_params = {
        'spatial_gate_hidden_dim': 128,
        'temporal_attention_hidden_dim': 128
    }
    mlp_params = {
        'hidden_dim': 128,
        'dropout': 0.5
    }

    model = DCGATT(
        num_nodes=num_nodes,
        seq_len=seq_len,
        num_classes=num_classes,
        cnn_params=cnn_params,
        gat_params=gat_params,
        transformer_params=transformer_params,
        pooling_params=pooling_params,
        mlp_params=mlp_params
    )

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")