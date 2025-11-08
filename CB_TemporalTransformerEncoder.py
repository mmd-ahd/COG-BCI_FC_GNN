import torch
import torch.nn as nn
from CB_PositionalEncoding import PositionalEncoding

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout, max_seq_len=60):
        super().__init__()
        self.input_dim = input_dim
        self.pos_encoder = PositionalEncoding(input_dim, dropout, max_len=max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, seq_node_embeddings):
        batch_size, num_nodes, seq_len, _ = seq_node_embeddings.shape

        x = seq_node_embeddings.permute(2, 0, 1, 3)
        x = x.reshape(seq_len, batch_size * num_nodes, self.input_dim)

        residual = x

        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)

        output = output + residual

        output = self.output_norm(output)

        output = output.reshape(seq_len, batch_size, num_nodes, self.input_dim)
        output = output.permute(1, 2, 0, 3)

        return output