import torch
import torch.nn as nn

from randomwalker_atom import RandomWalkerAtomEncoder


class GraphModel(nn.Module):
    def __init__(
        self,
        n_output=1,
        output_dim=128,
        dropout=0.2,
        walk_encoder="mamba",
        num_layers=3,
        walk_length=50,
        window_size=8,
    ):
        super().__init__()

        self.MolAtomEncoder = RandomWalkerAtomEncoder(
            in_node_dim=101,
            in_edge_dim=12,
            hidden_size=output_dim,
            num_layers=num_layers,
            walk_encoder=walk_encoder,
            walk_length=walk_length,
            window_size=window_size,
            dropout=dropout,
            global_pool="mean",
            global_mp_type="vn",
            local_mp_type="gin",
            d_conv=9,
            d_state=16,
            expand=2,
            mlp_ratio=2,
            proj_mlp_ratio=1,
            bidirection=True,
            use_positional_encoding=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_output),
        )

    def forward(self, data_mol):
        graph_embedding = self.MolAtomEncoder(data_mol)
        output = self.predictor(graph_embedding)
        return output