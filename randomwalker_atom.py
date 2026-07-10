
import torch
from torch import nn
import torch_geometric.nn as gnn
from randomwalker_layer import RandomWalkerLayer


class FeatureEncoder(nn.Module):
    def __init__(self, hidden_size, in_node_dim, in_edge_dim=None):
        super().__init__()

        self.use_edge_attr = in_edge_dim is not None

        # node feature projection
        self.node_embed = nn.Linear(
            in_node_dim,
            hidden_size,
            bias=False
        )

        # edge feature projection
        if self.use_edge_attr:
            self.edge_embed = nn.Linear(
                in_edge_dim,
                hidden_size,
                bias=False
            )

    def forward(self, batch):
        batch.x = self.node_embed(batch.x.float())

        edge_attr = getattr(batch, "edge_attr", None)

        if self.use_edge_attr and edge_attr is not None:
            batch.edge_attr = self.edge_embed(edge_attr.float())
        else:
            batch.edge_attr = None

        return batch
        
class RandomWalkerAtomEncoder(nn.Module):

    def __init__(
        self,
        in_node_dim=101,
        in_edge_dim=12,
        hidden_size=128,
        num_layers=3,
        walk_encoder="conv",          
        walk_length=50,
        window_size=8,
        pad_idx=-1,
        dropout=0.2,
        global_pool="mean",          
        global_mp_type="vn",
        local_mp_type="gin",
        d_state=16,
        d_conv=9,
        expand=2,
        mlp_ratio=2,
        use_encoder_norm=True,
        proj_mlp_ratio=1,
        use_positional_encoding=True,
        bidirection=True,
        num_heads=4,
        attn_dropout=0.0,
        vn_norm_first=True,
        vn_norm_type="batchnorm",
        vn_pooling="sum",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size

     
        #  x is float features (101 dims), edge_attr is float (11 dims)
    
        self.feature_encoder = FeatureEncoder(
            hidden_size=hidden_size,
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
        )

        self.walk_encoder = walk_encoder

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # if global_mp_type == 'vn', last layer often sets global_model_type=None
            global_model_type = None if (global_mp_type == "vn" and i == num_layers - 1) else global_mp_type
            self.blocks.append(
                RandomWalkerLayer(
                    hidden_size=hidden_size,
                    sequence_layer_type=walk_encoder,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    use_encoder_norm=use_encoder_norm,
                    proj_mlp_ratio=proj_mlp_ratio,
                    walk_length=walk_length,
                    use_positional_encoding=use_positional_encoding,
                    pos_embed=False,
                    window_size=window_size,
                    bidirection=bidirection,
                    layer_idx=i,
                    local_gnn_type=local_mp_type,
                    global_model_type=global_model_type,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    vn_norm_first=vn_norm_first,
                    vn_norm_type=vn_norm_type,
                    vn_pooling=vn_pooling,
                )
            )

        self.node_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        if global_pool == "mean":
            self.global_pool = gnn.global_mean_pool
        elif global_pool == "sum":
            self.global_pool = gnn.global_add_pool
        else:
            raise ValueError("global_pool must be 'mean' or 'sum'")

    def forward(self, batch):

        batch.walk_pe = torch.cat(
            [batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1
        )

        batch = self.feature_encoder(batch)

        for block in self.blocks:
            batch = block(batch)

        h = self.node_out(batch.x)

        # graph-level embedding
        hg = self.global_pool(h, batch.batch)
        return hg
