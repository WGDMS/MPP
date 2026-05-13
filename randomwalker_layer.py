import torch
from torch import nn
from einops.layers.torch import Rearrange
import math
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum, scatter_mean, scatter
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils

class WalkEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        sequence_layer_type='conv',
        d_state=16,
        d_conv=9,
        expand=2,
        num_heads=4,
        mlp_ratio=1,
        use_encoder_norm=True,
        proj_mlp_ratio=1,
        dropout=0.,
        walk_length=50,
        use_positional_encoding=True,
        pos_embed=False,
        window_size=8,
        bidirection=False,
        layer_idx=None,
    ):
        super().__init__()

        self.use_positional_encoding= use_positional_encoding
        walk_pe_dim = window_size * 2 - 1 if use_positional_encoding else 0
        self.bidirection = bidirection
        self.edge_proj = nn.Linear(hidden_size, hidden_size)
        self.walk_pe_proj = nn.Linear(walk_pe_dim, hidden_size)

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, walk_length, hidden_size), requires_grad=False
            )
            self.pos_embed.data.copy_(
                get_1d_sine_pos_embed(hidden_size, walk_length).unsqueeze(0)
            )

        # build sequence layer
        self.norm = None
        if use_encoder_norm:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.seq_layer_backward = None
        self.sequence_layer_type = sequence_layer_type
        if sequence_layer_type == "conv":
            self.seq_layer = nn.Sequential(
                Rearrange("a b c -> a c b"),
                nn.Conv1d(hidden_size, hidden_size, d_conv, groups=hidden_size, padding=d_conv // 2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1, padding=0),
                nn.ReLU(),
                Rearrange("a b c -> a c b")
            )
        elif sequence_layer_type == "s4":
            from .s4 import S4Block
            self.seq_layer = S4Block(
                d_model=hidden_size,
                lr=0.001
            )
            if bidirection:
                self.seq_layer_backward = S4Block(
                    d_model=hidden_size,
                    lr=0.001
                )
        elif sequence_layer_type == 'mamba':
            from mamba_ssm import Mamba
            self.seq_layer = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
            )
            if bidirection:
                self.seq_layer_backward = Mamba(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=layer_idx,
                )
        else:
            raise NotImplementedError(
                f"not supported sequence layer type: {sequence_layer_type}"
            )
        print(sequence_layer_type)
        self.out_node_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
            nn.BatchNorm1d(hidden_size * proj_mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * proj_mlp_ratio, hidden_size)
        )

        self.out_edge_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
            nn.BatchNorm1d(hidden_size * proj_mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * proj_mlp_ratio, hidden_size)
        )

    def forward(self, batch):
        x = batch.x
        edge_attr = batch.edge_attr
        walk_node_index, walk_edge_index = batch.walk_node_idx, batch.walk_edge_idx
        walk_node_mask, walk_edge_mask = batch.walk_node_mask, batch.walk_edge_mask
        walk_pe = batch.walk_pe

        walk_x = x[walk_node_index]
        walk_x = torch.where(walk_node_mask[:, :, None], 0., walk_x)


        use_edges = (edge_attr is not None) and (edge_attr.size(0) > 0)

        if use_edges:
       
            we = walk_edge_index.clone()
            valid = (we >= 0) & (we < edge_attr.size(0))
            we = we.clamp(min=0, max=edge_attr.size(0) - 1)

            walk_e = edge_attr[we]
          
            edge_mask = walk_edge_mask | (~valid)
            walk_e = torch.where(edge_mask[:, :, None], 0., walk_e)

            walk_x = walk_x + self.edge_proj(walk_e)
            del walk_e
            
        if self.use_positional_encoding and walk_pe is not None:
            walk_x = walk_x + self.walk_pe_proj(walk_pe)

        if self.pos_embed is not None:
            walk_x = walk_x + self.pos_embed

        if self.norm is not None:
            walk_x = self.norm(walk_x)

        if self.sequence_layer_type == 'transformer':
            walk_x_forward = self.seq_layer(walk_x, ~walk_node_mask)
        else:
            walk_x_forward = self.seq_layer(walk_x)

        if self.seq_layer_backward is not None:
            walk_x_backward = self.seq_layer_backward(walk_x.flip([1])).flip([1])
            walk_x_forward = (walk_x_forward + walk_x_backward) * 0.5
            del walk_x_backward

        walk_x = walk_x_forward

        node_agg = scatter_mean(
            walk_x[~walk_node_mask],
            walk_node_index[~walk_node_mask],
            dim=0,
            dim_size=batch.num_nodes,
        )

        x = x + self.out_node_proj(node_agg)

        del node_agg

        edge_agg = scatter_mean(
            walk_x[~walk_edge_mask],
            walk_edge_index[~walk_edge_mask],
            dim=0,
            dim_size=batch.edge_index.shape[-1],
        )

        if edge_attr is not None:
            edge_attr = edge_attr + self.out_edge_proj(edge_agg)
        else:
            edge_attr = self.out_edge_proj(edge_agg)

        batch.x = x
        batch.edge_attr = edge_attr
        return batch


class RandomWalkerLayer(nn.Module):
    

    def __init__(
        self,
        hidden_size,
        sequence_layer_type,
        d_state=16,
        d_conv=9,
        expand=2,
        mlp_ratio=1,
        use_encoder_norm=True,
        proj_mlp_ratio=1,
        walk_length=50,
        use_positional_encoding=True,
        pos_embed=False,
        window_size=8,
        bidirection=False,
        layer_idx=None,
        local_gnn_type='gin',
        global_model_type=None,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vn_norm_first=True,
        vn_norm_type='batchnorm',
        vn_pooling='sum',
    ):
        super().__init__()

        self.walk_encoder = WalkEncoder(
            hidden_size=hidden_size,
            sequence_layer_type=sequence_layer_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_encoder_norm=use_encoder_norm,
            proj_mlp_ratio=proj_mlp_ratio,
            dropout=dropout,
            walk_length=walk_length,
            use_positional_encoding=use_positional_encoding,
            pos_embed=pos_embed,
            window_size=window_size,
            bidirection=bidirection,
            layer_idx=layer_idx,
        )

        self.mp_layer = MessagePassingLayer(
            hidden_size=hidden_size,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling
        )

    def forward(self, batch):
        batch = self.walk_encoder(batch)
        batch = self.mp_layer(batch)
        return batch
        

def get_1d_sine_pos_embed(dim, length, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )
    pos = torch.arange(0, length)
    args = pos[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
    
class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        local_gnn_type='gin',
        global_model_type=None,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vn_norm_first=True,
        vn_norm_type='batchnorm',
        vn_pooling='sum',
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.local_mp = GINConv(
            hidden_size,
            dropout
        )

        self.global_model_type = global_model_type
        self.global_mp = get_global_layer(
            global_model_type,
            hidden_size,
            num_heads,
            attn_dropout,
            dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling,
        )

        self.use_attn = global_model_type is not None and global_model_type != 'vn'
        self.use_ff = self.use_attn

        if self.use_attn:
            self.norm1_local = nn.BatchNorm1d(hidden_size)
            self.norm1_attn = nn.BatchNorm1d(hidden_size)


        # Feed Forward block.
        if self.use_ff:
            self.ff_linear1 = nn.Linear(hidden_size, hidden_size * 2)
            self.ff_linear2 = nn.Linear(hidden_size * 2, hidden_size)
            self.act_fn_ff = nn.ReLU()
            self.norm2 = nn.BatchNorm1d(hidden_size)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in = h
        edge_attr = batch.edge_attr

        if h is None:
            raise ValueError("batch.x is None before local message passing")
        
        if self.local_mp is not None:
            h, edge_attr = self.local_mp(h, batch.edge_index, edge_attr)

        if h is None:
            raise ValueError("h became None after local message passing")


        if self.global_model_type is not None:
            if self.global_model_type == 'vn':
                h = self.global_mp(h, batch)
            else:
                h = self.norm1_local(h)
                h_dense, mask = to_dense_batch(h, batch.batch)
                if self.global_model_type == 'transformer':
                    h_attn = self._sa_block(h_dense, None, ~mask)[mask]
                elif self.global_model_type == 'performer':
                    h_attn = self.global_mp(h_dense, mask=mask)[mask]
                else:
                    raise RuntimeError(f"Unexpected {self.global_model_type}")

                h_attn = h_in + h_attn  # Residual connection.
                h = self.norm1_attn(h_attn)
                del h_attn, h_in

        if self.use_ff:
            h = self.norm2(h + self._ff_block(h))

        batch.x = h
        batch.edge_attr = edge_attr
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.global_mp(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


def get_global_layer(
    global_model_type,
    hidden_size,
    num_heads,
    attn_dropout,
    dropout,
    vn_norm_first=True,
    vn_norm_type='batchnorm',
    vn_pooling='sum',
):
    if global_model_type == 'transformer':
        return nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )
    elif global_model_type == 'performer':
        from performer_pytorch import SelfAttention
        return SelfAttention(
            dim=hidden_size, heads=num_heads,
            dropout=attn_dropout, causal=False
        )
    elif global_model_type == 'vn':
        return VirtualNodeLayer(
            hidden_size, dropout,
            norm_first=vn_norm_first, norm_type=vn_norm_type,
            pooling=vn_pooling
        )
    else:
        return None



class VirtualNodeLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.0, norm_first=True, norm_type='batchnorm', pooling='sum'):
        super().__init__()

        self.pooling = pooling
        norm_fn = nn.LayerNorm if norm_type == 'layernorm' else nn.BatchNorm1d

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            norm_fn(hidden_size) if norm_first else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            norm_fn(hidden_size) if not norm_first else nn.Identity()
        )

    def forward(self, x, batch):
        if x is None:
            raise ValueError("x is None inside VirtualNodeLayer.forward()")

        if batch.batch is None:
            raise ValueError("batch.batch is None inside VirtualNodeLayer.forward()")

        virtual_node = getattr(batch, 'virtual_node', None)

        if self.pooling == 'mean':
            h = scatter_mean(x, batch.batch, dim=0)
        else:
            h = scatter_sum(x, batch.batch, dim=0)

        if virtual_node is not None:
            h = h + virtual_node

        virtual_node = self.mlp(h)
        x = x + virtual_node[batch.batch]
        batch.virtual_node = virtual_node
        return x
  
  
class GINConv(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.model = gnn.GINEConv(mlp, train_eps=True, edge_dim=hidden_size)

    #def forward(self, x, edge_index, edge_attr):
     #   return x + self.model(x, edge_index, edge_attr), edge_attr
        
        
    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), x.size(-1)))

        out = self.model(x, edge_index, edge_attr)

        if out is None:
            raise ValueError("GINEConv returned None")

        return x + out, edge_attr