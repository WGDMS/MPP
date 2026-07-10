import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric import utils
from walks import get_random_walks

BOND_ORDER_WEIGHT = np.array([1.0, 2.0, 3.0, 1.5], dtype=np.float32)  # S,D,T,Arom


def compute_edge_weights(edge_attr, w_conj=0.5, w_ring=0.3, eps=1e-6):

    if edge_attr is None or edge_attr.size(0) == 0:
        return None

    ea = edge_attr.detach().cpu().numpy().astype(np.float32)

    bond_order = ea[:, 0:4] @ BOND_ORDER_WEIGHT
    conjugated = ea[:, 4]
    in_ring = ea[:, 5]

    weights = bond_order + w_conj * conjugated + w_ring * in_ring + eps

    return weights.astype(np.float32)

def sample_random_walks(
    edge_index,
    num_nodes=None,
    length=50,
    sample_rate=1.0,
    backtracking=False,
    strict=False,
    window_size=8,
    pad_value=-1,
    rng=None,
    edge_weights=None,         
    ):
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)

  
    if edge_weights is not None and edge_index.size(1) > 0:
        sp = utils.to_scipy_sparse_matrix(
            edge_index,
            edge_attr=torch.as_tensor(edge_weights, dtype=torch.float),
            num_nodes=num_nodes,
        ).tocsr()
        weighted_data = sp.data.astype(np.float32)          
        
        csr_matrix = sp.astype(np.int32)
        csr_matrix.data[:] = 1                               
    else:
        csr_matrix = utils.to_scipy_sparse_matrix(
            edge_index, num_nodes=num_nodes
        ).astype(np.int32).tocsr()
        weighted_data = None

    return get_random_walks(
        csr_matrix,
        length,
        sample_rate,
        pad_value,
        backtracking,
        strict,
        window_size,
        rng,
        edge_weights=weighted_data,   # NEW
    )


class WalkData(Data):
    def __init__(self, data=None):
        data_dict = {} if data is None else {k: v for k, v in data}
        super().__init__(**data_dict)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "walk_node_idx":
            return self.num_nodes
        if key == "walk_edge_idx":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class RandomWalkSampler:
    def __init__(
        self,
        length=50,
        sample_rate=1.0,
        backtracking=False,
        strict=False,
        pad_idx=-1,
        window_size=8,
        sampling_mode="uniform",
        w_conj=0.5,
        w_ring=0.3,
        rng=None,                      
    ):
        self.length = length
        self.sample_rate = sample_rate
        self.backtracking = backtracking
        self.strict = strict
        self.pad_idx = pad_idx
        self.window_size = window_size
        self.sampling_mode = sampling_mode
        self.w_conj = w_conj
        self.w_ring = w_ring
        self.rng = rng                  
        


    def __call__(self, data):
        if not data.is_coalesced():
            data = data.coalesce()

        edge_weights = None
        if self.sampling_mode == "chem":
            edge_weights = compute_edge_weights(
                getattr(data, "edge_attr", None),
                w_conj=self.w_conj,
                w_ring=self.w_ring,
            )

        (
            walk_node_index,
            walk_edge_index,
            walk_node_id_encoding,
            walk_node_adj_encoding,
        ) = sample_random_walks(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            length=self.length,
            sample_rate=self.sample_rate,
            backtracking=self.backtracking,
            strict=self.strict,
            window_size=self.window_size,
            pad_value=self.pad_idx,
            rng=self.rng,              
            edge_weights=edge_weights,
        )
        

        data.walk_node_idx = torch.tensor(np.asarray(walk_node_index), dtype=torch.long)
        data.walk_edge_idx = torch.tensor(np.asarray(walk_edge_index), dtype=torch.long)
        data.walk_node_id_encoding = torch.tensor(np.asarray(walk_node_id_encoding), dtype=torch.float)
        data.walk_node_adj_encoding = torch.tensor(np.asarray(walk_node_adj_encoding), dtype=torch.float)

        data.walk_node_mask = data.walk_node_idx == self.pad_idx
        data.walk_edge_mask = data.walk_edge_idx == self.pad_idx

        return WalkData(data)
