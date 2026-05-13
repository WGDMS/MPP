import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric import utils
from walks import get_random_walks

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
):
    if rng is None:
        rng = np.random.mtrand._rand

    if isinstance(rng, int):
        rng = np.random.RandomState(rng)

    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index,
        num_nodes=num_nodes
    ).astype(np.int32).tocsr()

    return get_random_walks(
        csr_matrix,
        length,
        sample_rate,
        pad_value,
        backtracking,
        strict,
        window_size,
        rng,
    )


class WalkData(Data):
    def __init__(self, data=None):
        if data is None:
            data_dict = {}
        else:
            data_dict = {key: item for key, item in data}

        super().__init__(**data_dict)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "walk_node_idx":
            return self.num_nodes

        if key == "walk_edge_idx":
            return self.edge_index.shape[1]

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
    ):
        self.length = length
        self.sample_rate = sample_rate
        self.backtracking = backtracking
        self.strict = strict
        self.pad_idx = pad_idx
        self.window_size = window_size

    def __call__(self, data):
        if not data.is_coalesced():
            data = data.coalesce()

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
        )

        data.walk_node_idx = torch.from_numpy(walk_node_index).long()
        data.walk_edge_idx = torch.from_numpy(walk_edge_index).long()

        data.walk_node_mask = data.walk_node_idx == self.pad_idx
        data.walk_edge_mask = data.walk_edge_idx == self.pad_idx

        data.walk_node_id_encoding = torch.from_numpy(
            walk_node_id_encoding
        ).float()

        data.walk_node_adj_encoding = torch.from_numpy(
            walk_node_adj_encoding
        ).float()

        return WalkData(data)