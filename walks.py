import numpy as np


def get_random_walks(
    csr_matrix,
    walk_length,
    sample_rate=1.0,
    pad_value=-1,
    backtracking=False,
    strict=False,
    window_size=0,
    rng=None,
):
    if rng is None:
        rng = np.random.RandomState(0)
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    indices = csr_matrix.indices
    indptr = csr_matrix.indptr
    num_nodes = csr_matrix.shape[0]

    degrees = np.asarray(csr_matrix.sum(axis=-1), dtype=np.int32).flatten()

    if sample_rate < 1.0:
        active_nodes = rng.rand(num_nodes) < sample_rate
        active_node_ids = np.where(active_nodes)[0]
    else:
        active_node_ids = np.arange(num_nodes)

    num_active_nodes = len(active_node_ids)

    walk_node_index = np.full(
        (num_active_nodes, walk_length),
        pad_value,
        dtype=np.int64,
    )

    walk_edge_index = np.full(
        (num_active_nodes, walk_length),
        pad_value,
        dtype=np.int64,
    )

    if window_size > 0:
        walk_node_id_encoding = np.zeros(
            (num_active_nodes, walk_length, window_size),
            dtype=np.uint8,
        )

        walk_node_adj_encoding = np.zeros(
            (num_active_nodes, walk_length, window_size - 1),
            dtype=np.uint8,
        )

    for walk_index, start_node in enumerate(active_node_ids):
        node_index = int(start_node)
        prev_index = -1

        walk_node_index[walk_index, 0] = node_index

        for j in range(walk_length - 1):
            degree = int(degrees[node_index])

            if degree == 0:
                break

            edge_start = int(indptr[node_index])
            edge_end = int(indptr[node_index + 1])

            neighbors = []
            edge_indices = []

            for edge_idx in range(edge_start, edge_end):
                neighbor = int(indices[edge_idx])

                if backtracking or neighbor != prev_index:
                    neighbors.append(neighbor)
                    edge_indices.append(edge_idx)

            if len(neighbors) == 0:
                if strict:
                    break
                next_index = prev_index
                edge_index = pad_value
            else:
                choice_index = rng.randint(len(neighbors))
                next_index = neighbors[choice_index]
                edge_index = edge_indices[choice_index]

            walk_node_index[walk_index, j + 1] = next_index
            walk_edge_index[walk_index, j] = edge_index

            prev_index = node_index
            node_index = next_index

            if window_size > 0:
                o = min(window_size, j + 1)

                for s in range(o):
                    adj_prev_node_index = walk_node_index[walk_index, j - s]

                    walk_node_id_encoding[
                        walk_index, j + 1, window_size - 1 - s
                    ] = int(next_index == adj_prev_node_index)

                    if s > 0:
                        adj_start = int(indptr[adj_prev_node_index])
                        adj_end = int(indptr[adj_prev_node_index + 1])

                        for k in range(adj_start, adj_end):
                            if indices[k] > next_index:
                                break
                            if indices[k] == next_index:
                                walk_node_adj_encoding[
                                    walk_index, j + 1, window_size - 1 - s
                                ] = 1
                                break

    if window_size > 0:
        return (
            walk_node_index,
            walk_edge_index,
            walk_node_id_encoding,
            walk_node_adj_encoding,
        )

    return walk_node_index, walk_edge_index