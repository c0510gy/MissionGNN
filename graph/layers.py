"""Custom message-passing layer with k-hop subgraph masking."""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

class GCNConvTarget(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k_hops: int):
        super().__init__(aggr="mean")
        self.k = k_hops
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sensor_idx = x.size(1) - 2  # second last is sensor, last is encoding\
        _, sub_edge_index, _, _ = k_hop_subgraph(sensor_idx, self.k, edge_index,
                                                 relabel_nodes=False, directed=True, flow='target_to_source')
        x = self.lin(x)
        return self.propagate(sub_edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return x_i * x_j
