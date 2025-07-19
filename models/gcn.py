from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from config import cfg
from graph.layers import GCNConvTarget

class KnowledgeGCN(nn.Module):
    """Per‑class hierarchical GCN operating over a fixed KG."""

    def __init__(self, vertices: torch.Tensor, edge_index: torch.Tensor):
        super().__init__()
        self.register_buffer("base_vertices", vertices)      # [V-2, 1024]
        self.register_buffer("edge_index", edge_index)       # [2, E]

        self.conv1 = GCNConvTarget(cfg.embed_dim, cfg.gnn_hidden, 1)
        self.bn1 = nn.BatchNorm1d(cfg.gnn_hidden)
        self.conv2 = GCNConvTarget(cfg.gnn_hidden, cfg.gnn_hidden, 2)
        self.bn2 = nn.BatchNorm1d(cfg.gnn_hidden)
        self.conv3 = GCNConvTarget(cfg.gnn_hidden, cfg.gnn_hidden, 3)
        self.bn3 = nn.BatchNorm1d(cfg.gnn_hidden)

    # def _forward_single(self, sensor: torch.Tensor) -> torch.Tensor:
    #     """Forward one graph (one video frame). sensor: [1024]"""
    #     zeros = torch.zeros_like(sensor)
    #     x = torch.cat([self.base_vertices, sensor.unsqueeze(0), zeros.unsqueeze(0)], dim=0)
    #     x = self.bn1(self.conv1(x, self.edge_index))
    #     x = self.bn2(self.conv2(x, self.edge_index))
    #     x = self.bn3(self.conv3(x, self.edge_index))
    #     return x[-1]  # encoding node

    def forward(self, sensor_batch: torch.Tensor) -> torch.Tensor:  # [B*seq, 1024]
        x = torch.stack([torch.concatenate([
            self.base_vertices, 
            sensor_input.reshape(1, -1), 
            torch.zeros(1024, device=sensor_batch.device).reshape(1, -1),
        ], dim=0) for sensor_input in sensor_batch])#NOTE Clone this!!
        
        batch_size, num_nodes, _ = x.shape
        
        x = F.elu(self.bn1(self.conv1(x, self.edge_index).reshape(batch_size*num_nodes, -1)).reshape(batch_size, num_nodes, -1))
        x = F.elu(self.bn2(self.conv2(x, self.edge_index).reshape(batch_size*num_nodes, -1)).reshape(batch_size, num_nodes, -1))
        x = F.elu(self.bn3(self.conv3(x, self.edge_index).reshape(batch_size*num_nodes, -1)).reshape(batch_size, num_nodes, -1))
        # x = F.elu(self.bn4(self.conv4(x, self.edge_index).reshape(batch_size*num_nodes, -1)).reshape(batch_size, num_nodes, -1))

        return x
        
        
        # iterate sample‑wise to keep indexing simple & safe on GPU
        outs = [self._forward_single(s) for s in sensor_batch]
        return torch.stack(outs)
