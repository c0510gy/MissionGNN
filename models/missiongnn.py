import torch
from torch import nn
from typing import List
from config import cfg
from graph.kg_loader import load_class_graph
from models.gcn import KnowledgeGCN
from models.temporal import ShortTermTemporal

class MissionGNN(nn.Module):
    def __init__(self, sequence_length: int = 30):
        super().__init__()
        # build one KG‑specific branch per anomaly class
        branches: List[KnowledgeGCN] = []
        for cls in cfg.classes:
            v, e = load_class_graph(cls)
            branches.append(KnowledgeGCN(v.to(cfg.device), e.to(cfg.device)))
        self.branches = nn.ModuleList(branches)
        self.temporal = ShortTermTemporal(sequence_length)

    def forward(self, sensor_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """sensor_seq: [B, T, 1024]  mask: [B, T]"""
        B, T, D = sensor_seq.shape
        per_class_feats = []
        for branch in self.branches:
            # flatten batch*time for branch, then reshape back
            x = sensor_seq.view(B*T, D)
            enc = branch(x)[:,-1,:].view(B, T, -1)
            per_class_feats.append(enc)
        concat = torch.cat(per_class_feats, dim=-1)  # [B, T, num_classes*hidden]
        return self.temporal(concat, mask)
