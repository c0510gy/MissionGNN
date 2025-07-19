import torch
from torch import nn
from config import cfg

class ShortTermTemporal(nn.Module):
    def __init__(self, sequence_length: int = 30):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.cls_head = nn.Linear(cfg.transformer_d_model, len(cfg.classes) + 1)
        self.sequence_length = sequence_length

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats: [B, T, D], mask: [B, T] (0 for padding)."""
        feats = feats * mask.unsqueeze(-1)  # zero‑out padded steps
        out = self.transformer(feats)[:, -1]  # last time‑step
        return self.cls_head(out)
