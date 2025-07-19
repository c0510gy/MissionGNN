"""Load vertices + edges for each anomaly class and build tensors ready for PyG."""
from pathlib import Path
from typing import List, Tuple
import torch
from imagebind.models.imagebind_model import ModalityType
import imagebind

from config import cfg

model = imagebind.models.imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(cfg.device)


def _embed_words(words: List[str]) -> torch.Tensor:
    """Return ImageBind text embeddings stacked into [len(words), 1024]."""
    inputs = {
        ModalityType.TEXT: imagebind.data.load_and_transform_text(
            [f"this CCTV footage is related to {w}" for w in words], cfg.device
        ),
    }
    with torch.no_grad():
        return model(inputs)[ModalityType.TEXT].cpu()


def load_class_graph(class_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (vertex_embeddings, edge_index) for a given anomaly class."""
    # -- read edge list
    with open(cfg.subgraph_dir / f"subgraph_{class_name}.txt", "r") as f:
        edges = [tuple(line.split("->")) for line in f.read().splitlines()]
    vertices = sorted({v for e in edges for v in e})
    
    # build edge index (source, target)
    edge_index = torch.tensor(
        [[vertices.index(s), vertices.index(t)] for s, t in edges], dtype=torch.long
    )

    # attach sensor + encoding nodes
    sensor_idx, encoding_idx = len(vertices), len(vertices) + 1
    # sensor connects TO each keyword node
    with open(cfg.subgraph_dir / f"keywords_{class_name}.txt", "r") as f:
        keywords = [w.strip().lower() for w in f.read().splitlines()]
    for kw in keywords:
        if kw not in vertices:
            print(f"[WARN] keyword '{kw}' missing in vertices for {class_name}")
            continue
        edge_index = torch.vstack([
            edge_index,
            torch.tensor([[sensor_idx, vertices.index(kw)]])
        ])
    # every non‑keyword leaf CONNECTS TO encoding node
    for v in vertices:
        if v not in keywords:
            edge_index = torch.vstack([
                edge_index,
                torch.tensor([[vertices.index(v), encoding_idx]])
            ])

    vertices.extend(["<SENSOR>", "<ENCODING>"])

    # embed vertices (keywords + subgraph) – encoding node gets zeros
    vertex_embeds = torch.vstack([
        _embed_words(vertices[:-2]),
        # torch.zeros(1, cfg.embed_dim)  # encoding node
    ])

    return vertex_embeds, edge_index.T.contiguous()

