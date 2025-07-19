import torch
from torchmetrics import AUROC, AveragePrecision
from sklearn.metrics import f1_score
from functools import partial

class MetricCollection:
    def __init__(self, num_classes: int):
        self.aurocs = [AUROC(task="binary") for _ in range(num_classes + 1)]
        self.f1 = partial(f1_score, average='micro')
        self.ap = [AveragePrecision(task="binary") for _ in range(num_classes + 1)]

    def __call__(self, scores: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            f1 = self.f1(targets, torch.max(scores, 1).indices)
            aucs = [m(scores[:, i], targets == i) for i, m in enumerate(self.aurocs)]
            aps = [m(scores[:, i], targets == i) for i, m in enumerate(self.ap)]
        return dict(f1=f1, mauc=sum(aucs[1:]).item() / len(aucs[1:]), map=sum(aps[1:]).item() / len(aps[1:]))
