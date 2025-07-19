from pathlib import Path
from typing import List, Optional
import random, glob
import torch
from torch.utils.data import Dataset
import numpy as np
from config import cfg

class SensorSequenceDataset(Dataset):
    def __init__(
        self,
        classes: List[str],
        positives: List[Path],
        negatives: Optional[List[Path]] = None,
        normals: Optional[List[Path]] = None,
        preload: bool = False,
    ):
        self.classes = classes
        self.files, self.targets = [], []
        rng = random.Random(cfg.seed)

        # positive class embeddings
        for positive in positives:
            for label, cls in enumerate(classes, start=1):
                files = sorted(positive.glob(f"{cls}*.pt"))
                rng.shuffle(files)
                # files = files[:100]
                self.files += files
                self.targets += [label] * len(files)

        # negatives (anomalous labels as normal)
        def _add_glob(paths: List[Path]):
            for p in paths:
                files = sorted(p.glob("*.pt"))
                rng.shuffle(files)
                # files = files[:100]
                self.files.extend(files)
                self.targets.extend([0] * len(files))

        if negatives:
            _add_glob(negatives)
        if normals:
            _add_glob(normals)

        # sort for deterministic temporal indexing
        idx = np.argsort([str(f) for f in self.files])
        self.files = [self.files[i] for i in idx]
        self.targets = torch.tensor([self.targets[i] for i in idx])

        # first index of each video name -> for temporal windows
        self.first_idx = []
        last_vid = ""
        for i, f in enumerate(self.files):
            vid = f.name.split("_")[0]
            if vid == last_vid:
                self.first_idx.append(self.first_idx[-1])
            else:
                self.first_idx.append(i)
            last_vid = vid

        self._cache = None
        if preload:
            print(len(self.files))
            self._cache = torch.stack([torch.load(fp) for fp in self.files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        start = max(self.first_idx[idx], idx - 29)
        sensors = (
            self._cache[start : idx + 1]
            if self._cache is not None
            else torch.stack([torch.load(fp) for fp in self.files[start : idx + 1]])
        )
        seq_len = sensors.size(0)
        # pad to 30
        if seq_len < 30:
            pad = torch.zeros(30 - seq_len, cfg.embed_dim)
            sensors = torch.cat([pad, sensors], dim=0)
        mask = torch.zeros(30)
        mask[-seq_len:] = 1
        return sensors, mask, self.targets[idx]
