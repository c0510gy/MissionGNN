from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import cfg
from datasets import SensorSequenceDataset
from models.missiongnn import MissionGNN
from utils import MetricCollection
import tqdm


def evaluate(checkpoint: Path):
    ds = SensorSequenceDataset(cfg.classes, [cfg.data_root / "Test_event"], preload=True)
    dl = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False)

    model = MissionGNN().to(cfg.device)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    model.eval()

    metrics = MetricCollection(len(cfg.classes))
    scores, targets = [], []
    with torch.no_grad():
        for sensors, mask, y in tqdm.tqdm(dl, desc="Val"):
            sensors, mask = sensors.to(cfg.device), mask.to(cfg.device)
            probs = torch.softmax(model(sensors, mask), dim=-1)
            scores.append(probs.cpu())
            targets.append(y)
    scores = torch.cat(scores)
    targets = torch.cat(targets)
    print(metrics(scores, targets))


if __name__ == "__main__":
    ckpt = Path("checkpoints/best.pt")
    evaluate(ckpt)
