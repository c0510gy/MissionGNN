import torch, json, copy
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

from config import cfg
from datasets import SensorSequenceDataset
from models.missiongnn import MissionGNN
from utils import MetricCollection
from sklearn.utils.class_weight import compute_class_weight


def make_dataloaders():
    root = cfg.data_root
    train_ds = SensorSequenceDataset(cfg.classes, [root / "Train_event", root / "Train"], [], [root / "NormalTrain"], preload=True)
    val_ds = SensorSequenceDataset(cfg.classes, [root / "Test_event"], [root / "Test"], [root / "NormalTest"], preload=True)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_train, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=cfg.batch_val, shuffle=False, num_workers=4),
    )


def compute_class_weights(targets: torch.Tensor) -> torch.Tensor:
    print(torch.unique(targets).numpy())
    weights = compute_class_weight("balanced", classes=torch.unique(targets).numpy(), y=targets.numpy())
    return torch.tensor(weights, dtype=torch.float32)


def train():
    torch.manual_seed(cfg.seed)
    train_dl, val_dl = make_dataloaders()
    model = MissionGNN().to(cfg.device)

    class_w = compute_class_weights(train_dl.dataset.targets).to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics = MetricCollection(len(cfg.classes))
    best_mauc = 0.0
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    # ---- decaying threshold state ----
    threshold = cfg.threshold_start

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm.tqdm(train_dl, desc=f"Train {epoch}")
        steps = 0
        for sensors, mask, y in pbar:
            sensors, mask = sensors.to(cfg.device), mask.to(cfg.device)
            y_mod = y.clone()  # will be edited in‑place for pseudo‑labelling

            # forward pass
            logits = model(sensors, mask)
            probs = torch.softmax(logits, dim=-1)

            # ---- apply decaying threshold to suppress uncertain positive labels ----
            # treat a positive sample as normal if anomaly confidence < (1‑threshold)
            # anomaly confidence is (1 – background prob)
            with torch.no_grad():
                anomaly_conf = 1.0 - probs[:, 0].detach().cpu()
                mask_suppress = (y_mod > 0) & (anomaly_conf <= 1.0 - threshold)
                y_mod[mask_suppress] = 0

            loss = criterion(logits, y_mod.to(cfg.device)) + (1 - probs[:, 0]).mean() * cfg.lambda_1
            optim.zero_grad(); loss.backward(); optim.step()

            # decay threshold after each optimisation step
            threshold *= cfg.threshold_decay

            pbar.set_postfix(loss=float(loss), thresh=threshold)
            
            if steps % 1000 == 0:
                break
            steps += 1

        # ---------------- validation ----------------
        model.eval()
        all_scores, all_targets = [], []
        with torch.no_grad():
            for sensors, mask, y in tqdm.tqdm(val_dl, desc="Val"):
                sensors, mask = sensors.to(cfg.device), mask.to(cfg.device)
                probs = torch.softmax(model(sensors, mask), dim=-1)
                all_scores.append(probs.cpu())
                all_targets.append(y)
        scores = torch.cat(all_scores)
        targets = torch.cat(all_targets)
        stats = metrics(scores, targets)
        print({"epoch": epoch, **stats})

        if stats["mauc"] > best_mauc:
            best_mauc = stats["mauc"]
            torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")

    print("Training complete. Best mAUC:", best_mauc)


if __name__ == "__main__":
    train()
