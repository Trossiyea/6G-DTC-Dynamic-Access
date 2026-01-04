#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train NS-GBS action scorer from dataset npz.
Classification target: argmax Delta_true per state.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Create env from environment_nn.yml and retry. "
        f"Import error: {exc}"
    )


class NSGBSDataset(Dataset):
    def __init__(self, features, labels, mean=None, std=None, window_dim=None, drop_window=False, drop_harq=False):
        self.features = features
        self.labels = labels
        self.mean = mean
        self.std = std
        self.window_dim = window_dim
        self.drop_window = drop_window
        self.drop_harq = drop_harq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.asarray(self.features[idx], dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"Invalid feature shape at idx={idx}: {x.shape}")
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        if self.drop_window and self.window_dim is not None:
            x[:, :self.window_dim] = 0.0
        if self.drop_harq and self.window_dim is not None:
            # feature tail: [k0, block_pred, delta_pred, pf_inv, retx_flag, cap_rem, kind_id]
            retx_idx = self.window_dim + 4
            if retx_idx < x.shape[1]:
                x[:, retx_idx] = 0.0
        y = int(self.labels[idx])
        return x, y


def collate_batch(batch):
    xs, ys = zip(*batch)
    max_actions = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    bsz = len(xs)

    x_pad = np.zeros((bsz, max_actions, feat_dim), dtype=np.float32)
    mask = np.zeros((bsz, max_actions), dtype=np.bool_)
    for i, x in enumerate(xs):
        n = x.shape[0]
        x_pad[i, :n, :] = x
        mask[i, :n] = True
    return (
        torch.from_numpy(x_pad),
        torch.tensor(ys, dtype=torch.long),
        torch.from_numpy(mask),
    )


class ActionScorer(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for i in range(max(1, depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [..., F] -> [..., 1]
        return self.net(x)


def compute_norm_stats(features, indices):
    sum_vec = None
    sumsq_vec = None
    count = 0
    for idx in indices:
        x = np.asarray(features[idx], dtype=np.float64)
        if x.size == 0:
            continue
        if sum_vec is None:
            sum_vec = x.sum(axis=0)
            sumsq_vec = (x * x).sum(axis=0)
        else:
            sum_vec += x.sum(axis=0)
            sumsq_vec += (x * x).sum(axis=0)
        count += x.shape[0]
    if count <= 0:
        raise ValueError("No features to compute normalization.")
    mean = sum_vec / float(count)
    var = sumsq_vec / float(count) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train NS-GBS scorer (classification).")
    parser.add_argument("--data", required=True, help="Path to dataset npz")
    parser.add_argument("--out", default="output/models/nsgbs_scorer.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-torchscript", action="store_true")
    parser.add_argument("--export-numpy", action="store_true", help="Export weights to numpy (.npz)")
    parser.add_argument("--drop-window", action="store_true", help="Zero out window features (ablation)")
    parser.add_argument("--drop-harq", action="store_true", help="Zero out HARQ retx feature (ablation)")
    args = parser.parse_args()

    set_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    if len(features) == 0:
        raise SystemExit("Empty dataset.")
    feat_dim = int(np.asarray(features[0]).shape[1])
    tail_dim = 7
    window_dim = feat_dim - tail_dim
    if window_dim <= 0:
        raise SystemExit(f"Invalid feature dim {feat_dim}; expected > {tail_dim}.")

    n = len(labels)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - args.val_split))
    train_idx = idx[:split]
    val_idx = idx[split:] if split < n else idx[:0]

    mean = std = None
    if not args.no_norm:
        mean, std = compute_norm_stats(features, train_idx)

    train_ds = NSGBSDataset(
        features[train_idx],
        labels[train_idx],
        mean,
        std,
        window_dim=window_dim,
        drop_window=args.drop_window,
        drop_harq=args.drop_harq,
    )
    val_ds = (
        NSGBSDataset(
            features[val_idx],
            labels[val_idx],
            mean,
            std,
            window_dim=window_dim,
            drop_window=args.drop_window,
            drop_harq=args.drop_harq,
        )
        if val_idx.size > 0
        else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch) if val_ds else None

    model = ActionScorer(feat_dim, hidden_dim=args.hidden, depth=args.depth, dropout=args.dropout).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_cnt = 0
        for x, y, mask in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            mask = mask.to(args.device)

            bsz, max_actions, fdim = x.shape
            logits = model(x.reshape(-1, fdim)).reshape(bsz, max_actions)
            logits = logits.masked_fill(~mask, -1e9)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -log_probs[torch.arange(bsz, device=args.device), y].mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * bsz
            pred = torch.argmax(logits, dim=1)
            total_acc += float((pred == y).sum().item())
            total_cnt += bsz

        train_loss = total_loss / max(1, total_cnt)
        train_acc = total_acc / max(1, total_cnt)

        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_acc = 0.0
            v_cnt = 0
            with torch.no_grad():
                for x, y, mask in val_loader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    mask = mask.to(args.device)
                    bsz, max_actions, fdim = x.shape
                    logits = model(x.reshape(-1, fdim)).reshape(bsz, max_actions)
                    logits = logits.masked_fill(~mask, -1e9)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = -log_probs[torch.arange(bsz, device=args.device), y].mean()
                    v_loss += float(loss.item()) * bsz
                    pred = torch.argmax(logits, dim=1)
                    v_acc += float((pred == y).sum().item())
                    v_cnt += bsz
            val_loss = v_loss / max(1, v_cnt)
            val_acc = v_acc / max(1, v_cnt)
        else:
            val_loss = None
            val_acc = None

        msg = f"[{epoch:03d}/{args.epochs:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            if best_val is None or val_loss < best_val:
                best_val = val_loss
        print(msg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    meta = {
        "feature_dim": feat_dim,
        "window_dim": window_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "mean": None if mean is None else mean.tolist(),
        "std": None if std is None else std.tolist(),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)) if val_ds is not None else 0,
        "seed": int(args.seed),
        "drop_window": bool(args.drop_window),
        "drop_harq": bool(args.drop_harq),
    }
    with open(str(out_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    if args.save_torchscript:
        ts_path = out_path.with_suffix(".ts")
        example = torch.zeros((1, feat_dim), dtype=torch.float32)
        scripted = torch.jit.trace(model.cpu(), example)
        scripted.save(str(ts_path))
        with open(str(ts_path) + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    if args.export_numpy:
        params = {}
        for name, tensor in model.state_dict().items():
            params[name] = tensor.detach().cpu().numpy()
        if mean is not None and std is not None:
            params["norm_mean"] = mean
            params["norm_std"] = std
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(npz_path, **params)

    print(f"[NS-GBS] saved model: {out_path}")


if __name__ == "__main__":
    main()
