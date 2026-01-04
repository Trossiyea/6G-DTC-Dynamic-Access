"""
NS-GBS torch inference utilities.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class ActionScorer(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NSGBSScorer:
    def __init__(self, model, mean: Optional[np.ndarray], std: Optional[np.ndarray], device: str):
        self.model = model
        self.device = device
        self.mean = None
        self.std = None
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
            self.std = torch.tensor(std, dtype=torch.float32, device=device)

    def score(self, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is not available for NS-GBS scoring.")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        t = torch.from_numpy(x).to(self.device)
        if self.mean is not None and self.std is not None:
            t = (t - self.mean) / self.std
        with torch.no_grad():
            out = self.model(t).squeeze(-1)
        return out.detach().cpu().numpy()


def _load_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nsgbs_scorer(cfg: dict) -> Optional[NSGBSScorer]:
    if torch is None:
        print("[NS-GBS] PyTorch not available; cannot load model.")
        return None
    model_path = cfg.get("nsgbs_model_path", None)
    if not model_path:
        return None
    device_cfg = cfg.get("nsgbs_device", None)
    device = str(device_cfg) if device_cfg else ("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    mean = std = None

    if str(model_path).endswith(".ts"):
        model = torch.jit.load(model_path, map_location=device)
        meta_path = str(model_path) + ".json"
        if os.path.exists(meta_path):
            meta = _load_meta(meta_path)
            mean = np.asarray(meta.get("mean"), dtype=np.float32) if meta.get("mean") is not None else None
            std = np.asarray(meta.get("std"), dtype=np.float32) if meta.get("std") is not None else None
        else:
            print("[NS-GBS] TorchScript meta not found; running without normalization.")
    else:
        meta_path = str(model_path) + ".json"
        if not os.path.exists(meta_path):
            print(f"[NS-GBS] Missing meta file: {meta_path}")
            return None
        meta = _load_meta(meta_path)
        feat_dim = int(meta["feature_dim"])
        hidden = int(meta.get("hidden", 128))
        depth = int(meta.get("depth", 2))
        dropout = float(meta.get("dropout", 0.0))
        model = ActionScorer(feat_dim, hidden_dim=hidden, depth=depth, dropout=dropout)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        mean = np.asarray(meta.get("mean"), dtype=np.float32) if meta.get("mean") is not None else None
        std = np.asarray(meta.get("std"), dtype=np.float32) if meta.get("std") is not None else None

    model.to(device)
    model.eval()
    return NSGBSScorer(model, mean, std, device)
