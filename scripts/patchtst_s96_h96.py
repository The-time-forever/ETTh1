#!/usr/bin/env python
r"""PatchTST：S 模式，96 -> 96，直接多步。

研究问题：让模型通过 patch + self-attention 直接学习不同历史时间片段
之间的关系，能否比 GRU/LSTM 的循环记忆更有效地预测未来 96 小时？

三个变体（同一套 patch/attention 结构，只改变水平/尺度处理）：
    Plain PatchTST    : 全局标准化，直接预测绝对 OT
    Anchored PatchTST : X'=X-l_t, Y'=Y-l_t, 预测后加回 l_t
    RevIN PatchTST    : 每个输入窗口做 instance normalization（原版 PatchTST 的做法）

另加一个控制基线：
    RevIN Ridge       : 同样的 per-window normalization + 线性 Ridge，
                        用来区分“attention 的收益”与“RevIN 的收益”。

协议与前面一致：3 seeds, Adam lr=1e-3, wd=1e-4, batch=256, MSE,
val MAE 早停（patience=20, max_epochs=300），梯度裁剪 1.0。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\patchtst_s96_h96.py

输出：
    outputs/patchtst/seed_metrics.csv
    outputs/patchtst/test_metrics.csv
    outputs/patchtst/per_step_metrics.csv
    outputs/patchtst/segment_metrics.csv
    outputs/patchtst/training_history.csv
    outputs/patchtst/summary.json
    outputs/patchtst/figures/*.png
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
RIDGE_DIR = ROOT / "outputs" / "ridge"
RIDGE_LEVEL_DIR = ROOT / "outputs" / "ridge_level"
MLP_DIR = ROOT / "outputs" / "mlp"
GRU_DIR = ROOT / "outputs" / "gru"
LSTM_DIR = ROOT / "outputs" / "lstm"
OUT_DIR = ROOT / "outputs" / "patchtst"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
SEEDS = [0, 1, 2]
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 64
N_HEADS = 4
E_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
HEAD_DROPOUT = 0.1
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
COLORS = {
    "Plain PatchTST": "#8e44ad",
    "Anchored PatchTST": "#c0392b",
    "RevIN PatchTST": "#16a085",
    "Plain GRU": "#a569bd",
    "Anchored GRU": "#e74c3c",
    "Plain LSTM": "#7d3c98",
    "Anchored LSTM": "#922b21",
    "Plain Ridge": "#5b2c6f",
    "Anchored Ridge": "#7b241c",
    "RevIN Ridge": "#117864",
    "B2a_local_mean_24": "#2980b9",
    "B1_persistence": "#7f8c8d",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        e_layers: int = E_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        head_dropout: float = HEAD_DROPOUT,
        output_dim: int = HORIZON,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(self.num_patches * d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        patches = x.unfold(1, self.patch_len, self.stride)  # (batch, num_patches, patch_len)
        z = self.patch_embed(patches) + self.pos_embed
        z = self.encoder(z)
        return self.head(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    mse = float(np.mean(err**2))
    centered = err - err.mean(axis=0)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mse_step_centered": float(np.mean(centered**2)),
        "mae_step_centered": float(np.mean(np.abs(centered))),
    }


def make_xy(z, split: str, ot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    return X, Y


def level_anchor(X: np.ndarray) -> np.ndarray:
    return X[:, -LEVEL_WINDOW:].mean(axis=1)


def raw_predict(model: nn.Module, X_t: torch.Tensor, to_raw) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(X_t).cpu().numpy()
    return to_raw(out)


def prepare(Xtr, Ytr, Xva, Yva, Xte, Yte, mode: str):
    if mode == "revin":
        def stats(X):
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True) + 1e-8
            return mu, sd

        mu_tr, sd_tr = stats(Xtr)
        mu_va, sd_va = stats(Xva)
        mu_te, sd_te = stats(Xte)
        Xtr_m = (Xtr - mu_tr) / sd_tr
        Xva_m = (Xva - mu_va) / sd_va
        Xte_m = (Xte - mu_te) / sd_te
        Ytr_m = (Ytr - mu_tr) / sd_tr
        to_raw_va = lambda p: p * sd_va + mu_va
        to_raw_te = lambda p: p * sd_te + mu_te
        return Xtr_m, Ytr_m, Xva_m, Xte_m, to_raw_va, to_raw_te

    if mode == "anchored":
        ltr, lva, lte = level_anchor(Xtr), level_anchor(Xva), level_anchor(Xte)
        Xtr_f = Xtr - ltr[:, None]
        Xva_f = Xva - lva[:, None]
        Xte_f = Xte - lte[:, None]
        Ytr_f = Ytr - ltr[:, None]
    else:
        lva = lte = None
        Xtr_f, Xva_f, Xte_f, Ytr_f = Xtr, Xva, Xte, Ytr

    sx = StandardScaler().fit(Xtr_f)
    sy = StandardScaler().fit(Ytr_f)
    Xtr_m = sx.transform(Xtr_f)
    Xva_m = sx.transform(Xva_f)
    Xte_m = sx.transform(Xte_f)
    Ytr_m = sy.transform(Ytr_f)

    def to_raw_va(p):
        raw = sy.inverse_transform(p)
        return raw + lva[:, None] if mode == "anchored" else raw

    def to_raw_te(p):
        raw = sy.inverse_transform(p)
        return raw + lte[:, None] if mode == "anchored" else raw

    return Xtr_m, Ytr_m, Xva_m, Xte_m, to_raw_va, to_raw_te


def train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, mode: str, seed: int) -> dict:
    set_seed(seed)
    Xtr_m, Ytr_m, Xva_m, Xte_m, to_raw_va, to_raw_te = prepare(Xtr, Ytr, Xva, Yva, Xte, Yte, mode)

    Xtr_t = torch.tensor(Xtr_m, dtype=torch.float32, device=DEVICE)
    Ytr_t = torch.tensor(Ytr_m, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva_m, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(Xte_m, dtype=torch.float32, device=DEVICE)

    model = PatchTSTForecaster().to(DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    n = len(Xtr_m)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        train_mse = total / n

        pred_val = raw_predict(model, Xva_t, to_raw_va)
        val_mae = float(np.mean(np.abs(pred_val - Yva)))
        history.append({"epoch": epoch, "train_mse_standardized": train_mse, "val_mae": val_mae})
        if not np.isfinite(val_mae):
            break

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_state)
    test_pred = raw_predict(model, Xte_t, to_raw_te)

    return {
        "test_pred": test_pred,
        "test_metrics": metrics(test_pred, Yte),
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "history": history,
        "seed": seed,
        "n_params": n_params,
    }


def revin_ridge(Xtr, Ytr, Xva, Yva, Xte, Yte):
    def stats(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        return mu, sd

    mu_tr, sd_tr = stats(Xtr)
    mu_va, sd_va = stats(Xva)
    mu_te, sd_te = stats(Xte)
    Xtr_n = (Xtr - mu_tr) / sd_tr
    Xva_n = (Xva - mu_va) / sd_va
    Xte_n = (Xte - mu_te) / sd_te
    Ytr_n = (Ytr - mu_tr) / sd_tr

    best = None
    rows = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_n, Ytr_n)
        pred_va = model.predict(Xva_n) * sd_va + mu_va
        pred_te = model.predict(Xte_n) * sd_te + mu_te
        m_va = metrics(pred_va, Yva)
        m_te = metrics(pred_te, Yte)
        rows.append({"alpha": alpha, "val_mae": m_va["mae"], "test_mae": m_te["mae"]})
        if best is None or m_va["mae"] < best["val_mae"]:
            best = {"alpha": alpha, "val_mae": m_va["mae"], "test_mae": m_te["mae"], "pred": pred_te}
    return best, pd.DataFrame(rows)


def block_bootstrap_ci(diff: np.ndarray, block: int = 96, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, n - block + 1, size=n_blocks)
        means[b] = np.concatenate([diff[s : s + block] for s in starts])[:n].mean()
    lo, med, hi = np.percentile(means, [2.5, 50, 97.5])
    return {"mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "median": float(med)}


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        RIDGE_DIR / "test_metrics.csv", RIDGE_DIR / "per_step_metrics.csv",
        RIDGE_LEVEL_DIR / "test_metrics.csv", RIDGE_LEVEL_DIR / "per_step_metrics.csv",
        MLP_DIR / "test_metrics.csv", MLP_DIR / "per_step_metrics.csv",
        GRU_DIR / "test_metrics.csv", GRU_DIR / "per_step_metrics.csv",
        LSTM_DIR / "test_metrics.csv", LSTM_DIR / "per_step_metrics.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run the previous experiments first")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    ot = df["OT"].to_numpy(dtype=float)
    z = np.load(SPLIT_DIR / f"windows_L{SEQ_LEN}.npz")

    Xtr, Ytr = make_xy(z, "train", ot)
    Xva, Yva = make_xy(z, "val", ot)
    Xte, Yte = make_xy(z, "test", ot)

    results: dict[str, dict] = {}
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for mode, label in [("plain", "Plain PatchTST"), ("anchored", "Anchored PatchTST"), ("revin", "RevIN PatchTST")]:
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, mode=mode, seed=seed)
            preds.append(r["test_pred"])
            histories.append(r["history"])
            seed_rows.append(
                {
                    "variant": mode,
                    "seed": seed,
                    "n_params": int(r["n_params"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_mae": float(r["best_val_mae"]),
                    **r["test_metrics"],
                }
            )
            for h in r["history"]:
                history_rows.append({"variant": mode, "seed": seed, **h})
            print(f"{label} seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['test_metrics']['mae']:.4f}")
        results[mode] = {"preds": preds, "histories": histories, "ensemble": np.mean(preds, axis=0)}

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    revin_ridge_best, revin_ridge_rows = revin_ridge(Xtr, Ytr, Xva, Yva, Xte, Yte)
    revin_ridge_rows.to_csv(OUT_DIR / "revin_ridge_alpha.csv", index=False, encoding="utf-8-sig")

    # ---------- 读取历史结果 ----------
    ridge_plain = pd.read_csv(RIDGE_DIR / "test_metrics.csv")
    ridge_anch = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    mlp_test = pd.read_csv(MLP_DIR / "test_metrics.csv")
    gru_test = pd.read_csv(GRU_DIR / "test_metrics.csv")
    lstm_test = pd.read_csv(LSTM_DIR / "test_metrics.csv")
    ps_plain = pd.read_csv(RIDGE_DIR / "per_step_metrics.csv")
    ps_anch = pd.read_csv(RIDGE_LEVEL_DIR / "per_step_metrics.csv")
    ps_mlp = pd.read_csv(MLP_DIR / "per_step_metrics.csv")
    ps_gru = pd.read_csv(GRU_DIR / "per_step_metrics.csv")
    ps_lstm = pd.read_csv(LSTM_DIR / "per_step_metrics.csv")

    def pick(df_, model):
        return df_[df_["model"] == model].iloc[0].to_dict()

    plain_ridge = pick(ridge_plain, "Ridge")
    anch_ridge = pick(ridge_anch, "Anchored Ridge")
    plain_mlp = pick(mlp_test, "Plain MLP (3-seed mean)")
    anch_mlp = pick(mlp_test, "Anchored MLP (3-seed mean)")
    plain_gru = pick(gru_test, "Plain GRU (3-seed mean)")
    anch_gru = pick(gru_test, "Anchored GRU (3-seed mean)")
    plain_lstm = pick(lstm_test, "Plain LSTM (3-seed mean)")
    anch_lstm = pick(lstm_test, "Anchored LSTM (3-seed mean)")
    b1 = pick(ridge_plain, "B1_persistence")
    b2a = pick(ridge_plain, "B2a_local_mean_24")

    # ---------- 汇总表 ----------
    rows: list[dict] = [
        {"model": "B1_persistence", "mse": b1["mse"], "mse_std": 0.0, "mae": b1["mae"], "mae_std": 0.0, "bias": b1["bias"]},
        {"model": "B2a_local_mean_24", "mse": b2a["mse"], "mse_std": 0.0, "mae": b2a["mae"], "mae_std": 0.0, "bias": b2a["bias"]},
        {"model": "Plain Ridge", "mse": plain_ridge["mse"], "mse_std": 0.0, "mae": plain_ridge["mae"], "mae_std": 0.0, "bias": plain_ridge["bias"], "rmse": plain_ridge["rmse"], "mse_step_centered": plain_ridge["mse_step_centered"], "mae_step_centered": plain_ridge["mae_step_centered"]},
        {"model": "Anchored Ridge", "mse": anch_ridge["mse"], "mse_std": 0.0, "mae": anch_ridge["mae"], "mae_std": 0.0, "bias": anch_ridge["bias"], "rmse": anch_ridge["rmse"], "mse_step_centered": anch_ridge["mse_step_centered"], "mae_step_centered": anch_ridge["mae_step_centered"]},
        {"model": "Plain MLP", "mse": plain_mlp["mse"], "mse_std": plain_mlp["mse_std"], "mae": plain_mlp["mae"], "mae_std": plain_mlp["mae_std"], "bias": plain_mlp["bias"], "rmse": plain_mlp["rmse"], "mse_step_centered": plain_mlp["mse_step_centered"], "mae_step_centered": plain_mlp["mae_step_centered"]},
        {"model": "Anchored MLP", "mse": anch_mlp["mse"], "mse_std": anch_mlp["mse_std"], "mae": anch_mlp["mae"], "mae_std": anch_mlp["mae_std"], "bias": anch_mlp["bias"], "rmse": anch_mlp["rmse"], "mse_step_centered": anch_mlp["mse_step_centered"], "mae_step_centered": anch_mlp["mae_step_centered"]},
        {"model": "Plain GRU", "mse": plain_gru["mse"], "mse_std": plain_gru["mse_std"], "mae": plain_gru["mae"], "mae_std": plain_gru["mae_std"], "bias": plain_gru["bias"], "rmse": plain_gru["rmse"], "mse_step_centered": plain_gru["mse_step_centered"], "mae_step_centered": plain_gru["mae_step_centered"]},
        {"model": "Anchored GRU", "mse": anch_gru["mse"], "mse_std": anch_gru["mse_std"], "mae": anch_gru["mae"], "mae_std": anch_gru["mae_std"], "bias": anch_gru["bias"], "rmse": anch_gru["rmse"], "mse_step_centered": anch_gru["mse_step_centered"], "mae_step_centered": anch_gru["mae_step_centered"]},
        {"model": "Plain LSTM", "mse": plain_lstm["mse"], "mse_std": plain_lstm["mse_std"], "mae": plain_lstm["mae"], "mae_std": plain_lstm["mae_std"], "bias": plain_lstm["bias"], "rmse": plain_lstm["rmse"], "mse_step_centered": plain_lstm["mse_step_centered"], "mae_step_centered": plain_lstm["mae_step_centered"]},
        {"model": "Anchored LSTM", "mse": anch_lstm["mse"], "mse_std": anch_lstm["mse_std"], "mae": anch_lstm["mae"], "mae_std": anch_lstm["mae_std"], "bias": anch_lstm["bias"], "rmse": anch_lstm["rmse"], "mse_step_centered": anch_lstm["mse_step_centered"], "mae_step_centered": anch_lstm["mae_step_centered"]},
        {"model": "RevIN Ridge", "mse": metrics(revin_ridge_best["pred"], Yte)["mse"], "mse_std": 0.0, "mae": metrics(revin_ridge_best["pred"], Yte)["mae"], "mae_std": 0.0, "bias": metrics(revin_ridge_best["pred"], Yte)["bias"], "rmse": metrics(revin_ridge_best["pred"], Yte)["rmse"], "mse_step_centered": metrics(revin_ridge_best["pred"], Yte)["mse_step_centered"], "mae_step_centered": metrics(revin_ridge_best["pred"], Yte)["mae_step_centered"]},
    ]
    for mode, label in [("plain", "Plain PatchTST"), ("anchored", "Anchored PatchTST"), ("revin", "RevIN PatchTST")]:
        sd = seed_df[seed_df["variant"] == mode]
        ens = metrics(results[mode]["ensemble"], Yte)
        rows.append(
            {
                "model": f"{label} (3-seed mean)",
                "mse": float(sd["mse"].mean()),
                "mse_std": float(sd["mse"].std(ddof=1)),
                "mae": float(sd["mae"].mean()),
                "mae_std": float(sd["mae"].std(ddof=1)),
                "bias": float(sd["bias"].mean()),
                "rmse": float(sd["rmse"].mean()),
                "mse_step_centered": float(sd["mse_step_centered"].mean()),
                "mae_step_centered": float(sd["mae_step_centered"].mean()),
            }
        )
        rows.append({"model": f"{label} (ensemble)", "mse_std": np.nan, "mae_std": np.nan, **ens})
    test_df = pd.DataFrame(rows)
    for ref, ref_row in [
        ("Plain Ridge", plain_ridge),
        ("Anchored Ridge", anch_ridge),
        ("Plain GRU", plain_gru),
        ("Anchored GRU", anch_gru),
        ("Plain LSTM", plain_lstm),
        ("Anchored LSTM", anch_lstm),
        ("RevIN Ridge", metrics(revin_ridge_best["pred"], Yte)),
        ("B2a", b2a),
    ]:
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    mapping = {
        "Plain Ridge": ("mae", ps_plain["Ridge_mae"]),
        "Anchored Ridge": ("mae", ps_anch["Anchored Ridge_mae"]),
        "Plain MLP": ("mae", ps_mlp["Plain MLP_mae"]),
        "Anchored MLP": ("mae", ps_mlp["Anchored MLP_mae"]),
        "Plain GRU": ("mae", ps_gru["Plain GRU_mae"]),
        "Anchored GRU": ("mae", ps_gru["Anchored GRU_mae"]),
        "Plain LSTM": ("mae", ps_lstm["Plain LSTM_mae"]),
        "Anchored LSTM": ("mae", ps_lstm["Anchored LSTM_mae"]),
        "B2a": ("mae", ps_plain["B2a_local_mean_24_mae"]),
    }
    for name, (_, series) in mapping.items():
        per_step[f"{name}_mae"] = series
    per_step["Plain MLP_mse"] = ps_mlp["Plain MLP_mse"]
    per_step["Plain MLP_bias"] = ps_mlp["Plain MLP_bias"]
    per_step["Anchored MLP_mse"] = ps_mlp["Anchored MLP_mse"]
    per_step["Anchored MLP_bias"] = ps_mlp["Anchored MLP_bias"]
    per_step["Plain Ridge_mse"] = ps_plain["Ridge_mse"]
    per_step["Plain Ridge_bias"] = ps_plain["Ridge_bias"]
    per_step["Anchored Ridge_mse"] = ps_anch["Anchored Ridge_mse"]
    per_step["Anchored Ridge_bias"] = ps_anch["Anchored Ridge_bias"]
    per_step["Plain GRU_mse"] = ps_gru["Plain GRU_mse"]
    per_step["Plain GRU_bias"] = ps_gru["Plain GRU_bias"]
    per_step["Anchored GRU_mse"] = ps_gru["Anchored GRU_mse"]
    per_step["Anchored GRU_bias"] = ps_gru["Anchored GRU_bias"]
    per_step["Plain LSTM_mse"] = ps_lstm["Plain LSTM_mse"]
    per_step["Plain LSTM_bias"] = ps_lstm["Plain LSTM_bias"]
    per_step["Anchored LSTM_mse"] = ps_lstm["Anchored LSTM_mse"]
    per_step["Anchored LSTM_bias"] = ps_lstm["Anchored LSTM_bias"]
    per_step["B2a_mse"] = ps_plain["B2a_local_mean_24_mse"]
    per_step["B2a_bias"] = ps_plain["B2a_local_mean_24_bias"]
    err_revin = revin_ridge_best["pred"] - Yte
    per_step["RevIN Ridge_mse"] = np.mean(err_revin**2, axis=0)
    per_step["RevIN Ridge_mae"] = np.mean(np.abs(err_revin), axis=0)
    per_step["RevIN Ridge_bias"] = np.mean(err_revin, axis=0)
    for mode, label in [("plain", "Plain PatchTST"), ("anchored", "Anchored PatchTST"), ("revin", "RevIN PatchTST")]:
        err = results[mode]["ensemble"] - Yte
        per_step[f"{label}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{label}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{label}_bias"] = np.mean(err, axis=0)
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    model_cols = [
        "Plain Ridge", "Anchored Ridge", "RevIN Ridge", "B2a",
        "Plain GRU", "Anchored GRU", "Plain LSTM", "Anchored LSTM",
        "Plain PatchTST", "Anchored PatchTST", "RevIN PatchTST",
    ]
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for m in model_cols:
            row[f"{m}_mse"] = float(seg[f"{m}_mse"].mean())
            row[f"{m}_mae"] = float(seg[f"{m}_mae"].mean())
            row[f"{m}_bias"] = float(seg[f"{m}_bias"].mean())
        for m, ref in [
            ("Plain PatchTST", "Plain Ridge"),
            ("Plain PatchTST", "Plain GRU"),
            ("Anchored PatchTST", "Anchored Ridge"),
            ("Anchored PatchTST", "Anchored GRU"),
            ("RevIN PatchTST", "RevIN Ridge"),
            ("Plain PatchTST", "B2a"),
            ("Anchored PatchTST", "B2a"),
            ("RevIN PatchTST", "B2a"),
        ]:
            row[f"{m}_mse_skill_vs_{ref}"] = 1 - row[f"{m}_mse"] / row[f"{ref}_mse"]
            row[f"{m}_mae_skill_vs_{ref}"] = 1 - row[f"{m}_mae"] / row[f"{ref}_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口胜负 / bootstrap ----------
    def window_mae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def window_mse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    ridge_plain_summary = json.loads((RIDGE_DIR / "summary.json").read_text(encoding="utf-8"))
    ridge_anch_summary = json.loads((RIDGE_LEVEL_DIR / "summary.json").read_text(encoding="utf-8"))
    alpha_plain = float(ridge_plain_summary["best_alpha"])
    alpha_anch = float(ridge_anch_summary["best_alpha"]["anchored"])

    def ridge_predictions(Xtr_, Ytr_, Xte_, anchored_: bool, alpha_: float) -> np.ndarray:
        if anchored_:
            ltr_, lte_ = level_anchor(Xtr_), level_anchor(Xte_)
            Xtr_f_ = Xtr_ - ltr_[:, None]
            Xte_f_ = Xte_ - lte_[:, None]
            Ytr_f_ = Ytr_ - ltr_[:, None]
        else:
            lte_ = None
            Xtr_f_, Xte_f_, Ytr_f_ = Xtr_, Xte_, Ytr_
        sx_ = StandardScaler().fit(Xtr_f_)
        model_ = Ridge(alpha=alpha_, fit_intercept=True).fit(sx_.transform(Xtr_f_), Ytr_f_)
        pred_ = model_.predict(sx_.transform(Xte_f_))
        if anchored_:
            pred_ = pred_ + lte_[:, None]
        return pred_

    plain_ridge_pred = ridge_predictions(Xtr, Ytr, Xte, False, alpha_plain)
    anch_ridge_pred = ridge_predictions(Xtr, Ytr, Xte, True, alpha_anch)
    xs_te = z[f"test_{HORIZON}_x_start"].astype(int)
    origins = xs_te + SEQ_LEN - 1
    b2a_pred = np.repeat(np.array([ot[o - 23 : o + 1].mean() for o in origins])[:, None], HORIZON, axis=1)

    win_stats = {
        "n_windows": int(len(Yte)),
        "plain_vs_Ridge_mae": block_bootstrap_ci(window_mae(results["plain"]["ensemble"]) - window_mae(plain_ridge_pred)),
        "plain_vs_Ridge_mse": block_bootstrap_ci(window_mse(results["plain"]["ensemble"]) - window_mse(plain_ridge_pred)),
        "anchored_vs_Ridge_mae": block_bootstrap_ci(window_mae(results["anchored"]["ensemble"]) - window_mae(anch_ridge_pred)),
        "anchored_vs_Ridge_mse": block_bootstrap_ci(window_mse(results["anchored"]["ensemble"]) - window_mse(anch_ridge_pred)),
        "revin_vs_RevIN_Ridge_mae": block_bootstrap_ci(window_mae(results["revin"]["ensemble"]) - window_mae(revin_ridge_best["pred"])),
        "revin_vs_RevIN_Ridge_mse": block_bootstrap_ci(window_mse(results["revin"]["ensemble"]) - window_mse(revin_ridge_best["pred"])),
        "plain_vs_B2a_mae": block_bootstrap_ci(window_mae(results["plain"]["ensemble"]) - window_mae(b2a_pred)),
        "plain_vs_B2a_mse": block_bootstrap_ci(window_mse(results["plain"]["ensemble"]) - window_mse(b2a_pred)),
        "anchored_vs_B2a_mae": block_bootstrap_ci(window_mae(results["anchored"]["ensemble"]) - window_mae(b2a_pred)),
        "anchored_vs_B2a_mse": block_bootstrap_ci(window_mse(results["anchored"]["ensemble"]) - window_mse(b2a_pred)),
        "revin_vs_B2a_mae": block_bootstrap_ci(window_mae(results["revin"]["ensemble"]) - window_mae(b2a_pred)),
        "revin_vs_B2a_mse": block_bootstrap_ci(window_mse(results["revin"]["ensemble"]) - window_mse(b2a_pred)),
        "plain_win_rate_vs_Ridge_pct": float(np.mean(window_mae(results["plain"]["ensemble"]) < window_mae(plain_ridge_pred)) * 100),
        "anchored_win_rate_vs_Ridge_pct": float(np.mean(window_mae(results["anchored"]["ensemble"]) < window_mae(anch_ridge_pred)) * 100),
        "revin_win_rate_vs_RevIN_Ridge_pct": float(np.mean(window_mae(results["revin"]["ensemble"]) < window_mae(revin_ridge_best["pred"])) * 100),
        "plain_win_rate_vs_B2a_pct": float(np.mean(window_mae(results["plain"]["ensemble"]) < window_mae(b2a_pred)) * 100),
        "anchored_win_rate_vs_B2a_pct": float(np.mean(window_mae(results["anchored"]["ensemble"]) < window_mae(b2a_pred)) * 100),
        "revin_win_rate_vs_B2a_pct": float(np.mean(window_mae(results["revin"]["ensemble"]) < window_mae(b2a_pred)) * 100),
        "ridge_recompute_check": {
            "plain_mae_diff": float(metrics(plain_ridge_pred, Yte)["mae"] - plain_ridge["mae"]),
            "anchored_mae_diff": float(metrics(anch_ridge_pred, Yte)["mae"] - anch_ridge["mae"]),
        },
    }

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, mode, label in [
        (axes[0], "plain", "Plain PatchTST"),
        (axes[1], "anchored", "Anchored PatchTST"),
        (axes[2], "revin", "RevIN PatchTST"),
    ]:
        for i, h in enumerate(results[mode]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    plain_models = ["B1_persistence", "B2a_local_mean_24", "Plain Ridge", "Plain MLP", "Plain GRU", "Plain LSTM",
                    "Plain PatchTST (3-seed mean)"]
    anchored_models = ["B2a_local_mean_24", "Anchored Ridge", "Anchored MLP", "Anchored GRU", "Anchored LSTM",
                       "Anchored PatchTST (3-seed mean)", "RevIN Ridge", "RevIN PatchTST (3-seed mean)"]
    for ax, models_, title in [(axes[0], plain_models, "Plain family"), (axes[1], anchored_models, "Anchored / RevIN family")]:
        vals = [test_df.loc[test_df["model"] == m, "mae"].iloc[0] for m in models_]
        errs = [test_df.loc[test_df["model"] == m, "mae_std"].iloc[0] if "mean" in m else 0 for m in models_]
        ax.bar(range(len(models_)), vals, yerr=errs, capsize=3)
        ax.set_xticks(range(len(models_)), models_, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("test MAE (C)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae_by_model.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, mode, label in [
        (axes[0], "plain", "Plain: PatchTST vs GRU/LSTM/Ridge"),
        (axes[1], "anchored", "Anchored: PatchTST vs GRU/LSTM/Ridge"),
        (axes[2], "revin", "RevIN: PatchTST vs RevIN Ridge"),
    ]:
        ax.plot(per_step["step"], per_step[f"{label.split(':')[0]} PatchTST_mae"], lw=1.2, color=COLORS[f"{label.split(':')[0]} PatchTST"], label="PatchTST")
        if mode == "plain":
            ax.plot(per_step["step"], per_step["Plain GRU_mae"], lw=0.9, ls="-.", color=COLORS["Plain GRU"], label="Plain GRU")
            ax.plot(per_step["step"], per_step["Plain Ridge_mae"], lw=1.0, ls="--", color=COLORS["Plain Ridge"], label="Plain Ridge")
        elif mode == "anchored":
            ax.plot(per_step["step"], per_step["Anchored GRU_mae"], lw=0.9, ls="-.", color=COLORS["Anchored GRU"], label="Anchored GRU")
            ax.plot(per_step["step"], per_step["Anchored Ridge_mae"], lw=1.0, ls="--", color=COLORS["Anchored Ridge"], label="Anchored Ridge")
        else:
            ax.plot(per_step["step"], per_step["RevIN Ridge_mae"], lw=1.0, ls="--", color=COLORS["RevIN Ridge"], label="RevIN Ridge")
        ax.plot(per_step["step"], per_step["B2a_mae"], lw=0.8, ls=":", color=COLORS["B2a_local_mean_24"], label="B2a")
        for a, _ in SEGMENTS[1:]:
            ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, mode in [(axes[0], "plain"), (axes[1], "anchored"), (axes[2], "revin")]:
        label = {"plain": "Plain", "anchored": "Anchored", "revin": "RevIN"}[mode]
        ax.plot(per_step["step"], per_step[f"{label} PatchTST_bias"], lw=1.2, color=COLORS[f"{label} PatchTST"], label="PatchTST")
        if mode == "plain":
            ax.plot(per_step["step"], per_step["Plain GRU_bias"], lw=0.9, ls="-.", color=COLORS["Plain GRU"], label="Plain GRU")
            ax.plot(per_step["step"], per_step["Plain Ridge_bias"], lw=1.0, ls="--", color=COLORS["Plain Ridge"], label="Plain Ridge")
        elif mode == "anchored":
            ax.plot(per_step["step"], per_step["Anchored GRU_bias"], lw=0.9, ls="-.", color=COLORS["Anchored GRU"], label="Anchored GRU")
            ax.plot(per_step["step"], per_step["Anchored Ridge_bias"], lw=1.0, ls="--", color=COLORS["Anchored Ridge"], label="Anchored Ridge")
        else:
            ax.plot(per_step["step"], per_step["RevIN Ridge_bias"], lw=1.0, ls="--", color=COLORS["RevIN Ridge"], label="RevIN Ridge")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("bias (C)")
        ax.set_title(f"{label}: per-step bias")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.26
    for ax, metric_name in [(axes[0], "mae"), (axes[1], "mse")]:
        for k, (label, ref, color) in enumerate([
            ("Plain PatchTST", "Plain Ridge", COLORS["Plain PatchTST"]),
            ("Anchored PatchTST", "Anchored Ridge", COLORS["Anchored PatchTST"]),
            ("RevIN PatchTST", "RevIN Ridge", COLORS["RevIN PatchTST"]),
        ]):
            vals = [seg_df.iloc[i][f"{label}_{metric_name}_skill_vs_{ref}"] for i in range(len(SEGMENTS))]
            ax.bar(x + (k - 1) * width, vals, width, label=f"{label} vs {ref}", color=color)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [s["segment"] for s in seg_rows])
        ax.set_ylabel(f"{metric_name.upper()} skill")
        ax.legend(fontsize=7)
    axes[0].set_title("Segment skill vs matching Ridge (MAE)")
    axes[1].set_title("Segment skill vs matching Ridge (MSE)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_improvement.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, results["anchored"]["ensemble"][i], color=COLORS["Anchored PatchTST"], lw=1.4, label="Anchored PatchTST")
    ax.plot(dates, results["plain"]["ensemble"][i], color=COLORS["Plain PatchTST"], lw=1.0, ls="--", label="Plain PatchTST")
    ax.plot(dates, b2a_pred[i], color=COLORS["B2a_local_mean_24"], lw=1.0, ls=":", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, (mode, label) in enumerate([("plain", "Plain PatchTST"), ("anchored", "Anchored PatchTST"), ("revin", "RevIN PatchTST")]):
        sd = seed_df[seed_df["variant"] == mode]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS[label], s=40, zorder=3, label=label)
    ax.axhline(plain_ridge["mae"], color=COLORS["Plain Ridge"], ls="--", lw=1.0, label="Plain Ridge")
    ax.axhline(anch_ridge["mae"], color=COLORS["Anchored Ridge"], ls="--", lw=1.0, label="Anchored Ridge")
    ax.axhline(metrics(revin_ridge_best["pred"], Yte)["mae"], color=COLORS["RevIN Ridge"], ls="--", lw=1.0, label="RevIN Ridge")
    ax.set_xticks([0, 1, 2], ["Plain", "Anchored", "RevIN"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Seed variability vs matching Ridge")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "architecture": f"PatchTST: patch={PATCH_LEN}, stride={STRIDE}, d_model={D_MODEL}, heads={N_HEADS}, layers={E_LAYERS}, d_ff={D_FF}",
            "training": "Adam lr=1e-3, weight_decay=1e-4, batch=256, MSE, grad clip 1.0, early stopping on val MAE",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seeds": SEEDS,
            "device": str(DEVICE),
            "variants": ["plain (global scaler)", "anchored (l_t)", "revin (per-window instance norm)"],
            "revin_ridge_alpha": float(revin_ridge_best["alpha"]),
        },
        "seed_metrics": seed_df.to_dict(orient="records"),
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\nwin stats:")
    print(json.dumps(win_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
