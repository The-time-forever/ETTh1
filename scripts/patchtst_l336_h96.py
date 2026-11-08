#!/usr/bin/env python
r"""长历史输入：PatchTST 336h -> 96h（对照 96h -> 96h）。

研究问题：给 PatchTST 更多历史（4 天 -> 14 天）是否有帮助？
同时用 Anchored Ridge 做同样的 96h vs 336h 对照，区分：
    - 历史信息本身的收益；
    - 神经网络结构（patch + attention）对长历史的利用能力。

模型：
    Anchored PatchTST（小配置）：patch=16, stride=8, d_model=16,
        4 heads, 3 layers, d_ff=32, dropout=0.1
    Anchored Ridge：96h / 336h 两套输入

注意：本轮配置同时改变了输入长度和 PatchTST 规模，定位为“更接近论文设置”
的实验，而不是严格单因素消融；因此额外加入 96h 的同配置 PatchTST 作为控制。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\patchtst_l336_h96.py

输出：
    outputs/patchtst_l336/alpha_selection.csv
    outputs/patchtst_l336/test_metrics.csv
    outputs/patchtst_l336/paired_deltas.csv
    outputs/patchtst_l336/per_step_metrics.csv
    outputs/patchtst_l336/segment_metrics.csv
    outputs/patchtst_l336/seed_metrics.csv
    outputs/patchtst_l336/summary.json
    outputs/patchtst_l336/figures/*.png
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
PATCHTST_DIR = ROOT / "outputs" / "patchtst"
OUT_DIR = ROOT / "outputs" / "patchtst_l336"
FIG_DIR = OUT_DIR / "figures"

HORIZON = 96
SEQ_LENS = [96, 336]
LEVEL_WINDOW = 24
SEEDS = [0, 1, 2]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 16
N_HEADS = 4
E_LAYERS = 3
D_FF = 32
DROPOUT = 0.1
HEAD_DROPOUT = 0.1
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Anchored Ridge L96": "#5b2c6f",
    "Anchored Ridge L336": "#8e44ad",
    "PatchTST L96": "#a569bd",
    "PatchTST L336": "#c0392b",
    "B2b": "#16a085",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchTSTForecaster(nn.Module):
    def __init__(self, seq_len: int, output_dim: int = HORIZON):
        super().__init__()
        self.patch_len = PATCH_LEN
        self.stride = STRIDE
        self.num_patches = (seq_len - PATCH_LEN) // STRIDE + 1
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=E_LAYERS)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(self.num_patches * D_MODEL, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, self.patch_len, self.stride)
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


def load_data(seq_len: int):
    zp = np.load(SPLIT_DIR / f"windows_L{seq_len}.npz")
    out = {}
    for split in ["train", "val", "test"]:
        xs = zp[f"{split}_{HORIZON}_x_start"].astype(int)
        ys = zp[f"{split}_{HORIZON}_y_start"].astype(int)
        ye = zp[f"{split}_{HORIZON}_y_end"].astype(int)
        out[split] = (xs, ys, ye)
    return zp, out


def build_xy(z, split: str, ot: np.ndarray, seq_len: int):
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + seq_len] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    level = X[:, -LEVEL_WINDOW:].mean(axis=1)
    return X, Y, level


def fit_ridge(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte):
    Xtr_f = Xtr - ltr[:, None]
    Xva_f = Xva - lva[:, None]
    Xte_f = Xte - lte[:, None]
    Ytr_f = Ytr - ltr[:, None]
    sx = StandardScaler().fit(Xtr_f)
    sy = StandardScaler().fit(Ytr_f)
    Xtr_s = sx.transform(Xtr_f)
    Xva_s = sx.transform(Xva_f)
    Xte_s = sx.transform(Xte_f)
    Ytr_s = sy.transform(Ytr_f)

    rows: list[dict] = []
    models: dict[float, Ridge] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_s, Ytr_s)
        models[alpha] = model
        for split, X, Y, level in [("val", Xva_s, Yva, lva), ("test", Xte_s, Yte, lte)]:
            pred = sy.inverse_transform(model.predict(X)) + level[:, None]
            rows.append({"alpha": alpha, "split": split, **metrics(pred, Y)})
    alpha_df = pd.DataFrame(rows)
    best_alpha = float(alpha_df[alpha_df["split"] == "val"].sort_values("mae").iloc[0]["alpha"])
    model = models[best_alpha]
    pred_val = sy.inverse_transform(model.predict(Xva_s)) + lva[:, None]
    pred_test = sy.inverse_transform(model.predict(Xte_s)) + lte[:, None]
    return alpha_df, best_alpha, pred_val, pred_test


def train_patchtst(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, seq_len: int, seed: int) -> dict:
    set_seed(seed)
    Xtr_f = Xtr - ltr[:, None]
    Xva_f = Xva - lva[:, None]
    Xte_f = Xte - lte[:, None]
    Ytr_f = Ytr - ltr[:, None]

    sx = StandardScaler().fit(Xtr_f)
    sy = StandardScaler().fit(Ytr_f)
    Xtr_s = sx.transform(Xtr_f).astype(np.float32)
    Xva_s = sx.transform(Xva_f).astype(np.float32)
    Xte_s = sx.transform(Xte_f).astype(np.float32)
    Ytr_s = sy.transform(Ytr_f).astype(np.float32)

    Xtr_t = torch.tensor(Xtr_s, device=DEVICE)
    Ytr_t = torch.tensor(Ytr_s, device=DEVICE)
    Xva_t = torch.tensor(Xva_s, device=DEVICE)
    Xte_t = torch.tensor(Xte_s, device=DEVICE)

    model = PatchTSTForecaster(seq_len=seq_len).to(DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    def raw_pred(X_t, level):
        model.eval()
        with torch.no_grad():
            out = model(X_t).cpu().numpy()
        return sy.inverse_transform(out) + level[:, None]

    n = len(Xtr_s)
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

        pred_val = raw_pred(Xva_t, lva)
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

    model.load_state_dict(best_state)
    pred_test = raw_pred(Xte_t, lte)
    return {
        "pred_test": pred_test,
        "metrics": metrics(pred_test, Yte),
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "history": history,
        "n_params": n_params,
    }


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
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    ot = df["OT"].to_numpy(dtype=float)

    datasets = {}
    z_ref = None
    for L in SEQ_LENS:
        z = np.load(SPLIT_DIR / f"windows_L{L}.npz")
        if z_ref is None:
            z_ref = z
        else:
            for split in ["val", "test"]:
                assert np.array_equal(z_ref[f"{split}_{HORIZON}_y_start"], z[f"{split}_{HORIZON}_y_start"])
                assert np.array_equal(z_ref[f"{split}_{HORIZON}_y_end"], z[f"{split}_{HORIZON}_y_end"])
        Xtr, Ytr, ltr = build_xy(z, "train", ot, L)
        Xva, Yva, lva = build_xy(z, "val", ot, L)
        Xte, Yte, lte = build_xy(z, "test", ot, L)
        datasets[L] = {
            "z": z, "Xtr": Xtr, "Ytr": Ytr, "ltr": ltr,
            "Xva": Xva, "Yva": Yva, "lva": lva,
            "Xte": Xte, "Yte": Yte, "lte": lte,
        }

    # ---------- Ridge L96 / L336 ----------
    ridge_results: dict[int, dict] = {}
    alpha_frames: list[pd.DataFrame] = []
    for L in SEQ_LENS:
        d = datasets[L]
        alpha_df, alpha, pred_val, pred_test = fit_ridge(
            d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], d["Xte"], d["Yte"], d["ltr"], d["lva"], d["lte"]
        )
        alpha_df["seq_len"] = L
        alpha_frames.append(alpha_df)
        ridge_results[L] = {"alpha": alpha, "pred_test": pred_test, "pred_val": pred_val}
    pd.concat(alpha_frames, ignore_index=True).to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    # ---------- PatchTST L96 / L336 ----------
    patch_results: dict[int, dict] = {}
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for L in SEQ_LENS:
        d = datasets[L]
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_patchtst(
                d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], d["Xte"], d["Yte"],
                d["ltr"], d["lva"], d["lte"], seq_len=L, seed=seed,
            )
            preds.append(r["pred_test"])
            histories.append(r["history"])
            seed_rows.append(
                {
                    "seq_len": L,
                    "seed": seed,
                    "n_params": int(r["n_params"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_mae": float(r["best_val_mae"]),
                    **r["metrics"],
                }
            )
            for h in r["history"]:
                history_rows.append({"seq_len": L, "seed": seed, **h})
            print(f"PatchTST L={L} seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['metrics']['mae']:.4f}")
        patch_results[L] = {"preds": preds, "histories": histories, "ensemble": np.mean(preds, axis=0)}
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    # ---------- 参考 ----------
    patch_large = pd.read_csv(PATCHTST_DIR / "test_metrics.csv")
    large_mean = patch_large[patch_large["model"] == "Anchored PatchTST (3-seed mean)"].iloc[0]
    large_ens = patch_large[patch_large["model"] == "Anchored PatchTST (ensemble)"].iloc[0]

    Yte = datasets[96]["Yte"]
    origins = datasets[96]["z"][f"test_{HORIZON}_y_start"].astype(int) - 1
    b2b = np.repeat(np.array([ot[o - 95 : o + 1].mean() for o in origins])[:, None], HORIZON, axis=1)

    rows: list[dict] = [{"model": "B2b_local_mean_96", **metrics(b2b, Yte)}]
    for L in SEQ_LENS:
        rows.append({"model": f"Anchored Ridge L{L}", **metrics(ridge_results[L]["pred_test"], Yte)})
    for L in SEQ_LENS:
        sd = seed_df[seed_df["seq_len"] == L]
        ens = metrics(patch_results[L]["ensemble"], Yte)
        rows.append(
            {
                "model": f"PatchTST small L{L} (3-seed mean)",
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
        rows.append({"model": f"PatchTST small L{L} (ensemble)", "mse_std": np.nan, "mae_std": np.nan, **ens})
    rows.append({"model": "PatchTST large L96 (reference, 3-seed mean)", "mse": float(large_mean["mse"]), "mae": float(large_mean["mae"]), "bias": float(large_mean["bias"])})
    test_df = pd.DataFrame(rows)
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- paired deltas ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    paired_rows: list[dict] = []
    for model in ["Ridge", "PatchTST"]:
        if model == "Ridge":
            p96, p336 = ridge_results[96]["pred_test"], ridge_results[336]["pred_test"]
        else:
            p96, p336 = patch_results[96]["ensemble"], patch_results[336]["ensemble"]
        paired_rows.append(
            {
                "model": model,
                "comparison": "L336 - L96",
                "mae_delta": float(wmae(p336).mean() - wmae(p96).mean()),
                "mae_ci_low": block_bootstrap_ci(wmae(p336) - wmae(p96))["ci95_low"],
                "mae_ci_high": block_bootstrap_ci(wmae(p336) - wmae(p96))["ci95_high"],
                "mse_delta": float(wmse(p336).mean() - wmse(p96).mean()),
                "mse_ci_low": block_bootstrap_ci(wmse(p336) - wmse(p96))["ci95_low"],
                "mse_ci_high": block_bootstrap_ci(wmse(p336) - wmse(p96))["ci95_high"],
                "l336_win_rate_pct": float(np.mean(wmae(p336) < wmae(p96)) * 100),
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT_DIR / "paired_deltas.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in [
        ("B2b", b2b),
        ("Anchored Ridge L96", ridge_results[96]["pred_test"]),
        ("Anchored Ridge L336", ridge_results[336]["pred_test"]),
        ("PatchTST L96", patch_results[96]["ensemble"]),
        ("PatchTST L336", patch_results[336]["ensemble"]),
    ]:
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in ["B2b", "Anchored Ridge L96", "Anchored Ridge L336", "PatchTST L96", "PatchTST L336"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, L in [(axes[0], 96), (axes[1], 336)]:
        for i, h in enumerate(patch_results[L]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(f"Anchored PatchTST L={L}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = ["B2b_local_mean_96", "Anchored Ridge L96", "Anchored Ridge L336",
             "PatchTST small L96 (3-seed mean)", "PatchTST small L336 (3-seed mean)",
             "PatchTST large L96 (reference, 3-seed mean)"]
    labels = ["B2b", "Ridge L96", "Ridge L336", "PatchTST L96", "PatchTST L336", "PatchTST large L96"]
    vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    errs = [test_df[test_df["model"] == n]["mae_std"].iloc[0] if "mean" in n and n != "PatchTST large L96 (reference, 3-seed mean)" else 0 for n in names]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=3,
           color=["#16a085", "#5b2c6f", "#8e44ad", "#a569bd", "#c0392b", "#7d3c98"])
    ax.set_xticks(range(len(names)), labels, rotation=15, fontsize=8)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Long history: L96 vs L336")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, model in [(axes[0], "Anchored Ridge"), (axes[1], "PatchTST")]:
        ax.plot(per_step["step"], per_step[f"{model} L96_mae"], lw=1.2, color=COLORS[f"{model} L96"], label=f"{model} L96")
        ax.plot(per_step["step"], per_step[f"{model} L336_mae"], lw=1.2, color=COLORS[f"{model} L336"], label=f"{model} L336")
        ax.plot(per_step["step"], per_step["B2b_mae"], lw=0.8, ls=":", color=COLORS["B2b"], label="B2b")
        for a, _ in SEGMENTS[1:]:
            ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("MAE (C)")
        ax.set_title(f"{model}: L96 vs L336")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, model in [(axes[0], "Anchored Ridge"), (axes[1], "PatchTST")]:
        ax.plot(per_step["step"], per_step[f"{model} L96_bias"], lw=1.1, color=COLORS[f"{model} L96"], label=f"{model} L96")
        ax.plot(per_step["step"], per_step[f"{model} L336_bias"], lw=1.1, color=COLORS[f"{model} L336"], label=f"{model} L336")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("bias (C)")
        ax.set_title(f"{model}: per-step bias")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(paired_df))
    width = 0.35
    ax.bar(x - width / 2, paired_df["mae_delta"], width, label="MAE delta", color="#c0392b")
    ax.bar(x + width / 2, paired_df["mse_delta"], width, label="MSE delta", color="#16a085")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, paired_df["model"])
    ax.set_ylabel("L336 - L96 (negative = more history better)")
    ax.set_title("Effect of longer history (paired, same test targets)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_input_length_effect.png")
    plt.close(fig)

    i = 0
    ys = datasets[96]["z"][f"test_{HORIZON}_y_start"].astype(int)
    ye = datasets[96]["z"][f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, patch_results[336]["ensemble"][i], color=COLORS["PatchTST L336"], lw=1.3, label="PatchTST L336")
    ax.plot(dates, patch_results[96]["ensemble"][i], color=COLORS["PatchTST L96"], lw=1.0, ls="--", label="PatchTST L96")
    ax.plot(dates, ridge_results[336]["pred_test"][i], color=COLORS["Anchored Ridge L336"], lw=1.0, ls="-.", label="Ridge L336")
    ax.plot(dates, b2b[i], color=COLORS["B2b"], lw=0.9, ls=":", label="B2b")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (H=96)")
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, L in enumerate(SEQ_LENS):
        sd = seed_df[seed_df["seq_len"] == L]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS[f"PatchTST L{L}"], s=40, zorder=3, label=f"PatchTST L{L}")
    ax.axhline(float(test_df[test_df["model"] == "Anchored Ridge L96"]["mae"].iloc[0]),
               color=COLORS["Anchored Ridge L96"], ls="--", lw=1.0, label="Ridge L96")
    ax.axhline(float(test_df[test_df["model"] == "Anchored Ridge L336"]["mae"].iloc[0]),
               color=COLORS["Anchored Ridge L336"], ls="--", lw=1.0, label="Ridge L336")
    ax.set_xticks([0, 1], ["L=96", "L=336"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Seed variability: input length")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "horizon": HORIZON,
            "seq_lens": SEQ_LENS,
            "patchtst": f"patch={PATCH_LEN}, stride={STRIDE}, d_model={D_MODEL}, heads={N_HEADS}, layers={E_LAYERS}, d_ff={D_FF}",
            "ridge_alphas": ALPHAS,
            "seeds": SEEDS,
            "device": str(DEVICE),
            "note": "L96 and L336 share identical val/test target windows; paired comparison is valid",
        },
        "test_metrics": test_df.to_dict(orient="records"),
        "paired_deltas": paired_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "seed_metrics": seed_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\npaired deltas (L336 - L96):")
    print(paired_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
