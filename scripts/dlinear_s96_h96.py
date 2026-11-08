#!/usr/bin/env python
r"""DLinear：S 模式，96 -> 96，直接多步。

DLinear = 移动平均分解（kernel=25） + 两个线性层（trend / seasonal）。
本轮比较：
    - Anchored Ridge（当前主要参考）
    - S_control（per-variable 标准化的 Anchored Ridge，作为最干净的对照）
    - Plain DLinear
    - Anchored DLinear

训练协议：
    - MSE loss, Adam lr=1e-3, weight_decay=1e-4, batch=256
    - val MAE 早停（patience=20, max_epochs=300）
    - 3 seeds, direct multi-step

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\dlinear_s96_h96.py

输出：
    outputs/dlinear/seed_metrics.csv
    outputs/dlinear/test_metrics.csv
    outputs/dlinear/per_step_metrics.csv
    outputs/dlinear/segment_metrics.csv
    outputs/dlinear/training_history.csv
    outputs/dlinear/summary.json
    outputs/dlinear/figures/*.png
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
RIDGE_LEVEL_DIR = ROOT / "outputs" / "ridge_level"
RIDGE_MS_DIR = ROOT / "outputs" / "ridge_ms"
BASELINE_DIR = ROOT / "outputs" / "baselines"
OUT_DIR = ROOT / "outputs" / "dlinear"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
KERNEL_SIZE = 25
SEEDS = [0, 1, 2]
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Plain DLinear": "#8e44ad",
    "Anchored DLinear": "#c0392b",
    "Anchored Ridge": "#922b21",
    "S_control": "#5b2c6f",
    "B2a_local_mean_24": "#2980b9",
    "B1_persistence": "#7f8c8d",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.unsqueeze(1)).squeeze(1)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, pred_len: int = HORIZON, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


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


def make_xy(z, split: str, ot: np.ndarray):
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    level = X[:, -LEVEL_WINDOW:].mean(axis=1)
    return X, Y, level


def train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, anchored: bool, seed: int) -> dict:
    set_seed(seed)
    if anchored:
        Xtr_f = Xtr - ltr[:, None]
        Xva_f = Xva - lva[:, None]
        Xte_f = Xte - lte[:, None]
        Ytr_f = Ytr - ltr[:, None]
    else:
        lva = lte = None
        Xtr_f, Xva_f, Xte_f, Ytr_f = Xtr, Xva, Xte, Ytr

    # Per-variable (scalar) standardization, fit on train only.
    sx = StandardScaler().fit(Xtr_f.reshape(-1, 1))
    sy = StandardScaler().fit(Ytr_f.reshape(-1, 1))
    Xtr_s = sx.transform(Xtr_f.reshape(-1, 1)).reshape(Xtr_f.shape).astype(np.float32)
    Xva_s = sx.transform(Xva_f.reshape(-1, 1)).reshape(Xva_f.shape).astype(np.float32)
    Xte_s = sx.transform(Xte_f.reshape(-1, 1)).reshape(Xte_f.shape).astype(np.float32)
    Ytr_s = sy.transform(Ytr_f.reshape(-1, 1)).reshape(Ytr_f.shape).astype(np.float32)

    Xtr_t = torch.tensor(Xtr_s, device=DEVICE)
    Ytr_t = torch.tensor(Ytr_s, device=DEVICE)
    Xva_t = torch.tensor(Xva_s, device=DEVICE)
    Xte_t = torch.tensor(Xte_s, device=DEVICE)

    model = DLinearForecaster().to(DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    def raw_pred(X_t, level):
        model.eval()
        with torch.no_grad():
            out = model(X_t).cpu().numpy()
        pred = sy.inverse_transform(out.reshape(-1, 1)).reshape(out.shape)
        if anchored:
            pred = pred + level[:, None]
        return pred

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
    z = np.load(SPLIT_DIR / f"windows_L{SEQ_LEN}.npz")

    Xtr, Ytr, ltr = make_xy(z, "train", ot)
    Xva, Yva, lva = make_xy(z, "val", ot)
    Xte, Yte, lte = make_xy(z, "test", ot)

    results: dict[str, dict] = {}
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for anchored, key, label in [(False, "plain", "Plain DLinear"), (True, "anchored", "Anchored DLinear")]:
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, anchored=anchored, seed=seed)
            preds.append(r["pred_test"])
            histories.append(r["history"])
            seed_rows.append(
                {
                    "variant": key,
                    "seed": seed,
                    "n_params": int(r["n_params"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_mae": float(r["best_val_mae"]),
                    **r["metrics"],
                }
            )
            for h in r["history"]:
                history_rows.append({"variant": key, "seed": seed, **h})
            print(f"{label} seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['metrics']['mae']:.4f}")
        results[key] = {"preds": preds, "histories": histories, "ensemble": np.mean(preds, axis=0)}

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    # ---------- 参考模型 ----------
    ridge_level = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    ridge_ms = pd.read_csv(RIDGE_MS_DIR / "test_metrics.csv")
    round1 = pd.read_csv(BASELINE_DIR / "round1_metrics.csv")
    ridge_anch = ridge_level[ridge_level["model"] == "Anchored Ridge"].iloc[0]
    s_control = ridge_ms[ridge_ms["model"] == "S_control (per-variable)"].iloc[0]
    h96 = round1[(round1["split"] == "test") & (round1["horizon"] == 96)].set_index("model")

    rows: list[dict] = [
        {"model": "B1_persistence", "mse": float(h96.loc["B1_persistence", "mse"]), "mae": float(h96.loc["B1_persistence", "mae"]), "bias": np.nan},
        {"model": "B2a_local_mean_24", "mse": float(h96.loc["B2a_local_mean_24", "mse"]), "mae": float(h96.loc["B2a_local_mean_24", "mae"]), "bias": np.nan},
        {"model": "B2b_local_mean_96", "mse": float(h96.loc["B2b_local_mean_96", "mse"]), "mae": float(h96.loc["B2b_local_mean_96", "mae"]), "bias": np.nan},
        {"model": "Anchored Ridge (per-lag)", "mse": float(ridge_anch["mse"]), "mae": float(ridge_anch["mae"]), "bias": float(ridge_anch["bias"]), "rmse": float(ridge_anch["rmse"]), "mse_step_centered": float(ridge_anch["mse_step_centered"]), "mae_step_centered": float(ridge_anch["mae_step_centered"])},
        {"model": "S_control (per-variable)", "mse": float(s_control["mse"]), "mae": float(s_control["mae"]), "bias": float(s_control["bias"]), "rmse": float(s_control["rmse"]), "mse_step_centered": float(s_control["mse_step_centered"]), "mae_step_centered": float(s_control["mae_step_centered"])},
    ]
    for key, label in [("plain", "Plain DLinear"), ("anchored", "Anchored DLinear")]:
        sd = seed_df[seed_df["variant"] == key]
        ens = metrics(results[key]["ensemble"], Yte)
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
        ("Anchored Ridge", ridge_anch),
        ("S_control", s_control),
        ("B2b", {"mse": float(h96.loc["B2b_local_mean_96", "mse"]), "mae": float(h96.loc["B2b_local_mean_96", "mae"])}),
    ]:
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    ps_ridge = pd.read_csv(RIDGE_LEVEL_DIR / "per_step_metrics.csv")
    ps_ms = pd.read_csv(RIDGE_MS_DIR / "per_step_metrics.csv")
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    per_step["Anchored Ridge_mae"] = ps_ridge["Anchored Ridge_mae"]
    per_step["Anchored Ridge_mse"] = ps_ridge["Anchored Ridge_mse"]
    per_step["Anchored Ridge_bias"] = ps_ridge["Anchored Ridge_bias"]
    per_step["S_control_mae"] = ps_ms["S_control_mae"]
    per_step["S_control_mse"] = ps_ms["S_control_mse"]
    per_step["S_control_bias"] = ps_ms["S_control_bias"]
    per_step["B2a_mae"] = ps_ridge["B2a_local_mean_24_mae"]
    per_step["B2a_mse"] = ps_ridge["B2a_local_mean_24_mse"]
    per_step["B2a_bias"] = ps_ridge["B2a_local_mean_24_bias"]
    for key, label in [("plain", "Plain DLinear"), ("anchored", "Anchored DLinear")]:
        err = results[key]["ensemble"] - Yte
        per_step[f"{label}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{label}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{label}_bias"] = np.mean(err, axis=0)
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in ["Anchored Ridge", "S_control", "B2a", "Plain DLinear", "Anchored DLinear"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        for model, ref in [
            ("Plain DLinear", "S_control"),
            ("Plain DLinear", "Anchored Ridge"),
            ("Anchored DLinear", "S_control"),
            ("Anchored DLinear", "Anchored Ridge"),
        ]:
            row[f"{model}_mse_skill_vs_{ref}"] = 1 - row[f"{model}_mse"] / row[f"{ref}_mse"]
            row[f"{model}_mae_skill_vs_{ref}"] = 1 - row[f"{model}_mae"] / row[f"{ref}_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口 ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    win_stats: dict = {"n_windows": int(len(Yte))}
    # S_control predictions are not saved; compare aggregate only.
    # Paired bootstrap is done against B2a and between the two DLinear variants.
    xs = z[f"test_{HORIZON}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    b2a_pred = np.repeat(np.array([ot[o - 23 : o + 1].mean() for o in origins])[:, None], HORIZON, axis=1)
    for model, pred in [("Plain DLinear", results["plain"]["ensemble"]), ("Anchored DLinear", results["anchored"]["ensemble"])]:
        win_stats[f"{model}_vs_B2a_mae"] = block_bootstrap_ci(wmae(pred) - wmae(b2a_pred))
        win_stats[f"{model}_vs_B2a_mse"] = block_bootstrap_ci(wmse(pred) - wmse(b2a_pred))
        win_stats[f"{model}_win_rate_vs_B2a_pct"] = float(np.mean(wmae(pred) < wmae(b2a_pred)) * 100)
    win_stats["Anchored_vs_Plain_DLinear_mae"] = block_bootstrap_ci(
        wmae(results["anchored"]["ensemble"]) - wmae(results["plain"]["ensemble"])
    )
    win_stats["Anchored_vs_Plain_DLinear_mse"] = block_bootstrap_ci(
        wmse(results["anchored"]["ensemble"]) - wmse(results["plain"]["ensemble"])
    )
    win_stats["Anchored_win_rate_vs_Plain_pct"] = float(
        np.mean(wmae(results["anchored"]["ensemble"]) < wmae(results["plain"]["ensemble"])) * 100
    )

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, label in [(axes[0], "plain", "Plain DLinear"), (axes[1], "anchored", "Anchored DLinear")]:
        for i, h in enumerate(results[key]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = ["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96",
             "Anchored Ridge (per-lag)", "S_control (per-variable)",
             "Plain DLinear (3-seed mean)", "Anchored DLinear (3-seed mean)"]
    labels = ["B1", "B2a", "B2b", "Anchored Ridge", "S_control", "Plain DLinear", "Anchored DLinear"]
    vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    errs = [test_df[test_df["model"] == n]["mae_std"].iloc[0] if "mean" in n else 0 for n in names]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=3,
           color=["#7f8c8d", "#2980b9", "#16a085", "#922b21", "#5b2c6f", "#8e44ad", "#c0392b"])
    ax.set_xticks(range(len(names)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("S 96->96: Ridge vs DLinear")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prefix, title in [
        (axes[0], "Plain", "Plain DLinear vs references"),
        (axes[1], "Anchored", "Anchored DLinear vs references"),
    ]:
        ax.plot(per_step["step"], per_step[f"{prefix} DLinear_mae"], lw=1.2, color=COLORS[f"{prefix} DLinear"], label=f"{prefix} DLinear")
        ax.plot(per_step["step"], per_step["S_control_mae"], lw=1.0, ls="--", color=COLORS["S_control"], label="S_control")
        ax.plot(per_step["step"], per_step["Anchored Ridge_mae"], lw=0.9, ls="-.", color=COLORS["Anchored Ridge"], label="Anchored Ridge")
        ax.plot(per_step["step"], per_step["B2a_mae"], lw=0.8, ls=":", color=COLORS["B2a_local_mean_24"], label="B2a")
        for a, _ in SEGMENTS[1:]:
            ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("MAE (C)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prefix in [(axes[0], "Plain"), (axes[1], "Anchored")]:
        ax.plot(per_step["step"], per_step[f"{prefix} DLinear_bias"], lw=1.2, color=COLORS[f"{prefix} DLinear"], label=f"{prefix} DLinear")
        ax.plot(per_step["step"], per_step["S_control_bias"], lw=1.0, ls="--", color=COLORS["S_control"], label="S_control")
        ax.plot(per_step["step"], per_step["Anchored Ridge_bias"], lw=0.9, ls="-.", color=COLORS["Anchored Ridge"], label="Anchored Ridge")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("bias (C)")
        ax.set_title(f"{prefix}: per-step bias")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    for ax, model in [(axes[0], "Plain DLinear"), (axes[1], "Anchored DLinear")]:
        ax.bar(x - width / 2, seg_df[f"{model}_mae_skill_vs_S_control"], width, label="vs S_control")
        ax.bar(x + width / 2, seg_df[f"{model}_mae_skill_vs_Anchored Ridge"], width, label="vs Anchored Ridge")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [s["segment"] for s in seg_rows])
        ax.set_ylabel("MAE skill")
        ax.set_title(f"{model}: segment MAE skill")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_skill.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, results["anchored"]["ensemble"][i], color=COLORS["Anchored DLinear"], lw=1.4, label="Anchored DLinear")
    ax.plot(dates, results["plain"]["ensemble"][i], color=COLORS["Plain DLinear"], lw=1.0, ls="--", label="Plain DLinear")
    ax.plot(dates, b2a_pred[i], color=COLORS["B2a_local_mean_24"], lw=1.0, ls=":", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (H=96)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (key, label) in enumerate([("plain", "Plain DLinear"), ("anchored", "Anchored DLinear")]):
        sd = seed_df[seed_df["variant"] == key]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS[label], s=40, zorder=3, label=label)
    ax.axhline(float(s_control["mae"]), color=COLORS["S_control"], ls="--", lw=1.0, label="S_control")
    ax.axhline(float(ridge_anch["mae"]), color=COLORS["Anchored Ridge"], ls="--", lw=1.0, label="Anchored Ridge")
    ax.set_xticks([0, 1], ["Plain DLinear", "Anchored DLinear"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Seed variability vs Ridge references")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "model": f"DLinear (decomp kernel={KERNEL_SIZE}, two linear heads)",
            "loss": "MSE",
            "optimizer": "Adam lr=1e-3, weight_decay=1e-4, batch=256",
            "early_stopping": "val MAE, patience=20, max_epochs=300",
            "seeds": SEEDS,
            "device": str(DEVICE),
            "standardization": "per-variable scalar, train only",
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
