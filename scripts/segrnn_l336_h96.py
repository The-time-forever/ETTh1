#!/usr/bin/env python
r"""SegRNN（GRU 核心）：S 模式，336h -> 96h。

Segment Length = 48；历史 segment 数 = 336/48 = 7；预测 segment 数 = 96/48 = 2；
Decoder = parallel multi-step（PMF）。

两个水平处理变体：
    Anchored SegRNN : X' = X - l_t (l_t = 最近 24h 均值)，Y' = Y - l_t；
                      这是我们自己的适配，便于和 Anchored Ridge 对照。
    Last-value SegRNN: X' = X - 最后观测值，Y' = Y - 最后观测值；
                      接近 SegRNN 官方默认的 seq_last 处理。

对照：
    Anchored Ridge 336 -> 96（主要参考）
    PatchTST small L336（上一轮长历史实验）
    B2b（局部水平）

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\segrnn_l336_h96.py

输出：
    outputs/segrnn/alpha_selection.csv
    outputs/segrnn/test_metrics.csv
    outputs/segrnn/paired_deltas.csv
    outputs/segrnn/per_step_metrics.csv
    outputs/segrnn/segment_metrics.csv
    outputs/segrnn/seed_metrics.csv
    outputs/segrnn/summary.json
    outputs/segrnn/figures/*.png
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
PATCHTST_L336_DIR = ROOT / "outputs" / "patchtst_l336"
OUT_DIR = ROOT / "outputs" / "segrnn"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 336
HORIZON = 96
SEG_LEN = 48
D_MODEL = 64
DROPOUT = 0.1
LEVEL_WINDOW = 24
SEEDS = [0, 1, 2]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Anchored SegRNN": "#c0392b",
    "Last-value SegRNN": "#8e44ad",
    "Anchored Ridge": "#922b21",
    "PatchTST L336": "#a569bd",
    "B2b": "#16a085",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegRNN(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, pred_len: int = HORIZON, seg_len: int = SEG_LEN,
                 d_model: int = D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        self.seg_len = seg_len
        self.seg_num_x = seq_len // seg_len
        self.seg_num_y = pred_len // seg_len
        self.value_embedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model))
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        batch = x.size(0)
        # (b, n, w) -> (b, n, d)
        x = self.value_embedding(x.reshape(batch, self.seg_num_x, self.seg_len))
        _, hn = self.rnn(x)  # (1, b, d)

        # parallel multi-step decoder: each forecast segment starts from the same hn
        pos = self.pos_emb.repeat(batch, 1).unsqueeze(1)  # (b*m, 1, d)
        hn_rep = hn.repeat(1, 1, self.seg_num_y).reshape(1, batch * self.seg_num_y, self.pos_emb.size(1))
        _, hy = self.rnn(pos, hn_rep)  # (1, b*m, d)
        y = self.predict(hy).reshape(batch, self.seg_num_y * self.seg_len)
        return y


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
    last = X[:, -1]
    return X, Y, level, last


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


def train_segrnn(Xtr, Ytr, Xva, Yva, Xte, Yte, base_tr, base_va, base_te, seed: int) -> dict:
    """base_* is the level added back: 24h anchor or last observation."""
    set_seed(seed)
    Xtr_f = Xtr - base_tr[:, None]
    Xva_f = Xva - base_va[:, None]
    Xte_f = Xte - base_te[:, None]
    Ytr_f = Ytr - base_tr[:, None]

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

    model = SegRNN().to(DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    def raw_pred(X_t, base):
        model.eval()
        with torch.no_grad():
            out = model(X_t).cpu().numpy()
        pred = sy.inverse_transform(out.reshape(-1, 1)).reshape(out.shape)
        return pred + base[:, None]

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

        pred_val = raw_pred(Xva_t, base_va)
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
    pred_test = raw_pred(Xte_t, base_te)
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

    Xtr, Ytr, ltr, last_tr = make_xy(z, "train", ot)
    Xva, Yva, lva, last_va = make_xy(z, "val", ot)
    Xte, Yte, lte, last_te = make_xy(z, "test", ot)

    # ---------- Anchored Ridge 336 ----------
    alpha_df, ridge_alpha, ridge_val, ridge_test = fit_ridge(
        Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte
    )
    alpha_df.to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    # ---------- SegRNN variants ----------
    results: dict[str, dict] = {}
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for key, label, base_tr, base_va, base_te in [
        ("anchored", "Anchored SegRNN", ltr, lva, lte),
        ("last", "Last-value SegRNN", last_tr, last_va, last_te),
    ]:
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_segrnn(
                Xtr, Ytr, Xva, Yva, Xte, Yte, base_tr, base_va, base_te, seed
            )
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

    # ---------- 参考结果 ----------
    ptst = pd.read_csv(PATCHTST_L336_DIR / "test_metrics.csv")
    ptst_mean = ptst[ptst["model"] == "PatchTST small L336 (3-seed mean)"].iloc[0]
    ptst_ens = ptst[ptst["model"] == "PatchTST small L336 (ensemble)"].iloc[0]
    ptst_ps = pd.read_csv(PATCHTST_L336_DIR / "per_step_metrics.csv")
    origins = z[f"test_{HORIZON}_y_start"].astype(int) - 1
    b2b = np.repeat(np.array([ot[o - 95 : o + 1].mean() for o in origins])[:, None], HORIZON, axis=1)

    rows: list[dict] = [
        {"model": "B2b_local_mean_96", **metrics(b2b, Yte)},
        {"model": "Anchored Ridge 336", **metrics(ridge_test, Yte)},
    ]
    for key, label in [("anchored", "Anchored SegRNN"), ("last", "Last-value SegRNN")]:
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
    rows.append({"model": "PatchTST small L336 (mean, reference)", "mse": float(ptst_mean["mse"]), "mae": float(ptst_mean["mae"]), "bias": float(ptst_mean["bias"])})
    rows.append({"model": "PatchTST small L336 (ensemble, reference)", "mse": float(ptst_ens["mse"]), "mae": float(ptst_ens["mae"]), "bias": float(ptst_ens["bias"])})
    test_df = pd.DataFrame(rows)
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- paired deltas ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    pairs = [
        ("Anchored SegRNN vs Anchored Ridge", results["anchored"]["ensemble"], ridge_test),
        ("Anchored SegRNN vs Last-value SegRNN", results["anchored"]["ensemble"], results["last"]["ensemble"]),
        ("Last-value SegRNN vs Anchored Ridge", results["last"]["ensemble"], ridge_test),
        ("Anchored SegRNN vs PatchTST L336", results["anchored"]["ensemble"], None),
    ]
    paired_rows: list[dict] = []
    for name, pred, ref in pairs:
        if ref is None:
            # PatchTST predictions are not saved; use aggregate only.
            continue
        paired_rows.append(
            {
                "comparison": name,
                "mae_delta": float(wmae(pred).mean() - wmae(ref).mean()),
                "mae_ci_low": block_bootstrap_ci(wmae(pred) - wmae(ref))["ci95_low"],
                "mae_ci_high": block_bootstrap_ci(wmae(pred) - wmae(ref))["ci95_high"],
                "mse_delta": float(wmse(pred).mean() - wmse(ref).mean()),
                "mse_ci_low": block_bootstrap_ci(wmse(pred) - wmse(ref))["ci95_low"],
                "mse_ci_high": block_bootstrap_ci(wmse(pred) - wmse(ref))["ci95_high"],
                "win_rate_pct": float(np.mean(wmae(pred) < wmae(ref)) * 100),
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT_DIR / "paired_deltas.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in [
        ("B2b", b2b),
        ("Anchored Ridge", ridge_test),
        ("Anchored SegRNN", results["anchored"]["ensemble"]),
        ("Last-value SegRNN", results["last"]["ensemble"]),
    ]:
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    per_step["PatchTST L336_mae"] = ptst_ps["PatchTST L336_mae"]
    per_step["PatchTST L336_bias"] = ptst_ps["PatchTST L336_bias"]
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in ["B2b", "Anchored Ridge", "Anchored SegRNN", "Last-value SegRNN", "PatchTST L336"]:
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean()) if f"{name}_mse" in seg else np.nan
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean()) if f"{name}_bias" in seg else np.nan
        row["Anchored_SegRNN_mae_skill_vs_Ridge"] = 1 - row["Anchored SegRNN_mae"] / row["Anchored Ridge_mae"]
        row["Anchored_SegRNN_mse_skill_vs_Ridge"] = 1 - row["Anchored SegRNN_mse"] / row["Anchored Ridge_mse"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, label in [(axes[0], "anchored", "Anchored SegRNN"), (axes[1], "last", "Last-value SegRNN")]:
        for i, h in enumerate(results[key]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    names = ["B2b_local_mean_96", "Anchored Ridge 336",
             "Anchored SegRNN (3-seed mean)", "Anchored SegRNN (ensemble)",
             "Last-value SegRNN (3-seed mean)", "Last-value SegRNN (ensemble)",
             "PatchTST small L336 (mean, reference)", "PatchTST small L336 (ensemble, reference)"]
    labels = ["B2b", "Ridge 336", "SegRNN-anch", "SegRNN-anch ens", "SegRNN-last", "SegRNN-last ens", "PatchTST mean", "PatchTST ens"]
    vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    ax.bar(range(len(names)), vals,
           color=["#16a085", "#922b21", "#c0392b", "#e74c3c", "#8e44ad", "#af7ac5", "#a569bd", "#7d3c98"])
    ax.set_xticks(range(len(names)), labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("SegRNN L336->H96 vs references")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored SegRNN", "#c0392b"), ("Last-value SegRNN", "#8e44ad"),
                        ("Anchored Ridge", "#922b21"), ("PatchTST L336", "#a569bd"), ("B2b", "#16a085")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.0, color=color, label=name)
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("MAE (C)")
    ax.set_title("Per-step MAE (H=96)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored SegRNN", "#c0392b"), ("Last-value SegRNN", "#8e44ad"),
                        ("Anchored Ridge", "#922b21"), ("PatchTST L336", "#a569bd")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.0, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("bias (C)")
    ax.set_title("Per-step bias (H=96)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    ax.bar(x - width / 2, seg_df["Anchored_SegRNN_mae_skill_vs_Ridge"], width, label="MAE skill", color="#c0392b")
    ax.bar(x + width / 2, seg_df["Anchored_SegRNN_mse_skill_vs_Ridge"], width, label="MSE skill", color="#16a085")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [s["segment"] for s in seg_rows])
    ax.set_ylabel("skill vs Anchored Ridge")
    ax.set_title("Anchored SegRNN: segment skill vs Ridge 336")
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
    ax.plot(dates, results["anchored"]["ensemble"][i], color="#c0392b", lw=1.3, label="Anchored SegRNN")
    ax.plot(dates, results["last"]["ensemble"][i], color="#8e44ad", lw=1.0, ls="--", label="Last-value SegRNN")
    ax.plot(dates, ridge_test[i], color="#922b21", lw=1.0, ls="-.", label="Anchored Ridge")
    ax.plot(dates, b2b[i], color="#16a085", lw=0.9, ls=":", label="B2b")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (H=96)")
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, key in enumerate(["anchored", "last"]):
        sd = seed_df[seed_df["variant"] == key]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS["Anchored SegRNN" if key == "anchored" else "Last-value SegRNN"], s=40, zorder=3,
                   label=("Anchored SegRNN" if key == "anchored" else "Last-value SegRNN"))
    ax.axhline(float(test_df[test_df["model"] == "Anchored Ridge 336"]["mae"].iloc[0]),
               color=COLORS["Anchored Ridge"], ls="--", lw=1.0, label="Anchored Ridge")
    ax.axhline(float(test_df[test_df["model"] == "PatchTST small L336 (mean, reference)"]["mae"].iloc[0]),
               color=COLORS["PatchTST L336"], ls=":", lw=1.0, label="PatchTST L336 mean")
    ax.set_xticks([0, 1], ["Anchored", "Last-value"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("SegRNN seed variability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "seg_len": SEG_LEN,
            "seg_num_x": SEQ_LEN // SEG_LEN,
            "seg_num_y": HORIZON // SEG_LEN,
            "d_model": D_MODEL,
            "decoder": "parallel multi-step (PMF)",
            "training": "MSE, Adam lr=1e-3, wd=1e-4, batch=256, grad clip 1.0, val MAE early stopping",
            "seeds": SEEDS,
            "device": str(DEVICE),
            "anchored": "l_t = mean(last 24h OT), our adaptation",
            "last_value": "official-style seq_last subtraction",
        },
        "ridge_alpha": ridge_alpha,
        "test_metrics": test_df.to_dict(orient="records"),
        "paired_deltas": paired_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "seed_metrics": seed_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"ridge alpha: {ridge_alpha}")
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\npaired deltas:")
    print(paired_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
