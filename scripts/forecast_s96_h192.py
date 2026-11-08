#!/usr/bin/env python
r"""Horizon 192：S 模式，96h 输入 -> 192h OT 输出。

只比较两类代表模型：
    - Anchored Ridge（线性 + 局部水平锚定）
    - Anchored PatchTST（patch + self-attention + 同一锚定）

基线：
    B1 persistence, B2a local mean 24, B2b local mean 96

研究问题：
    1) 随着 horizon 从 96 增到 192，Anchored Ridge 相对统计基线的优势是否下降？
    2) PatchTST 是否在 H=192 体现出 H=96 时没有的优势？

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\forecast_s96_h192.py

输出：
    outputs/h192/alpha_selection.csv
    outputs/h192/test_metrics.csv
    outputs/h192/per_step_metrics.csv
    outputs/h192/segment_metrics.csv
    outputs/h192/seed_metrics.csv
    outputs/h192/trend_h96_vs_h192.csv
    outputs/h192/summary.json
    outputs/h192/figures/*.png
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
BASELINE_DIR = ROOT / "outputs" / "baselines"
RIDGE_LEVEL_DIR = ROOT / "outputs" / "ridge_level"
PATCHTST_DIR = ROOT / "outputs" / "patchtst"
OUT_DIR = ROOT / "outputs" / "h192"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 192
LEVEL_WINDOW = 24
SEEDS = [0, 1, 2]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
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
SEGMENTS = [(1 + 24 * i, 24 * (i + 1)) for i in range(HORIZON // 24)]
COLORS = {
    "Anchored Ridge": "#922b21",
    "Anchored PatchTST": "#c0392b",
    "B1_persistence": "#7f8c8d",
    "B2a_local_mean_24": "#2980b9",
    "B2b_local_mean_96": "#16a085",
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


def make_xy(z, split: str, ot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    level = X[:, -LEVEL_WINDOW:].mean(axis=1)
    return X, Y, level


def make_baselines(z, ot: np.ndarray) -> dict[str, np.ndarray]:
    xs = z[f"test_{HORIZON}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    b1 = np.repeat(ot[origins][:, None], HORIZON, axis=1)
    mean24 = np.array([ot[o - 23 : o + 1].mean() for o in origins])
    mean96 = np.array([ot[o - 95 : o + 1].mean() for o in origins])
    return {
        "B1_persistence": b1,
        "B2a_local_mean_24": np.repeat(mean24[:, None], HORIZON, axis=1),
        "B2b_local_mean_96": np.repeat(mean96[:, None], HORIZON, axis=1),
    }


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
    best_model = models[best_alpha]
    pred_val = sy.inverse_transform(best_model.predict(Xva_s)) + lva[:, None]
    pred_test = sy.inverse_transform(best_model.predict(Xte_s)) + lte[:, None]
    return alpha_df, best_alpha, pred_val, pred_test


def train_patchtst(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, seed):
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

    model = PatchTSTForecaster(output_dim=HORIZON).to(DEVICE)
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
    z = np.load(SPLIT_DIR / f"windows_L{SEQ_LEN}.npz")

    Xtr, Ytr, ltr = make_xy(z, "train", ot)
    Xva, Yva, lva = make_xy(z, "val", ot)
    Xte, Yte, lte = make_xy(z, "test", ot)
    baselines = make_baselines(z, ot)

    # ---------- Anchored Ridge ----------
    alpha_df, ridge_alpha, ridge_val, ridge_test = fit_ridge(
        Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte
    )
    alpha_df.to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    # ---------- Anchored PatchTST ----------
    seed_rows: list[dict] = []
    histories: list[list[dict]] = []
    preds: list[np.ndarray] = []
    for seed in SEEDS:
        r = train_patchtst(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, seed)
        preds.append(r["pred_test"])
        histories.append(r["history"])
        seed_rows.append(
            {
                "seed": seed,
                "n_params": int(r["n_params"]),
                "best_epoch": int(r["best_epoch"]),
                "best_val_mae": float(r["best_val_mae"]),
                **r["metrics"],
            }
        )
        print(f"Anchored PatchTST seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['metrics']['mae']:.4f}")
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    patch_ens = np.mean(preds, axis=0)

    # ---------- Test 汇总 ----------
    rows: list[dict] = []
    for name, pred in baselines.items():
        rows.append({"model": name, **metrics(pred, Yte)})
    rows.append({"model": "Anchored Ridge", **metrics(ridge_test, Yte)})
    rows.append(
        {
            "model": "Anchored PatchTST (3-seed mean)",
            "mse": float(seed_df["mse"].mean()),
            "mse_std": float(seed_df["mse"].std(ddof=1)),
            "rmse": float(seed_df["rmse"].mean()),
            "mae": float(seed_df["mae"].mean()),
            "mae_std": float(seed_df["mae"].std(ddof=1)),
            "bias": float(seed_df["bias"].mean()),
            "mse_step_centered": float(seed_df["mse_step_centered"].mean()),
            "mae_step_centered": float(seed_df["mae_step_centered"].mean()),
        }
    )
    rows.append({"model": "Anchored PatchTST (ensemble)", **metrics(patch_ens, Yte)})
    test_df = pd.DataFrame(rows)
    for ref in ["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96", "Anchored Ridge"]:
        ref_row = test_df[test_df["model"] == ref].iloc[0]
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in {
        "B1": baselines["B1_persistence"],
        "B2a": baselines["B2a_local_mean_24"],
        "B2b": baselines["B2b_local_mean_96"],
        "Anchored Ridge": ridge_test,
        "Anchored PatchTST": patch_ens,
    }.items():
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
        for name in ["B1", "B2a", "B2b", "Anchored Ridge", "Anchored PatchTST"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        for model in ["Anchored Ridge", "Anchored PatchTST"]:
            for ref in ["B1", "B2a", "B2b"]:
                row[f"{model}_mse_skill_vs_{ref}"] = 1 - row[f"{model}_mse"] / row[f"{ref}_mse"]
                row[f"{model}_mae_skill_vs_{ref}"] = 1 - row[f"{model}_mae"] / row[f"{ref}_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口 / bootstrap ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    win_stats: dict = {"n_windows": int(len(Yte))}
    for name, pred in [("Ridge", ridge_test), ("PatchTST", patch_ens)]:
        for ref_name, ref_pred in [
            ("B1", baselines["B1_persistence"]),
            ("B2a", baselines["B2a_local_mean_24"]),
            ("B2b", baselines["B2b_local_mean_96"]),
        ]:
            win_stats[f"{name}_vs_{ref_name}_mae"] = block_bootstrap_ci(wmae(pred) - wmae(ref_pred))
            win_stats[f"{name}_vs_{ref_name}_mse"] = block_bootstrap_ci(wmse(pred) - wmse(ref_pred))
            win_stats[f"{name}_win_rate_vs_{ref_name}_pct"] = float(np.mean(wmae(pred) < wmae(ref_pred)) * 100)
    win_stats["PatchTST_vs_Ridge_mae"] = block_bootstrap_ci(wmae(patch_ens) - wmae(ridge_test))
    win_stats["PatchTST_vs_Ridge_mse"] = block_bootstrap_ci(wmse(patch_ens) - wmse(ridge_test))
    win_stats["PatchTST_win_rate_vs_Ridge_pct"] = float(np.mean(wmae(patch_ens) < wmae(ridge_test)) * 100)

    # ---------- H=96 vs H=192 skill ----------
    round1 = pd.read_csv(BASELINE_DIR / "round1_metrics.csv")
    h96 = round1[(round1["split"] == "test") & (round1["horizon"] == 96)].set_index("model")
    ridge96 = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    ridge96 = ridge96[ridge96["model"] == "Anchored Ridge"].iloc[0]
    patch96 = pd.read_csv(PATCHTST_DIR / "test_metrics.csv")
    patch96 = patch96[patch96["model"] == "Anchored PatchTST (3-seed mean)"].iloc[0]

    h96_metrics = {
        "B1": (float(h96.loc["B1_persistence", "mse"]), float(h96.loc["B1_persistence", "mae"])),
        "B2a": (float(h96.loc["B2a_local_mean_24", "mse"]), float(h96.loc["B2a_local_mean_24", "mae"])),
        "B2b": (float(h96.loc["B2b_local_mean_96", "mse"]), float(h96.loc["B2b_local_mean_96", "mae"])),
        "Ridge": (float(ridge96["mse"]), float(ridge96["mae"])),
        "PatchTST": (float(patch96["mse"]), float(patch96["mae"])),
    }
    h192_metrics = {
        "B1": (float(test_df[test_df["model"] == "B1_persistence"]["mse"].iloc[0]),
               float(test_df[test_df["model"] == "B1_persistence"]["mae"].iloc[0])),
        "B2a": (float(test_df[test_df["model"] == "B2a_local_mean_24"]["mse"].iloc[0]),
                float(test_df[test_df["model"] == "B2a_local_mean_24"]["mae"].iloc[0])),
        "B2b": (float(test_df[test_df["model"] == "B2b_local_mean_96"]["mse"].iloc[0]),
                float(test_df[test_df["model"] == "B2b_local_mean_96"]["mae"].iloc[0])),
        "Ridge": (float(test_df[test_df["model"] == "Anchored Ridge"]["mse"].iloc[0]),
                  float(test_df[test_df["model"] == "Anchored Ridge"]["mae"].iloc[0])),
        "PatchTST": (float(test_df[test_df["model"] == "Anchored PatchTST (3-seed mean)"]["mse"].iloc[0]),
                     float(test_df[test_df["model"] == "Anchored PatchTST (3-seed mean)"]["mae"].iloc[0])),
    }
    trend_rows: list[dict] = []
    for model in ["Ridge", "PatchTST"]:
        for ref in ["B1", "B2a", "B2b"]:
            trend_rows.append(
                {
                    "model": model,
                    "baseline": ref,
                    "h96_mse_skill": 1 - h96_metrics[model][0] / h96_metrics[ref][0],
                    "h96_mae_skill": 1 - h96_metrics[model][1] / h96_metrics[ref][1],
                    "h192_mse_skill": 1 - h192_metrics[model][0] / h192_metrics[ref][0],
                    "h192_mae_skill": 1 - h192_metrics[model][1] / h192_metrics[ref][1],
                }
            )
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(OUT_DIR / "trend_h96_vs_h192.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, ls in [("val", "-"), ("test", "--")]:
        d = alpha_df[alpha_df["split"] == split].sort_values("alpha")
        ax.plot(d["alpha"], d["mae"], marker="o", ms=3, ls=ls, label=split)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log)")
    ax.set_ylabel("MAE (C)")
    ax.set_title(f"Anchored Ridge alpha selection at H={HORIZON}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_alpha_selection.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = ["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96", "Anchored Ridge",
             "Anchored PatchTST (3-seed mean)", "Anchored PatchTST (ensemble)"]
    vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    errs = [test_df[test_df["model"] == n]["mae_std"].iloc[0] if "mean" in n else 0 for n in names]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=3,
           color=["#7f8c8d", "#2980b9", "#16a085", "#922b21", "#c0392b", "#c0392b"])
    ax.set_xticks(range(len(names)), ["B1", "B2a", "B2b", "Anchored Ridge", "PatchTST mean", "PatchTST ens"],
                  rotation=15)
    ax.set_ylabel("test MAE (C)")
    ax.set_title(f"H={HORIZON}: test MAE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("Anchored PatchTST", "#c0392b"),
                        ("B2b", "#16a085"), ("B2a", "#2980b9"), ("B1", "#7f8c8d")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.1, color=color, label=name)
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("forecast step (1..192)")
    ax.set_ylabel("MAE (C)")
    ax.set_title(f"Per-step MAE on test (H={HORIZON})")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("Anchored PatchTST", "#c0392b"),
                        ("B2b", "#16a085"), ("B2a", "#2980b9")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.1, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..192)")
    ax.set_ylabel("bias (C)")
    ax.set_title(f"Per-step bias on test (H={HORIZON})")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.27
    for ax, model in [(axes[0], "Anchored Ridge"), (axes[1], "Anchored PatchTST")]:
        for k, ref in enumerate(["B1", "B2a", "B2b"]):
            vals = [seg_df.iloc[i][f"{model}_mae_skill_vs_{ref}"] for i in range(len(SEGMENTS))]
            ax.bar(x + (k - 1) * width, vals, width, label=f"vs {ref}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [s["segment"] for s in seg_rows], rotation=45, fontsize=6)
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
    ax.plot(dates, Yte[i], color="k", lw=1.8, label="truth")
    ax.plot(dates, ridge_test[i], color="#922b21", lw=1.2, label="Anchored Ridge")
    ax.plot(dates, patch_ens[i], color="#c0392b", lw=1.0, ls="--", label="Anchored PatchTST")
    ax.plot(dates, baselines["B2a_local_mean_24"][i], color="#2980b9", lw=0.9, ls=":", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title(f"Example test window (H={HORIZON})")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, (label, sds) in enumerate([("PatchTST seeds", seed_df["mae"])]):
        xs = np.full(len(sds), 0, dtype=float) + np.linspace(-0.08, 0.08, len(sds))
        ax.scatter(xs, sds, color="#c0392b", s=40, label="PatchTST seeds")
    ax.axhline(float(test_df[test_df["model"] == "Anchored Ridge"]["mae"].iloc[0]),
               color="#922b21", ls="--", lw=1.0, label="Anchored Ridge")
    ax.axhline(float(test_df[test_df["model"] == "B2b_local_mean_96"]["mae"].iloc[0]),
               color="#16a085", ls=":", lw=1.0, label="B2b")
    ax.set_xticks([0], ["Anchored PatchTST"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title(f"H={HORIZON}: seed variability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    refs = ["B1", "B2a", "B2b"]
    x = np.arange(len(refs))
    width = 0.35
    for ax, model in [(axes[0], "Ridge"), (axes[1], "PatchTST")]:
        d = trend_df[trend_df["model"] == model].set_index("baseline").loc[refs]
        ax.bar(x - width / 2, d["h96_mae_skill"], width, label="H=96")
        ax.bar(x + width / 2, d["h192_mae_skill"], width, label="H=192")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, refs)
        ax.set_ylabel("MAE skill")
        ax.set_title(f"{model}: skill vs baselines, H=96 vs H=192")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_horizon_trend.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "models": ["Anchored Ridge", "Anchored PatchTST"],
            "baselines": ["B1", "B2a (local mean 24)", "B2b (local mean 96)"],
            "anchor": "l_t = mean(last 24h OT)",
            "ridge_alphas": ALPHAS,
            "patchtst": f"patch={PATCH_LEN}, stride={STRIDE}, d_model={D_MODEL}, heads={N_HEADS}, layers={E_LAYERS}",
            "seeds": SEEDS,
            "device": str(DEVICE),
        },
        "ridge_best_alpha": ridge_alpha,
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
        "trend_h96_vs_h192": trend_df.to_dict(orient="records"),
        "seed_metrics": seed_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"ridge best alpha: {ridge_alpha}")
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\ntrend H96 vs H192:")
    print(trend_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
