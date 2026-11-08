#!/usr/bin/env python
r"""Attention 消融：PatchTST vs No-Attention Patch Model（S 96->96）。

只替换 Attention 模块，其余结构/训练协议完全一致：
    Base PatchTST      : Patch Embedding + Self-Attention + FFN（3 层，d_model=16）
    No-Attention Model : Patch Embedding + Token-Mixing MLP + FFN（3 层，d_model=16）

Token-mixing MLP 的参数量与 Self-Attention 接近（见报告），因此差异不能简单
归因于参数减少或深度变化。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\patchtst_noattn_s96_h96.py

输出：
    outputs/attn_ablation/seed_metrics.csv
    outputs/attn_ablation/test_metrics.csv
    outputs/attn_ablation/paired_deltas.csv
    outputs/attn_ablation/per_step_metrics.csv
    outputs/attn_ablation/segment_metrics.csv
    outputs/attn_ablation/summary.json
    outputs/attn_ablation/figures/*.png
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
BASELINE_DIR = ROOT / "outputs" / "baselines"
OUT_DIR = ROOT / "outputs" / "attn_ablation"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 16
N_HEADS = 4
E_LAYERS = 3
D_FF = 32
DROPOUT = 0.1
HEAD_DROPOUT = 0.1
TOKEN_MIX_HIDDEN = 48
SEEDS = [0, 1, 2]
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Base PatchTST": "#c0392b",
    "No-Attention PatchTST": "#8e44ad",
    "Anchored Ridge": "#922b21",
    "B2b": "#16a085",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, D_MODEL) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, PATCH_LEN, STRIDE)
        return self.patch_embed(patches) + self.pos_embed


class PatchTSTBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PatchEmbed()
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
            nn.Linear(self.embed.num_patches * D_MODEL, HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        z = self.encoder(z)
        return self.head(z)


class TokenMixBlock(nn.Module):
    """Pre-norm block: token-mixing MLP across patches + the same FFN as attention."""

    def __init__(self, num_patches: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, TOKEN_MIX_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(TOKEN_MIX_HIDDEN, num_patches),
            nn.Dropout(DROPOUT),
        )
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, d)
        y = self.norm1(x).transpose(1, 2)  # (b, d, n)
        y = self.token_mix(y).transpose(1, 2)  # (b, n, d)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class PatchTSTNoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PatchEmbed()
        self.blocks = nn.ModuleList([TokenMixBlock(self.embed.num_patches) for _ in range(E_LAYERS)])
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(self.embed.num_patches * D_MODEL, HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        for block in self.blocks:
            z = block(z)
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


def make_xy(z, split: str, ot: np.ndarray):
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    level = X[:, -LEVEL_WINDOW:].mean(axis=1)
    return X, Y, level


def train_one(kind: str, Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, seed: int) -> dict:
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

    model = (PatchTSTBase() if kind == "base" else PatchTSTNoAttention()).to(DEVICE)
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

    results: dict[str, dict] = {}
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for kind, label in [("base", "Base PatchTST"), ("noattn", "No-Attention PatchTST")]:
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_one(kind, Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte, seed)
            preds.append(r["pred_test"])
            histories.append(r["history"])
            seed_rows.append(
                {
                    "variant": kind,
                    "seed": seed,
                    "n_params": int(r["n_params"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_mae": float(r["best_val_mae"]),
                    **r["metrics"],
                }
            )
            for h in r["history"]:
                history_rows.append({"variant": kind, "seed": seed, **h})
            print(f"{label} seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['metrics']['mae']:.4f}")
        results[kind] = {"preds": preds, "histories": histories, "ensemble": np.mean(preds, axis=0)}

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    # ---------- references ----------
    ridge_level = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    ridge = ridge_level[ridge_level["model"] == "Anchored Ridge"].iloc[0]
    round1 = pd.read_csv(BASELINE_DIR / "round1_metrics.csv")
    h96 = round1[(round1["split"] == "test") & (round1["horizon"] == 96)].set_index("model")

    rows: list[dict] = [
        {"model": "B2b_local_mean_96", "mse": float(h96.loc["B2b_local_mean_96", "mse"]), "mae": float(h96.loc["B2b_local_mean_96", "mae"]), "bias": np.nan},
        {"model": "Anchored Ridge", "mse": float(ridge["mse"]), "mae": float(ridge["mae"]), "bias": float(ridge["bias"]), "rmse": float(ridge["rmse"]), "mse_step_centered": float(ridge["mse_step_centered"]), "mae_step_centered": float(ridge["mae_step_centered"])},
    ]
    for kind, label in [("base", "Base PatchTST"), ("noattn", "No-Attention PatchTST")]:
        sd = seed_df[seed_df["variant"] == kind]
        ens = metrics(results[kind]["ensemble"], Yte)
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
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- paired base vs no-attn ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    paired_rows = [
        {
            "comparison": "No-Attention - Base",
            "mae_delta": float(wmae(results["noattn"]["ensemble"]).mean() - wmae(results["base"]["ensemble"]).mean()),
            "mae_ci_low": block_bootstrap_ci(wmae(results["noattn"]["ensemble"]) - wmae(results["base"]["ensemble"]))["ci95_low"],
            "mae_ci_high": block_bootstrap_ci(wmae(results["noattn"]["ensemble"]) - wmae(results["base"]["ensemble"]))["ci95_high"],
            "mse_delta": float(wmse(results["noattn"]["ensemble"]).mean() - wmse(results["base"]["ensemble"]).mean()),
            "mse_ci_low": block_bootstrap_ci(wmse(results["noattn"]["ensemble"]) - wmse(results["base"]["ensemble"]))["ci95_low"],
            "mse_ci_high": block_bootstrap_ci(wmse(results["noattn"]["ensemble"]) - wmse(results["base"]["ensemble"]))["ci95_high"],
            "noattn_win_rate_pct": float(np.mean(wmae(results["noattn"]["ensemble"]) < wmae(results["base"]["ensemble"])) * 100),
        }
    ]
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT_DIR / "paired_deltas.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in [
        ("B2b", np.repeat(np.array([ot[o - 95 : o + 1].mean() for o in z[f"test_{HORIZON}_y_start"].astype(int) - 1])[:, None], HORIZON, axis=1)),
        ("Anchored Ridge", None),
        ("Base PatchTST", results["base"]["ensemble"]),
        ("No-Attention", results["noattn"]["ensemble"]),
    ]:
        if pred is None:
            continue
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    # Ridge per-step from committed output
    ps_ridge = pd.read_csv(RIDGE_LEVEL_DIR / "per_step_metrics.csv")
    per_step["Anchored Ridge_mae"] = ps_ridge["Anchored Ridge_mae"]
    per_step["Anchored Ridge_mse"] = ps_ridge["Anchored Ridge_mse"]
    per_step["Anchored Ridge_bias"] = ps_ridge["Anchored Ridge_bias"]
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in ["B2b", "Anchored Ridge", "Base PatchTST", "No-Attention"]:
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        row["NoAttn_mae_skill_vs_Base"] = 1 - row["No-Attention_mae"] / row["Base PatchTST_mae"]
        row["NoAttn_mse_skill_vs_Base"] = 1 - row["No-Attention_mse"] / row["Base PatchTST_mse"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind, label in [(axes[0], "base", "Base PatchTST"), (axes[1], "noattn", "No-Attention PatchTST")]:
        for i, h in enumerate(results[kind]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = ["B2b_local_mean_96", "Anchored Ridge",
             "Base PatchTST (3-seed mean)", "Base PatchTST (ensemble)",
             "No-Attention PatchTST (3-seed mean)", "No-Attention PatchTST (ensemble)"]
    labels = ["B2b", "Anchored Ridge", "Base mean", "Base ens", "NoAttn mean", "NoAttn ens"]
    vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    ax.bar(range(len(names)), vals,
           color=["#16a085", "#922b21", "#c0392b", "#e74c3c", "#8e44ad", "#af7ac5"])
    ax.set_xticks(range(len(names)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Attention ablation (S 96->96)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Base PatchTST", "#c0392b"), ("No-Attention", "#8e44ad"),
                        ("Anchored Ridge", "#922b21"), ("B2b", "#16a085")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.0, color=color, label=name)
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("MAE (C)")
    ax.set_title("Per-step MAE: attention vs token mixing")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Base PatchTST", "#c0392b"), ("No-Attention", "#8e44ad"), ("Anchored Ridge", "#922b21")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.0, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("bias (C)")
    ax.set_title("Per-step bias: attention vs token mixing")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    ax.bar(x - width / 2, seg_df["NoAttn_mae_skill_vs_Base"], width, label="MAE skill", color="#8e44ad")
    ax.bar(x + width / 2, seg_df["NoAttn_mse_skill_vs_Base"], width, label="MSE skill", color="#16a085")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [s["segment"] for s in seg_rows])
    ax.set_ylabel("No-Attention relative to Base")
    ax.set_title("Segment skill: token mixing vs self-attention")
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
    ax.plot(dates, results["base"]["ensemble"][i], color="#c0392b", lw=1.3, label="Base PatchTST")
    ax.plot(dates, results["noattn"]["ensemble"][i], color="#8e44ad", lw=1.1, ls="--", label="No-Attention")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (H=96)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, kind in enumerate(["base", "noattn"]):
        sd = seed_df[seed_df["variant"] == kind]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS["Base PatchTST" if kind == "base" else "No-Attention PatchTST"], s=40, zorder=3,
                   label=("Base PatchTST" if kind == "base" else "No-Attention PatchTST"))
    ax.axhline(float(ridge["mae"]), color=COLORS["Anchored Ridge"], ls="--", lw=1.0, label="Anchored Ridge")
    ax.set_xticks([0, 1], ["Base", "No-Attention"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Seed variability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "d_model": D_MODEL,
            "layers": E_LAYERS,
            "patch": f"{PATCH_LEN}/{STRIDE}",
            "token_mix_hidden": TOKEN_MIX_HIDDEN,
            "training": "MSE, Adam lr=1e-3, wd=1e-4, batch=256, val MAE early stopping",
            "seeds": SEEDS,
            "device": str(DEVICE),
        },
        "seed_metrics": seed_df.to_dict(orient="records"),
        "test_metrics": test_df.to_dict(orient="records"),
        "paired_deltas": paired_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\npaired delta (No-Attention - Base):")
    print(paired_df.round(4).to_string(index=False))
    print("\nparams:")
    print(seed_df[["variant", "n_params"]].drop_duplicates().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
