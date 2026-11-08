#!/usr/bin/env python
r"""GRU：S 模式，96 -> 96，直接多步。

研究问题：显式的时序建模（循环隐藏状态）能否比 Ridge/MLP 更有效地
提取油温变化规律？

两个变体：
    Plain GRU      : GRU(X) -> last hidden -> Linear(96) -> Y
    Anchored GRU   : X'=X-l_t, Y'=Y-l_t -> GRU -> Linear(96) -> Y = Y' + l_t

训练协议与 MLP 完全一致：
    - 3 个 seed, Adam lr=1e-3, weight_decay=1e-4, batch=256, MSE
    - val MAE 早停（patience=20, max_epochs=300）
    - 输入全局标准化（train），目标标准化（train），预测反变换回 ℃

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\gru_s96_h96.py

输出：
    outputs/gru/seed_metrics.csv
    outputs/gru/test_metrics.csv
    outputs/gru/per_step_metrics.csv
    outputs/gru/segment_metrics.csv
    outputs/gru/training_history.csv
    outputs/gru/summary.json
    outputs/gru/figures/*.png
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
OUT_DIR = ROOT / "outputs" / "gru"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
SEEDS = [0, 1, 2]
HIDDEN = 64
NUM_LAYERS = 1
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 20
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Plain GRU": "#8e44ad",
    "Anchored GRU": "#c0392b",
    "Plain MLP": "#af7ac5",
    "Anchored MLP": "#e74c3c",
    "Plain Ridge": "#5b2c6f",
    "Anchored Ridge": "#922b21",
    "B2a_local_mean_24": "#2980b9",
    "B1_persistence": "#7f8c8d",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUForecaster(nn.Module):
    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, output_dim: int = HORIZON):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 96, 1)
        _, h = self.gru(x)
        return self.head(h[-1])


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


def raw_predict(model: nn.Module, X_t: torch.Tensor, sy: StandardScaler, level: np.ndarray | None, anchored: bool) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(X_t).cpu().numpy()
    pred = sy.inverse_transform(out)
    if anchored:
        pred = pred + level[:, None]
    return pred


def train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, anchored: bool, seed: int) -> dict:
    set_seed(seed)
    if anchored:
        ltr, lva, lte = level_anchor(Xtr), level_anchor(Xva), level_anchor(Xte)
        Xtr_f = Xtr - ltr[:, None]
        Xva_f = Xva - lva[:, None]
        Xte_f = Xte - lte[:, None]
        Ytr_f = Ytr - ltr[:, None]
    else:
        ltr = lva = lte = None
        Xtr_f, Xva_f, Xte_f, Ytr_f = Xtr, Xva, Xte, Ytr

    # Global (scalar) standardization for the GRU input sequence.
    sx = StandardScaler().fit(Xtr_f.reshape(-1, 1))
    Xtr_s = sx.transform(Xtr_f.reshape(-1, 1)).reshape(Xtr_f.shape)
    Xva_s = sx.transform(Xva_f.reshape(-1, 1)).reshape(Xva_f.shape)
    Xte_s = sx.transform(Xte_f.reshape(-1, 1)).reshape(Xte_f.shape)
    sy = StandardScaler().fit(Ytr_f)
    Ytr_s = sy.transform(Ytr_f)

    Xtr_t = torch.tensor(Xtr_s[:, :, None], dtype=torch.float32, device=DEVICE)
    Ytr_t = torch.tensor(Ytr_s, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva_s[:, :, None], dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(Xte_s[:, :, None], dtype=torch.float32, device=DEVICE)

    model = GRUForecaster().to(DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

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

        pred_val = raw_predict(model, Xva_t, sy, lva, anchored)
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
    test_pred = raw_predict(model, Xte_t, sy, lte, anchored)

    return {
        "test_pred": test_pred,
        "test_metrics": metrics(test_pred, Yte),
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "history": history,
        "seed": seed,
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
    required = [
        RIDGE_DIR / "test_metrics.csv",
        RIDGE_DIR / "per_step_metrics.csv",
        RIDGE_LEVEL_DIR / "test_metrics.csv",
        RIDGE_LEVEL_DIR / "per_step_metrics.csv",
        MLP_DIR / "test_metrics.csv",
        MLP_DIR / "per_step_metrics.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run the Ridge/MLP experiments first")

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
    for anchored, key, label in [(False, "plain", "Plain GRU"), (True, "anchored", "Anchored GRU")]:
        preds: list[np.ndarray] = []
        histories: list[list[dict]] = []
        for seed in SEEDS:
            r = train_one(Xtr, Ytr, Xva, Yva, Xte, Yte, anchored=anchored, seed=seed)
            preds.append(r["test_pred"])
            histories.append(r["history"])
            seed_rows.append(
                {
                    "variant": key,
                    "seed": seed,
                    "n_params": int(r["n_params"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_mae": float(r["best_val_mae"]),
                    **r["test_metrics"],
                }
            )
            for h in r["history"]:
                history_rows.append({"variant": key, "seed": seed, **h})
            print(f"{label} seed={seed} best_epoch={r['best_epoch']} val_mae={r['best_val_mae']:.4f} test_mae={r['test_metrics']['mae']:.4f}")
        results[key] = {"preds": preds, "histories": histories, "ensemble": np.mean(preds, axis=0)}

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    # ---------- 读取 Ridge / MLP 结果 ----------
    ridge_plain = pd.read_csv(RIDGE_DIR / "test_metrics.csv")
    ridge_anch = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    mlp_test = pd.read_csv(MLP_DIR / "test_metrics.csv")
    ps_plain = pd.read_csv(RIDGE_DIR / "per_step_metrics.csv")
    ps_anch = pd.read_csv(RIDGE_LEVEL_DIR / "per_step_metrics.csv")
    ps_mlp = pd.read_csv(MLP_DIR / "per_step_metrics.csv")

    def pick(df_, model):
        return df_[df_["model"] == model].iloc[0].to_dict()

    plain_ridge = pick(ridge_plain, "Ridge")
    anch_ridge = pick(ridge_anch, "Anchored Ridge")
    plain_mlp = pick(mlp_test, "Plain MLP (3-seed mean)")
    anch_mlp = pick(mlp_test, "Anchored MLP (3-seed mean)")
    b1 = pick(ridge_plain, "B1_persistence")
    b2a = pick(ridge_plain, "B2a_local_mean_24")

    # ---------- 汇总表 ----------
    rows: list[dict] = [
        {"model": "B1_persistence", "mse": b1["mse"], "mse_std": 0.0, "mae": b1["mae"], "mae_std": 0.0, "bias": b1["bias"]},
        {"model": "B2a_local_mean_24", "mse": b2a["mse"], "mse_std": 0.0, "mae": b2a["mae"], "mae_std": 0.0, "bias": b2a["bias"]},
        {"model": "Plain Ridge", "mse": plain_ridge["mse"], "mse_std": 0.0, "mae": plain_ridge["mae"], "mae_std": 0.0, "bias": plain_ridge["bias"], "rmse": plain_ridge["rmse"], "mse_step_centered": plain_ridge["mse_step_centered"], "mae_step_centered": plain_ridge["mae_step_centered"]},
        {"model": "Anchored Ridge", "mse": anch_ridge["mse"], "mse_std": 0.0, "mae": anch_ridge["mae"], "mae_std": 0.0, "bias": anch_ridge["bias"], "rmse": anch_ridge["rmse"], "mse_step_centered": anch_ridge["mse_step_centered"], "mae_step_centered": anch_ridge["mae_step_centered"]},
        {"model": "Plain MLP (3-seed mean)", "mse": plain_mlp["mse"], "mse_std": plain_mlp["mse_std"], "mae": plain_mlp["mae"], "mae_std": plain_mlp["mae_std"], "bias": plain_mlp["bias"], "rmse": plain_mlp["rmse"], "mse_step_centered": plain_mlp["mse_step_centered"], "mae_step_centered": plain_mlp["mae_step_centered"]},
        {"model": "Anchored MLP (3-seed mean)", "mse": anch_mlp["mse"], "mse_std": anch_mlp["mse_std"], "mae": anch_mlp["mae"], "mae_std": anch_mlp["mae_std"], "bias": anch_mlp["bias"], "rmse": anch_mlp["rmse"], "mse_step_centered": anch_mlp["mse_step_centered"], "mae_step_centered": anch_mlp["mae_step_centered"]},
    ]
    for key, label in [("plain", "Plain GRU"), ("anchored", "Anchored GRU")]:
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
        ("Plain Ridge", plain_ridge),
        ("Anchored Ridge", anch_ridge),
        ("Plain MLP", plain_mlp),
        ("Anchored MLP", anch_mlp),
        ("B2a", b2a),
    ]:
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    per_step["Plain Ridge_mae"] = ps_plain["Ridge_mae"]
    per_step["Plain Ridge_mse"] = ps_plain["Ridge_mse"]
    per_step["Plain Ridge_bias"] = ps_plain["Ridge_bias"]
    per_step["Anchored Ridge_mae"] = ps_anch["Anchored Ridge_mae"]
    per_step["Anchored Ridge_mse"] = ps_anch["Anchored Ridge_mse"]
    per_step["Anchored Ridge_bias"] = ps_anch["Anchored Ridge_bias"]
    per_step["Plain MLP_mae"] = ps_mlp["Plain MLP_mae"]
    per_step["Plain MLP_mse"] = ps_mlp["Plain MLP_mse"]
    per_step["Plain MLP_bias"] = ps_mlp["Plain MLP_bias"]
    per_step["Anchored MLP_mae"] = ps_mlp["Anchored MLP_mae"]
    per_step["Anchored MLP_mse"] = ps_mlp["Anchored MLP_mse"]
    per_step["Anchored MLP_bias"] = ps_mlp["Anchored MLP_bias"]
    per_step["B2a_mae"] = ps_plain["B2a_local_mean_24_mae"]
    per_step["B2a_mse"] = ps_plain["B2a_local_mean_24_mse"]
    per_step["B2a_bias"] = ps_plain["B2a_local_mean_24_bias"]
    for key, label in [("plain", "Plain GRU"), ("anchored", "Anchored GRU")]:
        err = results[key]["ensemble"] - Yte
        per_step[f"{label}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{label}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{label}_bias"] = np.mean(err, axis=0)
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    model_cols = ["Plain Ridge", "Anchored Ridge", "Plain MLP", "Anchored MLP", "B2a", "Plain GRU", "Anchored GRU"]
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for m in model_cols:
            row[f"{m}_mse"] = float(seg[f"{m}_mse"].mean())
            row[f"{m}_mae"] = float(seg[f"{m}_mae"].mean())
            row[f"{m}_bias"] = float(seg[f"{m}_bias"].mean())
        for m, ref in [
            ("Plain GRU", "Plain Ridge"),
            ("Plain GRU", "Plain MLP"),
            ("Anchored GRU", "Anchored Ridge"),
            ("Anchored GRU", "Anchored MLP"),
            ("Plain GRU", "B2a"),
            ("Anchored GRU", "B2a"),
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
        "plain_vs_B2a_mae": block_bootstrap_ci(window_mae(results["plain"]["ensemble"]) - window_mae(b2a_pred)),
        "plain_vs_B2a_mse": block_bootstrap_ci(window_mse(results["plain"]["ensemble"]) - window_mse(b2a_pred)),
        "anchored_vs_B2a_mae": block_bootstrap_ci(window_mae(results["anchored"]["ensemble"]) - window_mae(b2a_pred)),
        "anchored_vs_B2a_mse": block_bootstrap_ci(window_mse(results["anchored"]["ensemble"]) - window_mse(b2a_pred)),
        "plain_win_rate_vs_Ridge_pct": float(np.mean(window_mae(results["plain"]["ensemble"]) < window_mae(plain_ridge_pred)) * 100),
        "anchored_win_rate_vs_Ridge_pct": float(np.mean(window_mae(results["anchored"]["ensemble"]) < window_mae(anch_ridge_pred)) * 100),
        "plain_win_rate_vs_B2a_pct": float(np.mean(window_mae(results["plain"]["ensemble"]) < window_mae(b2a_pred)) * 100),
        "anchored_win_rate_vs_B2a_pct": float(np.mean(window_mae(results["anchored"]["ensemble"]) < window_mae(b2a_pred)) * 100),
        "ridge_recompute_check": {
            "plain_mae_diff": float(metrics(plain_ridge_pred, Yte)["mae"] - plain_ridge["mae"]),
            "anchored_mae_diff": float(metrics(anch_ridge_pred, Yte)["mae"] - anch_ridge["mae"]),
        },
    }

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, label in [(axes[0], "plain", "Plain GRU"), (axes[1], "anchored", "Anchored GRU")]:
        for i, h in enumerate(results[key]["histories"]):
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], lw=0.9, alpha=0.65, label=f"seed {SEEDS[i]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (C)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png")
    plt.close(fig)

    models_bar = [
        "B1_persistence", "B2a_local_mean_24",
        "Plain Ridge", "Plain MLP (3-seed mean)", "Plain GRU (3-seed mean)",
        "Anchored Ridge", "Anchored MLP (3-seed mean)", "Anchored GRU (3-seed mean)",
    ]
    vals = [test_df.loc[test_df["model"] == m, "mae"].iloc[0] for m in models_bar]
    errs = [test_df.loc[test_df["model"] == m, "mae_std"].iloc[0] if "mean" in m else 0 for m in models_bar]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(models_bar)), vals, yerr=errs, capsize=3,
           color=["#7f8c8d", "#2980b9", "#5b2c6f", "#af7ac5", "#8e44ad", "#922b21", "#e74c3c", "#c0392b"])
    ax.set_xticks(range(len(models_bar)), models_bar, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Test MAE: Ridge vs MLP vs GRU (mean +/- std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_mae_by_model.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prefix in [(axes[0], "Plain"), (axes[1], "Anchored")]:
        ax.plot(per_step["step"], per_step[f"{prefix} GRU_mae"], lw=1.2, color=COLORS[f"{prefix} GRU"], label=f"{prefix} GRU")
        ax.plot(per_step["step"], per_step[f"{prefix} MLP_mae"], lw=0.9, ls="-.", color=COLORS[f"{prefix} MLP"], label=f"{prefix} MLP")
        ax.plot(per_step["step"], per_step[f"{prefix} Ridge_mae"], lw=1.0, ls="--", color=COLORS[f"{prefix} Ridge"], label=f"{prefix} Ridge")
        ax.plot(per_step["step"], per_step["B2a_mae"], lw=0.8, ls=":", color=COLORS["B2a_local_mean_24"], label="B2a")
        for a, _ in SEGMENTS[1:]:
            ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
        ax.set_xlabel("forecast step (1..96)")
        ax.set_ylabel("MAE (C)")
        ax.set_title(f"{prefix}: per-step MAE")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prefix in [(axes[0], "Plain"), (axes[1], "Anchored")]:
        ax.plot(per_step["step"], per_step[f"{prefix} GRU_bias"], lw=1.2, color=COLORS[f"{prefix} GRU"], label=f"{prefix} GRU")
        ax.plot(per_step["step"], per_step[f"{prefix} MLP_bias"], lw=0.9, ls="-.", color=COLORS[f"{prefix} MLP"], label=f"{prefix} MLP")
        ax.plot(per_step["step"], per_step[f"{prefix} Ridge_bias"], lw=1.0, ls="--", color=COLORS[f"{prefix} Ridge"], label=f"{prefix} Ridge")
        ax.plot(per_step["step"], per_step["B2a_bias"], lw=0.8, ls=":", color=COLORS["B2a_local_mean_24"], label="B2a")
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
    width = 0.28
    for ax, metric_name in [(axes[0], "mae"), (axes[1], "mse")]:
        for k, (label, ref, color) in enumerate([
            ("Plain GRU", "Plain Ridge", COLORS["Plain GRU"]),
            ("Anchored GRU", "Anchored Ridge", COLORS["Anchored GRU"]),
        ]):
            vals = [seg_df.iloc[i][f"{label}_{metric_name}_skill_vs_{ref}"] for i in range(len(SEGMENTS))]
            ax.bar(x + (k - 0.5) * width, vals, width, label=f"{label} vs {ref}", color=color)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [s["segment"] for s in seg_rows])
        ax.set_ylabel(f"{metric_name.upper()} skill vs matching Ridge")
        ax.legend(fontsize=7)
    axes[0].set_title("Segment improvement over Ridge: MAE")
    axes[1].set_title("Segment improvement over Ridge: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_improvement.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, results["anchored"]["ensemble"][i], color=COLORS["Anchored GRU"], lw=1.4, label="Anchored GRU")
    ax.plot(dates, results["plain"]["ensemble"][i], color=COLORS["Plain GRU"], lw=1.0, ls="--", label="Plain GRU")
    ax.plot(dates, b2a_pred[i], color=COLORS["B2a_local_mean_24"], lw=1.0, ls=":", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (key, label) in enumerate([("plain", "Plain GRU"), ("anchored", "Anchored GRU")]):
        sd = seed_df[seed_df["variant"] == key]
        xs_plot = np.full(len(sd), j, dtype=float) + np.linspace(-0.08, 0.08, len(sd))
        ax.scatter(xs_plot, sd["mae"], color=COLORS[label], s=40, zorder=3, label=label)
    ax.axhline(plain_ridge["mae"], color=COLORS["Plain Ridge"], ls="--", lw=1.0, label="Plain Ridge")
    ax.axhline(anch_ridge["mae"], color=COLORS["Anchored Ridge"], ls="--", lw=1.0, label="Anchored Ridge")
    ax.axhline(plain_mlp["mae"], color=COLORS["Plain MLP"], ls=":", lw=1.0, label="Plain MLP")
    ax.axhline(anch_mlp["mae"], color=COLORS["Anchored MLP"], ls=":", lw=1.0, label="Anchored MLP")
    ax.set_xticks([0, 1], ["Plain GRU", "Anchored GRU"])
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Seed variability vs Ridge/MLP")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_seed_variability.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "architecture": "GRU(1 -> 64, 1 layer) -> last hidden -> Linear(96)",
            "training": "Adam lr=1e-3, weight_decay=1e-4, batch=256, MSE, grad clip 1.0, early stopping on val MAE",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seeds": SEEDS,
            "device": str(DEVICE),
            "anchored": "X'=X-l_t, Y'=Y-l_t, predict=GRU(X')+l_t",
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
