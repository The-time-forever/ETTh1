#!/usr/bin/env python
r"""Horizon 720：S 模式，96h 输入 -> 720h (30 天) OT 输出，Anchored Ridge。

Horizon 研究的收官实验。只做线性模型 + 两个统计基线：
    - Anchored Ridge
    - B1 persistence
    - B2b local mean 96

回答：
    当输入只有 4 天、预测未来 30 天时，学习型线性映射还能比局部水平
    估计多提供多少信息？增量收益如何随预测距离衰减？

另外做轻量的统一预测起点复核，把 H=96/192/336/720 放在同一批起点上比较。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\forecast_s96_h720.py

输出：
    outputs/h720/alpha_selection.csv
    outputs/h720/test_metrics.csv
    outputs/h720/per_step_metrics.csv
    outputs/h720/segment_metrics.csv
    outputs/h720/anchor_deviation.csv
    outputs/h720/horizon_skill.csv
    outputs/h720/fixed_origin_skill.csv
    outputs/h720/summary.json
    outputs/h720/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
BASELINE_DIR = ROOT / "outputs" / "baselines"
RIDGE_LEVEL_DIR = ROOT / "outputs" / "ridge_level"
H192_DIR = ROOT / "outputs" / "h192"
H336_DIR = ROOT / "outputs" / "h336"
OUT_DIR = ROOT / "outputs" / "h720"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 720
LEVEL_WINDOW = 24
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEGMENTS = [(1, 24), (25, 96), (97, 192), (193, 336), (337, 720)]
COLORS = {
    "Anchored Ridge": "#922b21",
    "B1_persistence": "#7f8c8d",
    "B2b_local_mean_96": "#16a085",
}


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


def make_xy(z, split: str, ot: np.ndarray, horizon: int):
    xs = z[f"{split}_{horizon}_x_start"].astype(int)
    ys = z[f"{split}_{horizon}_y_start"].astype(int)
    ye = z[f"{split}_{horizon}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    level = X[:, -LEVEL_WINDOW:].mean(axis=1)
    return X, Y, level


def make_baselines(z, ot: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    xs = z[f"test_{horizon}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    b1 = np.repeat(ot[origins][:, None], horizon, axis=1)
    mean96 = np.array([ot[o - 95 : o + 1].mean() for o in origins])
    return {
        "B1_persistence": b1,
        "B2b_local_mean_96": np.repeat(mean96[:, None], horizon, axis=1),
    }


def fit_ridge_anchored(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte):
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
    return {
        "alpha_df": alpha_df,
        "best_alpha": best_alpha,
        "model": model,
        "sx": sx,
        "sy": sy,
        "coef": model.coef_,
        "intercept": model.intercept_,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def fit_ridge_fixed_alpha(Xtr, Ytr, alpha):
    ltr = Xtr[:, -LEVEL_WINDOW:].mean(axis=1)
    Xtr_f = Xtr - ltr[:, None]
    Ytr_f = Ytr - ltr[:, None]
    sx = StandardScaler().fit(Xtr_f)
    sy = StandardScaler().fit(Ytr_f)
    model = Ridge(alpha=alpha, fit_intercept=True).fit(sx.transform(Xtr_f), sy.transform(Ytr_f))

    def predict(X, level):
        return sy.inverse_transform(model.predict(sx.transform(X - level[:, None]))) + level[:, None]

    return predict


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

    Xtr, Ytr, ltr = make_xy(z, "train", ot, HORIZON)
    Xva, Yva, lva = make_xy(z, "val", ot, HORIZON)
    Xte, Yte, lte = make_xy(z, "test", ot, HORIZON)
    baselines = make_baselines(z, ot, HORIZON)

    ridge = fit_ridge_anchored(Xtr, Ytr, Xva, Yva, Xte, Yte, ltr, lva, lte)
    ridge["alpha_df"].to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    test_rows: list[dict] = []
    for name, pred in baselines.items():
        test_rows.append({"model": name, **metrics(pred, Yte)})
    test_rows.append({"model": "Anchored Ridge", **metrics(ridge["pred_test"], Yte)})
    test_df = pd.DataFrame(test_rows)
    for ref in ["B1_persistence", "B2b_local_mean_96"]:
        ref_row = test_df[test_df["model"] == ref].iloc[0]
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in {
        "B1": baselines["B1_persistence"],
        "B2b": baselines["B2b_local_mean_96"],
        "Anchored Ridge": ridge["pred_test"],
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
        for name in ["B1", "B2b", "Anchored Ridge"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        for ref in ["B1", "B2b"]:
            row[f"Ridge_mse_skill_vs_{ref}"] = 1 - row["Anchored Ridge_mse"] / row[f"{ref}_mse"]
            row[f"Ridge_mae_skill_vs_{ref}"] = 1 - row["Anchored Ridge_mae"] / row[f"{ref}_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口 ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    win_stats: dict = {"n_windows": int(len(Yte))}
    for ref_name, ref_pred in [
        ("B1", baselines["B1_persistence"]),
        ("B2b", baselines["B2b_local_mean_96"]),
    ]:
        win_stats[f"Ridge_vs_{ref_name}_mae"] = block_bootstrap_ci(wmae(ridge["pred_test"]) - wmae(ref_pred))
        win_stats[f"Ridge_vs_{ref_name}_mse"] = block_bootstrap_ci(wmse(ridge["pred_test"]) - wmse(ref_pred))
        win_stats[f"Ridge_win_rate_vs_{ref_name}_pct"] = float(np.mean(wmae(ridge["pred_test"]) < wmae(ref_pred)) * 100)

    # ---------- 偏离锚定水平 ----------
    dev_pred = ridge["pred_test"] - lte[:, None]
    dev_true = Yte - lte[:, None]
    dev_b2b = baselines["B2b_local_mean_96"] - lte[:, None]
    mean_abs_dev_pred = np.mean(np.abs(dev_pred), axis=0)
    mean_abs_dev_true = np.mean(np.abs(dev_true), axis=0)
    mean_abs_dev_b2b = np.mean(np.abs(dev_b2b), axis=0)
    corr_dev = np.array(
        [
            np.corrcoef(dev_pred[:, h], dev_true[:, h])[0, 1]
            if np.std(dev_pred[:, h]) > 1e-12 and np.std(dev_true[:, h]) > 1e-12
            else np.nan
            for h in range(HORIZON)
        ]
    )
    coef_raw = ridge["coef"] / ridge["sx"].scale_
    sum_abs_coef = np.abs(coef_raw).sum(axis=1)
    intercept = ridge["intercept"]
    dev_df = pd.DataFrame(
        {
            "step": np.arange(1, HORIZON + 1),
            "mean_abs_dev_pred": mean_abs_dev_pred,
            "mean_abs_dev_true": mean_abs_dev_true,
            "mean_abs_dev_b2b": mean_abs_dev_b2b,
            "corr_dev_pred_true": corr_dev,
            "sum_abs_coef": sum_abs_coef,
            "intercept": intercept,
        }
    )
    dev_df.to_csv(OUT_DIR / "anchor_deviation.csv", index=False, encoding="utf-8-sig")

    # ---------- 跨 horizon skill ----------
    round1 = pd.read_csv(BASELINE_DIR / "round1_metrics.csv")
    h96 = round1[(round1["split"] == "test") & (round1["horizon"] == 96)].set_index("model")
    ridge96 = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")
    ridge96 = ridge96[ridge96["model"] == "Anchored Ridge"].iloc[0]
    h192 = pd.read_csv(H192_DIR / "test_metrics.csv")
    h192_ridge = h192[h192["model"] == "Anchored Ridge"].iloc[0]
    h192_b2b = h192[h192["model"] == "B2b_local_mean_96"].iloc[0]
    h336 = pd.read_csv(H336_DIR / "test_metrics.csv")
    h336_ridge = h336[h336["model"] == "Anchored Ridge"].iloc[0]
    h336_b2b = h336[h336["model"] == "B2b_local_mean_96"].iloc[0]

    horizon_rows: list[dict] = []
    for horizon, ridge_row, b2b_row in [
        (96, ridge96, h96.loc["B2b_local_mean_96"]),
        (192, h192_ridge, h192_b2b),
        (336, h336_ridge, h336_b2b),
        (720, test_df[test_df["model"] == "Anchored Ridge"].iloc[0],
         test_df[test_df["model"] == "B2b_local_mean_96"].iloc[0]),
    ]:
        horizon_rows.append(
            {
                "horizon": horizon,
                "ridge_mse": float(ridge_row["mse"]),
                "ridge_mae": float(ridge_row["mae"]),
                "b2b_mse": float(b2b_row["mse"]),
                "b2b_mae": float(b2b_row["mae"]),
                "mse_skill_vs_B2b": 1 - float(ridge_row["mse"]) / float(b2b_row["mse"]),
                "mae_skill_vs_B2b": 1 - float(ridge_row["mae"]) / float(b2b_row["mae"]),
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(OUT_DIR / "horizon_skill.csv", index=False, encoding="utf-8-sig")

    # ---------- 统一预测起点 ----------
    origins = z[f"test_{HORIZON}_x_start"].astype(int) + SEQ_LEN - 1
    X_common = np.stack([ot[o - (SEQ_LEN - 1) : o + 1] for o in origins])
    level_common = X_common[:, -LEVEL_WINDOW:].mean(axis=1)
    fixed_rows: list[dict] = []
    for horizon, alpha in [(96, 100.0), (192, 100.0), (336, 100.0), (HORIZON, ridge["best_alpha"])]:
        Y_common = np.stack([ot[o + 1 : o + 1 + horizon] for o in origins])
        Xtr_h, Ytr_h, _ = make_xy(z, "train", ot, horizon)
        predict = fit_ridge_fixed_alpha(Xtr_h, Ytr_h, alpha)
        pred = predict(X_common, level_common)
        b2b = np.repeat(X_common[:, -96:].mean(axis=1)[:, None], horizon, axis=1)
        m = metrics(pred, Y_common)
        mb = metrics(b2b, Y_common)
        fixed_rows.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "n_origins": int(len(origins)),
                "ridge_mse": m["mse"],
                "ridge_mae": m["mae"],
                "b2b_mse": mb["mse"],
                "b2b_mae": mb["mae"],
                "mae_skill_vs_B2b": 1 - m["mae"] / mb["mae"],
                "mse_skill_vs_B2b": 1 - m["mse"] / mb["mse"],
            }
        )
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(OUT_DIR / "fixed_origin_skill.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, ls in [("val", "-"), ("test", "--")]:
        d = ridge["alpha_df"][ridge["alpha_df"]["split"] == split].sort_values("alpha")
        ax.plot(d["alpha"], d["mae"], marker="o", ms=3, ls=ls, label=split)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log)")
    ax.set_ylabel("MAE (C)")
    ax.set_title(f"Anchored Ridge alpha selection at H={HORIZON}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_alpha_selection.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["B1_persistence", "B2b_local_mean_96", "Anchored Ridge"]
    labels = ["B1", "B2b", "Anchored Ridge"]
    axes[0].bar(labels, [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names],
                color=["#7f8c8d", "#16a085", "#922b21"])
    axes[0].set_ylabel("test MAE (C)")
    axes[0].set_title(f"H={HORIZON}: MAE")
    axes[1].bar(labels, [test_df[test_df["model"] == n]["mse"].iloc[0] for n in names],
                color=["#7f8c8d", "#16a085", "#922b21"])
    axes[1].set_ylabel("test MSE (C^2)")
    axes[1].set_title(f"H={HORIZON}: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("B2b", "#16a085"), ("B1", "#7f8c8d")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=0.9, color=color, label=name)
    for a, _ in [(25, 96), (97, 192), (193, 336), (337, 720)]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("forecast step (1..720)")
    ax.set_ylabel("MAE (C)")
    ax.set_title(f"Per-step MAE on test (H={HORIZON})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("B2b", "#16a085")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=0.9, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..720)")
    ax.set_ylabel("bias (C)")
    ax.set_title(f"Per-step bias on test (H={HORIZON})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    axes[0].bar(x - width / 2, seg_df["Ridge_mae_skill_vs_B2b"], width, label="MAE skill", color="#922b21")
    axes[0].bar(x + width / 2, seg_df["Ridge_mse_skill_vs_B2b"], width, label="MSE skill", color="#16a085")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, [s["segment"] for s in seg_rows], rotation=30, fontsize=7)
    axes[0].set_ylabel("skill vs B2b")
    axes[0].set_title("Ridge vs B2b by segment")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - width / 2, seg_df["Ridge_mae_skill_vs_B1"], width, label="vs B1 MAE", color="#7f8c8d")
    axes[1].bar(x + width / 2, seg_df["Ridge_mse_skill_vs_B1"], width, label="vs B1 MSE", color="#c0392b")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x, [s["segment"] for s in seg_rows], rotation=30, fontsize=7)
    axes[1].set_ylabel("skill")
    axes[1].set_title("Ridge vs B1 by segment")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_skill.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_pred"], lw=1.0, color="#922b21", label="Ridge |pred - l|")
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_true"], lw=0.9, ls="--", color="#2c3e50", label="truth |y - l|")
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_b2b"], lw=0.9, ls=":", color="#16a085", label="B2b |pred - l|")
    axes[0].set_xlabel("forecast step")
    axes[0].set_ylabel("mean |deviation| (C)")
    axes[0].set_title("Deviation from anchor l_t")
    axes[0].legend(fontsize=6)
    axes[1].plot(dev_df["step"], dev_df["sum_abs_coef"], lw=1.0, color="#922b21", label="sum |coef|")
    axes[1].set_xlabel("forecast step")
    axes[1].set_ylabel("sum |raw coef|")
    axes[1].set_title("Effective mapping strength")
    axes[1].legend(fontsize=7)
    axes[2].plot(dev_df["step"], dev_df["corr_dev_pred_true"], lw=1.0, color="#c0392b")
    axes[2].axhline(0, color="k", lw=0.7)
    axes[2].set_xlabel("forecast step")
    axes[2].set_ylabel("corr(pred dev, true dev)")
    axes[2].set_title("Signal in the deviation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_anchor_deviation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(horizon_df))
    width = 0.35
    ax.bar(x - width / 2, horizon_df["mae_skill_vs_B2b"], width, label="official test windows", color="#922b21")
    ax.bar(x + width / 2, fixed_df["mae_skill_vs_B2b"], width, label="common origins", color="#16a085")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"H={h}" for h in horizon_df["horizon"]])
    ax.set_ylabel("MAE skill vs B2b")
    ax.set_title("Horizon study: Ridge skill vs B2b")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_horizon_skill.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=1.5, label="truth")
    ax.plot(dates, ridge["pred_test"][i], color="#922b21", lw=1.1, label="Anchored Ridge")
    ax.plot(dates, baselines["B2b_local_mean_96"][i], color="#16a085", lw=0.9, ls=":", label="B2b")
    ax.set_ylabel("OT (C)")
    ax.set_title(f"Example test window (H={HORIZON})")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_example_forecast.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "model": "Anchored Ridge",
            "baselines": ["B1", "B2b (96h mean)"],
            "anchor": "l_t = mean(last 24h OT)",
            "alphas": ALPHAS,
            "alpha_selection": "val MAE",
        },
        "best_alpha": ridge["best_alpha"],
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
        "horizon_skill": horizon_df.to_dict(orient="records"),
        "fixed_origin_skill": fixed_df.to_dict(orient="records"),
        "anchor_deviation_summary": {
            "at_steps": [24, 96, 192, 336, 720],
            "mean_abs_dev_pred": [float(mean_abs_dev_pred[[23, 95, 191, 335, 719][k]]) for k in range(5)],
            "mean_abs_dev_true": [float(mean_abs_dev_true[[23, 95, 191, 335, 719][k]]) for k in range(5)],
            "mean_abs_dev_b2b": [float(mean_abs_dev_b2b[[23, 95, 191, 335, 719][k]]) for k in range(5)],
            "sum_abs_coef": [float(sum_abs_coef[[23, 95, 191, 335, 719][k]]) for k in range(5)],
            "abs_intercept": [float(abs(intercept[[23, 95, 191, 335, 719][k]])) for k in range(5)],
            "corr_dev": [float(corr_dev[[23, 95, 191, 335, 719][k]]) for k in range(5)],
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"ridge best alpha: {ridge['best_alpha']}")
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\nhorizon skill:")
    print(horizon_df.round(4).to_string(index=False))
    print("\nfixed origins:")
    print(fixed_df.round(4).to_string(index=False))
    print("\nsegments:")
    print(seg_df.round(4).to_string(index=False))
    print("\nanchor deviation summary:")
    print(json.dumps(summary["anchor_deviation_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
