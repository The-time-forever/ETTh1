#!/usr/bin/env python
r"""Ridge 预测器：S 模式，96 -> 96，直接多步，无额外变量/日历特征。

实验：
  - X = 输入窗口的 96 个 OT 值（展平）
  - Y = 未来 96 个 OT 值（直接多输出）
  - X 标准化，scaler 只在 train 拟合
  - alpha ∈ {0.1, 1, 10, 100}，用 val MAE 选择
  - test 上对比 Ridge / B1 persistence / B2a local mean 24
  - 96 步逐点误差、分段误差、逐窗口胜负、block bootstrap

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\ridge_s96_h96.py

输出：
    outputs/ridge/alpha_selection.csv
    outputs/ridge/test_metrics.csv
    outputs/ridge/per_step_metrics.csv
    outputs/ridge/segment_metrics.csv
    outputs/ridge/summary.json
    outputs/ridge/figures/*.png
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
OUT_DIR = ROOT / "outputs" / "ridge"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
ALPHAS = [0.1, 1.0, 10.0, 100.0]
MODELS = ["Ridge", "B1_persistence", "B2a_local_mean_24"]
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]


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


def baseline_preds(z, split: str, ot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    b1 = np.repeat(ot[origins][:, None], HORIZON, axis=1)
    local24 = np.array([ot[o - 23 : o + 1].mean() for o in origins])
    b2a = np.repeat(local24[:, None], HORIZON, axis=1)
    return b1, b2a


def block_bootstrap_ci(diff: np.ndarray, block: int = 96, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([diff[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo, med, hi = np.percentile(means, [2.5, 50, 97.5])
    return {"mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "median": float(med)}


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    ot = df["OT"].to_numpy(dtype=float)

    z = np.load(SPLIT_DIR / f"windows_L{SEQ_LEN}.npz")
    Xtr, Ytr = make_xy(z, "train", ot)
    Xva, Yva = make_xy(z, "val", ot)
    Xte, Yte = make_xy(z, "test", ot)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    b1_va, b2a_va = baseline_preds(z, "val", ot)
    b1_te, b2a_te = baseline_preds(z, "test", ot)

    # ---------- alpha 选择（val），test 仅作诊断 ----------
    alpha_rows: list[dict] = []
    models: dict[float, Ridge] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xtr_s, Ytr)
        models[alpha] = model
        for split, X, Y in [("val", Xva_s, Yva), ("test", Xte_s, Yte)]:
            m = metrics(model.predict(X), Y)
            alpha_rows.append({"alpha": alpha, "split": split, **m})
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    val_df = alpha_df[alpha_df["split"] == "val"].sort_values("mae")
    best_alpha = float(val_df.iloc[0]["alpha"])
    best_model = models[best_alpha]

    # ---------- test 对比 ----------
    ridge_te = best_model.predict(Xte_s)
    preds = {"Ridge": ridge_te, "B1_persistence": b1_te, "B2a_local_mean_24": b2a_te}
    test_rows: list[dict] = []
    for name, pred in preds.items():
        m = metrics(pred, Yte)
        test_rows.append({"model": name, "alpha": best_alpha if name == "Ridge" else "", **m})
    test_df = pd.DataFrame(test_rows)
    b1_mse = float(test_df.loc[test_df["model"] == "B1_persistence", "mse"].iloc[0])
    b1_mae = float(test_df.loc[test_df["model"] == "B1_persistence", "mae"].iloc[0])
    b2a_mse = float(test_df.loc[test_df["model"] == "B2a_local_mean_24", "mse"].iloc[0])
    b2a_mae = float(test_df.loc[test_df["model"] == "B2a_local_mean_24", "mae"].iloc[0])
    for row in test_rows:
        row["mse_skill_vs_B1"] = 1 - row["mse"] / b1_mse
        row["mae_skill_vs_B1"] = 1 - row["mae"] / b1_mae
        row["mse_skill_vs_B2a"] = 1 - row["mse"] / b2a_mse
        row["mae_skill_vs_B2a"] = 1 - row["mae"] / b2a_mae
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点误差 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in preds.items():
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    per_step["ridge_minus_b1_mae"] = per_step["Ridge_mae"] - per_step["B1_persistence_mae"]
    per_step["ridge_minus_b2a_mae"] = per_step["Ridge_mae"] - per_step["B2a_local_mean_24_mae"]
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段误差 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in MODELS:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        row["ridge_mse_skill_vs_B2a"] = 1 - row["Ridge_mse"] / row["B2a_local_mean_24_mse"]
        row["ridge_mae_skill_vs_B2a"] = 1 - row["Ridge_mae"] / row["B2a_local_mean_24_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口胜负与 bootstrap ----------
    mae_ridge_w = np.mean(np.abs(ridge_te - Yte), axis=1)
    mae_b1_w = np.mean(np.abs(b1_te - Yte), axis=1)
    mae_b2a_w = np.mean(np.abs(b2a_te - Yte), axis=1)
    mse_ridge_w = np.mean((ridge_te - Yte) ** 2, axis=1)
    mse_b2a_w = np.mean((b2a_te - Yte) ** 2, axis=1)
    win_stats = {
        "n_windows": int(len(mae_ridge_w)),
        "ridge_mae_better_than_B1_pct": float(np.mean(mae_ridge_w < mae_b1_w) * 100),
        "ridge_mae_better_than_B2a_pct": float(np.mean(mae_ridge_w < mae_b2a_w) * 100),
        "mean_window_mae_ridge": float(mae_ridge_w.mean()),
        "mean_window_mae_B2a": float(mae_b2a_w.mean()),
        "median_window_mae_diff_ridge_minus_B2a": float(np.median(mae_ridge_w - mae_b2a_w)),
        "bootstrap_mae_diff_vs_B2a": block_bootstrap_ci(mae_ridge_w - mae_b2a_w),
        "bootstrap_mse_diff_vs_B2a": block_bootstrap_ci(mse_ridge_w - mse_b2a_w),
    }

    # ---------- 系数诊断 ----------
    coef = best_model.coef_  # (H, L), standardized-X scale
    coef_raw = coef / scaler.scale_  # back to raw OT scale
    intercept_raw = best_model.intercept_ - coef_raw @ scaler.mean_
    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "strategy": "direct multi-output Ridge",
            "extra_features": "none (no loads, no calendar)",
            "standardization": "StandardScaler fit on train X only",
            "alpha_grid": ALPHAS,
            "alpha_selection": "val MAE",
        },
        "best_alpha": best_alpha,
        "alpha_table": alpha_df.to_dict(orient="records"),
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
        "coef": {
            "scale": "raw (coef / StandardScaler.scale_)",
            "mean_abs_by_input_lag": [float(v) for v in np.mean(np.abs(coef_raw), axis=0)],
            "weight_on_last_input_by_step": [float(v) for v in coef_raw[:, -1]],
            "sum_coef_by_step": [float(v) for v in coef_raw.sum(axis=1)],
            "intercept_by_step": [float(v) for v in intercept_raw],
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    colors = {"Ridge": "#c0392b", "B1_persistence": "#7f8c8d", "B2a_local_mean_24": "#2980b9"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for split, ls in [("val", "-"), ("test", "--")]:
        d = alpha_df[alpha_df["split"] == split].sort_values("alpha")
        axes[0].plot(d["alpha"], d["mae"], marker="o", ms=3, ls=ls, label=split)
        axes[1].plot(d["alpha"], d["mse"], marker="o", ms=3, ls=ls, label=split)
    for ax, ylabel in zip(axes, ["MAE (C)", "MSE (C^2)"]):
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log)")
        ax.set_ylabel(ylabel)
        ax.axvline(best_alpha, color="k", ls=":", lw=0.8)
        ax.legend()
    axes[0].set_title("Alpha selection: MAE (val used for selection)")
    axes[1].set_title("Alpha selection: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_alpha_selection.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name in MODELS:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.2, label=name, color=colors[name])
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.6)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("MAE (C)")
    ax.set_title("Per-step MAE on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(per_step["step"], per_step["ridge_minus_b2a_mae"], lw=1.2, color="#c0392b", label="Ridge - B2a")
    ax.plot(per_step["step"], per_step["ridge_minus_b1_mae"], lw=1.0, color="#7f8c8d", label="Ridge - B1")
    ax.axhline(0, color="k", lw=0.8)
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.6)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("MAE difference (C)")
    ax.set_title("Per-step MAE difference: negative = Ridge better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_diff.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(mae_ridge_w, bins=50, alpha=0.55, density=True, label="Ridge", color=colors["Ridge"])
    axes[0].hist(mae_b2a_w, bins=50, alpha=0.55, density=True, label="B2a", color=colors["B2a_local_mean_24"])
    axes[0].set_xlabel("per-window MAE (C)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Per-window MAE distribution")
    axes[0].legend()
    axes[1].hist(mae_ridge_w - mae_b2a_w, bins=60, alpha=0.8, color="#8e44ad")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Ridge MAE - B2a MAE (C)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Win rate vs B2a: {win_stats['ridge_mae_better_than_B2a_pct']:.1f}%")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_window_error.png")
    plt.close(fig)

    i = 0
    xs = z[f"test_{HORIZON}_x_start"].astype(int)
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, ridge_te[i], color=colors["Ridge"], lw=1.4, label="Ridge")
    ax.plot(dates, b1_te[i], color=colors["B1_persistence"], lw=1.0, ls="--", label="B1")
    ax.plot(dates, b2a_te[i], color=colors["B2a_local_mean_24"], lw=1.0, ls="--", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_example_forecast.png")
    plt.close(fig)

    lag_importance = np.mean(np.abs(coef_raw), axis=0)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(np.arange(1, SEQ_LEN + 1), lag_importance, lw=1.2, color="#2c3e50")
    ax.set_xlabel("input lag (1 = oldest, 96 = most recent)")
    ax.set_ylabel("mean |raw coef|")
    ax.set_title("Ridge coefficient importance by input lag (raw OT scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_coefficient_profile.png")
    plt.close(fig)

    print(f"best alpha (val MAE): {best_alpha}")
    print("\nalpha table:")
    print(alpha_df.pivot(index="alpha", columns="split", values="mae").round(4).to_string())
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\nsegments:")
    print(seg_df.round(4).to_string(index=False))
    print("\nwin stats:")
    print(json.dumps(win_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
