#!/usr/bin/env python
r"""ETTh1 第一轮基线（官方 12/4/4 划分）。

五组配置：
    B0  Global Mean           训练集 OT 均值
    B1  Persistence           最后一个观测值
    B2a Local Mean 24         最近 24h 均值
    B2b Local Mean 96         最近 96h 均值
    B3  Seasonal Naive 24     重复最后 24h 的变化模式

评估：官方 val/test 目标块，seq_len=96，direct multi-horizon，
      pred_len ∈ {96, 192, 336, 720}，原始 ℃ 上的 MSE/RMSE/MAE。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\baselines_round1.py

输出：
    outputs/baselines/round1_metrics.csv
    outputs/baselines/round1_summary.json
    outputs/baselines/figures/*.png
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

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
OUT_DIR = ROOT / "outputs" / "baselines"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
PRED_LENS = [96, 192, 336, 720]
MODELS = [
    "B0_global_mean",
    "B1_persistence",
    "B2a_local_mean_24",
    "B2b_local_mean_96",
    "B3_seasonal_naive_24",
]


def predict(model: str, origin: int, horizon: int, ot: np.ndarray, train_mean: float) -> np.ndarray:
    if model == "B0_global_mean":
        return np.full(horizon, train_mean, dtype=float)
    if model == "B1_persistence":
        return np.full(horizon, ot[origin], dtype=float)
    if model == "B2a_local_mean_24":
        return np.full(horizon, ot[origin - 23 : origin + 1].mean(), dtype=float)
    if model == "B2b_local_mean_96":
        return np.full(horizon, ot[origin - 95 : origin + 1].mean(), dtype=float)
    if model == "B3_seasonal_naive_24":
        pattern = ot[origin - 23 : origin + 1]
        return pattern[np.arange(horizon) % 24]
    raise ValueError(f"unknown model: {model}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    ot = df["OT"].to_numpy(dtype=float)
    train_mean = float(ot[:8640].mean())

    npz_path = SPLIT_DIR / f"windows_L{SEQ_LEN}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}; run scripts/make_splits.py first")
    z = np.load(npz_path)

    rows: list[dict] = []
    per_window_test_h96: dict[str, np.ndarray] = {}
    ys_h96: np.ndarray | None = None
    for split in ["val", "test"]:
        for horizon in PRED_LENS:
            xs = z[f"{split}_{horizon}_x_start"]
            ys = z[f"{split}_{horizon}_y_start"]
            ye = z[f"{split}_{horizon}_y_end"]
            truth = np.stack([ot[y0:y1] for y0, y1 in zip(ys, ye)])
            preds = {m: np.empty_like(truth) for m in MODELS}
            for i, x in enumerate(xs):
                origin = int(x) + SEQ_LEN - 1
                for model in MODELS:
                    preds[model][i] = predict(model, origin, horizon, ot, train_mean)

            mse_by_model: dict[str, float] = {}
            for model in MODELS:
                err = preds[model] - truth
                mse = float(np.mean(err**2))
                mse_by_model[model] = mse
                rows.append(
                    {
                        "split": split,
                        "seq_len": SEQ_LEN,
                        "horizon": horizon,
                        "model": model,
                        "n_samples": int(len(xs)),
                        "mse": mse,
                        "rmse": float(np.sqrt(mse)),
                        "mae": float(np.mean(np.abs(err))),
                    }
                )
            for row in rows[-len(MODELS) :]:
                row["skill_vs_B0"] = float(1 - row["mse"] / mse_by_model["B0_global_mean"])
                row["skill_vs_B1"] = float(1 - row["mse"] / mse_by_model["B1_persistence"])

            if split == "test" and horizon == 96:
                ys_h96 = ys.copy()
                for model in MODELS:
                    per_window_test_h96[model] = np.mean(np.abs(preds[model] - truth), axis=1)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "round1_metrics.csv", index=False, encoding="utf-8-sig")

    test = metrics[metrics["split"] == "test"].copy()
    val = metrics[metrics["split"] == "val"].copy()
    pivot_mse = test.pivot(index="model", columns="horizon", values="mse").loc[MODELS]
    pivot_mae = test.pivot(index="model", columns="horizon", values="mae").loc[MODELS]
    val_pivot_mse = val.pivot(index="model", columns="horizon", values="mse").loc[MODELS]
    val_pivot_mae = val.pivot(index="model", columns="horizon", values="mae").loc[MODELS]

    summary = {
        "protocol": {
            "split": "official_ETT_12_4_4",
            "seq_len": SEQ_LEN,
            "pred_lens": PRED_LENS,
            "evaluation": "direct multi-horizon on the official val/test target blocks",
            "units": "degrees Celsius (raw OT)",
            "note": "MAE(H) is averaged over all steps 1..H, not only the final step",
        },
        "train_mean": train_mean,
        "definitions": {
            "B0_global_mean": "predict train OT mean",
            "B1_persistence": "predict last observed OT",
            "B2a_local_mean_24": "predict mean of the last 24 observations",
            "B2b_local_mean_96": "predict mean of the last 96 observations",
            "B3_seasonal_naive_24": "repeat the last 24h OT pattern",
        },
        "test_mse": {str(h): {m: float(pivot_mse.loc[m, h]) for m in MODELS} for h in PRED_LENS},
        "test_mae": {str(h): {m: float(pivot_mae.loc[m, h]) for m in MODELS} for h in PRED_LENS},
        "val_mse": {str(h): {m: float(val_pivot_mse.loc[m, h]) for m in MODELS} for h in PRED_LENS},
        "val_mae": {str(h): {m: float(val_pivot_mae.loc[m, h]) for m in MODELS} for h in PRED_LENS},
        "test_skill_vs_B1": {
            str(h): {
                m: float(
                    test[(test["model"] == m) & (test["horizon"] == h)]["skill_vs_B1"].iloc[0]
                )
                for m in MODELS
            }
            for h in PRED_LENS
        },
        "best_by_horizon": {str(h): str(pivot_mae[h].idxmin()) for h in PRED_LENS},
    }
    (OUT_DIR / "round1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 图 1：误差随 horizon ----------
    colors = {
        "B0_global_mean": "#95a5a6",
        "B1_persistence": "#e74c3c",
        "B2a_local_mean_24": "#3498db",
        "B2b_local_mean_96": "#2ecc71",
        "B3_seasonal_naive_24": "#9b59b6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model in MODELS:
        g = test[test["model"] == model].sort_values("horizon")
        axes[0].plot(g["horizon"], g["mse"], marker="o", ms=3, label=model, color=colors[model])
        axes[1].plot(g["horizon"], g["mae"], marker="o", ms=3, label=model, color=colors[model])
    for ax, ylabel in zip(axes, ["MSE (C^2)", "MAE (C)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("horizon (hours, log)")
        ax.set_ylabel(ylabel + ", log")
        ax.legend(fontsize=7)
    axes[0].set_title("Round-1 baselines on test: MSE vs horizon")
    axes[1].set_title("Round-1 baselines on test: MAE vs horizon")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_error_vs_horizon.png")
    plt.close(fig)

    # ---------- 图 2：一个 test 窗口的预测示例 ----------
    horizon = 96
    xs = z[f"test_{horizon}_x_start"]
    ys = z[f"test_{horizon}_y_start"]
    ye = z[f"test_{horizon}_y_end"]
    i = 0
    origin = int(xs[i]) + SEQ_LEN - 1
    truth = ot[int(ys[i]) : int(ye[i])]
    dates = df["date"].iloc[int(ys[i]) : int(ye[i])]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, truth, color="k", lw=2.0, label="truth")
    for model in MODELS:
        pred = predict(model, origin, horizon, ot, train_mean)
        ax.plot(dates, pred, lw=1.1, ls="--", label=model, color=colors[model], alpha=0.9)
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_example_forecast.png")
    plt.close(fig)

    # ---------- 图 3：H=96 的分月 MAE（不含 B0） ----------
    if ys_h96 is None:
        raise RuntimeError("H=96 test windows were not captured")
    months = df["date"].iloc[ys_h96.astype(int)].dt.strftime("%Y-%m")
    month_df = pd.DataFrame({"month": months.to_numpy(), **per_window_test_h96})
    month_mae = month_df.groupby("month").mean(numeric_only=True)
    month_mae.to_csv(OUT_DIR / "monthly_mae_h96.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.2
    x = np.arange(len(month_mae.index))
    for k, model in enumerate(["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96", "B3_seasonal_naive_24"]):
        ax.bar(x + (k - 1.5) * width, month_mae[model], width, label=model, color=colors[model])
    ax.set_xticks(x, month_mae.index, rotation=0)
    ax.set_ylabel("mean per-window MAE (C)")
    ax.set_title("H=96 test MAE by target-start month")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_monthly_mae_h96.png")
    plt.close(fig)

    # ---------- 图 4：相对 persistence 的 skill ----------
    horizons = [96, 336, 720]
    models = ["B2a_local_mean_24", "B2b_local_mean_96", "B3_seasonal_naive_24"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.22
    x = np.arange(len(horizons))
    for k, model in enumerate(models):
        vals = [
            float(test[(test["model"] == model) & (test["horizon"] == h)]["skill_vs_B1"].iloc[0])
            for h in horizons
        ]
        ax.bar(x + (k - 1) * width, vals, width, label=model, color=colors[model])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"H={h}" for h in horizons])
    ax.set_ylabel("skill vs persistence (1 - MSE/MSE_B1)")
    ax.set_title("Does smoothing/seasonal beat persistence?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_skill_vs_persistence.png")
    plt.close(fig)

    print("train mean OT: %.4f" % train_mean)
    print("\nTEST MAE:")
    print(pivot_mae.round(4).to_string())
    print("\nTEST MSE:")
    print(pivot_mse.round(4).to_string())
    print("\nbest by horizon (MAE):")
    for h in PRED_LENS:
        print(f"  H={h}: {summary['best_by_horizon'][str(h)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
