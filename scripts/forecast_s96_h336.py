#!/usr/bin/env python
r"""Horizon 336：S 模式，96h 输入 -> 336h OT 输出，Anchored Ridge。

只做线性模型 + 统计基线，不训练神经网络、不加新特征。

回答：
    1) H=96/192/336，Ridge 相对基线的 skill 是否继续缩小（重点 B2b）？
    2) 优势是否越来越集中在较近的预测步长（1-24 / 25-96 / 97-192 / 193-336）？
    3) 远期预测是否逐渐退化为“局部水平”（预测偏离锚定水平越来越小）？

另外做一个轻量的“统一预测起点”对照：
    在 H=336 的 test 起点上，重新评估 H=96/192/336 Ridge 与 B2b，
    以区分样本差异与 horizon 差异。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\forecast_s96_h336.py

输出：
    outputs/h336/alpha_selection.csv
    outputs/h336/test_metrics.csv
    outputs/h336/per_step_metrics.csv
    outputs/h336/segment_metrics.csv
    outputs/h336/anchor_deviation.csv
    outputs/h336/horizon_skill.csv
    outputs/h336/fixed_origin_skill.csv
    outputs/h336/summary.json
    outputs/h336/figures/*.png
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
OUT_DIR = ROOT / "outputs" / "h336"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 336
LEVEL_WINDOW = 24
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEGMENTS = [(1, 24), (25, 96), (97, 192), (193, 336)]
COLORS = {
    "Anchored Ridge": "#922b21",
    "B1_persistence": "#7f8c8d",
    "B2a_local_mean_24": "#2980b9",
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
    mean24 = np.array([ot[o - 23 : o + 1].mean() for o in origins])
    mean96 = np.array([ot[o - 95 : o + 1].mean() for o in origins])
    return {
        "B1_persistence": b1,
        "B2a_local_mean_24": np.repeat(mean24[:, None], horizon, axis=1),
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

    def predict_common(X, level):
        Xf = X - level[:, None]
        return sy.inverse_transform(model.predict(sx.transform(Xf))) + level[:, None]

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
        "predict_common": predict_common,
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
    for ref in ["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96"]:
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
        for name in ["B1", "B2a", "B2b", "Anchored Ridge"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        for ref in ["B1", "B2a", "B2b"]:
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
        ("B2a", baselines["B2a_local_mean_24"]),
        ("B2b", baselines["B2b_local_mean_96"]),
    ]:
        win_stats[f"Ridge_vs_{ref_name}_mae"] = block_bootstrap_ci(wmae(ridge["pred_test"]) - wmae(ref_pred))
        win_stats[f"Ridge_vs_{ref_name}_mse"] = block_bootstrap_ci(wmse(ridge["pred_test"]) - wmse(ref_pred))
        win_stats[f"Ridge_win_rate_vs_{ref_name}_pct"] = float(np.mean(wmae(ridge["pred_test"]) < wmae(ref_pred)) * 100)

    # ---------- 问题 3：偏离锚定水平 ----------
    lte_2d = lte[:, None]
    dev_pred = ridge["pred_test"] - lte_2d
    dev_true = Yte - lte_2d
    dev_b2b = baselines["B2b_local_mean_96"] - lte_2d
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
    sum_coef = coef_raw.sum(axis=1)
    intercept = ridge["intercept"]
    dev_df = pd.DataFrame(
        {
            "step": np.arange(1, HORIZON + 1),
            "mean_abs_dev_pred": mean_abs_dev_pred,
            "mean_abs_dev_true": mean_abs_dev_true,
            "mean_abs_dev_b2b": mean_abs_dev_b2b,
            "corr_dev_pred_true": corr_dev,
            "sum_abs_coef": sum_abs_coef,
            "sum_coef": sum_coef,
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
    h192_b1 = h192[h192["model"] == "B1_persistence"].iloc[0]
    h192_b2a = h192[h192["model"] == "B2a_local_mean_24"].iloc[0]
    h192_b2b = h192[h192["model"] == "B2b_local_mean_96"].iloc[0]

    horizon_rows: list[dict] = []
    for horizon, base in [
        (96, {
            "B1": (float(h96.loc["B1_persistence", "mse"]), float(h96.loc["B1_persistence", "mae"])),
            "B2a": (float(h96.loc["B2a_local_mean_24", "mse"]), float(h96.loc["B2a_local_mean_24", "mae"])),
            "B2b": (float(h96.loc["B2b_local_mean_96", "mse"]), float(h96.loc["B2b_local_mean_96", "mae"])),
        }),
        (192, {
            "B1": (float(h192_b1["mse"]), float(h192_b1["mae"])),
            "B2a": (float(h192_b2a["mse"]), float(h192_b2a["mae"])),
            "B2b": (float(h192_b2b["mse"]), float(h192_b2b["mae"])),
        }),
        (336, {
            "B1": (float(test_df[test_df["model"] == "B1_persistence"]["mse"].iloc[0]),
                   float(test_df[test_df["model"] == "B1_persistence"]["mae"].iloc[0])),
            "B2a": (float(test_df[test_df["model"] == "B2a_local_mean_24"]["mse"].iloc[0]),
                    float(test_df[test_df["model"] == "B2a_local_mean_24"]["mae"].iloc[0])),
            "B2b": (float(test_df[test_df["model"] == "B2b_local_mean_96"]["mse"].iloc[0]),
                    float(test_df[test_df["model"] == "B2b_local_mean_96"]["mae"].iloc[0])),
        }),
    ]:
        if horizon == 96:
            ridge_mse, ridge_mae = float(ridge96["mse"]), float(ridge96["mae"])
        elif horizon == 192:
            ridge_mse, ridge_mae = float(h192_ridge["mse"]), float(h192_ridge["mae"])
        else:
            ridge_mse = float(test_df[test_df["model"] == "Anchored Ridge"]["mse"].iloc[0])
            ridge_mae = float(test_df[test_df["model"] == "Anchored Ridge"]["mae"].iloc[0])
        horizon_rows.append(
            {
                "horizon": horizon,
                "ridge_mse": ridge_mse,
                "ridge_mae": ridge_mae,
                "vs_B1_mse_skill": 1 - ridge_mse / base["B1"][0],
                "vs_B1_mae_skill": 1 - ridge_mae / base["B1"][1],
                "vs_B2a_mse_skill": 1 - ridge_mse / base["B2a"][0],
                "vs_B2a_mae_skill": 1 - ridge_mae / base["B2a"][1],
                "vs_B2b_mse_skill": 1 - ridge_mse / base["B2b"][0],
                "vs_B2b_mae_skill": 1 - ridge_mae / base["B2b"][1],
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(OUT_DIR / "horizon_skill.csv", index=False, encoding="utf-8-sig")

    # ---------- 统一预测起点（轻量） ----------
    origins = z[f"test_{HORIZON}_x_start"].astype(int) + SEQ_LEN - 1
    X_common = np.stack([ot[o - (SEQ_LEN - 1) : o + 1] for o in origins])
    level_common = X_common[:, -LEVEL_WINDOW:].mean(axis=1)
    fixed_rows: list[dict] = []
    for horizon, alpha in [(96, 100.0), (192, 100.0), (HORIZON, ridge["best_alpha"])]:
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    names = ["B1_persistence", "B2a_local_mean_24", "B2b_local_mean_96", "Anchored Ridge"]
    labels = ["B1", "B2a", "B2b", "Anchored Ridge"]
    mae_vals = [test_df[test_df["model"] == n]["mae"].iloc[0] for n in names]
    mse_vals = [test_df[test_df["model"] == n]["mse"].iloc[0] for n in names]
    axes[0].bar(labels, mae_vals, color=["#7f8c8d", "#2980b9", "#16a085", "#922b21"])
    axes[0].set_ylabel("test MAE (C)")
    axes[0].set_title(f"H={HORIZON}: test MAE")
    axes[1].bar(labels, mse_vals, color=["#7f8c8d", "#2980b9", "#16a085", "#922b21"])
    axes[1].set_ylabel("test MSE (C^2)")
    axes[1].set_title(f"H={HORIZON}: test MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("B2b", "#16a085"), ("B2a", "#2980b9"), ("B1", "#7f8c8d")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.0, color=color, label=name)
    for a, _ in [(25, 96), (97, 192), (193, 336)]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("forecast step (1..336)")
    ax.set_ylabel("MAE (C)")
    ax.set_title(f"Per-step MAE on test (H={HORIZON})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("Anchored Ridge", "#922b21"), ("B2b", "#16a085"), ("B2a", "#2980b9")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.0, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..336)")
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
    axes[0].set_xticks(x, [s["segment"] for s in seg_rows])
    axes[0].set_ylabel("skill vs B2b")
    axes[0].set_title("Ridge vs B2b by segment")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - width / 2, seg_df["Ridge_mae_skill_vs_B2a"], width, label="vs B2a MAE", color="#2980b9")
    axes[1].bar(x + width / 2, seg_df["Ridge_mae_skill_vs_B1"], width, label="vs B1 MAE", color="#7f8c8d")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x, [s["segment"] for s in seg_rows])
    axes[1].set_ylabel("MAE skill")
    axes[1].set_title("Ridge vs B2a / B1 by segment")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_skill.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_pred"], lw=1.2, color="#922b21", label="Ridge |pred - l|")
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_true"], lw=1.0, ls="--", color="#2c3e50", label="truth |y - l|")
    axes[0].plot(dev_df["step"], dev_df["mean_abs_dev_b2b"], lw=1.0, ls=":", color="#16a085", label="B2b |pred - l|")
    axes[0].set_xlabel("forecast step")
    axes[0].set_ylabel("mean |deviation| (C)")
    axes[0].set_title("Deviation from anchor l_t")
    axes[0].legend(fontsize=7)
    axes[1].plot(dev_df["step"], dev_df["sum_abs_coef"], lw=1.2, color="#922b21", label="sum |coef|")
    ax2 = axes[1].twinx()
    ax2.plot(dev_df["step"], np.abs(dev_df["intercept"]), lw=1.0, ls="--", color="#8e44ad", label="|intercept|")
    axes[1].set_xlabel("forecast step")
    axes[1].set_ylabel("sum |raw coef|")
    ax2.set_ylabel("|intercept|")
    axes[1].set_title("Effective mapping strength")
    axes[1].legend(fontsize=7, loc="upper right")
    ax2.legend(fontsize=7, loc="lower right")
    axes[2].plot(dev_df["step"], dev_df["corr_dev_pred_true"], lw=1.2, color="#c0392b")
    axes[2].axhline(0, color="k", lw=0.7)
    axes[2].set_xlabel("forecast step")
    axes[2].set_ylabel("corr(pred dev, true dev)")
    axes[2].set_title("Signal in the deviation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_anchor_deviation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, horizon_df["vs_B2b_mae_skill"], width, label="official test windows", color="#922b21")
    ax.bar(x + width / 2, fixed_df["mae_skill_vs_B2b"], width, label="common origins", color="#16a085")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"H={h}" for h in horizon_df["horizon"]])
    ax.set_ylabel("MAE skill vs B2b")
    ax.set_title("Ridge skill vs B2b: official vs fixed origins")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_horizon_skill.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=1.6, label="truth")
    ax.plot(dates, ridge["pred_test"][i], color="#922b21", lw=1.2, label="Anchored Ridge")
    ax.plot(dates, baselines["B2b_local_mean_96"][i], color="#16a085", lw=1.0, ls=":", label="B2b")
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
            "baselines": ["B1", "B2a (24h mean)", "B2b (96h mean)"],
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
            "at_steps": [24, 96, 192, 336],
            "mean_abs_dev_pred": [float(mean_abs_dev_pred[[23, 95, 191, 335][k]]) for k in range(4)],
            "mean_abs_dev_true": [float(mean_abs_dev_true[[23, 95, 191, 335][k]]) for k in range(4)],
            "mean_abs_dev_b2b": [float(mean_abs_dev_b2b[[23, 95, 191, 335][k]]) for k in range(4)],
            "sum_abs_coef": [float(sum_abs_coef[[23, 95, 191, 335][k]]) for k in range(4)],
            "abs_intercept": [float(abs(intercept[[23, 95, 191, 335][k]])) for k in range(4)],
            "corr_dev": [float(corr_dev[[23, 95, 191, 335][k]]) for k in range(4)],
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
