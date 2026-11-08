#!/usr/bin/env python
r"""MS 模式 Anchored Ridge：96h x 7 变量 -> 未来 96h OT。

研究问题：历史负载（6 个通道）能否提供仅靠 OT 历史无法获得的预测信息？

设计：
    - 模型：Anchored Ridge（与 S 模式相同的局部水平锚定）
    - 输入：96h x 7 变量（HUFL/HULL/MUFL/MULL/LUFL/LULL/OT）
    - 输出：未来 96h OT
    - 水平锚定：l_t = 输入窗口最后 24h OT 均值
    - 标准化：每个变量只用 train 统计量（7 个均值/标准差）
    - alpha：{0.1, 1, 10, 100, 1000}，按 val MAE 选择
    - 对照：
        1) S_control：同样的 per-variable 标准化 + 锚定，仅 OT 输入；
        2) 已有 S 模式 Anchored Ridge（per-lag 标准化）；
        3) B1 / B2a。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\ridge_ms_s96_h96.py

输出：
    outputs/ridge_ms/alpha_selection.csv
    outputs/ridge_ms/test_metrics.csv
    outputs/ridge_ms/per_step_metrics.csv
    outputs/ridge_ms/segment_metrics.csv
    outputs/ridge_ms/channel_importance.csv
    outputs/ridge_ms/summary.json
    outputs/ridge_ms/figures/*.png
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

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
RIDGE_DIR = ROOT / "outputs" / "ridge"
RIDGE_LEVEL_DIR = ROOT / "outputs" / "ridge_level"
OUT_DIR = ROOT / "outputs" / "ridge_ms"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"


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


def make_data(z, split: str, df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    arr = df[cols].to_numpy(dtype=float)
    X = np.stack([arr[x : x + SEQ_LEN] for x in xs])  # (n, 96, C)
    y_all = df[TARGET].to_numpy(dtype=float)
    Y = np.stack([y_all[a:b] for a, b in zip(ys, ye)])  # (n, 96)
    level = X[:, -LEVEL_WINDOW:, -1].mean(axis=1)  # last channel is OT in both S and MS
    return X, Y, level


def run_variant(name: str, cols: list[str], data: dict) -> dict:
    Xtr, Ytr, ltr = data["train"]
    Xva, Yva, lva = data["val"]
    Xte, Yte, lte = data["test"]

    def anchor(X, level):
        Z = X.copy()
        Z[:, :, -1] -= level[:, None]
        return Z

    Xtr_a = anchor(Xtr, ltr)
    Xva_a = anchor(Xva, lva)
    Xte_a = anchor(Xte, lte)

    mu = Xtr_a.mean(axis=(0, 1))
    sd = Xtr_a.std(axis=(0, 1)) + 1e-8

    def flatten(X):
        return ((X - mu[None, None, :]) / sd[None, None, :]).reshape(len(X), -1)

    Xtr_m = flatten(Xtr_a)
    Xva_m = flatten(Xva_a)
    Xte_m = flatten(Xte_a)

    Ytr_p = Ytr - ltr[:, None]
    mu_y = float(Ytr_p.mean())
    sd_y = float(Ytr_p.std()) + 1e-8
    Ytr_m = (Ytr_p - mu_y) / sd_y

    rows: list[dict] = []
    models: dict[float, Ridge] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_m, Ytr_m)
        models[alpha] = model
        for split, Xm, Y, level in [("val", Xva_m, Yva, lva), ("test", Xte_m, Yte, lte)]:
            pred = model.predict(Xm) * sd_y + mu_y + level[:, None]
            rows.append({"variant": name, "alpha": alpha, "split": split, **metrics(pred, Y)})
    alpha_df = pd.DataFrame(rows)

    val_df = alpha_df[alpha_df["split"] == "val"].sort_values("mae")
    best_alpha = float(val_df.iloc[0]["alpha"])
    model = models[best_alpha]
    pred_val = model.predict(Xva_m) * sd_y + mu_y + lva[:, None]
    pred_test = model.predict(Xte_m) * sd_y + mu_y + lte[:, None]

    return {
        "name": name,
        "cols": cols,
        "alpha_df": alpha_df,
        "best_alpha": best_alpha,
        "model": model,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "coef": model.coef_,
        "n_features": int(Xtr_m.shape[1]),
        "n_channels": len(cols),
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
    z = np.load(SPLIT_DIR / f"windows_L{SEQ_LEN}.npz")

    data_ms = {
        "train": make_data(z, "train", df, FEATURES),
        "val": make_data(z, "val", df, FEATURES),
        "test": make_data(z, "test", df, FEATURES),
    }
    data_s = {
        "train": make_data(z, "train", df, [TARGET]),
        "val": make_data(z, "val", df, [TARGET]),
        "test": make_data(z, "test", df, [TARGET]),
    }

    s_ctrl = run_variant("S_control", [TARGET], data_s)
    ms = run_variant("MS", FEATURES, data_ms)

    alpha_df = pd.concat([s_ctrl["alpha_df"], ms["alpha_df"]], ignore_index=True)
    alpha_df.to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    # ---------- 历史结果 ----------
    ridge_plain = pd.read_csv(RIDGE_DIR / "test_metrics.csv")
    ridge_anch = pd.read_csv(RIDGE_LEVEL_DIR / "test_metrics.csv")

    def pick(df_, model):
        return df_[df_["model"] == model].iloc[0].to_dict()

    s_existing = pick(ridge_anch, "Anchored Ridge")
    b1 = pick(ridge_plain, "B1_persistence")
    b2a = pick(ridge_plain, "B2a_local_mean_24")

    test_rows: list[dict] = [
        {"model": "B1_persistence", "mse": b1["mse"], "mae": b1["mae"], "bias": b1["bias"]},
        {"model": "B2a_local_mean_24", "mse": b2a["mse"], "mae": b2a["mae"], "bias": b2a["bias"]},
        {"model": "S_existing (per-lag)", "mse": s_existing["mse"], "mae": s_existing["mae"], "bias": s_existing["bias"]},
        {"model": "S_control (per-variable)", **metrics(s_ctrl["pred_test"], data_ms["test"][1])},
        {"model": "MS (7 channels)", **metrics(ms["pred_test"], data_ms["test"][1])},
    ]
    test_df = pd.DataFrame(test_rows)
    for ref in ["S_existing (per-lag)", "S_control (per-variable)", "B2a_local_mean_24"]:
        ref_row = test_df[test_df["model"] == ref].iloc[0]
        test_df[f"mse_skill_vs_{ref}"] = 1 - test_df["mse"] / ref_row["mse"]
        test_df[f"mae_skill_vs_{ref}"] = 1 - test_df["mae"] / ref_row["mae"]
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    Yte = data_ms["test"][1]
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in [
        ("S_control", s_ctrl["pred_test"]),
        ("MS", ms["pred_test"]),
    ]:
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    # B2a on the same windows
    xs = z[f"test_{HORIZON}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    ot = df[TARGET].to_numpy(dtype=float)
    b2a_pred = np.repeat(np.array([ot[o - 23 : o + 1].mean() for o in origins])[:, None], HORIZON, axis=1)
    err = b2a_pred - Yte
    per_step["B2a_mse"] = np.mean(err**2, axis=0)
    per_step["B2a_mae"] = np.mean(np.abs(err), axis=0)
    per_step["B2a_bias"] = np.mean(err, axis=0)
    per_step["MS_minus_S_mae"] = per_step["MS_mae"] - per_step["S_control_mae"]
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in ["S_control", "MS", "B2a"]:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        row["MS_mse_skill_vs_S"] = 1 - row["MS_mse"] / row["S_control_mse"]
        row["MS_mae_skill_vs_S"] = 1 - row["MS_mae"] / row["S_control_mae"]
        row["MS_mse_skill_vs_B2a"] = 1 - row["MS_mse"] / row["B2a_mse"]
        row["MS_mae_skill_vs_B2a"] = 1 - row["MS_mae"] / row["B2a_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口胜负 ----------
    def wmae(pred):
        return np.mean(np.abs(pred - Yte), axis=1)

    def wmse(pred):
        return np.mean((pred - Yte) ** 2, axis=1)

    win_stats = {
        "n_windows": int(len(Yte)),
        "ms_vs_s_mae": block_bootstrap_ci(wmae(ms["pred_test"]) - wmae(s_ctrl["pred_test"])),
        "ms_vs_s_mse": block_bootstrap_ci(wmse(ms["pred_test"]) - wmse(s_ctrl["pred_test"])),
        "ms_vs_b2a_mae": block_bootstrap_ci(wmae(ms["pred_test"]) - wmae(b2a_pred)),
        "ms_vs_b2a_mse": block_bootstrap_ci(wmse(ms["pred_test"]) - wmse(b2a_pred)),
        "ms_win_rate_vs_s_pct": float(np.mean(wmae(ms["pred_test"]) < wmae(s_ctrl["pred_test"])) * 100),
        "ms_win_rate_vs_b2a_pct": float(np.mean(wmae(ms["pred_test"]) < wmae(b2a_pred)) * 100),
    }

    # ---------- 通道重要性 ----------
    coef = ms["coef"]  # (96, 96*7)
    coef3 = coef.reshape(HORIZON, SEQ_LEN, len(FEATURES))
    importance = np.abs(coef3).mean(axis=(0, 1))
    imp_df = pd.DataFrame({"channel": FEATURES, "mean_abs_coef": importance})
    imp_df["is_target"] = [c == TARGET for c in FEATURES]
    imp_df.to_csv(OUT_DIR / "channel_importance.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for variant, color in [("S_control", "#5b2c6f"), ("MS", "#c0392b")]:
        d = alpha_df[alpha_df["variant"] == variant]
        for split, ls in [("val", "-"), ("test", "--")]:
            dd = d[d["split"] == split].sort_values("alpha")
            axes[0].plot(dd["alpha"], dd["mae"], marker="o", ms=3, ls=ls, color=color, label=f"{variant}-{split}")
            axes[1].plot(dd["alpha"], dd["mse"], marker="o", ms=3, ls=ls, color=color, label=f"{variant}-{split}")
    for ax, ylabel in zip(axes, ["MAE (C)", "MSE (C^2)"]):
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6)
    axes[0].set_title("Alpha selection (val = solid, test = dashed)")
    axes[1].set_title("Alpha selection: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_alpha_selection.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    labels = ["B1", "B2a", "S_existing", "S_control", "MS"]
    model_names = ["B1_persistence", "B2a_local_mean_24", "S_existing (per-lag)", "S_control (per-variable)", "MS (7 channels)"]
    mae_vals = [test_df.loc[test_df["model"] == m, "mae"].iloc[0] for m in model_names]
    mse_vals = [test_df.loc[test_df["model"] == m, "mse"].iloc[0] for m in model_names]
    axes[0].bar(labels, mae_vals, color=["#7f8c8d", "#2980b9", "#5b2c6f", "#8e44ad", "#c0392b"])
    axes[0].set_ylabel("test MAE (C)")
    axes[0].set_title("Test MAE: S vs MS")
    axes[1].bar(labels, mse_vals, color=["#7f8c8d", "#2980b9", "#5b2c6f", "#8e44ad", "#c0392b"])
    axes[1].set_ylabel("test MSE (C^2)")
    axes[1].set_title("Test MSE: S vs MS")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_test_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("MS", "#c0392b"), ("S_control", "#8e44ad"), ("B2a", "#2980b9")]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.2, color=color, label=name)
    for a, _ in SEGMENTS[1:]:
        ax.axvline(a - 0.5, color="k", ls=":", lw=0.6)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("MAE (C)")
    ax.set_title("Per-step MAE on test: MS vs S control vs B2a")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in [("MS", "#c0392b"), ("S_control", "#8e44ad"), ("B2a", "#2980b9")]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.2, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("bias (C)")
    ax.set_title("Per-step bias on test: MS vs S control vs B2a")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    axes[0].bar(x - width / 2, seg_df["MS_mae_skill_vs_S"], width, label="MAE skill", color="#c0392b")
    axes[0].bar(x + width / 2, seg_df["MS_mse_skill_vs_S"], width, label="MSE skill", color="#8e44ad")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, [s["segment"] for s in seg_rows])
    axes[0].set_ylabel("skill vs S_control")
    axes[0].set_title("MS improvement over S_control by segment")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - width / 2, seg_df["MS_mae_skill_vs_B2a"], width, label="MAE skill", color="#c0392b")
    axes[1].bar(x + width / 2, seg_df["MS_mse_skill_vs_B2a"], width, label="MSE skill", color="#8e44ad")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x, [s["segment"] for s in seg_rows])
    axes[1].set_ylabel("skill vs B2a")
    axes[1].set_title("MS improvement over B2a by segment")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_segment_improvement.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, ms["pred_test"][i], color="#c0392b", lw=1.4, label="MS Ridge")
    ax.plot(dates, s_ctrl["pred_test"][i], color="#8e44ad", lw=1.0, ls="--", label="S control")
    ax.plot(dates, b2a_pred[i], color="#2980b9", lw=1.0, ls=":", label="B2a")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if c == TARGET else "#2980b9" for c in imp_df["channel"]]
    ax.bar(imp_df["channel"], imp_df["mean_abs_coef"], color=colors)
    ax.set_ylabel("mean |coef|")
    ax.set_title("MS Ridge channel importance (standardized coefficients)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_channel_importance.png")
    plt.close(fig)

    summary = {
        "protocol": {
            "mode": "MS (7 input channels), target OT",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "anchor": "l_t = mean(last 24h OT)",
            "standardization": "per-channel, train statistics only",
            "alphas": ALPHAS,
            "alpha_selection": "val MAE",
            "primary_control": "S_control (per-variable scaler + anchor)",
            "secondary_control": "existing S anchored Ridge (per-lag scaler)",
        },
        "best_alpha": {"S_control": s_ctrl["best_alpha"], "MS": ms["best_alpha"]},
        "n_features": {"S_control": s_ctrl["n_features"], "MS": ms["n_features"]},
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
        "channel_importance": imp_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("best alpha:", summary["best_alpha"])
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\nsegments:")
    print(seg_df.round(4).to_string(index=False))
    print("\nwin stats:")
    print(json.dumps(win_stats, ensure_ascii=False, indent=2))
    print("\nchannel importance:")
    print(imp_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
