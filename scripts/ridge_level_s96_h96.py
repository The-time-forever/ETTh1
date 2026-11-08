#!/usr/bin/env python
r"""局部水平锚定的 Ridge：S 模式，96 -> 96，直接多步。

参考水平（平移锚）：
    l_t = mean(y[t-23 .. t])          # 输入窗口最后 24h
    X'  = X - l_t
    Y'  = Y - l_t
    Ridge:  Y' = W X' + b
    predict: Y = Y' + l_t

仍然是线性模型；不增加输入变量；对输入/输出的共同平移等变。

用法：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\ridge_level_s96_h96.py

输出：
    outputs/ridge_level/alpha_selection.csv
    outputs/ridge_level/test_metrics.csv
    outputs/ridge_level/per_step_metrics.csv
    outputs/ridge_level/segment_metrics.csv
    outputs/ridge_level/summary.json
    outputs/ridge_level/figures/*.png
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
OUT_DIR = ROOT / "outputs" / "ridge_level"
FIG_DIR = OUT_DIR / "figures"

SEQ_LEN = 96
HORIZON = 96
LEVEL_WINDOW = 24
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEGMENTS = [(1, 24), (25, 48), (49, 72), (73, 96)]
COLORS = {
    "Anchored Ridge": "#c0392b",
    "Plain Ridge": "#8e44ad",
    "B1_persistence": "#7f8c8d",
    "B2a_local_mean_24": "#2980b9",
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


def make_xy(z, split: str, ot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    ys = z[f"{split}_{HORIZON}_y_start"].astype(int)
    ye = z[f"{split}_{HORIZON}_y_end"].astype(int)
    X = np.stack([ot[x : x + SEQ_LEN] for x in xs])
    Y = np.stack([ot[a:b] for a, b in zip(ys, ye)])
    return X, Y


def level_anchor(X: np.ndarray) -> np.ndarray:
    return X[:, -LEVEL_WINDOW:].mean(axis=1)


def baseline_preds(z, split: str, ot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = z[f"{split}_{HORIZON}_x_start"].astype(int)
    origins = xs + SEQ_LEN - 1
    b1 = np.repeat(ot[origins][:, None], HORIZON, axis=1)
    local24 = np.array([ot[o - 23 : o + 1].mean() for o in origins])
    b2a = np.repeat(local24[:, None], HORIZON, axis=1)
    return b1, b2a


def fit_grid(Xtr, Ytr, Xva, Yva, Xte, Yte, anchored: bool):
    if anchored:
        ltr, lva, lte = level_anchor(Xtr), level_anchor(Xva), level_anchor(Xte)
        Xtr_f = Xtr - ltr[:, None]
        Xva_f = Xva - lva[:, None]
        Xte_f = Xte - lte[:, None]
        Ytr_f = Ytr - ltr[:, None]
    else:
        ltr = lva = lte = np.zeros(len(Xtr))
        Xtr_f, Xva_f, Xte_f, Ytr_f = Xtr, Xva, Xte, Ytr

    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s = scaler.transform(Xtr_f)
    Xva_s = scaler.transform(Xva_f)
    Xte_s = scaler.transform(Xte_f)

    rows: list[dict] = []
    models: dict[float, Ridge] = {}
    variant = "anchored" if anchored else "plain"
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_s, Ytr_f)
        models[alpha] = model
        for split, X, Y, level in [("val", Xva_s, Yva, lva), ("test", Xte_s, Yte, lte)]:
            pred = model.predict(X)
            if anchored:
                pred = pred + level[:, None]
            rows.append({"variant": variant, "alpha": alpha, "split": split, **metrics(pred, Y)})
    return pd.DataFrame(rows), models, scaler, (ltr, lva, lte)


def predict_variant(model, scaler, X, level, anchored: bool) -> np.ndarray:
    Xf = X - level[:, None] if anchored else X
    pred = model.predict(scaler.transform(Xf))
    if anchored:
        pred = pred + level[:, None]
    return pred


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

    Xtr, Ytr = make_xy(z, "train", ot)
    Xva, Yva = make_xy(z, "val", ot)
    Xte, Yte = make_xy(z, "test", ot)

    plain_df, plain_models, plain_scaler, _ = fit_grid(Xtr, Ytr, Xva, Yva, Xte, Yte, anchored=False)
    anch_df, anch_models, anch_scaler, (ltr, lva, lte) = fit_grid(
        Xtr, Ytr, Xva, Yva, Xte, Yte, anchored=True
    )
    alpha_df = pd.concat([plain_df, anch_df], ignore_index=True)
    alpha_df.to_csv(OUT_DIR / "alpha_selection.csv", index=False, encoding="utf-8-sig")

    best_plain = float(plain_df[plain_df["split"] == "val"].sort_values("mae").iloc[0]["alpha"])
    best_anch = float(anch_df[anch_df["split"] == "val"].sort_values("mae").iloc[0]["alpha"])

    plain_te = predict_variant(plain_models[best_plain], plain_scaler, Xte, np.zeros(len(Xte)), False)
    anch_te = predict_variant(anch_models[best_anch], anch_scaler, Xte, lte, True)
    b1_te, b2a_te = baseline_preds(z, "test", ot)
    preds = {
        "Anchored Ridge": anch_te,
        "Plain Ridge": plain_te,
        "B1_persistence": b1_te,
        "B2a_local_mean_24": b2a_te,
    }

    test_rows: list[dict] = []
    for name, pred in preds.items():
        test_rows.append({"model": name, **metrics(pred, Yte)})
    test_df = pd.DataFrame(test_rows)
    ref = {r["model"]: r for r in test_rows}
    for row in test_rows:
        for base in ["B1_persistence", "B2a_local_mean_24", "Plain Ridge"]:
            row[f"mse_skill_vs_{base}"] = 1 - row["mse"] / ref[base]["mse"]
            row[f"mae_skill_vs_{base}"] = 1 - row["mae"] / ref[base]["mae"]
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐点 ----------
    per_step = pd.DataFrame({"step": np.arange(1, HORIZON + 1)})
    for name, pred in preds.items():
        err = pred - Yte
        per_step[f"{name}_mse"] = np.mean(err**2, axis=0)
        per_step[f"{name}_mae"] = np.mean(np.abs(err), axis=0)
        per_step[f"{name}_bias"] = np.mean(err, axis=0)
    per_step["anchored_minus_plain_mae"] = (
        per_step["Anchored Ridge_mae"] - per_step["Plain Ridge_mae"]
    )
    per_step["anchored_minus_b2a_mae"] = (
        per_step["Anchored Ridge_mae"] - per_step["B2a_local_mean_24_mae"]
    )
    per_step.to_csv(OUT_DIR / "per_step_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 分段 ----------
    seg_rows: list[dict] = []
    for a, b in SEGMENTS:
        idx = slice(a - 1, b)
        seg = per_step.iloc[idx]
        row = {"segment": f"{a}-{b}", "steps": b - a + 1}
        for name in preds:
            row[f"{name}_mse"] = float(seg[f"{name}_mse"].mean())
            row[f"{name}_mae"] = float(seg[f"{name}_mae"].mean())
            row[f"{name}_bias"] = float(seg[f"{name}_bias"].mean())
        row["anchored_mse_skill_vs_b2a"] = 1 - row["Anchored Ridge_mse"] / row["B2a_local_mean_24_mse"]
        row["anchored_mae_skill_vs_b2a"] = 1 - row["Anchored Ridge_mae"] / row["B2a_local_mean_24_mae"]
        row["anchored_mse_skill_vs_plain"] = 1 - row["Anchored Ridge_mse"] / row["Plain Ridge_mse"]
        row["anchored_mae_skill_vs_plain"] = 1 - row["Anchored Ridge_mae"] / row["Plain Ridge_mae"]
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- 逐窗口胜负 / bootstrap ----------
    mae_w = {name: np.mean(np.abs(pred - Yte), axis=1) for name, pred in preds.items()}
    mse_w = {name: np.mean((pred - Yte) ** 2, axis=1) for name, pred in preds.items()}
    win_stats = {
        "n_windows": int(len(Yte)),
        "anchored_mae_better_than_b2a_pct": float(np.mean(mae_w["Anchored Ridge"] < mae_w["B2a_local_mean_24"]) * 100),
        "anchored_mae_better_than_plain_pct": float(np.mean(mae_w["Anchored Ridge"] < mae_w["Plain Ridge"]) * 100),
        "anchored_vs_b2a_mae_diff": block_bootstrap_ci(mae_w["Anchored Ridge"] - mae_w["B2a_local_mean_24"]),
        "anchored_vs_b2a_mse_diff": block_bootstrap_ci(mse_w["Anchored Ridge"] - mse_w["B2a_local_mean_24"]),
        "anchored_vs_plain_mae_diff": block_bootstrap_ci(mae_w["Anchored Ridge"] - mae_w["Plain Ridge"]),
        "anchored_vs_plain_mse_diff": block_bootstrap_ci(mse_w["Anchored Ridge"] - mse_w["Plain Ridge"]),
    }

    # ---------- 平移等变性检查 ----------
    eq_rows: list[dict] = []
    X0 = Xte[0]
    l0 = float(level_anchor(X0[None, :])[0])
    pred0 = predict_variant(anch_models[best_anch], anch_scaler, X0[None, :], np.array([l0]), True)[0]
    for c in [-10.0, 10.0]:
        Xs = X0 + c
        ls = float(level_anchor(Xs[None, :])[0])
        preds_c = predict_variant(anch_models[best_anch], anch_scaler, Xs[None, :], np.array([ls]), True)[0]
        eq_rows.append(
            {
                "shift_c": c,
                "max_abs_prediction_shift_error": float(np.max(np.abs((preds_c - pred0) - c))),
                "mean_prediction_shift": float(np.mean(preds_c - pred0)),
            }
        )

    # ---------- 系数诊断 ----------
    coef = anch_models[best_anch].coef_
    coef_raw = coef / anch_scaler.scale_
    sum_coef = coef_raw.sum(axis=1)
    summary = {
        "protocol": {
            "mode": "S (OT only)",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "level_window": LEVEL_WINDOW,
            "anchored": "X' = X - l_t, Y' = Y - l_t, predict = Y' + l_t",
            "extra_features": "none",
            "standardization": "StandardScaler fit on train X' only",
            "alpha_grid": ALPHAS,
            "alpha_selection": "val MAE",
        },
        "best_alpha": {"anchored": best_anch, "plain": best_plain},
        "alpha_table": alpha_df.to_dict(orient="records"),
        "test_metrics": test_df.to_dict(orient="records"),
        "segments": seg_df.to_dict(orient="records"),
        "win_stats": win_stats,
        "equivariance_checks": eq_rows,
        "coef": {
            "sum_coef_by_step": [float(v) for v in sum_coef],
            "level_weight_1_minus_sum": [float(v) for v in (1 - sum_coef)],
            "intercept_by_step": [float(v) for v in anch_models[best_anch].intercept_],
            "mean_abs_by_input_lag": [float(v) for v in np.mean(np.abs(coef_raw), axis=0)],
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for variant, color in [("plain", "#8e44ad"), ("anchored", "#c0392b")]:
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
    axes[0].axvline(best_anch, color="#c0392b", ls=":", lw=0.8)
    axes[0].axvline(best_plain, color="#8e44ad", ls=":", lw=0.8)
    axes[0].set_title("Alpha selection (val = solid, test = dashed)")
    axes[1].set_title("Alpha selection: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_alpha_selection.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name in ["Anchored Ridge", "Plain Ridge", "B2a_local_mean_24", "B1_persistence"]:
        ax.plot(per_step["step"], per_step[f"{name}_mae"], lw=1.2, label=name, color=COLORS[name])
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
    for name in ["Anchored Ridge", "Plain Ridge", "B2a_local_mean_24", "B1_persistence"]:
        ax.plot(per_step["step"], per_step[f"{name}_bias"], lw=1.2, label=name, color=COLORS[name])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forecast step (1..96)")
    ax.set_ylabel("bias (pred - truth, C)")
    ax.set_title("Per-step bias on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_step_bias.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(SEGMENTS))
    width = 0.35
    for k, variant in enumerate(["anchored", "plain"]):
        if variant == "anchored":
            mae_skill = seg_df["anchored_mae_skill_vs_b2a"].tolist()
            mse_skill = seg_df["anchored_mse_skill_vs_b2a"].tolist()
        else:
            mae_skill = (1 - seg_df["Plain Ridge_mae"] / seg_df["B2a_local_mean_24_mae"]).tolist()
            mse_skill = (1 - seg_df["Plain Ridge_mse"] / seg_df["B2a_local_mean_24_mse"]).tolist()
        color = "#c0392b" if variant == "anchored" else "#8e44ad"
        axes[0].bar(x + (k - 0.5) * width, mae_skill, width, label=variant, color=color)
        axes[1].bar(x + (k - 0.5) * width, mse_skill, width, label=variant, color=color)
    for ax, ylabel in zip(axes, ["MAE skill vs B2a", "MSE skill vs B2a"]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [s["segment"] for s in seg_rows])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    axes[0].set_title("Segment improvement over B2a: MAE")
    axes[1].set_title("Segment improvement over B2a: MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_segment_improvement.png")
    plt.close(fig)

    i = 0
    ys = z[f"test_{HORIZON}_y_start"].astype(int)
    ye = z[f"test_{HORIZON}_y_end"].astype(int)
    dates = df["date"].iloc[ys[i] : ye[i]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, Yte[i], color="k", lw=2, label="truth")
    ax.plot(dates, anch_te[i], color=COLORS["Anchored Ridge"], lw=1.4, label="Anchored Ridge")
    ax.plot(dates, plain_te[i], color=COLORS["Plain Ridge"], lw=1.0, ls="--", label="Plain Ridge")
    ax.plot(dates, b2a_te[i], color=COLORS["B2a_local_mean_24"], lw=1.0, ls="--", label="B2a")
    ax.plot(dates, b1_te[i], color=COLORS["B1_persistence"], lw=1.0, ls=":", label="B1")
    ax.set_ylabel("OT (C)")
    ax.set_title("Example test window (first H=96 forecast)")
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_example_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mae_w["Anchored Ridge"] - mae_w["B2a_local_mean_24"], bins=60, alpha=0.75, color="#c0392b", label="Anchored - B2a")
    ax.hist(mae_w["Anchored Ridge"] - mae_w["Plain Ridge"], bins=60, alpha=0.45, color="#8e44ad", label="Anchored - Plain")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-window MAE difference (C)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Win rate: vs B2a {win_stats['anchored_mae_better_than_b2a_pct']:.1f}%, "
        f"vs Plain {win_stats['anchored_mae_better_than_plain_pct']:.1f}%"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_window_error.png")
    plt.close(fig)

    cs = np.arange(-20, 21, 5)
    shifts = []
    for c in cs:
        Xs = X0 + c
        ls = float(level_anchor(Xs[None, :])[0])
        p = predict_variant(anch_models[best_anch], anch_scaler, Xs[None, :], np.array([ls]), True)[0]
        shifts.append(p - pred0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(cs, np.mean(shifts, axis=1), marker="o", color="#c0392b", label="mean prediction shift")
    ax.plot(cs, cs, color="k", ls="--", lw=0.8, label="identity")
    ax.set_xlabel("input window shift c (C)")
    ax.set_ylabel("prediction shift (C)")
    ax.set_title("Translation equivariance check")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_equivariance.png")
    plt.close(fig)

    print(f"best alpha: anchored={best_anch}, plain={best_plain}")
    print("\ntest metrics:")
    print(test_df.round(4).to_string(index=False))
    print("\nsegments:")
    print(seg_df.round(4).to_string(index=False))
    print("\nequivariance:")
    print(pd.DataFrame(eq_rows).round(12).to_string(index=False))
    print("\nwin stats:")
    print(json.dumps(win_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
