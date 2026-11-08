#!/usr/bin/env python
r"""ETTh1 探索性数据分析（只读，不修改原始 CSV）。

用法（必须用 ml 环境）：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\eda_etth1.py

输出：
    outputs/eda/summary.json
    outputs/eda/report.md
    outputs/eda/*.csv
    outputs/eda/figures/*.png
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
OUT_DIR = ROOT / "outputs" / "eda"
FIG_DIR = OUT_DIR / "figures"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"
COLS = FEATURES + [TARGET]
SPLITS = {"train": (0, 8640), "val": (8640, 11520), "test": (11520, 14400), "unused": (14400, 17420)}
HORIZONS = [1, 2, 6, 12, 24, 48, 96, 168, 336, 720]
SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = float(np.dot(x, x))
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = float(np.dot(x[:-k], x[k:]) / den)
    return out


def pacf_from_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    r = acf(x, nlags)
    pac = np.zeros(nlags + 1)
    pac[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    if nlags >= 1:
        phi[1, 1] = r[1]
        pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - float(np.sum(phi[k - 1, 1:k] * r[k - 1:0:-1]))
        den = 1.0 - float(np.sum(phi[k - 1, 1:k] * r[1:k]))
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pac[k] = phi[k, k]
    return pac


def cross_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> dict[int, float]:
    n = len(a)
    out: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        out[lag] = float(np.corrcoef(x, y)[0, 1])
    return out


def eta_sq(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    ss_tot = float(((values - grand) ** 2).sum())
    ss_bet = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_bet += float(m.sum()) * (values[m].mean() - grand) ** 2
    return ss_bet / ss_tot if ss_tot > 0 else 0.0


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    train = df.iloc[SPLITS["train"][0] : SPLITS["train"][1]].copy()

    summary: dict = {
        "meta": {
            "rows": int(len(df)),
            "start": str(df["date"].iloc[0]),
            "end": str(df["date"].iloc[-1]),
            "target": TARGET,
            "features": FEATURES,
            "split_sizes": {k: b - a for k, (a, b) in SPLITS.items()},
        }
    }

    # ---------- 1. 划分统计 ----------
    split_stats: dict = {}
    for name, (a, b) in SPLITS.items():
        sub = df.iloc[a:b]
        split_stats[name] = {
            "rows": int(len(sub)),
            "start": str(sub["date"].iloc[0]),
            "end": str(sub["date"].iloc[-1]),
            "OT_mean": float(sub[TARGET].mean()),
            "OT_std": float(sub[TARGET].std()),
            "OT_min": float(sub[TARGET].min()),
            "OT_q05": float(sub[TARGET].quantile(0.05)),
            "OT_q50": float(sub[TARGET].median()),
            "OT_q95": float(sub[TARGET].quantile(0.95)),
            "OT_max": float(sub[TARGET].max()),
            "OT_months": {
                str(k): int(v)
                for k, v in sub["date"].dt.strftime("%Y-%m").value_counts().sort_index().items()
            },
        }
    summary["split_stats"] = split_stats

    month_year = df.groupby([df["date"].dt.year, df["date"].dt.month])[TARGET].mean()
    summary["ot_month_year_mean"] = {
        f"{int(y)}-{int(m):02d}": float(v) for (y, m), v in month_year.items()
    }

    # ---------- 2. 季节性强度（训练段 eta^2） ----------
    hours = train["date"].dt.hour.to_numpy()
    dows = train["date"].dt.day_of_week.to_numpy()
    months = train["date"].dt.month.to_numpy()
    seasonal_strength: dict = {}
    for c in COLS:
        v = train[c].to_numpy(dtype=float)
        seasonal_strength[c] = {
            "hour_eta2": eta_sq(v, hours),
            "weekday_eta2": eta_sq(v, dows),
            "month_eta2": eta_sq(v, months),
        }
    summary["seasonal_strength"] = seasonal_strength

    # ---------- 3. 日历分解（月 + 小时 + 星期，训练段估计） ----------
    s_train = train.set_index("date")[TARGET].astype(float)
    month_eff = s_train.groupby(s_train.index.month).mean()
    r_month = s_train - s_train.index.month.map(month_eff)
    hour_eff = r_month.groupby(s_train.index.hour).mean()
    r_hour = r_month - s_train.index.hour.map(hour_eff)
    dow_eff = r_hour.groupby(s_train.index.day_of_week).mean()
    r_dow = r_hour - s_train.index.day_of_week.map(dow_eff)

    var_s = float(s_train.var())
    decomposition = {
        "r2_month": float(1 - r_month.var() / var_s),
        "r2_hour_given_month": float(1 - r_hour.var() / r_month.var()),
        "r2_weekday_given_prev": float(1 - r_dow.var() / r_hour.var()),
        "residual_std": float(r_dow.std()),
        "month_effect": {str(k): float(v) for k, v in month_eff.items()},
        "hour_effect": {str(k): float(v) for k, v in hour_eff.items()},
        "weekday_effect": {str(k): float(v) for k, v in dow_eff.items()},
        "note": "训练段估计的 sequential R2；月=年周期，小时=日周期，星期=周周期",
    }
    summary["decomposition"] = decomposition

    decomposition_by_col: dict = {}
    for c in COLS:
        v = train.set_index("date")[c].astype(float)
        me = v.groupby(v.index.month).mean()
        r1 = v - v.index.month.map(me)
        he = r1.groupby(v.index.hour).mean()
        r2 = r1 - v.index.hour.map(he)
        de = r2.groupby(v.index.day_of_week).mean()
        r3 = r2 - v.index.day_of_week.map(de)
        vt = float(v.var())
        decomposition_by_col[c] = {
            "r2_month": float(1 - r1.var() / vt),
            "r2_hour_given_month": float(1 - r2.var() / r1.var()),
            "r2_weekday_given_prev": float(1 - r3.var() / r2.var()),
            "residual_std": float(r3.std()),
        }
    summary["decomposition_by_col"] = decomposition_by_col

    # ---------- 4. 自相关 / 偏自相关 ----------
    ot = df[TARGET].to_numpy(dtype=float)
    ot_train = train[TARGET].to_numpy(dtype=float)
    max_lag = 336
    acf_ot = acf(ot, max_lag)
    pacf_ot = pacf_from_acf(ot, 72)
    acf_table = {c: acf(df[c].to_numpy(dtype=float), max_lag) for c in COLS}
    d_ot = np.diff(ot_train)
    acf_d = acf(d_ot, 48)

    def first_below(arr: np.ndarray, thr: float) -> int | None:
        idx = np.where(np.abs(arr[1:]) < thr)[0]
        return int(idx[0] + 1) if len(idx) else None

    summary["autocorrelation"] = {
        "ot_acf": {str(k): float(acf_ot[k]) for k in [1, 2, 24, 48, 168, 336]},
        "ot_first_lag_abs_below_0_5": first_below(acf_ot, 0.5),
        "ot_first_negative_lag": int(np.where(acf_ot < 0)[0][0]) if (acf_ot < 0).any() else None,
        "ot_pacf_abs_gt_0_1": [int(k) for k in range(1, len(pacf_ot)) if abs(pacf_ot[k]) > 0.1],
        "diff_ot_acf_lag1": float(acf_d[1]),
        "diff_ot_acf_lag24": float(acf_d[24]),
        "acf_by_col": {
            c: {str(k): float(acf_table[c][k]) for k in [1, 24, 168, 336]} for c in COLS
        },
    }

    # ---------- 5. 负荷与 OT 的领先/滞后相关 ----------
    train_ot = train[TARGET].to_numpy(dtype=float)
    cross: dict = {}
    for c in FEATURES:
        cc = cross_corr(train[c].to_numpy(dtype=float), train_ot, 72)
        best_lag = max(cc, key=lambda k: abs(cc[k]))
        cross[c] = {
            "r_at_lag0": cc[0],
            "best_lag": int(best_lag),
            "r_at_best_lag": cc[best_lag],
            "r_at_plus24": cc[24],
            "r_at_minus24": cc[-24],
        }
    summary["cross_correlation"] = cross

    # ---------- 5b. 90 天滚动相关（稳定性） ----------
    roll_corr_stats: dict = {}
    roll_corr_series: dict[str, pd.Series] = {}
    for c in FEATURES:
        rc = df[c].rolling(24 * 90, min_periods=24 * 30).corr(df[TARGET])
        roll_corr_series[c] = rc
        sgn = np.sign(rc.dropna())
        roll_corr_stats[c] = {
            "min": float(rc.min()),
            "max": float(rc.max()),
            "std": float(rc.std()),
            "sign_changes": int((sgn.diff().abs() > 0).sum()),
        }
    summary["rolling_correlation_90d"] = roll_corr_stats

    # ---------- 5c. 伪影敏感性 ----------
    const_days = {
        d
        for d, g in df.groupby(df["date"].dt.normalize())
        if len(g) == 24 and bool((g[COLS].nunique() == 1).all())
    }
    bad = df["date"].dt.normalize().isin(const_days)
    bad |= (df["date"] >= "2016-12-05 07:00:00") & (df["date"] <= "2016-12-08 07:00:00")
    bad |= (df["date"] >= "2017-07-23 21:00:00") & (df["date"] <= "2017-07-29 17:00:00")
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[SPLITS["train"][0] : SPLITS["train"][1]] = True
    clean = df.loc[train_mask & ~bad.to_numpy()].set_index("date")
    clean_train = train.set_index("date").loc[clean.index]
    me_clean = clean_train[TARGET].groupby(clean_train.index.month).mean()
    r_month_clean = clean_train[TARGET] - clean_train.index.month.map(me_clean)
    summary["sensitivity"] = {
        "excluded_rows_in_train": int((train_mask & bad.to_numpy()).sum()),
        "OT_mean_all": float(train[TARGET].mean()),
        "OT_mean_clean": float(clean_train[TARGET].mean()),
        "OT_std_all": float(train[TARGET].std()),
        "OT_std_clean": float(clean_train[TARGET].std()),
        "r2_month_all": float(decomposition["r2_month"]),
        "r2_month_clean": float(1 - r_month_clean.var() / clean_train[TARGET].var()),
        "HULL_OT_r_all": float(train["HULL"].corr(train[TARGET])),
        "HULL_OT_r_clean": float(clean_train["HULL"].corr(clean_train[TARGET])),
        "MULL_OT_r_all": float(train["MULL"].corr(train[TARGET])),
        "MULL_OT_r_clean": float(clean_train["MULL"].corr(clean_train[TARGET])),
    }

    # ---------- 6. 互信息与 PCA ----------
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train), size=min(4000, len(train)), replace=False)
    Xtr = train[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(train[FEATURES].to_numpy(dtype=float))
    mi = mutual_info_regression(
        Z[idx], train[TARGET].to_numpy(dtype=float)[idx], random_state=0
    )
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    summary["mutual_info_with_OT"] = {c: float(v) for c, v in zip(FEATURES, mi)}
    summary["pca"] = {
        "explained_variance_ratio": [float(v) for v in evr],
        "cumulative": [float(v) for v in cum],
        "n_components_90": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_95": int(np.searchsorted(cum, 0.95) + 1),
        "n_components_99": int(np.searchsorted(cum, 0.99) + 1),
        "loadings": {
            f"PC{i + 1}": {c: float(pca.components_[i, j]) for j, c in enumerate(FEATURES)}
            for i in range(len(FEATURES))
        },
    }

    # ---------- 7. 预测难度：naive 基线 ----------
    y = ot
    test_start, test_end = SPLITS["test"]
    train_mean = float(train[TARGET].mean())
    baseline_rows: list[dict] = []
    for h in HORIZONS:
        origins = np.arange(test_start, test_end - h)
        yt = y[origins + h]

        def seasonal_offset(horizon: int, period: int) -> int:
            return horizon - int(np.ceil(horizon / period)) * period

        preds = {
            "mean": np.full(len(yt), train_mean),
            "persistence": y[origins],
            "seasonal_24": y[origins + seasonal_offset(h, 24)],
            "seasonal_168": y[origins + seasonal_offset(h, 168)],
        }
        mse_mean = float(np.mean((yt - train_mean) ** 2))
        for name, p in preds.items():
            mse = float(np.mean((yt - p) ** 2))
            mae = float(np.mean(np.abs(yt - p)))
            baseline_rows.append(
                {
                    "horizon": int(h),
                    "model": name,
                    "n": int(len(yt)),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "mae": mae,
                    "skill_vs_mean": float(1 - mse / mse_mean),
                }
            )
    summary["baselines"] = baseline_rows

    # ---------- 8. 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    colors = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c", "unused": "#95a5a6"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(df["date"], df[TARGET], lw=0.5, color="#c0392b")
    for name, (a, b) in SPLITS.items():
        axes[0].axvspan(df["date"].iloc[a], df["date"].iloc[b - 1], color=colors[name], alpha=0.12, label=name)
    axes[0].set_ylabel("OT (C)")
    axes[0].set_title("OT full series with standard 12/4/4 splits")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    monthly = df.groupby(df["date"].dt.to_period("M"))[TARGET].agg(["mean", "std", "count"])
    mx = monthly.index.to_timestamp()
    axes[1].errorbar(mx, monthly["mean"], yerr=monthly["std"], fmt="o-", ms=3, lw=0.8, capsize=2, color="#2c3e50")
    axes[1].set_ylabel("OT monthly mean +/- std")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_ot_overview.png")
    plt.close(fig)

    train["hour"] = train["date"].dt.hour
    train["dow"] = train["date"].dt.day_of_week
    train["season"] = train["date"].dt.month.map(SEASON)
    prof = train.groupby(["season", "hour"])[TARGET].mean().unstack(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    order = ["DJF", "MAM", "JJA", "SON"]
    for season in order:
        if season in prof.columns:
            axes[0].plot(prof.index, prof[season], marker="o", ms=2.5, label=season)
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean OT (C)")
    axes[0].set_title("OT diurnal profile by season (train)")
    axes[0].legend()
    dowprof = train.groupby("dow")[TARGET].agg(["mean", "std"])
    axes[1].bar(dowprof.index, dowprof["mean"], yerr=dowprof["std"], capsize=3, color="#3498db", alpha=0.85)
    axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_ylabel("mean OT (C)")
    axes[1].set_title("OT weekday profile (train, +/- std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_ot_profiles.png")
    plt.close(fig)

    dates = df["date"]
    month_comp = dates.dt.month.map(month_eff)
    hour_comp = dates.dt.hour.map(hour_eff)
    dow_comp = dates.dt.day_of_week.map(dow_eff)
    fitted = month_comp + hour_comp + dow_comp
    resid_full = df[TARGET] - fitted
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(dates, df[TARGET], lw=0.35, color="#c0392b")
    axes[0].set_ylabel("observed")
    axes[0].set_title(
        f"OT calendar decomposition (train effects): month R2={decomposition['r2_month']:.3f}, "
        f"+hour R2={decomposition['r2_hour_given_month']:.3f}, "
        f"+weekday R2={decomposition['r2_weekday_given_prev']:.3f}"
    )
    axes[1].plot(dates, month_comp, lw=0.8, color="#2c3e50")
    axes[1].set_ylabel("month effect")
    axes[2].plot(dates, hour_comp + dow_comp, lw=0.4, color="#16a085")
    axes[2].set_ylabel("hour+weekday")
    axes[3].plot(dates, resid_full, lw=0.3, color="#7f8c8d")
    axes[3].axvline(pd.Timestamp("2017-06-26"), color="k", ls="--", lw=0.7)
    axes[3].set_ylabel("residual")
    axes[3].set_title("residual: note the 2017 summer level shift (train-year seasonal profile applied)")
    axes[3].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_decomposition.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    lags = np.arange(max_lag + 1)
    axes[0].plot(lags, acf_ot, lw=1.8, color="#c0392b", label="OT")
    for c in FEATURES:
        axes[0].plot(lags, acf_table[c], lw=0.7, alpha=0.7, label=c)
    for k in (24, 168):
        axes[0].axvline(k, color="k", ls="--", lw=0.7)
    axes[0].set_xlabel("lag (hours)")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Autocorrelation (train+val+test)")
    axes[0].legend(ncol=2, fontsize=7)
    plags = np.arange(len(pacf_ot))
    axes[1].stem(plags, pacf_ot, basefmt="k-", linefmt="C0-", markerfmt="C0o")
    axes[1].axhline(0, color="k", lw=0.6)
    for k in (24, 48):
        axes[1].axvline(k, color="k", ls="--", lw=0.7)
    axes[1].set_xlabel("lag (hours)")
    axes[1].set_ylabel("PACF")
    axes[1].set_title("OT partial autocorrelation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_acf_pacf.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    lags = np.arange(-72, 73)
    for c in FEATURES:
        cc = cross_corr(train[c].to_numpy(dtype=float), train_ot, 72)
        ax.plot(lags, [cc[k] for k in lags], lw=0.9, label=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (hours): >0 means load leads OT")
    ax.set_ylabel("Pearson r")
    ax.set_title("Cross-correlation with OT (train)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_crosscorr.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, len(evr) + 1), cum, marker="o", color="#8e44ad")
    axes[0].axhline(0.9, color="g", ls="--", lw=0.8)
    axes[0].axhline(0.95, color="orange", ls="--", lw=0.8)
    axes[0].set_xlabel("components")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_title("PCA on 6 standardized load features (train)")
    axes[0].set_ylim(0, 1.02)
    im = axes[1].imshow(pca.components_, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(FEATURES)), FEATURES, rotation=45)
    axes[1].set_yticks(range(len(FEATURES)), [f"PC{i + 1}" for i in range(len(FEATURES))])
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            axes[1].text(j, i, f"{pca.components_[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("PCA loadings")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_pca.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in FEATURES:
        axes[0].plot(prof.index, train.groupby("hour")[c].mean(), marker="o", ms=2, label=c)
        axes[1].plot(range(7), train.groupby("dow")[c].mean(), marker="o", ms=2, label=c)
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean load")
    axes[0].set_title("Load diurnal profile (train)")
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_ylabel("mean load")
    axes[1].set_title("Load weekday profile (train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_load_profiles.png")
    plt.close(fig)

    base = pd.DataFrame(baseline_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model, g in base.groupby("model"):
        axes[0].plot(g["horizon"], g["mse"], marker="o", ms=3, label=model)
        axes[1].plot(g["horizon"], g["mae"], marker="o", ms=3, label=model)
    for ax, ylabel in zip(axes, ["MSE (C^2)", "MAE (C)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("horizon (hours, log)")
        ax.set_ylabel(ylabel + ", log")
        ax.legend(fontsize=7)
    axes[0].set_title("Naive baselines on test: MSE vs horizon")
    axes[1].set_title("Naive baselines on test: MAE vs horizon")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_baselines.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ["train", "val", "test"]:
        a, b = SPLITS[name]
        vals = df[TARGET].iloc[a:b]
        axes[0].hist(vals, bins=60, density=True, alpha=0.45, label=name)
    axes[0].set_xlabel("OT (C)")
    axes[0].set_ylabel("density")
    axes[0].set_title("OT distribution by split")
    axes[0].legend()
    axes[1].boxplot(
        [df[TARGET].iloc[a:b] for a, b in [SPLITS["train"], SPLITS["val"], SPLITS["test"]]],
        tick_labels=["train", "val", "test"],
        showfliers=False,
    )
    axes[1].set_ylabel("OT (C)")
    axes[1].set_title("OT boxplot by split (no outliers)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_split_distribution.png")
    plt.close(fig)

    zdf = df.copy()
    for c in COLS:
        zdf[c] = (zdf[c] - train[c].mean()) / train[c].std()
    monthly_z = zdf.groupby(zdf["date"].dt.to_period("M"))[COLS].mean()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for c in COLS:
        ax.plot(monthly_z.index.to_timestamp(), monthly_z[c], marker="o", ms=2.5, label=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(pd.Timestamp("2016-12-01"), color="r", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp("2017-07-23"), color="darkorange", ls="--", lw=0.8)
    ax.set_ylabel("monthly mean (train z-score)")
    ax.set_title("Monthly regime (dashed = known artifacts: 2016-12 outage, 2017-07 LUFL/LULL freeze)")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_monthly_regimes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for c in FEATURES:
        ax.plot(df["date"], roll_corr_series[c], lw=0.9, label=c)
    ax.axhline(0, color="k", lw=0.6)
    for name in ["val", "test"]:
        ax.axvline(df["date"].iloc[SPLITS[name][0]], color="k", ls="--", lw=0.7)
    ax.set_ylabel("90-day rolling Pearson r")
    ax.set_title("Rolling correlation between loads and OT (90-day window)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_rolling_correlation.png")
    plt.close(fig)

    # ---------- 9. 输出 ----------
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=py), encoding="utf-8"
    )
    base.to_csv(OUT_DIR / "baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seasonal_strength).T.to_csv(OUT_DIR / "seasonal_strength.csv", encoding="utf-8-sig")
    lines = [
        "# ETTh1 EDA summary",
        "",
        f"- rows: {len(df)}",
        f"- OT train mean/std: {split_stats['train']['OT_mean']:.2f} / {split_stats['train']['OT_std']:.2f}",
        f"- OT test mean/std: {split_stats['test']['OT_mean']:.2f} / {split_stats['test']['OT_std']:.2f}",
        "",
        "## Seasonal strength (train, eta^2)",
        "",
        "| variable | hour | weekday | month |",
        "|---|---:|---:|---:|",
    ]
    for c in COLS:
        ss = seasonal_strength[c]
        lines.append(f"| {c} | {ss['hour_eta2']:.3f} | {ss['weekday_eta2']:.3f} | {ss['month_eta2']:.3f} |")
    lines += [
        "",
        "## Calendar decomposition (train, sequential R2)",
        "",
        "| variable | month | +hour | +weekday | residual std |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in COLS:
        d = decomposition_by_col[c]
        lines.append(
            f"| {c} | {d['r2_month']:.3f} | {d['r2_hour_given_month']:.3f} | "
            f"{d['r2_weekday_given_prev']:.3f} | {d['residual_std']:.3f} |"
        )
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "split_stats": split_stats,
                "seasonal_strength": seasonal_strength,
                "decomposition": decomposition,
                "pca": summary["pca"]["n_components_95"],
                "baselines_h96": [r for r in baseline_rows if r["horizon"] == 96],
            },
            ensure_ascii=True,
            indent=1,
            default=py,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
