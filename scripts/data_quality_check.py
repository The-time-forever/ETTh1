#!/usr/bin/env python
r"""ETTh1 数据质量审计（只读，不修改原始 CSV）。

用法（必须用 ml 环境）：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\data_quality_check.py

输出：
    outputs/data_quality/summary.json   全部量化结果
    outputs/data_quality/report.md      自动生成的检查表
    outputs/data_quality/figures/*.png  诊断图
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
OUT_DIR = ROOT / "outputs" / "data_quality"
FIG_DIR = OUT_DIR / "figures"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"
COLS = FEATURES + [TARGET]
SPLITS = {"train": (0, 8640), "val": (8640, 11520), "test": (11520, 14400)}
NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


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


def plateau_stats(values: np.ndarray, tol: float = 1e-9) -> dict:
    """连续相同取值的平台段统计（传感器卡死/量化检测）。"""
    diff = np.abs(np.diff(values))
    eq = diff <= tol
    lengths: list[int] = []
    starts: list[int] = []
    i = 0
    n = len(eq)
    while i < n:
        if eq[i]:
            j = i
            while j < n and eq[j]:
                j += 1
            lengths.append(j - i + 1)
            starts.append(i)
            i = j
        else:
            i += 1
    if not lengths:
        return {"max_plateau_h": 1, "n_plateaus_ge_6h": 0, "longest_start": None}
    k = int(np.argmax(lengths))
    return {
        "max_plateau_h": int(lengths[k]),
        "n_plateaus_ge_6h": int(sum(1 for x in lengths if x >= 6)),
        "longest_start": int(starts[k]),
    }


def negative_segments(values: np.ndarray) -> list[dict]:
    neg = values < 0
    segs: list[dict] = []
    i = 0
    n = len(neg)
    while i < n:
        if neg[i]:
            j = i
            while j < n and neg[j]:
                j += 1
            segs.append({"start": i, "end": j - 1, "n": j - i})
            i = j
        else:
            i += 1
    return segs


def top_jumps(series: pd.Series, dates: pd.Series, k: int = 5) -> list[dict]:
    d = series.diff().abs()
    idx = d.nlargest(k).index
    out = []
    for i in idx:
        out.append(
            {
                "date": str(dates.iloc[i]),
                "from": float(series.iloc[i - 1]),
                "to": float(series.iloc[i]),
                "abs_delta": float(d.iloc[i]),
            }
        )
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_bytes = DATA_PATH.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()

    raw = pd.read_csv(DATA_PATH, dtype=str)
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    summary: dict = {}
    summary["file"] = {
        "path": str(DATA_PATH),
        "bytes": len(raw_bytes),
        "sha256": sha,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    # ---------- 1. 结构与解析 ----------
    summary["schema"] = {c: str(df[c].dtype) for c in df.columns}
    summary["parse"] = {
        "bad_timestamp_format": int((~raw["date"].str.match(TIME_RE)).sum()),
        "unparseable_timestamps": int(df["date"].isna().sum()),
        "bad_numeric_cells": {
            c: int((~raw[c].str.match(NUM_RE)).sum()) for c in COLS
        },
        "nan_cells": {c: int(df[c].isna().sum()) for c in COLS},
        "inf_cells": {
            c: int(np.isinf(df[c].to_numpy(dtype=float)).sum()) for c in COLS
        },
    }

    # ---------- 2. 时间轴 ----------
    ts = df["date"]
    expected = int((ts.max() - ts.min()).total_seconds() // 3600) + 1
    diffs = ts.diff().dt.total_seconds()
    gap_idx = diffs[diffs > 3600]
    summary["time"] = {
        "start": str(ts.min()),
        "end": str(ts.max()),
        "monotonic_increasing": bool(ts.is_monotonic_increasing),
        "unique_timestamps": int(ts.nunique()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "expected_hours": expected,
        "missing_timestamps": int(expected - len(df)),
        "gaps_gt_1h": int(len(gap_idx)),
        "gaps": [
            {"after": str(ts.iloc[i - 1]), "before": str(ts.iloc[i]), "hours": float(gap_idx.loc[i] / 3600)}
            for i in gap_idx.index
        ],
        "day_count": int(ts.dt.date.nunique()),
        "days_with_less_than_24h": int((df.groupby(ts.dt.date).size() != 24).sum()),
        "rows_per_hour_of_day": {
            int(h): int(v) for h, v in ts.dt.hour.value_counts().sort_index().items()
        },
        "rows_per_weekday": {
            int(d): int(v) for d, v in ts.dt.day_of_week.value_counts().sort_index().items()
        },
    }

    # ---------- 3. 分布与异常值 ----------
    desc = df[COLS].describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).T
    dist: dict = {}
    for c in COLS:
        s = df[c].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        med = s.median()
        mad = (s - med).abs().median()
        rz = (s - med).abs() / (1.4826 * mad) if mad > 0 else pd.Series(np.zeros(len(s)))
        dist[c] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p001": float(desc.loc[c, "0.1%"]),
            "p01": float(desc.loc[c, "1%"]),
            "p05": float(desc.loc[c, "5%"]),
            "median": float(med),
            "p95": float(desc.loc[c, "95%"]),
            "p99": float(desc.loc[c, "99%"]),
            "p999": float(desc.loc[c, "99.9%"]),
            "max": float(s.max()),
            "skew": float(stats.skew(s, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(s, bias=False)),
            "zeros": int((s == 0).sum()),
            "negatives": int((s < 0).sum()),
            "negative_pct": float((s < 0).mean() * 100),
            "iqr15_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "iqr30_outliers": int(((s < q1 - 3.0 * iqr) | (s > q3 + 3.0 * iqr)).sum()),
            "robust_z_gt3_5": int((rz > 3.5).sum()),
            "zscore_gt4": int(((s - s.mean()).abs() > 4 * s.std()).sum()),
            "n_unique": int(s.nunique()),
            "plateau": plateau_stats(s.to_numpy()),
        }
    summary["distributions"] = dist

    # ---------- 4. 重复与常量 ----------
    summary["duplicates"] = {
        "exact_rows_all_columns": int(df.duplicated().sum()),
        "exact_feature_vectors": int(df.duplicated(subset=COLS).sum()),
        "constant_columns": [c for c in COLS if df[c].nunique() <= 1],
    }

    # ---------- 5. 负值分段 ----------
    neg: dict = {}
    for c in FEATURES:
        segs = negative_segments(df[c].to_numpy(dtype=float))
        segs_sorted = sorted(segs, key=lambda x: x["n"], reverse=True)
        neg[c] = {
            "negative_points": int((df[c] < 0).sum()),
            "negative_pct": float((df[c] < 0).mean() * 100),
            "n_segments": len(segs),
            "longest_segments": [
                {
                    "start": str(ts.iloc[s["start"]]),
                    "end": str(ts.iloc[s["end"]]),
                    "hours": s["n"],
                }
                for s in segs_sorted[:3]
            ],
        }
    summary["negative_loads"] = neg

    # ---------- 5b. 冻结块与零值块 ----------
    def mask_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
        runs: list[tuple[int, int, int]] = []
        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                runs.append((i, j - 1, j - i))
                i = j
            else:
                i += 1
        return runs

    V = df[COLS].to_numpy(dtype=float)
    same_prev = np.zeros(len(df), dtype=bool)
    same_prev[1:] = np.all(np.abs(np.diff(V, axis=0)) <= 1e-9, axis=1)
    freeze_runs = mask_runs(same_prev)
    freeze_segments = [
        {
            "start": str(ts.iloc[s]),
            "end": str(ts.iloc[e]),
            "hours": int(L),
            "near_month_end": bool(ts.iloc[s].day >= 28),
        }
        for s, e, L in freeze_runs
    ]
    all_constant_days = [
        str(day)
        for day, g in df.groupby(ts.dt.date)
        if len(g) == 24 and bool((g[COLS].nunique() == 1).all())
    ]
    load_zero = (V[:, : len(FEATURES)] == 0).all(axis=1)
    ot_zero_month = (
        df.loc[df[TARGET] == 0, "date"].dt.strftime("%Y-%m").value_counts().sort_index()
    )
    summary["freeze"] = {
        "rows_equal_previous_all7": int(same_prev.sum()),
        "rows_equal_previous_all7_pct": float(same_prev.mean() * 100),
        "segments": freeze_segments,
        "max_segment_h": int(max((s["hours"] for s in freeze_segments), default=0)),
        "all_constant_days": all_constant_days,
        "all_constant_day_rows": int(len(all_constant_days) * 24),
        "all_load_columns_zero": {
            "rows": int(load_zero.sum()),
            "segments": [
                {"start": str(ts.iloc[s]), "end": str(ts.iloc[e]), "hours": int(L)}
                for s, e, L in mask_runs(load_zero)
                if L >= 3
            ],
        },
        "zero_runs_ge_3h": {
            c: [
                {"start": str(ts.iloc[s]), "end": str(ts.iloc[e]), "hours": int(L)}
                for s, e, L in sorted(mask_runs(V[:, k] == 0), key=lambda x: -x[2])[:3]
                if L >= 3
            ]
            for k, c in enumerate(COLS)
        },
        "ot_zero_by_month": {str(k): int(v) for k, v in ot_zero_month.items()},
    }

    # ---------- 6. 跳变 ----------
    jumps: dict = {}
    thresholds = {TARGET: [5.0, 8.0, 10.0], **{c: [5.0, 10.0] for c in FEATURES}}
    for c in COLS:
        d = df[c].diff().abs()
        jumps[c] = {
            "max_abs_delta": float(d.max()),
            "p9999_abs_delta": float(d.quantile(0.9999)),
            "counts_over_threshold": {
                str(t): int((d > t).sum()) for t in thresholds[c]
            },
            "top5": top_jumps(df[c], ts, 5),
        }
    summary["jumps"] = jumps

    # ---------- 7. 相关性与共线性 ----------
    pear = df[COLS].corr(method="pearson")
    spear = df[COLS].corr(method="spearman")
    pairs = [
        {"a": a, "b": b, "pearson": float(pear.loc[a, b]), "spearman": float(spear.loc[a, b])}
        for i, a in enumerate(COLS)
        for b in COLS[i + 1 :]
    ]
    pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)
    X = df[COLS].to_numpy(dtype=float)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    vif: dict = {}
    for i, c in enumerate(COLS):
        y = Xs[:, i]
        Xo = np.delete(Xs, i, axis=1)
        Xo1 = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo1, y, rcond=None)
        resid = y - Xo1 @ beta
        r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        vif[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    summary["correlation"] = {
        "pearson_matrix": {a: {b: float(pear.loc[a, b]) for b in COLS} for a in COLS},
        "spearman_matrix": {a: {b: float(spear.loc[a, b]) for b in COLS} for a in COLS},
        "top_pairs": pairs[:10],
        "pairs_abs_pearson_ge_0_8": [p for p in pairs if abs(p["pearson"]) >= 0.8],
        "condition_number": float(np.linalg.cond(Xs)),
        "vif": vif,
    }

    # ---------- 8. 划分与分布漂移 ----------
    split_info: dict = {}
    for name, (a, b) in SPLITS.items():
        sub = df.iloc[a:b]
        split_info[name] = {
            "rows": int(len(sub)),
            "start": str(sub["date"].iloc[0]),
            "end": str(sub["date"].iloc[-1]),
            "OT": {
                "mean": float(sub[TARGET].mean()),
                "std": float(sub[TARGET].std()),
                "min": float(sub[TARGET].min()),
                "max": float(sub[TARGET].max()),
            },
            "months": {
                str(k): int(v)
                for k, v in sub["date"].dt.strftime("%Y-%m").value_counts().sort_index().items()
            },
        }
    drift: dict = {}
    for c in COLS:
        tr = df[c].iloc[SPLITS["train"][0] : SPLITS["train"][1]]
        va = df[c].iloc[SPLITS["val"][0] : SPLITS["val"][1]]
        te = df[c].iloc[SPLITS["test"][0] : SPLITS["test"][1]]
        ksv = stats.ks_2samp(tr, va)
        kst = stats.ks_2samp(tr, te)
        drift[c] = {
            "ks_train_vs_val": {"stat": float(ksv.statistic), "p": float(ksv.pvalue)},
            "ks_train_vs_test": {"stat": float(kst.statistic), "p": float(kst.pvalue)},
            "mean_train": float(tr.mean()),
            "mean_test": float(te.mean()),
            "test_outside_train_range": int(((te < tr.min()) | (te > tr.max())).sum()),
        }
    key = df[COLS].round(9).astype(str).agg("|".join, axis=1)
    train_keys = set(key.iloc[SPLITS["train"][0] : SPLITS["train"][1]])
    test_keys = set(key.iloc[SPLITS["test"][0] : SPLITS["test"][1]])
    summary["splits"] = {
        "detail": split_info,
        "drift": drift,
        "exact_feature_vectors_shared_train_test": int(len(train_keys & test_keys)),
    }

    # ---------- 9. 质量结论 ----------
    flags: list[dict] = []

    def flag(check: str, ok: bool, detail: str, severity: str = "warn") -> None:
        flags.append({"check": check, "status": "pass" if ok else severity, "detail": detail})

    flag("时间戳可解析", summary["parse"]["unparseable_timestamps"] == 0,
         f"unparseable={summary['parse']['unparseable_timestamps']}", "fail")
    flag("时间轴严格小时级", summary["time"]["gaps_gt_1h"] == 0 and summary["time"]["missing_timestamps"] == 0,
         f"missing={summary['time']['missing_timestamps']}, gaps>1h={summary['time']['gaps_gt_1h']}", "fail")
    flag("时间戳唯一且递增", summary["time"]["duplicate_timestamps"] == 0 and summary["time"]["monotonic_increasing"],
         f"dups={summary['time']['duplicate_timestamps']}, monotonic={summary['time']['monotonic_increasing']}", "fail")
    flag("无缺失值/无穷值", sum(summary["parse"]["nan_cells"].values()) == 0 and sum(summary["parse"]["inf_cells"].values()) == 0,
         f"nan={sum(summary['parse']['nan_cells'].values())}, inf={sum(summary['parse']['inf_cells'].values())}", "fail")
    flag("无数值格式异常", sum(summary["parse"]["bad_numeric_cells"].values()) == 0,
         f"bad_cells={sum(summary['parse']['bad_numeric_cells'].values())}", "fail")
    flag("无整行重复", summary["duplicates"]["exact_rows_all_columns"] == 0,
         f"duplicated_rows={summary['duplicates']['exact_rows_all_columns']}")
    flag("无常量列", len(summary["duplicates"]["constant_columns"]) == 0,
         f"constant={summary['duplicates']['constant_columns']}", "fail")
    flag("负荷列无负值", all(v["negative_points"] == 0 for v in neg.values()),
         "; ".join(f"{c}={v['negative_points']}({v['negative_pct']:.1f}%)" for c, v in neg.items()))
    flag("无长平台/传感器卡死", all(v["plateau"]["n_plateaus_ge_6h"] == 0 for v in dist.values()),
         "; ".join(f"{c}={v['plateau']['n_plateaus_ge_6h']}" for c, v in dist.items()))
    flag("OT 无 >8C/h 的物理跳变", jumps[TARGET]["counts_over_threshold"]["8.0"] == 0,
         f">5C/h={jumps[TARGET]['counts_over_threshold']['5.0']}, >8C/h={jumps[TARGET]['counts_over_threshold']['8.0']}, >10C/h={jumps[TARGET]['counts_over_threshold']['10.0']}")
    n_const_days = len(all_constant_days)
    flag("无整日冻结（每月 31 日）", n_const_days == 0,
         f"all-constant days={n_const_days}, rows={len(all_constant_days) * 24} "
         f"({100 * len(all_constant_days) * 24 / len(df):.2f}%); e.g. {', '.join(all_constant_days[:3])}")
    n_zero_rows = summary["freeze"]["all_load_columns_zero"]["rows"]
    max_zero_seg = max(
        (s["hours"] for s in summary["freeze"]["all_load_columns_zero"]["segments"]),
        default=0,
    )
    flag("负荷通道无零值停机块", n_zero_rows == 0,
         f"all-load-zero rows={n_zero_rows}, max_segment={max_zero_seg}h (2016-12-05)")
    max_zero_run = max(
        (s["hours"] for runs in summary["freeze"]["zero_runs_ge_3h"].values() for s in runs),
        default=0,
    )
    flag("无长时间单通道零值/冻结", max_zero_run < 24,
         f"max single-channel zero run={max_zero_run}h (LULL, 2017-07-23..29)")
    flag("相关性不构成严重共线性",
         summary["correlation"]["condition_number"] < 30 and max(vif.values()) < 10,
         f"cond={summary['correlation']['condition_number']:.1f}, max_VIF={max(vif.values()):.1f}")
    max_ks = max(v["ks_train_vs_test"]["stat"] for v in drift.values())
    flag("训练/测试分布漂移可接受", max_ks < 0.2,
         f"max KS={max_ks:.3f} ({max(drift, key=lambda c: drift[c]['ks_train_vs_test']['stat'])})")
    flag("训练/测试无重复特征向量", summary["splits"]["exact_feature_vectors_shared_train_test"] == 0,
         f"shared={summary['splits']['exact_feature_vectors_shared_train_test']}")
    summary["flags"] = flags

    # ---------- 10. 图 ----------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    split_dates = [df["date"].iloc[SPLITS["train"][1]], df["date"].iloc[SPLITS["val"][1]]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(df["date"], df[TARGET], lw=0.5, color="#c0392b")
    axes[0].set_ylabel("OT (C)")
    axes[0].set_title("ETTh1 overview: OT (top) and loads (bottom)")
    for ax in axes:
        for d in split_dates:
            ax.axvline(d, color="k", ls="--", lw=0.8)
    for c in FEATURES:
        axes[1].plot(df["date"], df[c], lw=0.4, label=c, alpha=0.8)
    axes[1].set_ylabel("Load")
    axes[1].legend(ncol=6, fontsize=7, loc="upper left")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_overview.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    for ax, c in zip(axes.ravel(), COLS):
        ax.hist(df[c], bins=70, color="#2c7fb8", alpha=0.85)
        ax.axvline(df[c].median(), color="k", ls="--", lw=0.8)
        ax.set_title(c, fontsize=9)
        ax.grid(False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distributions (dashed = median)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_distributions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(pear.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(COLS)), COLS, rotation=45)
    ax.set_yticks(range(len(COLS)), COLS)
    for i in range(len(COLS)):
        for j in range(len(COLS)):
            ax.text(j, i, f"{pear.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_correlation.png")
    plt.close(fig)

    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.strftime("%Y-%m")
    neg_month = tmp.groupby("month")[FEATURES].apply(lambda g: (g < 0).mean() * 100)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(neg_month.to_numpy().T, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_xticks(range(len(neg_month.index)), neg_month.index, rotation=90, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% negative")
    ax.set_title("Share of negative values by month")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_negative_by_month.png")
    plt.close(fig)

    tmp["hour"] = tmp["date"].dt.hour
    piv = tmp.pivot_table(index="month", columns="hour", values=TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    ax.set_xticks(range(24), range(24), fontsize=7)
    ax.set_xlabel("hour of day")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean OT (C)")
    ax.set_title("OT seasonal profile: month x hour")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_ot_month_hour.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)
    axes[0].plot(df["date"], df[TARGET].diff().abs(), lw=0.4, color="#7f8c8d")
    axes[0].axhline(8, color="r", ls="--", lw=0.8)
    axes[0].set_ylabel("|dOT| (C/h)")
    axes[0].set_title("Absolute hourly change of OT (red = 8 C/h)")
    axes[1].hist(df[TARGET].diff().dropna(), bins=120, color="#8e44ad", alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("dOT (C/h)")
    axes[1].set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_ot_delta.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(df["date"], df[TARGET], lw=0.5, color="#c0392b", label="OT")
    for s in freeze_segments:
        axes[0].axvspan(pd.Timestamp(s["start"]), pd.Timestamp(s["end"]), color="orange", alpha=0.6, lw=0)
    for s in summary["freeze"]["all_load_columns_zero"]["segments"]:
        axes[0].axvspan(pd.Timestamp(s["start"]), pd.Timestamp(s["end"]), color="red", alpha=0.6, lw=0)
    axes[0].set_ylabel("OT (C)")
    axes[0].set_title("Quality masks: orange = all-7 frozen rows, red = all-load zero block")
    x0, x1 = mdates.date2num(df["date"].iloc[0]), mdates.date2num(df["date"].iloc[-1])
    axes[1].imshow(
        (V == 0).T,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        extent=[x0, x1, len(COLS), 0],
    )
    axes[1].set_yticks(np.arange(len(COLS)) + 0.5, COLS)
    axes[1].set_xlabel("date")
    axes[1].set_title("Exact zeros per column (black = 0)")
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_quality_masks.png")
    plt.close(fig)

    # ---------- 11. 输出 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=py), encoding="utf-8"
    )
    pd.DataFrame(freeze_segments).to_csv(OUT_DIR / "freeze_segments.csv", index=False, encoding="utf-8-sig")
    lines = ["# ETTh1 data quality checks", "", f"- file: `{DATA_PATH}`", f"- sha256: `{sha}`", f"- rows: {len(df)}", "", "| check | status | detail |", "|---|---|---|"]
    for f in flags:
        lines.append(f"| {f['check']} | {f['status']} | {f['detail']} |")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"flags": flags, "sha256": sha, "rows": len(df)}, ensure_ascii=True, indent=2, default=py))
    return 0


if __name__ == "__main__":
    sys.exit(main())
