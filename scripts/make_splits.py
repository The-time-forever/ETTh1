#!/usr/bin/env python
r"""构建 ETTh1 官方 12/4/4 划分与窗口索引（只读原始 CSV）。

官方协议（Autoformer / Time-Series-Library 的 Dataset_ETT_hour）：
    border1s = [0, 12*30*24 - L, 12*30*24 + 4*30*24 - L]
    border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
其中 L = seq_len。注意：
  - 月份按 30 天计，不是日历月；
  - val/test 的输入窗口包含前一段末尾的 L 行 lookback；
  - 目标块本身仍是 4 个月（2,880 行）。

用法（必须用 ml 环境）：
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\make_splits.py
    & "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\make_splits.py --seq-lens 96,336 --pred-lens 96,192,336,720

输出：
    outputs/splits/split_manifest.json
    outputs/splits/split_summary.csv
    outputs/splits/row_split_labels.csv
    outputs/splits/windows_L{L}.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ETTh1.csv"
OUT_DIR = ROOT / "outputs" / "splits"

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30
MONTH_HOURS = HOURS_PER_DAY * DAYS_PER_MONTH  # 720
TRAIN_ROWS = 12 * MONTH_HOURS  # 8640
VAL_ROWS = 4 * MONTH_HOURS  # 2880
TEST_ROWS = 4 * MONTH_HOURS  # 2880
USED_ROWS = TRAIN_ROWS + VAL_ROWS + TEST_ROWS  # 14400


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def block_info(df: pd.DataFrame, start: int, end: int) -> dict:
    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "rows": int(end - start),
        "start": str(df["date"].iloc[start]),
        "end": str(df["date"].iloc[end - 1]),
    }


def build_windows(
    border1: int, border2: int, seq_len: int, pred_len: int, label_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = (border2 - border1) - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError(f"no samples: border1={border1}, border2={border2}, L={seq_len}, H={pred_len}")
    x_start = border1 + np.arange(n, dtype=np.int32)
    dec_start = x_start + seq_len - label_len
    y_start = x_start + seq_len
    y_end = y_start + pred_len
    return x_start, dec_start, y_start, y_end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="96", help="comma-separated lookback lengths")
    ap.add_argument("--pred-lens", default="96,192,336,720", help="comma-separated horizons")
    ap.add_argument("--label-len", type=int, default=48)
    args = ap.parse_args()

    seq_lens = parse_ints(args.seq_lens)
    pred_lens = parse_ints(args.pred_lens)
    label_len = args.label_len

    raw = DATA_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if len(df) != 17420:
        raise ValueError(f"unexpected row count: {len(df)}")
    if len(df) < USED_ROWS:
        raise ValueError("dataset shorter than the standard 14,400-row protocol")

    nominal = {
        "train": (0, TRAIN_ROWS),
        "val": (TRAIN_ROWS, TRAIN_ROWS + VAL_ROWS),
        "test": (TRAIN_ROWS + VAL_ROWS, USED_ROWS),
        "unused": (USED_ROWS, len(df)),
    }
    row_blocks = {name: block_info(df, a, b) for name, (a, b) in nominal.items()}
    target_blocks = {
        "val": block_info(df, TRAIN_ROWS, TRAIN_ROWS + VAL_ROWS),
        "test": block_info(df, TRAIN_ROWS + VAL_ROWS, USED_ROWS),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": {
            "path": str(DATA_PATH),
            "sha256": sha,
            "rows": int(len(df)),
            "start": str(df["date"].iloc[0]),
            "end": str(df["date"].iloc[-1]),
        },
        "protocol": {
            "name": "official_ETT_12_4_4",
            "month_days": DAYS_PER_MONTH,
            "train_months": 12,
            "val_months": 4,
            "test_months": 4,
            "label_len": label_len,
            "seq_lens": seq_lens,
            "pred_lens": pred_lens,
        },
        "row_blocks": row_blocks,
        "target_blocks": target_blocks,
        "sample_arrays": {},
        "counts": {},
        "caveats": [
            "月份按 30 天计，不是日历月；切分点是行索引，不对齐月初/月末。",
            "val/test 的输入窗口（border1）包含前一段末尾 L 行 lookback；目标块仍是 2,880 行。",
            "标准协议只使用前 14,400 行；末尾 3,020 行不参与。",
            "test 覆盖 2017-10-24 至 2018-02-20，只有冬季，不能代表全年。",
        ],
    }

    rows: list[dict] = []
    for L in seq_lens:
        borders = {
            "train": (0, TRAIN_ROWS),
            "val": (TRAIN_ROWS - L, TRAIN_ROWS + VAL_ROWS),
            "test": (TRAIN_ROWS + VAL_ROWS - L, USED_ROWS),
        }
        manifest["sample_arrays"][str(L)] = {}
        manifest["counts"][str(L)] = {}
        arrays: dict[str, np.ndarray] = {}
        for split, (b1, b2) in borders.items():
            if split == "train":
                nominal_start = L
            elif split == "val":
                nominal_start = TRAIN_ROWS
            else:
                nominal_start = TRAIN_ROWS + VAL_ROWS
            manifest["counts"][str(L)][split] = {}
            manifest["sample_arrays"][str(L)][split] = {
                "border1": int(b1),
                "border2": int(b2),
                "rows": int(b2 - b1),
                "lookback_start": str(df["date"].iloc[b1]),
                "target_start_idx": int(nominal_start),
                "target_end_idx": int(b2),
                "target_start": str(df["date"].iloc[nominal_start]),
                "target_end": str(df["date"].iloc[b2 - 1]),
            }
            for H in pred_lens:
                x_start, dec_start, y_start, y_end = build_windows(b1, b2, L, H, label_len)
                if int(y_start[0]) != nominal_start:
                    raise AssertionError(f"{split} L={L} H={H}: y_start[0]={y_start[0]} != {nominal_start}")
                if int(y_end[-1]) != b2:
                    raise AssertionError(f"{split} L={L} H={H}: y_end[-1]={y_end[-1]} != {b2}")
                if int(dec_start[0]) < 0:
                    raise AssertionError(f"{split} L={L} H={H}: negative decoder start")
                if not np.all(np.diff(x_start) == 1):
                    raise AssertionError(f"{split} L={L} H={H}: non-contiguous windows")
                manifest["counts"][str(L)][split][str(H)] = int(len(x_start))
                arrays[f"{split}_{H}_x_start"] = x_start
                arrays[f"{split}_{H}_dec_start"] = dec_start
                arrays[f"{split}_{H}_y_start"] = y_start
                arrays[f"{split}_{H}_y_end"] = y_end
                rows.append(
                    {
                        "seq_len": L,
                        "split": split,
                        "pred_len": H,
                        "n_samples": int(len(x_start)),
                        "x_start_first": int(x_start[0]),
                        "x_start_last": int(x_start[-1]),
                        "y_start_first": int(y_start[0]),
                        "y_start_last": int(y_start[-1]),
                        "y_end_last": int(y_end[-1]),
                        "target_start": str(df["date"].iloc[int(y_start[0])]),
                        "target_end": str(df["date"].iloc[int(y_end[-1] - 1)]),
                    }
                )
        np.savez_compressed(OUT_DIR / f"windows_L{L}.npz", **arrays)

    labels = pd.DataFrame(
        {
            "row": np.arange(len(df), dtype=np.int32),
            "date": df["date"].astype(str),
            "nominal_split": np.select(
                [
                    np.arange(len(df)) < TRAIN_ROWS,
                    np.arange(len(df)) < TRAIN_ROWS + VAL_ROWS,
                    np.arange(len(df)) < USED_ROWS,
                ],
                ["train", "val", "test"],
                default="unused",
            ),
            "in_standard_protocol": np.arange(len(df)) < USED_ROWS,
        }
    )

    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(OUT_DIR / "row_split_labels.csv", index=False, encoding="utf-8-sig")

    print(f"source rows: {len(df)}  sha256: {sha}")
    print("\nrow blocks:")
    for name, info in row_blocks.items():
        print(f"  {name:6s} [{info['start_idx']:5d}, {info['end_idx']:5d})  {info['start']}  ->  {info['end']}")
    print("\ntarget blocks (first seq_len used as lookback for train):")
    for name, info in target_blocks.items():
        print(f"  {name:6s} [{info['start_idx']:5d}, {info['end_idx']:5d})  {info['start']}  ->  {info['end']}")
    print("\nsample counts (seq_len=96):")
    table = pd.DataFrame(rows)
    print(table[table["seq_len"] == seq_lens[0]].pivot(index="split", columns="pred_len", values="n_samples").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
