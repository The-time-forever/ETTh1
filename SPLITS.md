# ETTh1 官方 12/4/4 数据划分

> 脚本：`scripts/make_splits.py`（只读原始 CSV）
> 输出：`outputs/splits/split_manifest.json`、`split_summary.csv`、`row_split_labels.csv`、`windows_L96.npz`
> 依据：Autoformer / Time-Series-Library 的 `Dataset_ETT_hour`

## 1. 为什么先做划分

后续的标准化、特征工程、基线、模型选择都依赖划分。先把**行索引、目标块、窗口索引**固定下来，可以避免：

- 用全量数据拟合 scaler 造成泄漏；
- 把 val/test 的 lookback 误当成目标；
- 用日历月代替 30 天月，导致切分点与官方不一致；
- 在 test 上调参。

## 2. 官方公式（30 天/月）

TSLib 的 ETT 小时级划分：

```python
border1s = [0, 12*30*24 - L, 12*30*24 + 4*30*24 - L]
border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
```

其中 `L = seq_len`。代入：

| 划分 | border1 | border2 | 数组行数 | 目标块 |
|---|---:|---:|---:|---|
| train | 0 | 8,640 | 8,640 | `[L, 8640)` |
| val | 8,640 − L | 11,520 | 2,880 + L | `[8640, 11520)` |
| test | 11,520 − L | 14,400 | 2,880 + L | `[11520, 14400)` |

关键点：

- **月份按 30 天计**（720 h），不是日历月；所以切分点是行索引，不对齐月初/月末。
- val/test 的 `border1` 比目标块起点早 `L` 行：这 `L` 行是**输入 lookback**，来自前一段末尾。
- 因此 val/test 的 2,880 行是**目标块**，不是数组长度；数组长度是 2,880 + L。
- train 的前 `L` 行只作为第一个样本的 lookback，不作为目标。

## 3. 行块与日期

| 块 | 行索引 | 行数 | 起 | 止 |
|---|---|---:|---|---|
| train | `[0, 8640)` | 8,640 | 2016-07-01 00:00 | 2017-06-25 23:00 |
| val | `[8640, 11520)` | 2,880 | 2017-06-26 00:00 | 2017-10-23 23:00 |
| test | `[11520, 14400)` | 2,880 | 2017-10-24 00:00 | 2018-02-20 23:00 |
| unused | `[14400, 17420)` | 3,020 | 2018-02-21 00:00 | 2018-06-26 19:00 |

- 标准协议只使用前 14,400 行；末尾 3,020 行（约 4 个月）不参与。
- test 只覆盖秋末到冬末，**没有夏季**；这是评估解释的关键限制（见 `EDA.md`）。

## 4. 窗口索引（`seq_len=96`，`label_len=48`）

每个样本包含：

```text
[ x_start, x_start + L )                 输入窗口
[ dec_start, dec_start + L + H )         解码器输入（前 label_len 已知 + H 待预测）
[ y_start, y_end )                       预测目标，长度 H
```

其中：

- `x_start = border1 + s`
- `dec_start = x_start + L − label_len`
- `y_start = x_start + L`
- `y_end = y_start + H`

`outputs/splits/windows_L96.npz` 保存了每个 `split × pred_len` 的四个索引数组，例如：

```python
import numpy as np
z = np.load("outputs/splits/windows_L96.npz")
z["val_96_x_start"]   # 输入窗口起点
z["val_96_dec_start"] # 解码器输入起点
z["val_96_y_start"]   # 目标起点
z["val_96_y_end"]     # 目标终点（exclusive）
```

验证值（`L=96`）：

| split | 首个 x_start | 首个 dec_start | 首个 y_start | 末个 y_end |
|---|---:|---:|---:|---:|
| train | 0 | 48 | 96 | 8,640 |
| val | 8,544 | 8,592 | 8,640 | 11,520 |
| test | 11,424 | 11,472 | 11,520 | 14,400 |

- val 的 lookback 从 2017-06-22 00:00 开始；test 的 lookback 从 2017-10-20 00:00 开始。

## 5. 样本数

`n_samples = (border2 − border1) − L − H + 1`。

| seq_len | split | H=96 | H=192 | H=336 | H=720 |
|---:|---|---:|---:|---:|---:|
| 96 | train | 8,449 | 8,353 | 8,209 | 7,825 |
| 96 | val | 2,785 | 2,689 | 2,545 | 2,161 |
| 96 | test | 2,785 | 2,689 | 2,545 | 2,161 |

规律：

- **val/test 的样本数与 `L` 无关**：数组长度 = 2,880 + L，代入后 `n = 2881 − H`。
- **train 的样本数随 `L` 增大而减少**：`n = 8640 − L − H + 1`。
- 目标块覆盖 2,880 行，但样本窗口互相重叠，所以样本数是 2,785（H=96）而不是 2,880。

## 6. 使用方式

```python
import json
import numpy as np
import pandas as pd

man = json.load(open("outputs/splits/split_manifest.json", encoding="utf-8"))
df = pd.read_csv("ETTh1.csv", parse_dates=["date"])
z = np.load("outputs/splits/windows_L96.npz")

L, H = 96, 96
i = 0
xs = z[f"val_{H}_x_start"][i]
ys = z[f"val_{H}_y_start"][i]
ye = z[f"val_{H}_y_end"][i]

x = df.iloc[xs : xs + L]          # 输入
y = df.iloc[ys:ye]                # 目标，长度 H
```

- 需要其他 lookback 长度时，重跑脚本：`--seq-lens 336`；**目标块不变，只有 lookback 变化**。
- 标准化只在 train 上 `fit`；val/test 只 `transform`。
- 不要随机打乱或随机划分；样本窗口在时间上重叠。

## 7. 已知限制

- 官方 12/4/4 是**固定切分**，只有一次 train→test 的外推；结论对切分敏感。
- test 是冬季、低方差，且比 train 的冬季更冷；绝对 MSE 会偏小，建议同时报告 `skill vs persistence` 与分月误差。
- 每月 31 日整日冻结、2016-12 停机、2017-07 通道冻结等伪影落在不同划分中（见 `DATA_QUALITY.md`）；建议另做掩码版本。
- 尾部 3,020 行未使用；如果做 rolling-origin 或扩展评估，需要另建协议。

## 8. 复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\make_splits.py"
# 其他 lookback：
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\make_splits.py" --seq-lens 96,336
```
