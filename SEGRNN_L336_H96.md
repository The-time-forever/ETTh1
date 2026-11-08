# SegRNN（GRU 核心）：S 模式，336 → 96

> 脚本：`scripts/segrnn_l336_h96.py`
> 数据：官方 12/4/4；`seq_len=336`、`pred_len=96`
> Segment Length = 48；历史 7 段；预测 2 段；parallel multi-step (PMF)
> 模型：SegRNN（GRU，d_model=64，参数量 31,344）
> 对照：Anchored Ridge 336→96、PatchTST small L336、B2b

## 1. 变体与官方实现

本轮同时测试两种水平处理：

| 变体 | 处理 | 说明 |
|---|---|---|
| **Anchored SegRNN** | X' = X − ℓ_t，Y' = Y − ℓ_t；ℓ_t = 最近 24h OT 均值 | **我们自己的适配**，便于和 Anchored Ridge 对照 |
| **Last-value SegRNN** | X' = X − y_T，Y' = Y − y_T | 接近 SegRNN 官方默认的 `seq_last` 处理 |

官方 SegRNN 的 PMF 解码器：encoder GRU 处理 7 个 segment，最终 hidden state 复制到 2 个预测 segment，
每个 segment 加位置嵌入后并行过 GRU，再接 `Linear(d_model → 48)`；输出 2×48=96。
本实现按此结构，单通道，`d_model=64`。

训练：MSE、Adam lr=1e-3、wd=1e-4、batch=256、grad clip 1.0、val MAE 早停、3 seeds。

## 2. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE |
|---|---:|---:|---:|---:|
| B2b Local Mean 96 | 5.6172 | 1.8214 | +0.2257 | 5.5604 |
| **Anchored Ridge 336** | **4.4683** | **1.6250** | +0.0547 | **4.4642** |
| Anchored SegRNN（mean） | 5.0052 ± 0.2252 | 1.7878 ± 0.0492 | **−0.5140** | 4.7094 |
| Anchored SegRNN（ensemble） | 4.9503 | 1.7791 | −0.5140 | 4.6797 |
| **Last-value SegRNN（mean）** | 4.6138 ± 0.0901 | 1.6937 ± 0.0207 | −0.4145 | 4.4223 |
| **Last-value SegRNN（ensemble）** | 4.5760 | 1.6867 | −0.4145 | **4.3899** |
| PatchTST small L336（mean） | 4.9958 | 1.7577 | −0.2090 | — |
| PatchTST small L336（ensemble） | 4.6121 | 1.6918 | −0.2090 | — |

相对变化：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Anchored SegRNN（mean） vs Anchored Ridge | −12.0% | −10.0% |
| Last-value SegRNN（mean） vs Anchored Ridge | −3.3% | −4.2% |
| **Last-value vs Anchored SegRNN** | **+7.8%** | **+5.3%** |
| Last-value SegRNN（ensemble） vs PatchTST L336（ensemble） | +0.8% | +0.3% |

配对 bootstrap：

| 对比 | MAE 差 | 95% CI | MSE 差 | 95% CI | 胜率 |
|---|---:|---|---:|---|---:|
| Anchored SegRNN − Anchored Ridge | +0.154 | **[+0.049, +0.249]** | +0.482 | [−0.124, +0.985] | 37.6% |
| Anchored SegRNN − Last-value SegRNN | +0.092 | **[+0.060, +0.120]** | +0.374 | **[+0.203, +0.513]** | 36.8% |
| Last-value SegRNN − Anchored Ridge | +0.062 | [−0.028, +0.144] | +0.108 | [−0.402, +0.543] | 41.8% |

## 3. 关键发现

### 3.1 24h Anchoring 对 SegRNN 是负收益

- Anchored SegRNN：MAE 1.788（mean）/ 1.779（ensemble），MSE 5.005 / 4.950。
- Last-value SegRNN：MAE 1.694 / 1.687，MSE 4.614 / 4.576。
- 配对比较：Last-value 比 Anchored 好 **5.3% MAE / 7.8% MSE**，CI 不跨 0。
- 这与我们的预期相反：把 Anchored Ridge 的 24h 锚定直接搬到 SegRNN 上并不合适。SegRNN 的 segment/GRU 结构更依赖“最后观测值”作为参考；24h 均值把最近水平抹平，模型需要学习更大的修正，反而产生 −0.514 的负偏差。

### 3.2 官方 last-value 处理更好，但仍不敌 Anchored Ridge

- Last-value SegRNN 与 Anchored Ridge 的差异不显著（MAE +0.062，CI 跨 0；MSE +0.108，CI 跨 0），点估计更差，胜率 41.8%。
- 也就是说：**SegRNN 没有超过 Anchored Ridge**，最多打平。
- Last-value SegRNN 与 PatchTST L336 ensemble 基本打平（MAE 1.687 vs 1.692，MSE 4.576 vs 4.612）；两者都差于 Anchored Ridge 1.625。

### 3.3 偏差：SegRNN 系统性低估

| 模型 | bias |
|---|---:|
| Anchored Ridge 336 | +0.055 |
| Anchored SegRNN | **−0.514** |
| Last-value SegRNN | **−0.415** |
| PatchTST L336 | −0.209 |

- 两种 SegRNN 都明显低估；24h anchoring 使低估更严重。
- last-value 参考更接近当前水平，偏差减小，但仍未完全校正。

### 3.4 训练行为

- Anchored SegRNN 最优 epoch 4/4/4；Last-value 6/6/4；val MAE 在 ~2.17–2.27 后很快上升，存在早期过拟合。
- 参数量 31,344，与 PatchTST small L336（70.7k）和 Anchored Ridge（~33k）同一量级。
- 没有出现 PatchTST L336 那种 epoch 1 即早停的极端情况，但也没有训练出更好的长期映射。

## 4. 回答研究问题

**SegRNN（最佳变体）没有超过 Anchored Ridge 336→96。**

- 24h anchoring 是我们的适配，但对 SegRNN 不利；官方 last-value 更好。
- Last-value SegRNN 与 Anchored Ridge 统计上打平、点估计更差；与 PatchTST L336 ensemble 也基本打平。
- 因此，RNN 方向没有提供超过锚定线性模型的证据；如果继续 SegRNN，应使用 last-value 或 RevIN，而不是 24h anchoring。

## 5. 下一步建议

1. **如果继续 SegRNN**：以 last-value 或 RevIN 为默认；可以再试 segment length 24/12 与 d_model=32/128，但预期提升有限。
2. **如果继续 RNN 方向**：优先研究“归一化/参考水平”与循环结构的交互，而不是继续堆层数。
3. **当前最佳仍是 Anchored Ridge 336→96**（4.468/1.625）；SegRNN/PatchTST 都没有超过它。
4. 更高优先级仍是外生变量（天气）、水平/季节建模与不确定性预测。

## 6. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\segrnn_l336_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/segrnn/alpha_selection.csv` | Ridge alpha × val/test |
| `outputs/segrnn/test_metrics.csv` | B2b/Ridge/SegRNN/PatchTST 对比 |
| `outputs/segrnn/paired_deltas.csv` | 配对 MAE/MSE delta 与 CI |
| `outputs/segrnn/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/segrnn/segment_metrics.csv` | 4 段指标 |
| `outputs/segrnn/seed_metrics.csv` | 6 个 seed 的参数量/best epoch/指标 |
| `outputs/segrnn/summary.json` | 协议、test、paired、segment、seed |
| `figures/01_training_curves.png` | Anchored vs Last-value 训练曲线 |
| `02_test_mae.png` | SegRNN 与参考模型 |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias |
| `05_segment_skill.png` | 分段 skill vs Ridge |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 分布 |

## 7. 局限

- 只测了一个 SegRNN 配置（seg_len=48, d_model=64, 1 层 GRU）；其他 segment length/d_model 可能改变数值。
- 未运行官方 RevIN 变体；本轮聚焦 24h anchoring 与 last-value 的对照。
- 只评估 S 模式、H=96、单一切分；未做伪影掩码。
- 24h anchoring 是我们自己的适配，不是 SegRNN 论文的默认设置；结论应表述为“在本任务/本配置下，24h anchoring 不如 last-value”，而不是对 SegRNN 原论文的否定。
