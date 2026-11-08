# GRU：S 模式，96 → 96

> 脚本：`scripts/gru_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 数据：S 模式，仅 OT；无负荷变量、无日历特征
> 训练：PyTorch，3 个 seed；按 val MAE 早停

## 1. 研究问题

前面的路径是：

- Ridge：96 个历史值 → 96 个未来值（线性，直接映射）；
- MLP：同样的映射，加非线性；
- GRU：按时间顺序处理 96 个观测值，用门控隐藏状态学习时序依赖。

本轮只加入“显式时序结构”这一个因素，回答：

> 循环隐藏状态能否比全连接网络更有效地提取油温变化规律？

## 2. 模型与训练

```text
GRU(input=1, hidden=64, num_layers=1) -> last hidden -> Linear(64 -> 96)
```

- 参数量 **19,104**（MLP 128-128 约 41k）；更小的模型，若更好也不是靠容量。
- 输入为 `(batch, 96, 1)`；只用最后时刻的隐藏状态，直接输出 96 步（非递归解码）。
- 输入全局标准化（train），目标标准化（train），预测反变换回 ℃。
- 损失 MSE；Adam `lr=1e-3`、`weight_decay=1e-4`、batch=256、梯度裁剪 1.0。
- 早停：每 epoch 计算 **val MAE（原始 ℃）**，`patience=20`、`max_epochs=300`，恢复最优 epoch。
- Seeds：0/1/2；报告 3-seed mean ± std 与 3-seed ensemble。

两个变体：

| 变体 | 输入 | 输出 | 恢复 |
|---|---|---|---|
| Plain GRU | X | Y | 直接输出 |
| Anchored GRU | X' = X − ℓ_t | Y' = Y − ℓ_t | Ŷ = GRU(X') + ℓ_t |

`ℓ_t` = 输入窗口最后 24 h 均值，与 Anchored Ridge/MLP 完全一致。

## 3. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE |
|---|---:|---:|---:|---:|
| B1 Persistence | 5.8326 | 1.8654 | +0.1215 | 5.8119 |
| B2a Local Mean 24 | 5.4544 | 1.8198 | +0.1443 | 5.4277 |
| Plain Ridge | 5.1067 | 1.6698 | +0.7805 | **4.4000** |
| Plain MLP（3-seed mean） | 12.1981 ± 1.3135 | 2.8527 ± 0.1861 | +2.6776 | 4.4800 |
| **Plain GRU（3-seed mean）** | **17.8776 ± 1.0400** | **3.5318 ± 0.1197** | **+3.3314** | **5.4713** |
| Plain GRU（ensemble） | 17.8173 | 3.5266 | +3.3314 | 5.4283 |
| Anchored Ridge | 4.6127 | 1.6324 | +0.1051 | **4.5981** |
| Anchored MLP（3-seed mean） | 4.6802 ± 0.0403 | 1.6633 ± 0.0062 | −0.0337 | 4.6671 |
| **Anchored GRU（3-seed mean）** | **4.7759 ± 0.2076** | **1.6707 ± 0.0242** | −0.0298 | 4.7233 |
| Anchored GRU（ensemble） | **4.5254** | **1.6244** | −0.0298 | 4.5201 |

相对提升：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Plain GRU（mean） vs Plain Ridge | **−250.1%** | **−111.5%** |
| Plain GRU（mean） vs Plain MLP | **−46.6%** | **−23.8%** |
| Anchored GRU（mean） vs Anchored Ridge | −3.54% | −2.35% |
| Anchored GRU（ensemble） vs Anchored Ridge | +1.89% | +0.49% |
| Anchored GRU（ensemble） vs Anchored MLP | +3.31% | +2.34% |
| Anchored GRU（ensemble） vs B2a | +17.03% | +10.74% |

## 4. 关键发现

### 4.1 Plain GRU 比 Plain MLP 更差

- Test MAE 3.53 vs MLP 2.85、Ridge 1.67；bias **+3.33 ℃**。
- 步内去偏 MSE 5.47，也差于 Plain Ridge 4.40 与 Plain MLP 4.48——**不仅水平更差，形状也更差**。
- 在只有 8,640 个训练窗口、输入 96 h 的条件下，循环结构没有带来更好的短期规律，反而增加了优化/外推难度。
- 结论：**“有记忆”不等于“更会预测”**；在绝对温度上做循环建模，仍然逃不过水平漂移。

### 4.2 Anchored GRU 修掉了偏差，但单个 seed 不如 Anchored Ridge

- bias 从 +3.33 降到 **−0.030**；锚定同样修复了 GRU 的水平问题。
- 单个 seed 的 test MAE：1.686 / 1.684 / 1.643；均值 1.671，差于 Anchored Ridge 的 1.632。
- 单个 seed 的 MSE：4.996 / 4.747 / 4.584；均值 4.776，差于 Anchored Ridge 的 4.613。
- 步内去偏 MSE 4.72，仍差于 Anchored Ridge 4.60、Anchored MLP 4.67——**没有形状增益**。

### 4.3 3-seed ensemble 略好，但不是稳健的“时序增益”

- 3-seed ensemble：MSE 4.525、MAE 1.624，点估计略优于 Anchored Ridge（4.613 / 1.632）。
- 但 bootstrap：

| 对比 | MAE 差均值 | 95% CI | MSE 差均值 | 95% CI |
|---|---:|---|---:|---|
| Anchored GRU − Anchored Ridge | −0.008 | [−0.073, +0.063] | −0.087 | [−0.398, +0.258] |
| Anchored GRU − B2a | −0.195 | **[−0.279, −0.100]** | −0.929 | **[−1.387, −0.465]** |

- 与 Anchored Ridge 的差异**不显著**；逐窗口胜率仅 **51.8%**（几乎五五开）。
- 单个 seed 的 bias 分别为 +0.233 / −0.243 / −0.080：ensemble 的“提升”很大程度上是 **seed 间偏差相互抵消**，而不是循环结构学得更好。
- 与 B2a 的改善显著（MAE +10.7%、MSE +17.0%），但这与 Anchored Ridge/MLP 的结论一致：**收益来自锚定，不是循环**。

### 4.4 训练过程

- Plain GRU 的最优 epoch：31/30/29；val MAE 2.31–2.34；test 3.39–3.60。
- Anchored GRU 的最优 epoch：42/29/17；val MAE 2.29–2.34；test 1.643–1.686。
- val 曲线在 2.3–2.5 之间波动；val 早停仍然无法预测冬季 test 的排序（Plain GRU 的 val 并不离谱，但 test 崩溃）。

## 5. 回答“显式时序结构是否有增益”

**没有可复现的稳健增益。**

- 不加锚定：GRU 比 MLP 更差，水平和形状都退化；循环结构没有帮助。
- 加锚定：单个 seed 不如 Anchored Ridge；3-seed ensemble 点估计略好，但 CI 跨 0、胜率 51.8%，且提升主要来自 seed 平均。
- 从 Ridge → MLP → GRU 的路径看，**决定性能的是水平锚定，而不是模型族**；在锚定后的相对序列上，线性模型的形状甚至略好于 MLP/GRU。

这不是“GRU 永远无用”，而是在当前数据规模、输入（仅 96 个 OT 滞后）、任务（96→96）与评估协议下，**没有证据支持用循环结构替换线性/锚定基线**。

## 6. 下一步建议

1. **先加日历特征**：hour/day/month 的正弦特征，可能比换模型更有效；本轮仍遵守“无日历”约束。
2. **扩展到 H=336/720**：长 horizon 的偏差更大，锚定的价值可能更高；循环结构也可能在更长依赖上才有用。
3. **残差/集成**：Anchored Ridge 为主模型，GRU/MLP 只学残差；或对三者做简单平均，检验互补性。
4. **冬季验证 / rolling-origin**：避免 val 夏秋误导早停与模型选择。
5. **如果仍要测 LSTM/Transformer**：应在锚定表示上、与 Anchored Ridge 严格对比，并报告 seed 方差与 bootstrap CI；不要仅凭单次运行下结论。

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\gru_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/gru/seed_metrics.csv` | 每个 seed 的参数量、best epoch、val/test 指标 |
| `outputs/gru/test_metrics.csv` | Ridge/MLP/GRU/B1/B2a 汇总与相对提升 |
| `outputs/gru/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/gru/segment_metrics.csv` | 4 段指标与相对提升 |
| `outputs/gru/training_history.csv` | 逐 epoch val MAE |
| `outputs/gru/summary.json` | 协议、seed 指标、test/segment、胜负、bootstrap |
| `figures/01_training_curves.png` | 6 条训练曲线 |
| `02_test_mae_by_model.png` | Ridge/MLP/GRU 的 test MAE（带 seed std） |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias（Plain GRU 明显上扬） |
| `05_segment_improvement.png` | 分段相对匹配 Ridge 的提升 |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 间 test MAE 分布 |

## 8. 局限

- 只测了一个 GRU 架构（hidden=64, 1 layer）与 3 个 seed；更大/更深的 GRU 可能改变小数点后的结果。
- 只用了最后隐藏状态；序列到序列或注意力池化可能不同，但那是另一个因素。
- 未做伪影掩码；未加日历/负荷特征；只评估 H=96。
- 早停用 val MAE，val 是夏秋、test 是冬季；这对所有神经模型都不利，也解释了部分 val/test 反差。
