# 长历史输入：PatchTST 336h → 96h

> 脚本：`scripts/patchtst_l336_h96.py`
> 数据：官方 12/4/4；`pred_len=96`；输入 `seq_len=96` 或 `336`
> 模型：Anchored PatchTST（小配置）+ Anchored Ridge
> Seeds：3 个；MSE / Adam / val MAE 早停

## 1. 研究问题

之前所有实验的输入都只有 96h（4 天）。PatchTST 的核心设计目标之一，是通过
Patching 更有效地利用**更长历史**。本轮直接比较：

- **96h → 96h**：4 天历史预测未来 4 天；
- **336h → 96h**：14 天历史预测未来 4 天。

同时用 Anchored Ridge 做同样的 96h vs 336h 对照，区分“历史信息”与“网络结构”。

**重要技术事实**：L=96 与 L=336 的 val/test **目标窗口完全相同**（2,785 个），
只是输入 lookback 不同；因此可以做严格配对的 96 vs 336 比较。train 的目标集合不同
（L=336 的前 336 行只作 lookback）。

## 2. 配置

| 项 | 设置 |
|---|---|
| PatchTST | patch=16, stride=8, d_model=16, 4 heads, **3 layers**, d_ff=32, dropout=0.1 |
| Patch 数 | L=96 → 11；L=336 → 41 |
| 参数量 | L=96: 24,112；L=336: 70,672 |
| Ridge | per-lag X / per-step Y，alpha 按 val MAE 选择 |
| 训练 | MSE, Adam lr=1e-3, wd=1e-4, batch=256, grad clip 1.0, patience=20 |
| Seeds | 0/1/2 |

注意：该配置同时改变了输入长度、patch 数量和 head 宽度，**不是严格的单因素消融**；
同配置的 96h PatchTST 用来尽量分离“历史长度”和“模型规模”。

## 3. Test 结果

| 模型 | seq_len | MSE | MAE | bias | 参数量 |
|---|---:|---:|---:|---:|---:|
| B2b Local Mean 96 | — | 5.6172 | 1.8214 | +0.2257 | — |
| **Anchored Ridge** | 96 | **4.6127** | **1.6324** | +0.1051 | ~9k |
| **Anchored Ridge** | 336 | **4.4683** | **1.6250** | +0.0547 | ~33k |
| **PatchTST small**（mean） | 96 | 4.8422 ± 0.2862 | 1.6438 ± 0.0335 | +0.3009 | 24,112 |
| PatchTST small（ensemble） | 96 | 4.6319 | 1.6003 | +0.3009 | 24,112 |
| **PatchTST small**（mean） | 336 | 4.9958 ± 0.3569 | 1.7577 ± 0.0899 | −0.2090 | 70,672 |
| PatchTST small（ensemble） | 336 | 4.6121 | 1.6918 | −0.2090 | 70,672 |
| PatchTST large L96（参考） | 96 | 4.7770 | 1.6553 | +0.0842 | 136k |

配对比较（同一批 test 目标窗口，L336 − L96）：

| 模型 | MAE Δ | 95% CI | MSE Δ | 95% CI | L336 胜率 |
|---|---:|---|---:|---|---:|
| Anchored Ridge | −0.0074 | [−0.1518, +0.1124] | −0.1443 | [−0.7782, +0.3561] | 49.6% |
| PatchTST small | **+0.0916** | [−0.0696, +0.2297] | −0.0198 | [−1.0578, +0.8149] | 40.8% |

## 4. 关键发现

### 4.1 对 Ridge：更长历史只有微小、不显著的改善

- MSE 4.613 → 4.468（−3.1%），MAE 1.632 → 1.625（−0.45%）。
- 配对 CI 都跨 0；逐窗口胜率 49.6%（几乎五五开）。
- alpha 从 100 变成 1000，说明 336 个滞后特征更共线，需要更强正则；收益很小。
- 结论：**更多历史对线性模型有一点帮助，但不足以改变结论**。

### 4.2 对 PatchTST：更长历史反而变差

- mean：MAE 1.644 → 1.758（**+6.9%**），MSE 4.842 → 4.996（+3.2%）。
- ensemble：MAE 1.600 → 1.692（**+5.7%**），MSE 4.632 → 4.612（基本持平）。
- 配对 MAE Δ = **+0.092**（更长历史更差），CI 跨 0 但胜率只有 **40.8%**。
- 也就是说：在相同小模型配置下，**14 天历史没有比 4 天历史带来任何收益**。

### 4.3 原因：336 输入的 PatchTST 在 epoch 1 就过拟合

| 输入 | 最优 epoch（3 seeds） | 参数量 | val MAE 起点 |
|---|---|---|---:|
| L=96 | 7 / 10 / 9 | 24,112 | ~2.3 |
| L=336 | **1 / 1 / 1** | 70,672 | ~2.22 |

- 336 输入产生 41 个 patch，head 输入维度 41×16=656，head 参数约 63k；模型总参数从 24k 增到 71k。
- 早停在第 1 个 epoch，说明训练一开始就过拟合；额外的历史信息和 head 容量没有被有效利用。
- 因此，这个负结果**不能完全归因于“长历史无用”**，也包含了“长历史 + 更大 head + 不足正则”的复合因素。这正是本轮定位为“更接近论文配置”而非严格消融的原因。

### 4.4 同配置的 96h PatchTST 是竞争性的，但不稳健

- L=96 的 3-seed ensemble：MAE **1.6003**（略好于 Anchored Ridge 1.6324），MSE 4.6319（略差于 4.6127）。
- 但 3-seed mean：MAE 1.6438、MSE 4.8422，差于 Ridge；ensemble 的优势主要来自 seed 平均。
- 所以即使 96h PatchTST 也没有稳定超过 Anchored Ridge；加上长历史后更差。

## 5. 回答“更多历史是否有帮助”

**在当前配置下：没有。**

- 线性模型：336h 只有 −0.45% MAE / −3.1% MSE 的微小改善，统计上不显著。
- PatchTST：336h 明显更差（mean MAE +6.9%，ensemble MAE +5.7%），且在第 1 个 epoch 就过拟合。
- 这说明“更多历史”本身不是免费的收益；要利用长历史，需要同时控制模型容量、正则化和 head 结构。

## 6. 下一步建议

如果还要继续长历史方向，建议先做**容量受控**的 PatchTST：

1. **固定 head 宽度**：对 patch 表示做平均/池化后再接固定 `Linear(d_model, 96)`，使 L=96 与 L=336 参数量接近。
2. **更强正则**：dropout 0.2–0.3、weight decay 1e-3、较小 lr（3e-4），或早停 patience 更小。
3. **d_model=32 / 2 layers**：在控制参数量的前提下增加表示宽度。
4. **必要时做实例归一化（RevIN）**：长历史的水平漂移更明显，先处理水平再比较结构。
5. 如果这些控制后 336h 仍不能超过 96h，则可以明确结论：**这个数据上的额外历史没有可被 PatchTST 利用的信息**。

不建议直接放大 d_model/层数；当前 336 配置已经明显过拟合。

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\patchtst_l336_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/patchtst_l336/alpha_selection.csv` | Ridge L96/L336 × alpha |
| `outputs/patchtst_l336/test_metrics.csv` | B2b/Ridge/PatchTST 对比 |
| `outputs/patchtst_l336/paired_deltas.csv` | L336 − L96 的 MAE/MSE delta 与 CI |
| `outputs/patchtst_l336/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/patchtst_l336/segment_metrics.csv` | 4 段指标 |
| `outputs/patchtst_l336/seed_metrics.csv` | 6 个 seed 的 best epoch/参数量/指标 |
| `outputs/patchtst_l336/summary.json` | 协议、test、paired delta、segment、seed |
| `figures/01_training_curves.png` | L96/L336 训练曲线（可见 L336 在 epoch 1 后上升） |
| `02_test_mae.png` | 长历史对比 |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias |
| `05_input_length_effect.png` | L336 − L96 的 delta |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 分布 |

## 8. 局限

- 配置同时改变输入长度、patch 数和 head 宽度，不是严格单因素消融；容量因素与历史长度混杂。
- 336h 模型在 epoch 1 早停，说明正则化/容量不匹配；不能据此断言 PatchTST 永远无法利用长历史。
- 只测了 S 模式、H=96、单一切分；未做伪影掩码。
- 数据层面：val/test 目标窗口相同，配对比较有效；train 目标集合不同，训练样本量也从 8,449 降到 8,209。
