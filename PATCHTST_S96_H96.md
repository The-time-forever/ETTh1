# PatchTST：S 模式，96 → 96

> 脚本：`scripts/patchtst_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 数据：S 模式，仅 OT；无负荷变量、无日历特征
> 训练：PyTorch，3 个 seed；按 val MAE 早停

## 1. 研究问题

前面的模型：

- Ridge：线性直接映射；
- MLP：非线性全连接；
- GRU/LSTM：循环隐藏状态；
- PatchTST：**时间片段（patch）+ self-attention**。

本轮只改变“如何利用历史”这一因素，回答：

> 让模型直接学习不同历史时间片段之间的关系，能否比循环记忆更有效地预测未来 96 小时？

## 2. 模型与训练

```text
96 -> patch(16, stride=8) -> 11 patches
   -> Linear(16 -> 64) + learnable positional embedding
   -> 2 x TransformerEncoderLayer(d_model=64, heads=4, d_ff=128, GELU, pre-norm, dropout=0.1)
   -> Flatten -> Dropout(0.1) -> Linear(11*64 -> 96)
```

- 参数量 **136,416**（GRU 19,104；LSTM 23,392；MLP 约 41k）。
- 三个变体，只改变水平/尺度处理：
  - **Plain PatchTST**：全局标准化，直接预测绝对 OT；
  - **Anchored PatchTST**：X' = X − ℓ_t，Y' = Y − ℓ_t，预测后加回 ℓ_t；
  - **RevIN PatchTST**：每个输入窗口做 instance normalization（原版 PatchTST 的做法）。
- 控制基线 **RevIN Ridge**：同样的 per-window normalization + 线性 Ridge，用来区分“attention 的收益”与“RevIN 的收益”。
- 训练协议与前几轮一致：Adam `lr=1e-3`、`weight_decay=1e-4`、batch=256、MSE、梯度裁剪 1.0、val MAE 早停（patience=20，max_epochs=300）、3 个 seed。

## 3. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE | 参数量 |
|---|---:|---:|---:|---:|---:|
| B1 Persistence | 5.8326 | 1.8654 | +0.1215 | 5.8119 | — |
| B2a Local Mean 24 | 5.4544 | 1.8198 | +0.1443 | 5.4277 | — |
| Plain Ridge | 5.1067 | 1.6698 | +0.7805 | **4.4000** | ~9k |
| Plain GRU | 17.8776 ± 1.0400 | 3.5318 ± 0.1197 | +3.3314 | 5.4713 | 19k |
| Plain LSTM | 18.2429 ± 0.0312 | 3.5648 ± 0.0026 | +3.3388 | 5.7430 | 23k |
| **Plain PatchTST（mean）** | **11.2502 ± 1.6994** | **2.7124 ± 0.2622** | **+2.3305** | 5.1495 | 136k |
| Plain PatchTST（ensemble） | 10.8036 | 2.6728 | +2.3305 | 4.9623 | 136k |
| Anchored Ridge | **4.6127** | **1.6324** | +0.1051 | **4.5981** | ~9k |
| Anchored GRU | 4.7759 ± 0.2076 | 1.6707 ± 0.0242 | −0.0298 | 4.7233 | 19k |
| Anchored LSTM | 5.0765 ± 0.2517 | 1.7586 ± 0.0442 | −0.2022 | 5.0018 | 23k |
| **Anchored PatchTST（mean）** | 4.7770 ± 0.1232 | 1.6553 ± 0.0352 | +0.0842 | 4.7222 | 136k |
| **Anchored PatchTST（ensemble）** | **4.6168** | **1.6233** | +0.0842 | 4.5942 | 136k |
| **RevIN Ridge（控制）** | **4.6102** | **1.6363** | +0.0539 | **4.6050** | ~9k |
| RevIN PatchTST（mean） | 4.7665 ± 0.0539 | 1.6689 ± 0.0060 | +0.1969 | 4.7164 | 136k |
| RevIN PatchTST（ensemble） | 4.6869 | 1.6525 | +0.1969 | 4.6435 | 136k |

相对提升：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Plain PatchTST（mean） vs Plain Ridge | −120.3% | −62.4% |
| Plain PatchTST（mean） vs Plain GRU | +37.1% | +23.2% |
| Plain PatchTST（mean） vs Plain LSTM | +38.3% | +23.9% |
| Anchored PatchTST（mean） vs Anchored Ridge | −3.56% | −1.40% |
| Anchored PatchTST（ensemble） vs Anchored Ridge | −0.09% | **+0.56%** |
| Anchored PatchTST（ensemble） vs Anchored GRU | +3.33% | +2.84% |
| RevIN PatchTST（mean） vs RevIN Ridge | −3.39% | −1.99% |
| RevIN PatchTST（ensemble） vs RevIN Ridge | −1.66% | −0.99% |
| Anchored PatchTST（ensemble） vs B2a | +15.36% | +10.80% |

## 4. 关键发现

### 4.1 Plain PatchTST 比 Plain GRU/LSTM 好，但仍远差于 Plain Ridge

- MAE 2.71 vs GRU 3.53、LSTM 3.56、Ridge 1.67。
- bias **+2.33 ℃**（GRU +3.33、LSTM +3.34），说明 patch 表示确实缓解了一部分水平外推问题，但没有解决。
- 步内去偏 MSE 5.15，仍差于 Plain Ridge 4.40；**形状也没有超过线性模型**。
- 换句话说：patch attention 让“失败”没那么惨，但没有把失败变成成功。

### 4.2 Anchored PatchTST 与 Anchored Ridge 基本打平

- 3-seed mean：MSE 4.777、MAE 1.655；Anchored Ridge 4.613 / 1.632（mean 略差）。
- 3-seed ensemble：MSE 4.617、MAE **1.623**；Anchored Ridge 4.613 / 1.632（点估计略好）。
- 逐窗口 bootstrap：

| 对比 | MAE 差均值 | 95% CI | MSE 差均值 | 95% CI |
|---|---:|---|---:|---|
| Anchored PatchTST − Anchored Ridge | −0.009 | [−0.033, +0.018] | +0.004 | [−0.135, +0.182] |
| RevIN PatchTST − RevIN Ridge | +0.016 | [−0.024, +0.049] | +0.077 | [−0.116, +0.251] |

- 两者 CI 都跨 0；Anchored PatchTST 对 Anchored Ridge 的逐窗口胜率仅 **50.3%**。
- 结论：**attention 没有带来统计上可复现的增益**；ensemble 的 0.56% MAE 优势不足以支撑“更好”。

### 4.3 RevIN PatchTST 没有超过 RevIN 线性控制

- RevIN PatchTST mean/ensemble：MAE 1.669 / 1.653；RevIN Ridge：1.636。
- RevIN PatchTST 的 bias 还更大（+0.197 vs +0.054）。
- 逐窗口胜率 43.7%（低于五五开）。
- 说明：**per-window normalization 本身有效，但 attention 在这之上没有增加价值**。这是本轮最重要的控制实验。

### 4.4 与循环模型相比

- Anchored PatchTST ensemble 1.623 vs Anchored GRU 1.671、Anchored LSTM 1.759：点估计略好（相对 GRU +2.8% MAE）。
- 但 GRU 的 seed std 为 0.024，PatchTST 为 0.035；差异很小，且没有做配对 bootstrap（GRU 预测未保存）。
- 更关键的是：PatchTST 并没有超过 Anchored Ridge（1.632），所以“比循环好”最多是“比最弱的循环变体好一点”，而不是“比线性/锚定基线好”。

### 4.5 早停几乎在初始化处：严重过拟合

| 变体 | 最优 epoch（3 seeds） | 之后 val MAE |
|---|---|---|
| Plain PatchTST | 2 / 1 / 2 | 从 ~2.5 升到 3.0–4.0 |
| Anchored PatchTST | 2 / 1 / 3 | 从 ~2.3 升到 2.7–2.8 |
| RevIN PatchTST | 1 / 1 / 1 | 从 ~2.28 升到 2.4–2.7 |

- 136k 参数、8,449 个训练窗口；模型在 1–3 个 epoch 后就开始过拟合，val MAE 持续上升。
- 这意味着本轮比较的是“**几乎未训练的 PatchTST**”，而不是充分训练后的 attention 表示。
- 因此，负结果应表述为：**在当前数据规模、模型大小、正则化与早停协议下，PatchTST 没有带来超过线性/锚定的收益**；不能据此断言“attention 永远无用”。更小/更强正则的 PatchTST 仍值得一次后续检验。

## 5. 回答“PatchTST 是否比循环记忆更有效”

- 对 Plain 家族：PatchTST 比 GRU/LSTM 好（MAE 2.71 vs 3.53/3.56），但仍远差于 Plain Ridge（1.67），且 bias +2.33；只是“失败得轻一些”。
- 对 Anchored 家族：PatchTST 与 Anchored Ridge 统计上打平，未超过；比 Anchored GRU 略好但差距很小。
- 对 RevIN 家族：PatchTST 没有超过 RevIN Ridge，说明收益来自 normalization，而不是 attention。

**结论：没有可复现的证据表明 patch + self-attention 比循环记忆或锚定线性模型更有效。** 当前任务的瓶颈仍然是水平/分布漂移与数据规模，而不是“历史片段之间的关系”。

## 6. 下一步建议

1. **先做更小/更强正则的 PatchTST**：`d_model=32`、1 层、dropout 0.2、head dropout 0.2；当前 136k 参数在 8.4k 窗口上明显过大。
2. **保留 RevIN Ridge 作为控制**：任何 normalization 变体都必须与“同样 normalization 的线性模型”比较，否则会把 normalization 的收益误归因于 attention。
3. **扩展到 H=336/720**：patch attention 的长程优势可能只有在更长 horizon 才显现。
4. **日历特征**：在 anchored/RevIN 表示上加入 hour/day/month，可能比换模型更有效。
5. **残差/集成**：以 Anchored Ridge 为主模型，PatchTST 只学残差，或做简单平均。
6. **完整 Transformer / iTransformer**：只有在更小 PatchTST 仍显示潜力时才值得进入。

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\patchtst_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/patchtst/seed_metrics.csv` | 每个 seed 的参数量、best epoch、val/test 指标 |
| `outputs/patchtst/test_metrics.csv` | 所有模型的汇总与相对提升 |
| `outputs/patchtst/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/patchtst/segment_metrics.csv` | 4 段指标与相对提升 |
| `outputs/patchtst/training_history.csv` | 逐 epoch val MAE |
| `outputs/patchtst/revin_ridge_alpha.csv` | RevIN Ridge 的 alpha 选择 |
| `outputs/patchtst/summary.json` | 协议、seed 指标、test/segment、胜负、bootstrap |
| `figures/01_training_curves.png` | 9 条训练曲线（3 变体 × 3 seed） |
| `02_test_mae_by_model.png` | Plain / Anchored / RevIN 家族对比 |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias |
| `05_segment_improvement.png` | 分段相对匹配 Ridge 的提升 |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 间 test MAE 分布 |

## 8. 局限

- 只测了一个 PatchTST 配置（136k 参数、2 层、d_model=64）；明显过拟合，负结果不能外推到更小/更强正则的 PatchTST。
- 只用了最后层的 flatten head；其他 head（池化、query-based）可能不同。
- 未做伪影掩码；未加日历/负荷特征；只评估 H=96。
- 早停用 val MAE，val 是夏秋、test 是冬季；但 Plain PatchTST 的 val/test 都差，结论不依赖单一偏差。
