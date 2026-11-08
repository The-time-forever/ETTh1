# LSTM：S 模式，96 → 96

> 脚本：`scripts/lstm_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 数据：S 模式，仅 OT；无负荷变量、无日历特征
> 训练：PyTorch，3 个 seed；按 val MAE 早停

## 1. 研究问题

GRU 与 LSTM 的区别主要在门控结构：

- GRU：reset/update 两个门，隐藏状态同时承担记忆；
- LSTM：input/forget/output 三个门 + 独立的 cell state。

本轮只改变这一点，回答：

> LSTM 的独立记忆单元能否比 GRU 更好地提取油温的时间依赖？

## 2. 模型与训练

```text
LSTM(input=1, hidden=64, num_layers=1) -> last hidden -> Linear(64 -> 96)
```

- 参数量 **23,392**（GRU 19,104；MLP 128-128 约 41k）。
- 输入 `(batch, 96, 1)`；只用最后时刻的隐藏状态，直接输出 96 步。
- 输入全局标准化（train），目标标准化（train），预测反变换回 ℃。
- 损失 MSE；Adam `lr=1e-3`、`weight_decay=1e-4`、batch=256、梯度裁剪 1.0。
- 早停：每 epoch 计算 **val MAE（原始 ℃）**，`patience=20`、`max_epochs=300`，恢复最优 epoch。
- Seeds：0/1/2；报告 3-seed mean ± std 与 3-seed ensemble。

两个变体：

| 变体 | 输入 | 输出 | 恢复 |
|---|---|---|---|
| Plain LSTM | X | Y | 直接输出 |
| Anchored LSTM | X' = X − ℓ_t | Y' = Y − ℓ_t | Ŷ = LSTM(X') + ℓ_t |

## 3. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE |
|---|---:|---:|---:|---:|
| B1 Persistence | 5.8326 | 1.8654 | +0.1215 | 5.8119 |
| B2a Local Mean 24 | 5.4544 | 1.8198 | +0.1443 | 5.4277 |
| Plain Ridge | 5.1067 | 1.6698 | +0.7805 | **4.4000** |
| Plain MLP | 12.1981 ± 1.3135 | 2.8527 ± 0.1861 | +2.6776 | 4.4800 |
| Plain GRU | 17.8776 ± 1.0400 | 3.5318 ± 0.1197 | +3.3314 | 5.4713 |
| **Plain LSTM** | **18.2429 ± 0.0312** | **3.5648 ± 0.0026** | **+3.3388** | **5.7430** |
| Plain LSTM（ensemble） | 18.2040 | 3.5617 | +3.3388 | 5.7157 |
| Anchored Ridge | **4.6127** | **1.6324** | +0.1051 | **4.5981** |
| Anchored MLP | 4.6802 ± 0.0403 | 1.6633 ± 0.0062 | −0.0337 | 4.6671 |
| Anchored GRU | 4.7759 ± 0.2076 | 1.6707 ± 0.0242 | −0.0298 | 4.7233 |
| **Anchored LSTM** | **5.0765 ± 0.2517** | **1.7586 ± 0.0442** | **−0.2022** | **5.0018** |
| Anchored LSTM（ensemble） | 4.8640 | 1.7219 | −0.2022 | 4.8071 |

相对提升：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Plain LSTM vs Plain GRU | −2.04% | −0.93% |
| Plain LSTM vs Plain Ridge | −257.2% | −113.5% |
| Anchored LSTM（mean） vs Anchored GRU | **−6.30%** | **−5.26%** |
| Anchored LSTM（mean） vs Anchored Ridge | **−10.06%** | **−7.73%** |
| Anchored LSTM（ensemble） vs Anchored GRU | −1.85% | −3.06% |
| Anchored LSTM（ensemble） vs Anchored Ridge | −5.45% | −5.48% |
| Anchored LSTM（ensemble） vs B2a | +10.82% | +5.38% |

## 4. 关键发现

### 4.1 Plain LSTM 与 Plain GRU 一样失败，且略差

- Test MAE 3.565 vs GRU 3.532、Ridge 1.670；bias **+3.34 ℃**。
- 步内去偏 MSE 5.743，也差于 GRU 5.471、Ridge 4.400——**水平和形状都没有改善**。
- 说明：在绝对温度上做循环建模，LSTM 的额外 cell state 没有解决水平漂移，也没有带来更好的短期规律。

### 4.2 Anchored LSTM 明显差于 Anchored GRU 与 Anchored Ridge

- 3-seed mean：MSE 5.077、MAE 1.759；Anchored GRU 是 4.776 / 1.671，Anchored Ridge 是 4.613 / 1.632。
- 3-seed ensemble：MSE 4.864、MAE 1.722，仍差于 Anchored Ridge/GRU。
- 相对 Anchored GRU：MSE **−6.3%**、MAE **−5.3%**；相对 Anchored Ridge：MSE **−10.1%**、MAE **−7.7%**。
- **LSTM 的独立记忆单元没有带来收益，反而更差。**

### 4.3 不是 test 漂移造成的：val 上 LSTM 也更差

| 模型 | val MAE 范围（3 seeds） | test MAE 范围 |
|---|---:|---:|
| Plain GRU | 2.312–2.344 | 3.394–3.604 |
| Plain LSTM | 2.322–2.353 | 3.562–3.567 |
| Anchored GRU | 2.294–2.337 | 1.643–1.686 |
| Anchored LSTM | **2.357–2.375** | **1.724–1.808** |

- Anchored LSTM 在 **val（夏秋）** 上就已经差于 Anchored GRU，说明不是冬季 shift 才暴露的问题，而是**优化/拟合能力本身不足**。
- 参数量更多（23k vs 19k），但结果更差；不是容量问题，而是小数据上的优化难度/归纳偏置不匹配。

### 4.4 锚定仍有用，但 LSTM 的残差偏差更大

- Plain LSTM bias +3.34 → Anchored LSTM bias **−0.202**；锚定同样修复了大部分水平问题。
- 但三个 anchored seed 的 bias 全是负的：−0.251 / −0.049 / −0.307；不是 seed 噪声，而是**系统性低估**。
- Anchored GRU 的 bias 是 −0.030（seed 间正负抵消），明显更接近 0。
- 因此，锚定对 LSTM 的帮助不如对 GRU/Ridge 充分；残余水平误差仍然明显。

### 4.5 与 Ridge/B2a 的显著性

| 对比 | MAE 差均值 | 95% CI | MSE 差均值 | 95% CI |
|---|---:|---|---:|---|
| Anchored LSTM − Anchored Ridge | +0.089 | **[+0.024, +0.141]** | +0.251 | [−0.168, +0.580] |
| Anchored LSTM − B2a | −0.098 | [−0.214, +0.019] | −0.590 | [−1.196, +0.013] |

- 与 Anchored Ridge 的 MAE 差异显著，方向是 **LSTM 更差**。
- 与 B2a 的 MAE/MSE 差异 CI 都跨 0（MSE 上界仅 +0.013），**不能认为 Anchored LSTM 稳健优于 B2a**。
- 逐窗口胜率：Anchored LSTM 优于 Anchored Ridge 仅 **37.6%**，优于 B2a **55.2%**。

## 5. 回答“LSTM 是否比 GRU 更好地提取时间依赖”

**没有。**

- 不加锚定：LSTM 与 GRU 都失败，LSTM 略差（MAE 3.565 vs 3.532；去偏 MSE 5.743 vs 5.471）。
- 加锚定：LSTM 明显差于 GRU（MSE 5.077 vs 4.776，MAE 1.759 vs 1.671），也差于 Anchored Ridge。
- val 上同样更差，说明不是 test 漂移的偶然。
- 额外参数（23k vs 19k）与额外门控没有带来收益；在当前数据规模与 96→96 任务下，**GRU 是更合适的循环单元，而锚定线性模型仍然最好**。

## 6. 下一步建议

1. **停止在循环单元上继续加复杂度**：GRU 已经优于 LSTM；再上 LSTM 变体/双向/堆叠的收益预期很低。
2. **先加日历特征**：hour/day/month 的正弦特征可能比换模型更有效。
3. **扩展到 H=336/720**：长 horizon 的偏差更大，锚定 + 线性可能仍有优势；循环模型也可能在更长依赖上才有空间，但需要单独验证。
4. **残差/集成**：以 Anchored Ridge 为主模型，让 LSTM/GRU 只学残差；或对三者做简单平均。
5. **冬季验证 / rolling-origin**：避免 val 夏秋误导早停与模型选择。
6. **Transformer**：只有在锚定表示上、与 Anchored Ridge 严格对比，并报告 seed 方差与 bootstrap CI 才值得做；当前证据不支持把它当作默认下一步。

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\lstm_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/lstm/seed_metrics.csv` | 每个 seed 的参数量、best epoch、val/test 指标 |
| `outputs/lstm/test_metrics.csv` | Ridge/MLP/GRU/LSTM/B1/B2a 汇总与相对提升 |
| `outputs/lstm/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/lstm/segment_metrics.csv` | 4 段指标与相对提升 |
| `outputs/lstm/training_history.csv` | 逐 epoch val MAE |
| `outputs/lstm/test_predictions.npz` | plain/anchored 的 seed 预测与 ensemble 预测 |
| `outputs/lstm/summary.json` | 协议、seed 指标、test/segment、胜负、bootstrap |
| `figures/01_training_curves.png` | 6 条训练曲线 |
| `02_test_mae_by_model.png` | Ridge/MLP/GRU/LSTM 的 test MAE |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias |
| `05_segment_improvement.png` | 分段相对 Ridge/GRU 的提升 |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 间 test MAE 分布 |

## 8. 局限

- 只测了一个 LSTM 架构（hidden=64, 1 layer）与 3 个 seed；更大/更深/双向的 LSTM 可能改变小数点后的结果。
- 只用了最后隐藏状态；序列到序列或注意力池化是另一个因素。
- 未做伪影掩码；未加日历/负荷特征；只评估 H=96。
- 早停用 val MAE，val 是夏秋、test 是冬季；但本轮 LSTM 在 val 上也更差，结论不依赖这一偏差。
