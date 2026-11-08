# 轻量 MLP：S 模式，96 → 96

> 脚本：`scripts/mlp_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 数据：S 模式，仅 OT；无负荷变量、无日历特征
> 训练：PyTorch，3 个 seed；按 val MAE 早停

## 1. 实验目的

上一轮 Ridge 是线性映射。本轮只加入一个因素：**非线性激活**。

- 如果 MLP 能超过 Ridge，说明数据中存在线性模型无法表达的关系；
- 如果 MLP 不能超过 Ridge，说明瓶颈不是线性/非线性，而是水平漂移或其他结构。

两个变体，保证与 Ridge 的输入/输出/划分/指标完全一致：

| 变体 | 输入 | 输出 | 恢复 |
|---|---|---|---|
| Plain MLP | X（96 个 OT 滞后） | Y（未来 96 步） | 直接输出 |
| Anchored MLP | X' = X − ℓ_t | Y' = Y − ℓ_t | Ŷ = MLP(X') + ℓ_t |

`ℓ_t` = 输入窗口最后 24 h 均值，与 Anchored Ridge 完全一致。

## 2. 网络与训练

```text
96 -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(96)
```

- 无 dropout、无 BatchNorm、无 RNN、无 attention；输出层线性。
- 输入/目标均用训练段 `StandardScaler`；预测反变换回 ℃。
- 损失：MSE（标准化目标）；Adam `lr=1e-3`、`weight_decay=1e-4`、batch=256。
- 早停：每 epoch 计算 **val MAE（原始 ℃）**，`patience=20`，`max_epochs=300`，恢复最优 epoch。
- Seeds：0/1/2；报告 3-seed mean ± std 与 3-seed ensemble。
- 设备：CUDA。

## 3. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE | 步内去偏 MAE |
|---|---:|---:|---:|---:|---:|
| Plain Ridge | 5.1067 | 1.6698 | +0.7805 | **4.4000** | **1.6037** |
| **Plain MLP（3-seed mean）** | **12.1981 ± 1.3135** | **2.8527 ± 0.1861** | **+2.6776** | 4.4800 | 1.6078 |
| Plain MLP（ensemble） | 11.9473 | 2.8355 | +2.6776 | 4.4192 | 1.5946 |
| Anchored Ridge | **4.6127** | **1.6324** | +0.1051 | **4.5981** | **1.6399** |
| Anchored MLP（3-seed mean） | 4.6802 ± 0.0403 | 1.6633 ± 0.0062 | −0.0337 | 4.6671 | 1.6579 |
| Anchored MLP（ensemble） | 4.6412 | 1.6553 | −0.0337 | 4.6365 | 1.6516 |
| B1 Persistence | 5.8326 | 1.8654 | +0.1215 | 5.8119 | 1.8724 |
| B2a Local Mean 24 | 5.4544 | 1.8198 | +0.1443 | 5.4277 | 1.8288 |

相对提升：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Plain MLP（mean） vs Plain Ridge | **−138.9%** | **−70.8%** |
| Anchored MLP（mean） vs Anchored Ridge | −1.46% | −1.89% |
| Anchored MLP（ensemble） vs Anchored Ridge | −0.62% | −1.40% |
| Anchored MLP（ensemble） vs B2a | +14.9% | +9.0% |

## 4. 关键发现

### 4.1 Plain MLP 在冬季 test 上崩溃

- test MAE 2.65–3.02（3 个 seed），远差于 Plain Ridge 的 1.67。
- bias **+2.68 ℃**：模型把预测拉向训练段的绝对温度水平；冬季 test 均值只有 4.85 ℃，于是系统性高估。
- 但它的**步内去偏 MSE 是 4.48**，与 Plain Ridge 的 4.40 很接近——说明 Plain MLP 学到的“形状”并不差，**崩溃几乎全部来自绝对水平偏差**。
- 这再次证明：在当前数据上，绝对水平外推是主要风险，非线性并不能解决它。

### 4.2 Anchored MLP 与 Anchored Ridge 基本打平，但没有超过

- bias 降到 **−0.034 ℃**（Plain MLP 是 +2.68），锚定同样修复了 MLP 的水平偏差。
- Test MAE 1.663（mean）/ 1.655（ensemble），Anchored Ridge 是 1.632；MSE 4.680/4.641 vs 4.613。
- 逐窗口 bootstrap：

| 对比 | MAE 差均值 | 95% CI | MSE 差均值 | 95% CI |
|---|---:|---|---:|---|
| Anchored MLP − Anchored Ridge | +0.023 | [−0.018, +0.056] | +0.029 | [−0.195, +0.204] |
| Anchored MLP − B2a | −0.165 | **[−0.282, −0.040]** | −0.813 | **[−1.395, −0.201]** |

- 与 Anchored Ridge 的差异在统计上不显著，点估计还略偏向 Ridge；与 B2a 的改善则显著。
- 逐窗口胜率：Anchored MLP 优于 Anchored Ridge **48.6%**，优于 B2a **58.2%**。

### 4.3 非线性没有改善“形状”

| 模型 | 原始 MSE | 步内去偏 MSE | 原始 MAE | 步内去偏 MAE |
|---|---:|---:|---:|---:|
| Plain Ridge | 5.107 | **4.400** | 1.670 | **1.604** |
| Plain MLP | 12.198 | 4.480 | 2.853 | 1.608 |
| Anchored Ridge | **4.613** | **4.598** | **1.632** | **1.640** |
| Anchored MLP | 4.680 | 4.667 | 1.663 | 1.658 |

- 去掉水平偏差后，Plain MLP 的 MSE 4.480 仍略差于 Plain Ridge 的 4.400；Anchored MLP 的 4.667 也略差于 Anchored Ridge 的 4.598。
- 也就是说：**MLP 没有学到比线性模型更好的相对变化关系**；它只是把水平问题换了一种方式表达。
- 结论不是“非线性一定无用”，而是在当前输入（仅 96 个 OT 滞后）、当前数据量、当前训练协议下，**非线性没有带来可复现的增益**。

### 4.4 早停与 val 的局限

- Anchored MLP 的最优 epoch 只有 **5/3/3**（seed 0/1/2）；Plain MLP 是 **11/24/9**。val MAE 很快进入平台（约 2.27–2.29），随后波动。
- val（夏秋）与 test（冬季）的分布不同：Plain MLP 的 val MAE 约 2.27，看似不差，但 test 崩溃。**val MAE 不能可靠预测冬季外推表现**。
- 这也是为什么锚定模型在 val 上略差、在 test 上更好：收益来自抗漂移，而不是 val 平均精度。

## 5. 回答“非线性是否有增益”

**没有可复现的增益。**

- 不加锚定：MLP 因水平偏差在 test 上崩溃（MAE 2.85 vs Ridge 1.67）；这不是非线性带来的预测能力，而是绝对水平外推的失败。
- 加锚定：MLP 与 Anchored Ridge 打平（MAE 1.663 vs 1.632；CI 跨 0，点估计略差），没有超过线性模型。
- 形状层面：MLP 的步内去偏误差略差于对应的 Ridge。

因此，当前瓶颈是**水平/分布漂移**，不是“线性映射容量不足”。在这个结论下，直接跳到 GRU/LSTM/Transformer 的理由不充分；如果要做序列模型，应在**锚定后的相对序列**上做，并且先证明它能超过 Anchored Ridge。

## 6. 下一步建议

1. **先补日历特征**：本轮仍遵守“无日历”的约束；在锚定模型上加入 hour/day/month 的正弦特征，可能比换网络更有效。
2. **扩展到 H=336/720**：Anchored Ridge 在 H=96 的收益主要来自去偏，长 horizon 的偏差更大，锚定可能更有价值。
3. **残差/集成**：把 Anchored Ridge 作为主模型，MLP 只学其残差；或者对 Anchored Ridge 与 MLP 做简单平均，检验是否有互补性。
4. **冬季验证 / rolling-origin**：用 2016-12—2017-02 作为 shift 验证集，避免 val 夏秋误导模型选择。
5. **如果仍要做循环模型**：GRU/LSTM 应作为“序列结构”的独立假设，在锚定表示上测试；不要与“非线性”混在一起。

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\mlp_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/mlp/seed_metrics.csv` | 每个 seed 的 best epoch/val MAE/test 指标 |
| `outputs/mlp/test_metrics.csv` | Ridge/MLP/B1/B2a 的汇总与相对提升 |
| `outputs/mlp/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/mlp/segment_metrics.csv` | 4 段指标与相对 Ridge/B2a 的提升 |
| `outputs/mlp/training_history.csv` | 每个 seed 的逐 epoch val MAE |
| `outputs/mlp/summary.json` | 协议、seed 指标、test/segment、胜负、bootstrap |
| `figures/01_training_curves.png` | 6 条训练曲线（2 变体 × 3 seed） |
| `02_test_mae_by_model.png` | Ridge vs MLP 的 test MAE（带 seed std） |
| `03_per_step_mae.png` | Plain/Anchored 的逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias（Plain MLP 明显上扬） |
| `05_segment_improvement.png` | 分段相对匹配 Ridge 的提升 |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 间 test MAE 分布 |

## 8. 局限

- 只测了一个架构（128-128）与 3 个 seed；更宽的搜索可能改变小数点后的结论，但“Plain MLP 崩溃 + Anchored MLP 打平”的模式很稳定。
- 训练损失为 MSE、早停用 val MAE；换损失/学习率可能略改结果。
- 未做伪影掩码；月末冻结日可能影响训练与评估。
- 未加入日历/负荷特征；这些可能比“非线性”更关键。
- 未做多 horizon（336/720）；结论只适用于 96→96。
