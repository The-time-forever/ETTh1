# DLinear：S 模式，96 → 96

> 脚本：`scripts/dlinear_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 数据：S 模式，仅 OT
> 模型：DLinear（移动平均分解，Kernel Size=25 + trend/seasonal 两个线性头）
> 训练：MSE、Adam、val MAE 早停、3 seeds

## 1. 对照设计

| 模型 | 输入 | 标准化 | 说明 |
|---|---|---|---|
| Anchored Ridge（per-lag） | 96h OT | per-lag | 之前的主要参考 |
| S_control（per-variable） | 96h OT | per-variable | 与 DLinear 相同的标准化，最干净的对照 |
| **Plain DLinear** | 96h OT | per-variable | 检验显式趋势分解 |
| **Anchored DLinear** | X' = X − ℓ_t | per-variable | 检验分解 + 水平锚定 |

- ℓ_t = 输入窗口最后 24h OT 均值。
- DLinear：`trend = AvgPool1d(25)`（边缘填充），`seasonal = x − trend`；两个 `Linear(96→96)` 分别映射后相加。
- 参数量 **18,624**；3 seeds；best epoch 28–74（Plain）/38–64（Anchored），没有出现 MLP/PatchTST 那种 1–3 epoch 的剧烈过拟合。

## 2. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE | 步内去偏 MAE |
|---|---:|---:|---:|---:|---:|
| B1 Persistence | 5.8326 | 1.8654 | — | — | — |
| B2a Local Mean 24 | 5.4544 | 1.8198 | — | — | — |
| B2b Local Mean 96 | 5.6172 | 1.8214 | — | — | — |
| Anchored Ridge（per-lag） | **4.6127** | **1.6324** | +0.1051 | **4.5981** | **1.6399** |
| S_control（per-variable） | **4.6140** | **1.6327** | +0.1051 | **4.5994** | **1.6402** |
| **Plain DLinear（mean）** | 6.2245 ± 0.2452 | 1.8432 ± 0.0359 | **+1.2641** | **4.3074** | **1.5855** |
| Plain DLinear（ensemble） | 6.1812 | 1.8345 | +1.2641 | 4.3003 | 1.5841 |
| **Anchored DLinear（mean）** | 4.6162 ± 0.0045 | 1.6350 ± 0.0029 | **+0.0932** | 4.6039 | 1.6419 |
| Anchored DLinear（ensemble） | **4.6069** | **1.6331** | +0.0932 | 4.5956 | 1.6401 |

相对变化：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Plain DLinear（mean） vs Anchored Ridge | −34.9% | −12.9% |
| Plain DLinear（mean） vs S_control | −34.9% | −12.9% |
| Anchored DLinear（mean） vs Anchored Ridge | −0.08% | −0.16% |
| Anchored DLinear（ensemble） vs Anchored Ridge | +0.13% | −0.04% |
| Anchored DLinear（mean） vs S_control | −0.05% | −0.15% |
| Anchored DLinear（ensemble） vs B2b | +17.9% | +10.2% |

## 3. 关键发现

### 3.1 Plain DLinear：形状更好，但水平偏差更大

- 步内去偏 MSE **4.3074**，好于 Plain Ridge 的 4.4000（约 2.1%）；去偏 MAE 1.5855 也好于 Plain Ridge 1.6037。
- 说明 **移动平均分解确实改善了相对形状**：trend/seasonal 两个线性头比单一线性映射更容易表达短期结构。
- 但 bias 从 Plain Ridge 的 +0.781 增大到 **+1.264**；原始 MSE/MAE 反而更差（6.22/1.843 vs 5.11/1.670）。
- 也就是说：**分解能学形状，但不能学水平**；没有锚定时，水平漂移仍然主导误差。

### 3.2 Anchored DLinear：水平修好了，分解的额外价值消失

- bias 从 +1.264 降到 **+0.093**，与 Anchored Ridge 的 +0.105 几乎相同。
- 原始 MSE/MAE：4.616/1.635（mean）、4.607/1.633（ensemble），与 Anchored Ridge 4.613/1.632、S_control 4.614/1.633 **统计上完全打平**。
- 步内去偏 MSE 4.604，也与 Anchored Ridge 4.598 基本一致——**锚定后，DLinear 没有比简单线性映射更好的形状**。
- 解释：DLinear 的 trend 分解与“局部水平锚定”在功能上是**替代关系**：两者都在处理水平/趋势；一旦锚定把水平拿掉，分解的增量价值就消失了。

### 3.3 锚定仍然是最重要的因素

| 对比 | MAE 差 | 95% CI | MSE 差 | 95% CI | 胜率 |
|---|---:|---|---:|---|---:|
| Anchored DLinear − Plain DLinear | −0.201 | [−0.390, +0.001] | −1.574 | [−2.853, −0.408] | 63.1% |
| Plain DLinear − B2a | +0.015 | [−0.151, +0.191] | +0.727 | [−0.381, +2.047] | 51.2% |
| Anchored DLinear − B2a | −0.187 | [−0.286, −0.076] | −0.848 | [−1.349, −0.284] | 62.5% |

- 锚定对 DLinear 的 MSE 改善显著，MAE 改善处于边界（CI 上界 +0.001）。
- Plain DLinear 与 B2a 打平：不加锚定时，DLinear 的原始表现并不比“最近 24h 均值”更好。
- Anchored DLinear 与 Anchored Ridge 的差异在 0.2% 以内，且 seed std 很小（MSE 0.0045、MAE 0.0029），没有证据表明 DLinear 更优。

### 3.4 训练稳定性

- Plain DLinear：test MAE 1.804–1.874（3 seeds）；Anchored DLinear：1.632–1.638。
- 与 MLP/PatchTST 的 1–3 epoch 过拟合不同，DLinear 的 val 曲线平稳，best epoch 在 38–74；说明它在这个数据规模上是一个稳定的线性模型。
- 但稳定性不等于更强：稳定地达到与 Anchored Ridge 相同的误差。

## 4. 结论

1. **显式趋势分解有帮助，但只体现在“形状”上**：Plain DLinear 的去偏 MSE 4.307 好于 Plain Ridge 4.400。
2. **没有锚定时，水平偏差抵消了形状收益**：Plain DLinear 的 bias +1.264，原始 MAE 1.843，反而差于 Plain Ridge 1.670。
3. **加锚定后，DLinear 与 Anchored Ridge 打平**（MSE 4.617/4.607 vs 4.613；MAE 1.635/1.633 vs 1.632）；分解没有带来额外增益。
4. **DLinear 的 trend 分解与局部水平锚定是替代品，不是互补品**；真正决定性能的是水平处理。
5. 对当前 S 96→96 任务，Anchored Ridge 仍是更简单、同等精度的选择；DLinear 是一个稳定的替代基线，但不值得继续做变体。

## 5. 下一步建议

- 不再围绕 DLinear 做结构变体（moving-average kernel、individual head 等）；本轮的结论已经足够清楚。
- 如果要继续线性模型方向，优先做 **水平/外生信息**（日历、天气），而不是趋势分解。
- 如果要继续神经网络方向，应把重点放在“锚定 + 学习残差”或概率预测，而不是再换一个线性/注意力架构。
- 需要的话，可以把 Anchored DLinear 作为 H=192 的一个轻量对照，但预期仍是与 Anchored Ridge 打平。

## 6. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\dlinear_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/dlinear/seed_metrics.csv` | 3 seeds 的 best epoch/val/test 指标 |
| `outputs/dlinear/test_metrics.csv` | Ridge/DLinear/B1/B2a/B2b 对比 |
| `outputs/dlinear/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/dlinear/segment_metrics.csv` | 4 段指标与相对 skill |
| `outputs/dlinear/training_history.csv` | 逐 epoch val MAE |
| `outputs/dlinear/summary.json` | 协议、seed、test/segment、胜负 |
| `figures/01_training_curves.png` | 6 条训练曲线 |
| `02_test_mae.png` | Ridge vs DLinear |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias（Plain DLinear 明显上扬） |
| `05_segment_skill.png` | 分段 skill |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 间 MAE |

## 7. 局限

- 只测了 kernel=25、单层线性头、per-variable 标准化的 DLinear；其他 kernel/individual head 可能略改数值。
- 只评估 S 96→96；未做伪影掩码；test 是冬季。
- Anchored Ridge 与 Anchored DLinear 的差异小于 0.2%，没有做配对 bootstrap（Ridge 预测未保存）；但 seed std 与逐点曲线都支持“打平”的结论。
