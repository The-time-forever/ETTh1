# MS 模式 Anchored Ridge：96h × 7 → 未来 96h OT

> 脚本：`scripts/ridge_ms_s96_h96.py`
> 划分：官方 12/4/4；`seq_len=96`、`pred_len=96`，直接多步
> 目标：OT；输入：HUFL/HULL/MUFL/MULL/LUFL/LULL/OT（7 通道）
> 水平锚定：ℓ_t = 输入窗口最后 24h OT 均值
> 标准化：每个变量只用 train 统计量（7 个均值/标准差）

## 1. 研究问题

前面的 S 模式只使用 OT 历史。本轮只增加 6 个历史负载通道，保持：

- 模型：Anchored Ridge（与 S 模式相同）；
- 输出：未来 96h OT；
- 锚定：最近 24h OT 均值；
- 标准化、alpha 选择、评估协议不变。

回答：

> 历史负载数据能否提供仅靠 OT 历史无法获得的预测信息？

## 2. 对照设计

| 模型 | 输入 | 标准化 | 说明 |
|---|---|---|---|
| **S_control** | 96h OT | per-variable（OT） | 与 MS 完全相同的预处理/锚定，仅 OT |
| **MS** | 96h × 7 | per-variable（7 通道） | 本轮主实验 |
| S_existing | 96h OT | per-lag（96 列） | 之前已提交的 S 模式 Anchored Ridge |
| B1 / B2a | — | — | 持续性与 24h 均值基线 |

- S_control 与 MS 的唯一区别是 6 个负载通道。
- S_control 与 S_existing 用来确认“per-variable vs per-lag 标准化”的影响可忽略。
- alpha 网格：`{0.1, 1, 10, 100, 1000}`，按 val MAE 选择。

## 3. Alpha 选择

| 模型 | 最佳 alpha | val MAE | test MAE |
|---|---:|---:|---:|
| S_control | 100 | 2.2561 | 1.6327 |
| MS | **1000（网格边界）** | 2.4265 | 1.7953 |

- MS 需要非常强的正则化（1000）才能把误差压下来，val/test 曲线在 1000 处仍在下降。
- 这说明 672 个负载+OT 特征高度共线/噪声大；更强正则只会把 MS 进一步压向 B2a，而不是产生超过 S 的增益。
- S_control 的 alpha 曲线在 100 附近已稳定。

## 4. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE |
|---|---:|---:|---:|---:|
| B1 Persistence | 5.8326 | 1.8654 | +0.1215 | 5.8119 |
| B2a Local Mean 24 | 5.4544 | 1.8198 | +0.1443 | 5.4277 |
| S_existing（per-lag） | **4.6127** | **1.6324** | +0.1051 | **4.5981** |
| S_control（per-variable） | **4.6140** | **1.6327** | +0.1051 | **4.5994** |
| **MS（7 channels）** | **5.4115** | **1.7953** | +0.0213 | **5.4089** |

相对变化：

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| S_control vs S_existing | −0.03% | −0.01% |
| **MS vs S_control** | **−17.29%** | **−9.96%** |
| MS vs B2a | +0.79% | +1.35% |

- S_control 与 S_existing 几乎完全一致（MSE 4.6140 vs 4.6127，MAE 1.6327 vs 1.6324），说明 per-variable vs per-lag 标准化的差异可忽略，S vs MS 的比较是干净的。
- **MS 显著差于 S_control**：MSE 5.41 vs 4.61，MAE 1.795 vs 1.633。
- MS 与 B2a 基本打平（略好一点点，但 CI 跨 0）。
- MS 的 bias 虽接近 0（+0.021），但步内去偏 MSE 5.409 仍远差于 S 的 4.599——**负载通道破坏了“相对变化”的形状，而不是水平**。

## 5. 逐窗口与显著性

| 对比 | MAE 差均值 | 95% CI | MSE 差均值 | 95% CI |
|---|---:|---|---:|---|
| MS − S_control | +0.163 | **[+0.113, +0.226]** | +0.798 | **[+0.526, +1.135]** |
| MS − B2a | −0.025 | [−0.153, +0.131] | −0.043 | [−0.717, +0.751] |

- MS 相对 S 的退化在 MAE 与 MSE 上都显著（CI 不跨 0）。
- MS 相对 B2a 的差异不显著；逐窗口胜率 **50.8%**（几乎五五开）。
- MS 相对 S_control 的逐窗口胜率仅 **27.9%**——加入负载后，大多数窗口都变差。

## 6. 分段误差

| 步长段 | S_control MAE | MS MAE | B2a MAE | MS vs S MAE | MS vs B2a MAE |
|---|---:|---:|---:|---:|---:|
| 1–24 | **1.130** | 1.267 | 1.374 | **−12.1%** | +7.8% |
| 25–48 | **1.629** | 1.748 | 1.816 | −7.3% | +3.8% |
| 49–72 | **1.843** | 2.009 | 2.000 | −9.0% | −0.5% |
| 73–96 | **1.930** | 2.157 | 2.089 | −11.8% | −3.3% |

MSE 相对变化：

| 步长段 | MS vs S MSE | MS vs B2a MSE |
|---|---:|---:|
| 1–24 | −18.0% | +19.5% |
| 25–48 | −13.3% | +7.2% |
| 49–72 | −16.4% | −3.6% |
| 73–96 | −20.6% | −8.9% |

- 在所有分段，MS 都差于 S_control。
- MS 只在 1–48 步优于 B2a；49 步之后连 B2a 都不如。
- 这与“负载提供短期信息、但不足以超过 OT 自身历史”的解释一致；但更直接的是：**负载特征在 96h 任务上没有稳定增益**。

## 7. 通道重要性

MS 的标准化系数平均绝对值：

| 通道 | mean \|coef\| |
|---|---:|
| HULL | 0.0169 |
| MULL | 0.0146 |
| **OT** | 0.0128 |
| LULL | 0.0083 |
| LUFL | 0.0049 |
| MUFL | 0.0034 |
| HUFL | 0.0032 |

- 负载通道确实获得了非零权重（HULL/MULL 最大），但模型性能反而下降。
- 六个负载的系数质量合计约 0.051，是 OT（0.013）的 4 倍；这不是“负载无用”的系数证据，而是**过拟合/共线性的证据**：模型把权重分给了不稳定的负载关系。
- 与 EDA 的结论一致：负荷–OT 相关跨期变号、HUFL–MUFL 与 HULL–MULL 强共线；线性模型很容易学到不迁移的模式。

## 8. 回答“历史负载是否提供增量信息”

**没有可复现的增量价值；加入负载反而显著变差。**

- MS vs S_control：MSE −17.3%、MAE −10.0%，CI 不跨 0。
- MS 仅与 B2a 打平，远差于 S 模式。
- MS 的系数确实使用了负载，但性能没有提升，说明这些关系不迁移（与 EDA 的不稳定相关一致）。
- 因此，之前 S 模式的瓶颈**不是“缺少负载输入”**；至少在线性 Anchored Ridge 框架下，增加 6 个负载通道没有帮助。

## 9. 下一步建议

1. **不要在 Ridge 上继续加负载通道**；如果还要用负载，先做特征选择/PCA 或按 horizon 限制（1–24 步），并重新验证。
2. **检查短 horizon**：MS 在 1–48 步略优于 B2a，但仍不如 S；可以做 H=24 的专门实验，确认负载是否有短期价值。
3. **转向更有希望的外生变量**：环境温度/天气可能比负载更有信息量；当前数据没有这个变量。
4. **MS + 非线性/序列模型**：可以作为后续对照，但第一轮线性证据是负的，优先级应低于日历/天气特征。
5. **掩码伪影**：2016-12 停机、31 日冻结、2017-07 通道冻结可能进一步放大负载特征的噪声；MS 实验应在掩码版本上复核。

## 10. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\ridge_ms_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/ridge_ms/alpha_selection.csv` | S_control/MS × alpha × val/test 指标 |
| `outputs/ridge_ms/test_metrics.csv` | B1/B2a/S_existing/S_control/MS 对比 |
| `outputs/ridge_ms/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/ridge_ms/segment_metrics.csv` | 4 段指标与相对提升 |
| `outputs/ridge_ms/channel_importance.csv` | 7 通道的标准化系数重要性 |
| `outputs/ridge_ms/summary.json` | 协议、alpha、test/segment、胜负、通道重要性 |
| `figures/01_alpha_selection.png` | alpha 曲线（MS 在 1000 仍下降） |
| `02_test_metrics.png` | S vs MS 的 MAE/MSE |
| `03_per_step_mae.png` | 逐点 MAE（MS 仅前期优于 B2a） |
| `04_per_step_bias.png` | 逐点 bias |
| `05_segment_improvement.png` | 分段相对 S/B2a 的提升 |
| `06_example_forecast.png` | test 窗口示例 |
| `07_channel_importance.png` | 通道系数重要性 |

## 11. 局限

- 只测了线性 Anchored Ridge 与一个 MS 特征构造（96h 原始负载 + OT）；非线性 MS 模型或其他负载特征（差分、滚动统计、PCA）可能不同。
- 标准化为 per-variable；per-lag 的 S_existing 与 S_control 几乎一致，说明这个选择不影响主要结论。
- alpha 网格上界 1000 是 MS 的最优点；更强正则可能让 MS 更接近 B2a，但不会超过 S。
- 未做伪影掩码；test 是冬季；只评估 H=96。
- 负载与 OT 的关系在 EDA 中已被证明跨期不稳定；本结果是该现象的模型层面确认。
