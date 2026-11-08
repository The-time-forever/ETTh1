# Attention 消融：PatchTST vs No-Attention Patch Model（S 96→96）

> 脚本：`scripts/patchtst_noattn_s96_h96.py`
> 数据：官方 12/4/4；`seq_len=96`、`pred_len=96`；S 模式，仅 OT
> 两个模型只替换 token mixing 模块，其余结构/训练协议完全一致
> 3 seeds；MSE / Adam / val MAE 早停

## 1. 消融设计

| 模型 | Token mixing | 其余结构 |
|---|---|---|
| **Base PatchTST** | Self-Attention（4 heads） | Patch Embedding + FFN，3 层，d_model=16，d_ff=32 |
| **No-Attention Patch Model** | Token-Mixing MLP（跨 patch） | 同样的 Patch Embedding + FFN，3 层，d_model=16，d_ff=32 |

- Patch/stride：16/8；L=96 → 11 个 patch。
- No-Attention 的 token mixing：`(b,n,d) → transpose → Linear(n,48) → GELU → Linear(48,n) → transpose`，
  每个通道独立；再接与 Attention 版完全相同的 FFN、pre-norm、残差、dropout、head。
- **参数量**：Base 24,112；No-Attention **24,193**（+0.34%）。因此差异不能简单归因于“参数更少”或“深度变浅”。
- 两者都使用 Anchored 处理（ℓ_t = 最近 24h OT 均值），训练/早停/seeds 完全一致。

## 2. Test 结果

| 模型 | MSE | MAE | bias | 步内去偏 MSE | 步内去偏 MAE |
|---|---:|---:|---:|---:|---:|
| B2b Local Mean 96 | 5.6172 | 1.8214 | — | — | — |
| Anchored Ridge（参考） | 4.6127 | 1.6324 | +0.1051 | 4.5981 | 1.6399 |
| **Base PatchTST（mean）** | 4.8422 ± 0.2862 | **1.6438 ± 0.0335** | +0.3009 | 4.6896 | 1.6403 |
| **Base PatchTST（ensemble）** | 4.6319 | **1.6003** | +0.3009 | 4.5343 | 1.6086 |
| **No-Attention（mean）** | **4.7088 ± 0.0280** | 1.6843 ± 0.0047 | −0.0097 | 4.6818 | 1.6760 |
| **No-Attention（ensemble）** | **4.5852** | 1.6596 | −0.0097 | 4.5693 | 1.6531 |

配对比较（No-Attention − Base，同一批 test 窗口）：

| 指标 | 差值 | 95% CI | 胜率 |
|---|---:|---|---:|
| MAE | +0.0593 | [−0.0411, +0.1421] | 44.6% |
| MSE | −0.0467 | [−0.7652, +0.5038] | 44.6% |

## 3. 关键发现

### 3.1 去掉 Attention 没有明显退化

- MAE：No-Attention 比 Base 差 **2.5%（mean）/ 3.7%（ensemble）**；但配对 CI 跨 0。
- MSE：No-Attention 反而好 **2.7%（mean）/ 1.0%（ensemble）**；CI 同样跨 0。
- 逐窗口胜率 44.6%，接近五五开。
- 因此：**没有统计证据表明 self-attention 提供了不可替代的建模能力**；一个简单的 token-mixing MLP 已经能达到同一水平。

### 3.2 两种机制各有小幅优势

- Base 的 MAE 更好（尤其 ensemble 1.6003），但偏差更大（+0.301 vs −0.010）。
- No-Attention 的 MSE 更好、偏差几乎为 0，且 **seed 稳定性远好于 Base**：
  - Base：MSE std 0.2862、MAE std 0.0335；
  - No-Attention：MSE std 0.0280、MAE std 0.0047。
- 这说明 Attention 版可能在某些 seed 上更好，但方差更大；token mixing 更稳定。

### 3.3 与 Anchored Ridge 的关系

| 对比 | MSE 变化 | MAE 变化 |
|---|---:|---:|
| Base PatchTST（ensemble） vs Ridge | −0.42% | +1.97% |
| No-Attention（ensemble） vs Ridge | +0.60% | −1.67% |

- 两者都贴着 Anchored Ridge：Base 的 MAE 更好，No-Attention 的 MSE 更好；
- 没有哪个神经模型稳定超过 Ridge（1.632 / 4.613）。

### 3.4 训练行为

- Base 最优 epoch 7/10/9；No-Attention 5/3/8；两者都在 ~10 epoch 后 val MAE 上升。
- 参数量几乎相同（24,112 vs 24,193），深度相同（3 层），FFN/head 相同；因此这是一个较干净的“Attention vs Token Mixing”对照。

## 4. 回答研究问题

**Attention 在这个 PatchTST/ETTh1 96→96 设置中不是关键组件。**

- 替换为参数量接近的 token-mixing MLP 后，MAE 小幅变差（2.5–3.7%），MSE 小幅变好（1–2.7%），CI 全部跨 0。
- 两者都接近 Anchored Ridge；Attention 没有带来统计上可复现的优势。
- 因此，若继续研究 Attention，应优先改变**规模/历史长度/数据**，而不是继续在同一小模型上做注意力变体；当前任务的瓶颈不在 token mixing 机制。

## 5. 下一步建议

1. **不要把 Attention 作为默认重点**；token mixing 已经证明是可行替代。
2. 如果继续架构研究，优先在 **L=336 长历史**上比较 Base vs No-Attention（本轮 L=96；长历史下 attention 可能更有用，但 PatchTST L336 本身过拟合严重）。
3. 更高优先级仍是：外生变量（天气）、水平/季节建模、概率预测。
4. 若做更大模型，需先解决过拟合/正则化；否则 Attention 与 token mixing 的差异会被 seed 方差淹没。

## 6. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\patchtst_noattn_s96_h96.py"
```

| 文件 | 内容 |
|---|---|
| `outputs/attn_ablation/seed_metrics.csv` | 6 个 seed 的参数量/best epoch/指标 |
| `outputs/attn_ablation/test_metrics.csv` | Base/No-Attention/Ridge/B2b |
| `outputs/attn_ablation/paired_deltas.csv` | No-Attention − Base 的 MAE/MSE/CI |
| `outputs/attn_ablation/per_step_metrics.csv` | 96 步逐点 MSE/MAE/bias |
| `outputs/attn_ablation/segment_metrics.csv` | 4 段 skill |
| `outputs/attn_ablation/summary.json` | 协议、seed、test、paired、segment |
| `figures/01_training_curves.png` | Base vs No-Attention 训练曲线 |
| `02_test_mae.png` | 消融结果 |
| `03_per_step_mae.png` | 逐点 MAE |
| `04_per_step_bias.png` | 逐点 bias |
| `05_segment_skill.png` | 分段 skill |
| `06_example_forecast.png` | test 窗口示例 |
| `07_seed_variability.png` | seed 分布 |

## 7. 局限

- 只测了一个小配置（d_model=16、3 层、11 patches、96→96）；更大模型/更长历史下 Attention 的作用可能不同。
- Token-mixing MLP 与 Attention 参数接近，但不是严格同分布/同归纳偏置；GELU token mixing 与 content-dependent attention 机制不同。
- Base 的 seed 方差较大；单个 seed 的优劣不应过度解释。
- 只评估 S 模式、H=96、单一切分；未做伪影掩码。
