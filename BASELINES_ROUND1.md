# 第一轮基线结果（B0–B3）

> 脚本：`scripts/baselines_round1.py`
> 划分：官方 12/4/4（`SPLITS.md` / `outputs/splits/`）
> 口径：`seq_len=96`，direct multi-horizon，`pred_len ∈ {96,192,336,720}`，原始 ℃
> 数据：官方 val / test 目标块；未做标准化、未做伪影掩码
> 指标：MSE / RMSE / MAE。**MAE(H) 是 1..H 所有步长的平均，不是第 H 步的单点误差**

## 1. 五组配置

| 编号 | 基线 | 预测规则 | 研究目的 |
|---|---|---|---|
| B0 | Global Mean | 用训练集 OT 均值预测 | 最基本的参照，检验分布漂移量级 |
| B1 | Persistence | 用最后一个观测值预测 | 检验油温持续性 |
| B2a | Local Mean 24 | 最近 24 h 均值 | 检验短期平滑 |
| B2b | Local Mean 96 | 最近 96 h 均值 | 检验更长窗口的平滑 |
| B3 | Seasonal Naive | 重复最后 24 h 的变化模式 | 检验日周期 |

实现细节：

- B2a = `mean(OT[t-23 : t+1])`；B2b = `mean(OT[t-95 : t+1])`，其中 `t` 是输入窗口最后一个观测。
- B3 = 取 `OT[t-23 : t+1]` 作为 24 点模式，按 `h mod 24` 重复到整个 horizon。
- 所有基线都是**直接多步**（direct multi-horizon）：每个窗口一次输出 H 步，不做递归。

## 2. 测试集结果（官方 test，2017-10-24 — 2018-02-20）

### 2.1 MAE（℃）

| 模型 | H=96 | H=192 | H=336 | H=720 |
|---|---:|---:|---:|---:|
| B0 Global Mean | 12.3241 | 12.4076 | 12.5283 | 12.7577 |
| B1 Persistence | **1.8654** | 2.1627 | 2.4336 | 2.6007 |
| B2a Local Mean 24 | **1.8198** | 2.0819 | 2.3416 | 2.4878 |
| B2b Local Mean 96 | 1.8214 | **2.0632** | **2.2751** | **2.3952** |
| B3 Seasonal Naive 24 | 1.9318 | 2.1733 | 2.4172 | 2.5660 |

### 2.2 MSE（℃²）

| 模型 | H=96 | H=192 | H=336 | H=720 |
|---|---:|---:|---:|---:|
| B0 Global Mean | 161.4961 | 163.2060 | 165.8165 | 170.4930 |
| B1 Persistence | 5.8326 | 7.7441 | 9.5386 | 10.8779 |
| B2a Local Mean 24 | **5.4544** | 7.1608 | 8.7678 | 9.9171 |
| B2b Local Mean 96 | 5.6172 | **7.0268** | **8.2159** | **9.2025** |
| B3 Seasonal Naive 24 | 6.0169 | 7.7114 | 9.3330 | 10.5450 |

### 2.3 相对 persistence 的 MSE skill

`skill = 1 − MSE_model / MSE_B1`，正数表示优于 B1。

| 模型 | H=96 | H=192 | H=336 | H=720 |
|---|---:|---:|---:|---:|
| B2a Local Mean 24 | +6.48% | +7.53% | +8.08% | +8.83% |
| B2b Local Mean 96 | +3.69% | +9.26% | +13.87% | **+15.40%** |
| B3 Seasonal Naive 24 | **−3.16%** | +0.42% | +2.16% | +3.06% |

## 3. 验证集对照（官方 val，2017-06-26 — 2017-10-23）

MAE（℃）：

| 模型 | H=96 | H=192 | H=336 | H=720 |
|---|---:|---:|---:|---:|
| B0 Global Mean | 4.2988 | 4.2649 | 4.1548 | 3.8555 |
| B1 Persistence | 2.5988 | 2.9185 | 3.1212 | 3.6345 |
| B2a Local Mean 24 | 2.5488 | 2.7889 | 2.9539 | 3.4822 |
| B2b Local Mean 96 | 2.6098 | 2.7004 | 2.8561 | 3.3782 |
| B3 Seasonal Naive 24 | 2.6215 | 2.8568 | 3.0297 | 3.5475 |

- B0 在 val 上 MAE 约 4.3 ℃，在 test 上却是 12.3–12.8 ℃：**这不是模型变化，而是 train→test 的水平漂移**（train 均值 17.13，val 15.58，test 4.85）。
- B1/B2a/B2b 的排序在 val 与 test 上基本一致，说明“平滑优于单点”不是 test 偶然。

## 4. 关键发现

1. **B0 只能做 sanity check。** test 上 MAE 12–13 ℃，说明任何模型都必须显式处理冬季水平漂移；B0 在 val 上看似“正常”，恰好暴露了切分的季节偏差。
2. **B1 是真正的第一道门槛。** MAE 从 1.87 ℃（H=96）升到 2.60 ℃（H=720），误差增长很慢；这是油温强持续性的直接体现。
3. **短期平滑有小幅收益，长期平滑收益更大。**
   - H=96：B2a 最好（1.8198），比 B1 低 2.4% MAE、6.5% MSE。
   - H=192/336/720：B2b 最好（2.0632 / 2.2751 / 2.3952），MSE 比 B1 低 9.3% / 13.9% / 15.4%。
   - 这说明 OT 的短期波动主要是噪声，平滑 24–96 h 可以降噪；长 horizon 下“近期水平”比“最后一点”更稳。
4. **B3 日周期基线没有超过 B1。** H=96 时 MSE 反而比 B1 差 3.2%，MAE 差 3.6%；只在长 horizon 上比 B1 好 2–3%。这与 EDA 的结论一致：**日周期很弱，重复最后 24 h 模式不是有效信号**。
5. **1 月是唯一 B2b 明显变差的月份。** H=96 分月 MAE：

| 月份 | B1 | B2a | B2b | B3 |
|---|---:|---:|---:|---:|
| 2017-10 | 2.246 | 2.002 | **1.678** | 2.030 |
| 2017-11 | 2.060 | 2.019 | **2.009** | 2.122 |
| 2017-12 | 1.617 | 1.540 | **1.448** | 1.612 |
| 2018-01 | **1.992** | 2.071 | 2.359 | 2.234 |
| 2018-02 | 1.545 | 1.411 | **1.225** | 1.561 |

   - B2b 在 10/11/12/2 月最好，但在 1 月最差：96 h 均值在快速降温/寒潮时滞后，24 h 均值或 persistence 更快适应。
   - 这提示下一轮需要“**可自适应水平的平滑**”，而不是固定 96 h 平均。

## 5. 对下一轮的约束

- 第一轮的地板不是 B1，而是 **B2b（长 horizon）/ B2a（H=96）**。后续模型必须同时报告对 B1 与对 B2b 的 skill。
- 下一轮优先做“**水平 + 日历**”的强基线，而不是神经模型：
  - B4：Local level + hour/month 效应（在近期均值上叠加日历项）；
  - B5：Ridge / AR(p) on lags + calendar（直接多步）；
  - B6：DLinear（可学习趋势/季节分解）；
  - B7：LightGBM（滞后 + 滚动 + 日历特征）。
- 在这些基线出来之前，不上 Transformer；如果 B5/B6 已接近最好，神经模型的收益需要严格证明。
- 需要补两项稳健性检查：
  - 伪影掩码（31 日冻结、2016-12 停机、2017-07 冻结）后的排名；
  - 对 B2a/B2b/B3 的差异做 block bootstrap，确认 2–4% 的 MAE 差距是否显著。

## 6. 图

| 图 | 内容 |
|---|---|
| `outputs/baselines/figures/01_error_vs_horizon.png` | 5 组基线 MSE/MAE 随 horizon 变化（log-log） |
| `02_example_forecast.png` | 一个 test 窗口的预测示例（可见 B3 重复 24 h 模式） |
| `03_monthly_mae_h96.png` | H=96 的分月 MAE（1 月 B2b 变差） |
| `04_skill_vs_persistence.png` | B2a/B2b/B3 相对 B1 的 MSE skill |

## 7. 产物与复现

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" "D:\Document\ETTh1\scripts\baselines_round1.py"
```

- `outputs/baselines/round1_metrics.csv`：val/test × horizon × model 的 MSE/RMSE/MAE/skill。
- `outputs/baselines/round1_summary.json`：协议、定义、test/val 表、best-by-horizon。
- `outputs/baselines/monthly_mae_h96.csv`：H=96 分月 MAE。
- `outputs/baselines/figures/`：4 张图。

## 8. 局限

- 只评估了官方 val/test；test 是冬季，结论对季节敏感。
- 未做伪影掩码；月末冻结日可能让 persistence/local mean 略微占优。
- 未做显著性检验；2–4% 的 MAE 差距需要后续确认。
- B3 只测了 24 h 周期；168 h 版本预计更弱（EDA 周周期 < 2.3%），暂不优先。
