# ETTh1 相关工作与论文清单

> 用途：记录 ETTh1 的背景文献、模型主线与领域应用，供后续选题与写作引用。
> 检索与核验日期：2026-11-08。BibTeX 见同目录 `references.bib`。
> 引用次数来自 OpenAlex（检索当日快照），仅用于判断影响力，不作为质量结论。

## A. 数据集与基准

| 论文 |  venue / 年份 | 作用 | 标识 |
|---|---|---|---|
| Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting | AAAI 2021（最佳论文） | **ETDataset / ETTh1 的出处**；提出 ProbSparse 注意力、自注意力蒸馏、生成式解码器 | DOI: 10.1609/aaai.v35i12.17325；arXiv:2012.07436；被引 8,524 |
| Informer 官方代码与数据仓库 | GitHub | `zhouhaoyi/ETDataset`（数据）、`zhouhaoyi/Informer2020`（代码） | — |
| Time-Series-Library (TSLib) | GitHub / 持续更新 | 事实上的统一评测框架；定义 ETT 的 12/4/4 划分与 MSE/MAE 流程 | `thuml/Time-Series-Library` |

## B. 长序列预测（LSTF）模型主线

以下模型均在 ETTh1 上报告过结果，是最常见的对比基线。

| 论文 | venue / 年份 | 关键思路 | 与 ETTh1 的关系 | 标识 |
|---|---|---|---|---|
| Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting | NeurIPS 2021 | 序列分解 + 自相关机制，替代逐点自注意力 | 其数据管道与 12/4/4 划分被后续工作广泛沿用 | arXiv:2106.13008；被引 2,208 |
| FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting | ICML 2022 | 频域增强 + 分解，随机/低秩频域注意力 | ETTh1 上的代表性强基线 | arXiv:2201.12740；被引 1,366 |
| Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting | ICLR 2022 (Oral) | 金字塔式多分辨率注意力，O(L) 复杂度 | 以 ETT 等为长程预测基准 | ICLR 2022；作者 Liu, Yu, Liao, Li, Lin, Liu, Dustdar |
| Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting | ICLR 2023 | 跨维度依赖；DSW 嵌入 + 两阶段注意力 | 面向多变量 ETT 的通道建模路线 | ICLR 2023；作者 Zhang & Yan |
| Are Transformers Effective for Time Series Forecasting?（DLinear） | AAAI 2023 | 简单线性 + 分解即可超过多数 Transformer | 对 ETTh1 上“Transformer 必要性”的关键质疑，必跑基线 | DOI: 10.1609/aaai.v37i9.26317；被引 4,235 |
| A Time Series is Worth 64 Words: Long-term Forecasting with Transformers（PatchTST） | ICLR 2023 |  patch 化 + 通道独立 | ETTh1 上强基线；说明通道混合收益有限 | DOI: 10.48550/arXiv.2211.14730；被引 2,062 |
| TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ICLR 2023 | FFT 发现周期，将 1D 序列折叠为 2D 变化建模 | 通用任务框架，ETT 为主要基准 | DOI: 10.48550/arXiv.2210.02186；被引 1,349 |
| iTransformer: Inverted Transformers Are Effective for Time Series Forecasting | ICLR 2024 | 将变量维作为 token，注意力跨变量 | 多变量 ETT 的当前强基线之一 | DOI: 10.48550/arXiv.2310.06625；被引 1,733 |
| TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting | KDD 2023 | 时间维与通道维交替 MLP-Mixer | 轻量级强基线 | DOI: 10.1145/3580305.3599533；被引 379 |
| NHITS: Neural Hierarchical Interpolation for Time Series Forecasting | AAAI 2023 | 多速率采样 + 分层插值 | 长horizon 强基线 | DOI: 10.1609/aaai.v37i6.25854；被引 760 |
| SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction | NeurIPS 2022 | 递归下采样 + 样本卷积交互 | ETTh1 上的非 Transformer 基线 | DOI: 10.52202/068431-0421；被引 532 |
| TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting | ICLR 2024 | 可分解多尺度混合 | 在 ETT 上同时报告 look-back 96 / 搜索设置 | arXiv:2405.14616 |
| TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | ICLR 2024 | 外生变量预测范式 | 与 ETTh1“负荷 → 油温”的外生变量设定直接相关 | arXiv:2402.19072 |
| Is Mamba Effective for Time Series Forecasting?（S-Mamba） | Neurocomputing 2025 | 状态空间模型（Mamba）用于时序 | ETTh1 上的新一代对照 | DOI: 10.1016/j.neucom.2024.129178；被引 358 |

## C. 综述与方法论

| 论文 | venue / 年份 | 用途 | 标识 |
|---|---|---|---|
| Transformers in Time Series: A Survey | IJCAI 2023 | Transformer 时序模型的分类与脉络 | DOI: 10.24963/ijcai.2023/759；被引 1,921 |
| Long sequence time-series forecasting with deep learning: A survey | Information Fusion 2023 | LSTF 专门综述，含 ETT 基准讨论 | DOI: 10.1016/j.inffus.2023.101819；被引 374 |
| Forecast evaluation for data scientists: common pitfalls and best practices | Data Mining and Knowledge Discovery 2022 | 预测评估的常见错误（泄漏、划分、指标） | DOI: 10.1007/s10618-022-00894-5；被引 407 |
| Foundation Models for Time Series Analysis: A Tutorial and Survey | KDD 2024 | 时序基础模型与零样本评测 | DOI: 10.1145/3637528.3671451；被引 368 |

## D. 面向“变压器油温”的领域应用

| 论文 | venue / 年份 | 内容 | 标识 |
|---|---|---|---|
| A new deep learning architecture with inductive bias balance for transformer oil temperature forecasting | Journal of Big Data 2023 | 面向变压器油温的归纳偏置平衡架构 | DOI: 10.1186/s40537-023-00745-0 |
| Data-driven forecasting of transformer oil temperature using deep learning and ensemble models on multivariate time-series data | Discover Applied Sciences 2026 | 多变量油温预测：深度学习 + 集成 | DOI: 10.1007/s42452-026-09536-7 |
| The Transformer Oil Temperature Prediction Method Based on the CEEMDAN-PatchTST-Transformer Hybrid Model | ISCCN 2025 | CEEMDAN 分解 + PatchTST + Transformer 混合模型 | DOI: 10.1145/3732945.3732960 |

> 注意：第三类文献多数使用自有变压器数据；ETTh1 本身来自变压器油温场景，但在方法论文中主要作为通用 LSTF 基准使用，而非领域专用数据集。

## E. 建议阅读顺序

1. **Informer (2021)**：搞清楚数据来源、任务定义、ETTh1 的列语义与原始评测设置。
2. **DLinear (2023)** + **PatchTST (2023)**：理解“简单模型/通道独立”为何在 ETT 上强，避免重复造轮子。
3. **Autoformer (2021) / FEDformer (2022) / TimesNet (2023)**：分解、频域、2D 变化三条主流归纳偏置。
4. **iTransformer (2024) / TimeMixer (2024) / TimeXer (2024)**：跨变量建模、多尺度、外生变量三个近期方向。
5. **综述 + 评估陷阱 (2022–2024)**：确定实验协议与统计显著性，避免仅比较小数点后两位。
6. **领域油温论文 (2023–2026)**：把通用方法与变压器运维需求（越限预警、极限负载）对齐。

## F. 检索记录与限制

- **OpenAlex Works API**：`search=` 主题检索；`filter=cites:W3177318507&sort=cited_by_count:desc` 获取 Informer 的 8,443 篇被引文献中引用量最高的 25 篇（2026-11-08）。引用次数取自该快照。
- **DOI 元数据**：通过 `https://doi.org/<DOI>` 的 `Accept: application/x-bibtex` 内容协商获取（Crossref / DataCite），作者与页码以此为准。`references.bib` 中的条目均由该方式生成或经 OpenReview/ICLR 页面核对。
- **Semantic Scholar**：本次返回 HTTP 429（限流），未采用。
- **DBLP**：受 Anubis 反爬验证拦截，未采用。
- **ICLR 论文**（Pyraformer、Crossformer）无 Crossref DOI，作者与会议信息经 OpenReview / ICLR virtual 页面核对。
- 未做 exhaustive 检索；本清单是“主线 + 领域应用”的策展列表，不宣称覆盖全部 ETTh1 相关论文。
