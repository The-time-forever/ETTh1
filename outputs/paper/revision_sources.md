# 论文任务与文献核对

本轮将论文组织为 OT 预测问题、已有方法依据、方案构建与最终预测器。完整模型推导保留在学术附录；源文件路径和构建记录移出论文。

## 支持论文表述的原始来源

| 来源 | 已核对内容 | 在论文中的用途 |
|---|---|---|
| [Informer 官方实现](https://github.com/zhouhaoyi/Informer2020#usage) | `features` 的 S 为单变量到单变量，MS 为多变量到单变量，目标默认 OT | 引言中的任务命名，脚注给出官方链接 |
| [Zeng 等，arXiv:2205.13504v3](https://arxiv.org/html/2205.13504v3) | LTSF-Linear、NLinear 末值恢复；附录 C.1 的 ETT 单变量预测及分布变化讨论 | S 模式依据、线性模型候选、局部水平恢复的前人工作 |
| [Nie 等，arXiv:2211.14730v2](https://arxiv.org/html/2211.14730v2#A3) | 附录 A.3 明确以 ETT 油温为单变量预测目标；片段与通道独立设计 | 任务依据及片段模型对照 |
| [Han 等，arXiv:2304.05206](https://arxiv.org/abs/2304.05206) | 通道独立模型把多变量序列拆开建模，讨论容量和分布变化的关系 | 单变量与通道独立的联系和区别 |
| [Jiménez-Navarro 等，2023](https://doi.org/10.1186/s40537-023-00745-0) | SRCNet 面向变压器油温，比较误差、参数量和训练开销 | 引言中的油温预测应用研究 |

引用沿用已核对的 `paper/references.bib` 条目。本文误差为反标准化的 ℃/℃²，不与上述论文表格中的标准化误差直接排名。本文的 64/16 维 PatchTST 指嵌入宽度，与原论文 PatchTST/64 等按片段数的命名不同。

## 写作依据

- `C:\Users\Adam\.codex\skills\chinese-writing\SKILL.md`：直接表达、面向论文读者、控制重复、分析给出判断。
- [Academic Writing Suite](https://github.com/The-time-forever/academic-writing-suite)：采用引言的背景—已有工作—本任务—本文研究顺序，方法说明保留可复现的定义，结尾落到研究结果和方案。
- `C:\Users\Adam\.agents\skills\ml-paper-writing\SKILL.md`：以任务与证据组织叙述，核对引用，区分已有技术与本任务的组合使用。

## 结果核对

- 原始 CSV SHA256 不变。
- 按原验证集的最低 MAE 选择 `alpha=100`，重建既有 96→96 锚定岭回归，MSE/MAE 为 4.612680353051901 / 1.632422411561528，与保存结果相同。
- 保留三种子预测集成在部分指标上更低的结果；最终模型选择表述为具体单模型方案。
- 336 h 历史岭回归的测试 MAE 为 1.624980；结合其配对区间与输入长度，最终方案保留 96 h 输入。
- 新增曲线只用于展示固定日历窗口；评价仍覆盖全部测试窗口。
