# 论文构建与本轮修改

主文件为 `main.tex`，阅读稿为 `main.pdf`。本轮依据 Git 中的数据检查、基线、各模型与消融实验记录扩写，保留原始 CSV 和训练结果。

## 修改内容

- 正文与图表使用“最后值”“24 h 均值”“96 h 均值”等规则名称，清理 B1/B2a/B2b、plain、small 等开发简称。S/MS 等预测任务缩写在首次出现处定义；复现路径保留在附录。
- 方法部分补充窗口构造、训练尺度、岭回归目标函数、GRU/LSTM 状态更新、DLinear 分解、PatchTST 张量尺寸与编码块、片段混合 MLP、SegRNN 共享 GRU 和并行预测过程。
- 区分 64 维两层与 16 维三层 PatchTST，列出实际参数量；区分三种子指标均值与预测集成。
- 重绘四幅旧图，新增数据划分图和 PatchTST/SegRNN 结构图。结果比较采用点和标准差，避免截断柱形夸大差异。长预测使用共同起点曲线，并补充逐预测步曲线。
- 修正 96 h Ridge 的目标标准化说明、最强统计基线、参数量比例、第 1 轮取得最优参数与实际停止训练的区别。
- 删除“零系数保证安全修正”“Ridge 始终最优”“远期误差饱和”等超出现有结果的判断。明确长历史容量变化、伪影敏感性尚待计算及长预测对照范围。

语言使用 `D:\Document\Memory\chinese-writing\SKILL.md`。直接陈述模型、处理过程和结果，统一术语，缩短长句，清理假想反驳与开发过程用语。检查器为 0 ERROR、3 WARN；三项均为 192/336/720 等任务长度的必要重复。正文的开发代号已清理；英文摘要保留常用术语 anchored，附录保留实际路径。

## 构建

在工作空间根目录运行：

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\paper_figures.py
Push-Location paper
latexmk -norc -xelatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

图表脚本读取保存的 CSV，核验原始 CSV SHA256，在 `outputs/paper/` 写入图、数据来源清单，并同步论文使用的图和两个主结果表到 `paper/figures/`、`paper/tables/`。该过程不启动训练。图片数据源以 `outputs/paper/source_manifest.json` 为准；其余数值说明参见各实验目录的 `summary.json`、`seed_metrics.csv`、`paired_deltas.csv` 及实验脚本。

## 检查

- 原始数据 SHA256 与项目记录一致。
- XeLaTeX/BibTeX 构建通过，阅读稿为 15 页；最终日志无未定义引用、Overfull 或 Underfull 提示。
- 已将 PDF 全页栅格化检查，重点核对公式、结构图、主结果表及附录路径换行。
- `outputs/paper/language_check.json` 保存语言检查结果；`git diff --check` 用于检查补丁空白问题。

原稿与各轮实验历史保留在 Git 中，可通过 `0999185` 比较本轮论文修改。

## 中文预览字体

本稿采用 `fontset=windows`，与可在内置预览中正常显示的花卉分类论文相同。对比发现，原 Fandol 中文字体的 `pdffonts` 检查为 `CID Type 0C`、`emb=yes`、`uni=no`；花卉论文使用的宋体、黑体为 `CID TrueType`、`emb=yes`、`uni=yes`。

切换后，本稿宋体、黑体和楷体均已嵌入并带 Unicode 映射。重新编译后保持 15 页，日志无溢出和未定义引用。本地渲染检查通过；内置预览的显示结果需以重新打开文件后的实际页面为准。
