# 越复杂越好？

阅读稿为 `main.pdf`，LaTeX 主文件为 `main.tex`。论文以单变量 S 模式的油温预测为研究对象，最终选用 Anchored Ridge（局部水平锚定岭回归），输入 96 h OT 历史，以 24 h 均值为参考。当前阅读稿为 19 页。

## 内容安排

论文按照数据分析与处理、五个统计基线、线性模型构建与比较、非线性模型、实验结果、讨论及最终模型的顺序展开。对照模型的结构细节保留在学术附录。

## 文献与写作依据

- Huang 等的 arXiv:2606.27282 比较 Ridge、局部归一化及非线性模型；论文区分其多变量通道独立设置与本文 OT 单变量任务。
- Informer 官方实现明确 S/MS 的输入输出任务定义。
- DLinear 论文附录 C.1 给出 ETT 单变量比较及分布变化的讨论；NLinear 提供参考水平恢复的直接先例。
- PatchTST 附录 A.3 明确以 ETT 油温作为单变量预测目标。
- Han 等的通道独立研究用于解释独立建模的相关背景；正文区分单一 OT 输入与多通道共享参数。
- 语言遵循 `C:\Users\Adam\.codex\skills\chinese-writing\SKILL.md`；结构参考用户提供的 [Academic Writing Suite](https://github.com/The-time-forever/academic-writing-suite) 及本机 ML 论文写作技能。

具体来源与核验说明见 `outputs/paper/revision_sources.md`。

## 构建

在工作空间根目录运行：

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\paper_figures.py
Push-Location paper
latexmk -norc -xelatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

图表脚本读取已保存的结果，核验原始 CSV SHA256，生成七幅图及两个主结果表。为绘制最终预测曲线，它按原验证集选择的 `alpha=100` 重建既有线性预测器，断言其全测试集 MSE/MAE 与原结果一致；该步骤不启动神经网络训练，不进行新的模型搜索。

输入文件校验值见 `outputs/paper/source_manifest.json`，预测重建结果见 `outputs/paper/solution_verification.json`。曲线的两个首个目标时刻固定为 2017-11-06 与 2018-01-06 的 00:00，按日历指定，未按误差筛选。

## 验证与字体

XeLaTeX/BibTeX 编译及全页渲染检查通过，日志无溢出或未定义引用。补充语言检查器结果为 0 ERROR、3 WARN，三项为 192/336/720 等任务长度的必要重复，见 `outputs/paper/language_check.json`。

采用 `fontset=windows`，宋体、黑体和楷体已嵌入且带 Unicode 映射，与内置预览正常的花卉论文保持相同字体设置。
