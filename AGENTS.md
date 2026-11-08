# AGENTS.md

## 项目

`D:\Document\ETTh1` 是一个单数据集工作目录：中国某省电力变压器的小时级油温/负荷数据（ETTh1），用于长序列时序预测（LSTF）的基准研究。当前阶段只有数据、调研文档与两个只读分析脚本（数据质量检查 + EDA），**没有模型代码、没有训练实验**。

## Python / ML 环境（重要）

本机唯一 ML 环境是 conda 的 `ml` 环境，**没有加入 PATH**。因此：

- 一律使用绝对路径调用解释器，不要直接运行 `python`、`pip`、`conda`（会失败或指向错误解释器）。
- 不要新建 venv、不要使用系统 Python、不要改动 base 环境。
- 需要装包时用该环境的 pip：`& "D:\SoftWare\Anaconda\envs\ml\python.exe" -m pip install <pkg>`。

| 项 | 绝对路径 |
|---|---|
| Python 解释器 | `D:\SoftWare\Anaconda\envs\ml\python.exe` |
| conda | `D:\SoftWare\Anaconda\Scripts\conda.exe` |
| conda 激活脚本 | `D:\SoftWare\Anaconda\condabin\conda.bat` |

PowerShell 示例：

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" -c "import torch; print(torch.__version__)"
& "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\train.py
```

cmd 示例：

```bat
"D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\train.py
```

环境事实（2026-11-08 实测）：

- Python 3.12.15（Anaconda 打包）
- numpy 2.5.4、pandas 3.1.0、scipy 1.18.1、scikit-learn 1.9.1、matplotlib 3.11.2、seaborn 0.13.2
- torch 2.14.1+cu130、torchvision 0.29.1+cu130，CUDA 13.0 可用
- lightgbm 4.7.0、xgboost 3.4.1、tqdm 4.70.1、ipykernel 7.4.0
- 未安装：statsmodels、jupyterlab、transformers、pyarrow（需要时先确认再装）
- GPU：NVIDIA GeForce RTX 5090 Laptop GPU，24 GB 显存，驱动 617.14

快速自检：

```powershell
& "D:\SoftWare\Anaconda\envs\ml\python.exe" -c "import pandas, numpy, torch; print(pandas.__version__, numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

## 数据

- 唯一数据文件：`ETTh1.csv`（17,420 行，2016-07-01 00:00 — 2018-06-26 19:00，小时级，无 NaN、无重复时间戳；但缺失被编码为常量/0，见下）。
- 列：`date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`；`OT`（油温）是通常的预测目标。
- SHA256：`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- 统计、基准划分与注意事项：`DATASET.md`
- 数据质量检查结论与已知伪影：`DATA_QUALITY.md`（脚本 `scripts/data_quality_check.py`，输出 `outputs/data_quality/`）
- 探索性分析与预测难度基线：`EDA.md`（脚本 `scripts/eda_etth1.py`，输出 `outputs/eda/`）
- 论文与背景：`RELATED_PAPERS.md`；BibTeX：`references.bib`

## 约定

- 任何读写数据的脚本都用 `ml` 环境运行；系统 Python 没有 pandas。
- 原始 CSV 视为只读，不要覆盖或就地清洗。
- 缺失在原始数据中常被编码为常量/0（每月 31 日整日冻结、2016-12 负荷零值停机、LUFL/LULL 141 h 卡死），不要把这些 0 当成真实观测；详见 `DATA_QUALITY.md`。
- 新脚本放 `scripts/`，输出（图、表、日志、checkpoint）放 `outputs/`，不要污染根目录。
- 质量检查脚本：`& "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\data_quality_check.py`。
- 若做预测实验，遵守 `DATASET.md` 里的 12/4/4 个月划分，标准化只在训练段拟合，指标为反归一化后的 MSE/MAE。
- 本阶段只做数据理解与文献调研；未获明确指示前不要启动训练实验。
