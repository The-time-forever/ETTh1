# AGENTS.md

## 项目

`D:\Document\ETTh1` 是一个单数据集工作目录：中国某省电力变压器的小时级油温/负荷数据（ETTh1），用于长序列时序预测（LSTF）的基准研究。当前阶段有数据、调研文档、三个只读分析/准备脚本（质量检查、EDA、官方划分）与基线/Ridge 实验脚本，**没有神经网络训练实验**。

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
- 官方 12/4/4 划分与窗口索引：`SPLITS.md`（脚本 `scripts/make_splits.py`，输出 `outputs/splits/`）
- 第一轮基线（B0–B3）：`BASELINES_ROUND1.md`（脚本 `scripts/baselines_round1.py`，输出 `outputs/baselines/`）
- Ridge（S 模式 96→96）：`RIDGE_S96_H96.md`（脚本 `scripts/ridge_s96_h96.py`，输出 `outputs/ridge/`）
- 局部水平锚定 Ridge（S 模式 96→96）：`RIDGE_LEVEL_S96_H96.md`（脚本 `scripts/ridge_level_s96_h96.py`，输出 `outputs/ridge_level/`）
- 轻量 MLP（S 模式 96→96）：`MLP_S96_H96.md`（脚本 `scripts/mlp_s96_h96.py`，输出 `outputs/mlp/`）
- GRU（S 模式 96→96）：`GRU_S96_H96.md`（脚本 `scripts/gru_s96_h96.py`，输出 `outputs/gru/`）
- LSTM（S 模式 96→96）：`LSTM_S96_H96.md`（脚本 `scripts/lstm_s96_h96.py`，输出 `outputs/lstm/`）
- PatchTST（S 模式 96→96）：`PATCHTST_S96_H96.md`（脚本 `scripts/patchtst_s96_h96.py`，输出 `outputs/patchtst/`）
- MS Anchored Ridge（96h×7→96h OT）：`RIDGE_MS_S96_H96.md`（脚本 `scripts/ridge_ms_s96_h96.py`，输出 `outputs/ridge_ms/`）
- Horizon 192（S 96→192，Ridge/PatchTST）：`S96_H192.md`（脚本 `scripts/forecast_s96_h192.py`，输出 `outputs/h192/`）
- Horizon 336（S 96→336，Anchored Ridge）：`S96_H336.md`（脚本 `scripts/forecast_s96_h336.py`，输出 `outputs/h336/`）
- 论文与背景：`RELATED_PAPERS.md`；BibTeX：`references.bib`

## 约定

- 任何读写数据的脚本都用 `ml` 环境运行；系统 Python 没有 pandas。
- 原始 CSV 视为只读，不要覆盖或就地清洗。
- 缺失在原始数据中常被编码为常量/0（每月 31 日整日冻结、2016-12 负荷零值停机、LUFL/LULL 141 h 卡死），不要把这些 0 当成真实观测；详见 `DATA_QUALITY.md`。
- 新脚本放 `scripts/`，输出（图、表、日志、checkpoint）放 `outputs/`，不要污染根目录。
- 质量检查脚本：`& "D:\SoftWare\Anaconda\envs\ml\python.exe" scripts\data_quality_check.py`。
- 若做预测实验，遵守 `DATASET.md` 里的 12/4/4 个月划分，标准化只在训练段拟合，指标为反归一化后的 MSE/MAE。
- 本阶段做数据理解、文献调研、基线与线性模型；未获明确指示前不要启动神经网络训练。
- 每轮实验/步骤完成后自动提交；可用 `/commit-round <name>` 执行标准流程（见下节）。

## 提交约定（项目级）

本项目要求：**每完成一轮实验或一个明确步骤，立即提交一次，不必再次询问**。这是对本项目默认“仅在用户要求时提交”的覆盖。

一轮的完成标准：

1. 脚本成功运行，输出写入 `outputs/`。
2. 对应的报告/说明（如 `*_ROUND*.md`）已更新。
3. `git status --short` 与 `git diff --stat` 已检查。

提交流程：

1. 只 stage 本轮相关文件。
2. 不提交 `__pycache__/`、`*.pyc`、临时文件、密钥、checkpoint（除非明确需要）。
3. 提交信息：
   - 标题：`<type>: <summary>`
   - `type`：`data` / `exp` / `docs` / `feat` / `fix` / `chore`
   - 正文：脚本、输出目录、关键配置（seq_len/pred_len/alpha 等）、关键指标（MSE/MAE/skill）。
4. 提交后运行 `git log --oneline -1` 与 `git status --short`，确认工作区干净。
5. 在最终回复中报告 commit hash。

失败处理：若实验失败或输出不完整，不要提交半成品；先修复，或用 `fix:` / `docs:` 单独记录失败原因。
