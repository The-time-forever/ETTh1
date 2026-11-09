---
description: Commit the current experiment/data/script step with a standard message (not for paper edits)
---

提交当前这一轮实验/数据/脚本步骤（$ARGUMENTS）。

**不要用它逐次提交论文修改。** `paper/**` 只在用户明确要求，或完成一个事先约定的阶段（初稿、投稿版等）时提交。

1. 运行 `git status --short` 与 `git diff --stat`，确认本轮改动。
2. 只 stage 本轮相关文件；不要 stage `__pycache__/`、`*.pyc`、临时文件、密钥、checkpoint。
3. 如果 `$ARGUMENTS` 为空，根据改动生成简短名称（如 `ridge-s96-h96`）。
4. 提交信息：
   - 标题：`exp: <name>`；若只是文档或规则改动，用 `docs:` 或 `chore:`。
   - 正文：脚本、输出目录、关键配置（seq_len/pred_len/alpha 等）、关键指标（MSE/MAE/skill）。
5. 提交后运行 `git log --oneline -1` 和 `git status --short`，确认工作区干净。
6. 在回复中报告 commit hash 与变更文件。

若实验未成功或输出缺失，停止并说明，不要提交半成品。
