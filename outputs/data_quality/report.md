# ETTh1 data quality checks

- file: `D:\Document\ETTh1\ETTh1.csv`
- sha256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- rows: 17420

| check | status | detail |
|---|---|---|
| 时间戳可解析 | pass | unparseable=0 |
| 时间轴严格小时级 | pass | missing=0, gaps>1h=0 |
| 时间戳唯一且递增 | pass | dups=0, monotonic=True |
| 无缺失值/无穷值 | pass | nan=0, inf=0 |
| 无数值格式异常 | pass | bad_cells=0 |
| 无整行重复 | pass | duplicated_rows=0 |
| 无常量列 | pass | constant=[] |
| 负荷列无负值 | warn | HUFL=2318(13.3%); HULL=2342(13.4%); MUFL=2943(16.9%); MULL=5001(28.7%); LUFL=1(0.0%); LULL=1669(9.6%) |
| 无长平台/传感器卡死 | warn | HUFL=15; HULL=15; MUFL=17; MULL=17; LUFL=16; LULL=20; OT=20 |
| OT 无 >8C/h 的物理跳变 | warn | >5C/h=34, >8C/h=2, >10C/h=0 |
| 无整日冻结（每月 31 日） | warn | all-constant days=14, rows=336 (1.93%); e.g. 2016-07-31, 2016-08-31, 2016-10-31 |
| 负荷通道无零值停机块 | warn | all-load-zero rows=58, max_segment=58h (2016-12-05) |
| 无长时间单通道零值/冻结 | warn | max single-channel zero run=141h (LULL, 2017-07-23..29) |
| 相关性不构成严重共线性 | warn | cond=21.8, max_VIF=100.7 |
| 训练/测试分布漂移可接受 | warn | max KS=0.687 (OT) |
| 训练/测试无重复特征向量 | pass | shared=0 |
