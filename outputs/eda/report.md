# ETTh1 EDA summary

- rows: 17420
- OT train mean/std: 17.13 / 9.18
- OT test mean/std: 4.85 / 3.15

## Seasonal strength (train, eta^2)

| variable | hour | weekday | month |
|---|---:|---:|---:|
| HUFL | 0.260 | 0.001 | 0.263 |
| HULL | 0.083 | 0.005 | 0.597 |
| MUFL | 0.283 | 0.001 | 0.248 |
| MULL | 0.066 | 0.004 | 0.677 |
| LUFL | 0.066 | 0.007 | 0.186 |
| LULL | 0.018 | 0.001 | 0.469 |
| OT | 0.011 | 0.005 | 0.811 |

## Calendar decomposition (train, sequential R2)

| variable | month | +hour | +weekday | residual std |
|---|---:|---:|---:|---:|
| HUFL | 0.263 | 0.353 | 0.001 | 4.011 |
| HULL | 0.597 | 0.207 | 0.018 | 1.172 |
| MUFL | 0.248 | 0.377 | 0.001 | 3.776 |
| MULL | 0.677 | 0.203 | 0.019 | 0.968 |
| LUFL | 0.186 | 0.082 | 0.010 | 0.881 |
| LULL | 0.469 | 0.034 | 0.003 | 0.451 |
| OT | 0.811 | 0.057 | 0.023 | 3.834 |
