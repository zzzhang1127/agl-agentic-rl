# SWE-smith 训练诊断工具

这些工具用于解释训练行为和发现退化，不能替代固定题集的最终评测。脚本保留实验时的路径/日志格式假设，使用前需核对配置。

| 工具 | 用途 |
|---|---|
| `s3_watch.py` | 从日志监测零方差组、回复长度、熵、奖励与格式错误等信号 |
| `s4_trend.py` | 探针序列趋势与噪声分析 |
| `s4_reward_audit.py` | 核对奖励构成与长度相关信号 |
| `weight_drift.py` | 对比训练权重相对起点的漂移 |
| `s2_verdict.py`、`s4_process.py` | 历史运行的结果诊断与过程汇总 |

阅读示例见 [SWE-smith历史实验](../README.md)。评测口径见 [结果页](../../docs/results.md)；配对比较入口为 [paired_cmp.py](../harness/paired_cmp.py)。
