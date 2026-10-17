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

## 云端只读核查

[swanlab_health.py](swanlab_health.py) 读取 SwanLab 的学习率、KL、梯度、零优势比例和训练 token 数，并检查配置及 metadata 中的凭据字段是否脱敏。报告不会导出完整 profile 或凭据值，已有报告文件不会被覆盖。它不能替代真实进程、checkpoint 校验或固定验证集评测。

使用已有 SwanLab 环境运行 `python swe_smith/probes/swanlab_health.py --help` 查看参数；`--run` 使用 `workspace/project/run-id`，`--output` 指定新的 JSON 文件。认证使用 `SWANLAB_API_KEY` 环境变量或 SwanLab 已保存的认证，避免把密钥写进命令行或仓库。

该工具已用原生 GRPO 活跃运行的 step1–4 真实云端数据验证；本地回归覆盖敏感值不导出、脱敏标记、缺失 profile、固定指标选择和未解析配置。实现依赖已安装 SDK 的 `Api.run` / `Experiment.metrics` 接口，不自动安装或升级依赖。

## 已完成轨迹的读取

[archive_json.py](archive_json.py) 提供 `iter_json_archives(root)` 和 `read_json_archive(path)`，兼容UTF-8 JSON与gzip归档。压缩发布期间同一文件的两种表示只计一次；遍历后原JSON被删除时，会读取已经发布的gzip。损坏或截断的gzip抛出异常，调用方必须单列读取错误，不能补成空轨迹或零奖励。

该模块只读取，不压缩或删除文件。启用历史归档压缩前，必须先核对所有消费者的兼容性，验证解压SHA及gzip CRC，再发布gzip并替换原文件。当前训练run的原始归档尚未压缩：其现有监督器仍按纯JSON扫描，不能只更新一个读取工具就对所有活跃数据启用压缩。
