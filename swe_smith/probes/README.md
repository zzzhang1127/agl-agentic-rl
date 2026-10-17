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

该模块只读取，不压缩或删除文件。启用归档压缩前，必须先核对所有消费者的兼容性，验证解压SHA及gzip CRC，再发布gzip并替换原文件。首次发布读取器时监督器仍只支持JSON；后续实际部署的缓存保护已验证872份gzip与其余JSON混合读取。不同部署需要各自核验，不能只更新一个读取工具就启用所有活跃数据的压缩。

## 策略损失诊断的CPU准备检查

[policy_objective_contract.py](policy_objective_contract.py) 使用已安装verl及正式配置指定的loss，对一个很小的合成logits样例检查Advantage方向、mask、KL、梯度裁剪和AdamW更新的测量链路。它不加载模型、真实轨迹或检查点，不启动Ray或调用worker注册钩子。报告明确标记 `real_model_loaded=false`、`real_trajectories_fitted=false`，不能据此宣称真实模型或训练链路已通过拟合诊断。

从实际actor配置导出以下字段到JSON：`clip_ratio`、`clip_ratio_low`、`clip_ratio_high`、`clip_ratio_c`、`loss_agg_mode`、`policy_loss.loss_mode`、`use_kl_loss`、`kl_loss_type`、`kl_loss_coef`、`grad_clip`、`optim.weight_decay`。如果正式使用`per_rollout_mean`，还需提供受信任正式快照的`verified_custom_loss_source`及对应`verified_custom_loss_sha256`；工具校验哈希后注册同一loss，并先按同一函数归一化Advantage。不能用普通vanilla PPO测试代替实际自定义损失。

```bash
CUDA_VISIBLE_DEVICES='' python -m swe_smith.probes.policy_objective_contract \
  --actor-config /data/agl-example/actor_loss.json \
  --synthetic-contract-only --output /data/agl-example/objective_contract.json
```

默认在CPU上用学习率`1e-7`重复32次，固定old logprob、reference和归一化后的Advantage；记录policy loss/KL、裁剪前后梯度、参数RMS变化与正负Advantage动作logprob，已有报告不会覆盖。当前正式`per_rollout_mean`设置的CPU样例已验证正方向上升、负方向下降和mask外梯度为0；真实冻结批、模型前向、分布式缩放和重新rollout效果仍未由这个工具验证。PPO进入裁剪区或存在KL拉力时，不能要求每个token严格单调。

测试依赖已有torch、omegaconf和verl，不自动安装；缺少依赖时明确skip。设置`POLICY_CONTRACT_ACTOR_CONFIG`可运行指定正式loss的额外检查，该环境变量只应指向自己的脱敏配置，配置文件不会随工具发布。
