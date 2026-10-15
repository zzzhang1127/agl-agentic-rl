# SWE-smith 历史训练脚本导航

这里保留 GRPO/HMPO、数据准备与运行配置。当前固定 actor 的 PPO critic
入口在独立的 [critic_warmup](../critic_warmup/README.md) 包中，不从本目录启动。

## 按用途阅读

- 训练与配置：`train_smith_agent.py`、`restart_s5_smith_trainer.sh`、`s5_cycle.sh`。
- HMPO 与奖励实验：`launch_hmpo.py`、`hmpo_reward.py`、`env_state_reward.py`。
- 数据与模板检查：`download_dataset.sh`、`make_dataset_s3.py`、
  `screen_tasks.py`、`preflight_templates.py`。
- 集群下发与环境：`refresh_agent_configmap.sh`、`setup_cu124.sh`、
  `start_agl_server.sh`。
- 历史配置：`lineages/` 保留 s1–s4 的差异，供理解实验演进。

这些脚本按实验时的上游目录、模型路径、镜像和运行状态编写，不是可直接用于
任意主机的部署工具；部分历史权重已退役。先看 [复现指南](../../docs/reproduction.md)，
再审查脚本中的路径、GPU、端口和进程/服务操作。不要将历史运维方式当成共享
服务器上的默认授权。

结果与训练起点见 [SWE-smith 实验](../README.md) 和
[结果口径](../../docs/results.md)。
