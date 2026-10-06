# GSM8K：AGL 官方示例的单轮 GRPO，能训出效果

这是 AGL 自带的 `examples/gsm8k`（agent 只做一次 LLM 调用，奖励为最终答案精确匹配），
本目录只放本项目**新增**的脚本，agent 与训练入口（`gsm8k_agent.py`、`train_gsm8k_agent.py`）用上游原版。

## 结果

| | 基座 Qwen2.5-1.5B-Instruct | step 1550 | 提升 |
|---|---|---|---|
| GSM8K test 全量 1319 题 | 825/1319 = **62.55%** | 1031/1319 = **78.17%** | **+15.62 pp** |

- 两比例 z 检验 z = 8.78（非配对，保守）。训练中每 10 步在固定 200 题（seed 42）上评测，200 题探针序列的趋势检验 z = 5.70；200 题的 1σ ≈ 3.4 pp，所以单次探针差 < 6.7 pp 不能当作涨跌（`curve_gsm8k.py` 头注里写了这条判据）。
- 原始日志：[`results/eval_full1319.log`](results/eval_full1319.log)（`val/reward` 即准确率）。

必须一起报告的事实：

1. **未开 KL。** 官方示例默认 `use_kl_loss=False, kl_loss_coef=0`，本次沿用，没有改。
2. **step 1550 不是峰值。** 200 题探针在 800–1000 步附近更高，但 `max_actor_ckpt_to_keep=2` 把那些 ckpt 轮掉了，只剩 1550/1572 两个；报的是现存 ckpt 的全量评测，不是探针峰值。
3. **这是单轮 RLVR**（一次调用、可验证奖励），与多轮 agentic RL 不是一个口径，不能与 `swe_smith/` 的结果混报。

## 超参

全部沿用官方 `train_gsm8k_agent.py` 默认：GRPO、`rollout.n=4`、lr 1e-6、`train_batch_size=8`、`ppo_mini_batch_size=8`、
clip 0.2/0.28、`max_response_length=1024`、2 个 epoch（共 1868 步）。本次命令行只加了
`--val-size 200 --seed 42 trainer.test_freq=10 trainer.save_freq=50 trainer.max_actor_ckpt_to_keep=2 trainer.val_before_train=True`。
训练停在 step 1572（手动停，30.3 s/步，单卡）。

## 脚本

| 文件 | 作用 |
|---|---|
| `run_gsm8k.sh` | 替代官方 `run_local.sh`。官方脚本会 `pkill` 机器上所有 `agl-server`/`agl-controller`、`ray stop --force`、往 `/tmp` 写东西，在共享机上会误伤别人；这里只按自己记录的 PID 清理、Ray 用独立端口与临时目录、启动前检查磁盘/权重完整性/端口占用、`no_proxy` 按网卡自动生成（否则本机代理会吃掉 vLLM 的上游连接）。 |
| `eval_gsm8k_full.sh` | `trainer.val_only=True --val-size 0` 跑全量 1319 题；头部注释给了 FSDP ckpt → HF 权重的合并命令。 |
| `eval_both_full.sh` | 顺序评测基座与指定 step，汇总两行 `val/reward`。 |
| `curve_gsm8k.py` | 从 trainer 日志抽 200 题探针曲线和训练奖励滑动平均，并给出噪声底线判据。 |
