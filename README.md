# agl-agentic-rl

> Agentic RL on top of [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) (verl + GRPO):
> one single-turn run that works (GSM8K, +15.6 pp on the full test set), one multi-turn coding-agent run
> (SWE-smith, 2.5B model) that has **not** produced a real gain yet, and everything built along the way —
> framework patches, collapse probes, the official-harness evaluation fleet, a teacher-distillation pipeline, and
> a 3 800-line lab notebook of what went wrong and why. Numbers below are reported as measured; nothing is cherry-picked.

用 Agent Lightning（下称 AGL）做 agentic RL 的全部工程记录。两条线：

| 任务 | 模型 | 方法 | 基线 | 训练后 | 统计 | 口径与必须说明的事 |
|---|---|---|---|---|---|---|
| **GSM8K**（单轮） | Qwen2.5-1.5B-Instruct | 官方示例 GRPO，n=4，lr 1e-6 | 62.55%（825/1319） | **78.17%**（1031/1319，step 1550） | 两比例 z=8.78 | 全量 test 1319 题，精确匹配；**未开 KL**（官方默认 `use_kl_loss=False`）；step 1550 不是峰值（峰值在 800–1000 步附近，ckpt 被 `keep=2` 轮掉）；单轮 RLVR，不是多轮 agentic |
| **SWE-smith**（多轮 coding agent） | MiniCPM5-2B（2.5B，Llama 架构） | 官方 smith harness + GRPO，五条血统 s1–s5 | 134/474 | s5 step 10：132/474（p=0.90）；step 20 进行中 | 无显著增益 | 固定 474 题验证集，temperature 0.6，真实上限 470（4 题的镜像任何模型都做不出）；此前四条血统要么平线要么坍塌，根因都已定位（见下） |
| **SWE-smith 蒸馏** | 基座 → 教师轨迹 SFT | deepseek-v4-flash 轨迹，只留做对的题 | 基座 123/474 | SFT ep3 127/474（同一旧模板，差异不显著）；模板修复后重测 134 | — | 这是蒸馏口径，不是 RL 增益；第 5/6 批教师轨迹（官方 harness 格式）正在采，下一轮 SFT 待训 |

截至 2026-10-06。两条线的详细结果、血统表和根因见 [`gsm8k/README.md`](gsm8k/README.md) 与 [`swe_smith/README.md`](swe_smith/README.md)。

## 仓库结构

```
gsm8k/            可复现的单轮 GRPO：启动脚本、全量评测脚本、曲线/显著性工具、评测日志
swe_smith/
  agents/         跑在任务容器里的 agent（官方 smith 协议：THOUGHT + 单个 bash 块）及其单测
  training/       RL 训练入口、s5 启动脚本、k3s Job 模板、ConfigMap 刷新、数据集构造与筛题
    lineages/     s1–s4 的启动脚本与模板（逐字保留，便于对照每条血统改了什么）
  probes/         坍塌探针（训练日志 Tier-1 看门狗）、探针序列显著性、奖励审计、权重漂移
  harness/        官方 harness 上的 474 题全量评测：vLLM 多卡 fleet、分片 sweep、配对显著性
  distillation/   教师轨迹采集：OpenAI 兼容代理（归一化教师怪癖）、批次驱动、拒绝采样拼装
  sft/            教师轨迹 SFT：与推理逐字相同的模板渲染、只盖 assistant 的 loss mask、启动脚本
  containers/     SWE-smith 任务镜像的规划与预拉
  opencode_lineage/  更早的 OpenCode-harness 血统（v2–v4）与它的教师流水线，已被 smith harness 取代
patches/          对 agent-lightning 核心库的 diff（+1016/−49）、配套单测、verl dp_group 补丁
docs/             踩坑与经验（§1–§66，按「现象 → 根因 → 解决 → 经验」写）、MiniCPM5-2B 选型与架构问答
```

## 方法概览

AGL 的结构是：agent 在隔离环境里跑（这里是每道题一个 docker 容器，由 k3s Job 拉起），所有 LLM 调用经过
AGL 的 OpenAI 兼容代理（`agl-server`），代理把 prompt/response 的 token 序列采集下来交给 verl trainer 做 GRPO；
trainer 与 vLLM 推理共置在同一组 GPU 上，每步先 rollout 再更新。agent 代码本身不用改成「训练专用」，这是 AGL 的卖点，
也是本项目大部分时间都花在哪里的原因：**采集链路上每一处「代理看到的序列 ≠ 模型真正看到的序列」都会悄悄把训练变成噪声**。

本项目在框架层做的事（全部在 [`patches/`](patches/README.md)）：

- 修采集：工具调用轮被 chat template 渲染异常静默丢掉，采集率 8.1%；代理丢 `chat_template_kwargs` 导致多轮 episode 合并 89% 失败；chat 模板 `<think>` 不对称导致 episode 被打碎成「一轮一行」（前向浪费 18.3×）。
- 加闸门：采集率低于阈值直接失败；一个 batch 零 LLM 调用直接失败；模板 preflight 不过不起训。
- 训练器：DAPO 风格动态采样（零优势组整组丢弃并补采）、mini-batch 适配、行打乱、原子化 ckpt。
- verl：`dp_actor` 漏传 `dp_group` 导致 NCCL all-gather 挂死 3600s 的补丁。

在任务层做的事（[`swe_smith/`](swe_smith/README.md)）：稠密奖励（F2P 测试通过比例 + P2P 硬闸）、筛题、
坍塌探针与「探针显著下降即停」的自动循环、固定 474 题的配对显著性评测、教师轨迹拒绝采样 SFT。

## 复现

1. 环境：按 AGL 官方文档装 `agent-lightning @ 218f1f7c` + verl 0.7.1 + vLLM 0.8.5（`swe_smith/training/setup_cu124.sh` 是本项目的安装脚本，CUDA 12.4），然后套 `patches/`。
2. GSM8K：`gsm8k/run_gsm8k.sh`（训练）→ `gsm8k/eval_both_full.sh`（基座与 ckpt 的全量 1319 题评测）。
3. SWE-smith：`swe_smith/training/download_dataset.sh` 拿官方数据集 → `containers/pull_images.py` 预拉镜像 → `training/refresh_agent_configmap.sh` → `training/restart_s5_smith_trainer.sh`；评测用 `harness/smith_fleet.sh` 起 vLLM，再 `harness/run_smith_sweep.py` 分片跑 474 题，`harness/paired_cmp.py` 做配对检验。

脚本里的 `/workspace/...` 是脱敏后的占位路径（原始路径是服务器上的个人目录），按自己的目录改即可。

## 脱敏说明

公开前统一替换了：服务器个人目录 → `/workspace`，内网 IP/代理 → `<LAN_IP>`/`<CORP_PROXY>`，进程号 → `<N>`，
同机其他用户与进程名 → `<other-user>`/`<neighbour-process>`，个人 HF/SwanLab 句柄 → `<hf-user>`/`<swanlab-run-url>`。
所有密钥（教师 API、HF token）只以 `.env` 文件形式存在于服务器本地，从未入库，`.gitignore` 也拦住了它们。
如果发现漏网的信息，请开 issue。

## 之后会更新什么

- s5 的后续探针结果；若仍无增益，转「教师轨迹 SFT（第 5/6 批，官方 harness 格式）→ 再 RL」的路线。
- 蒸馏第 6 批（全训练集 5338 题）的采集结果与下一轮 SFT 的 474 题评测。
- `docs/pitfalls-and-lessons.md` 持续追加。
