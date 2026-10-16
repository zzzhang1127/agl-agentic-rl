# 复现与代码验证

本仓库提供实验脚本、框架补丁和部分结果记录，**不是完整的上游源码或一键部署包**。先选择实验线，再建立独立上游checkout；不要把两套补丁叠加。

## 环境与外部资源

历史验证环境围绕 Linux、CUDA12.4、PyTorch2.6.0+cu124、vLLM0.8.5、verl0.7.1及Agent Lightning指定基线。具体安装脚本见 [setup_cu124.sh](../swe_smith/training/setup_cu124.sh)：它包含Python3.12/Linux轮子和依赖安装操作，应先阅读再按环境调整，不适用于任意机器。

GPU训练还需模型、相应chat template/tool parser、数据划分、任务Docker镜像、AGL服务与k3s/ConfigMap。模型权重、原始轨迹、服务密钥不在仓库中，历史checkpoint未必仍保留。

历史脚本的 `/workspace` 和critic包的 `/data/agl-example` 是脱敏路径，需指向自己的数据盘。不要向代码或公开配置中填入密钥；运行所需凭据留在环境变量或私有文件中。

## 路径A：critic-only PPO（当前工程线）

从本仓库根目录执行，先在自己的数据盘准备独立上游checkout：

```bash
git clone https://github.com/microsoft/agent-lightning.git /data/agl-example/critic-code
git -C /data/agl-example/critic-code checkout 218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4
python swe_smith/critic_warmup/prepare_checkout.py --checkout /data/agl-example/critic-code
```

上一条只检查，不修改目标。确认后显式应用：

```bash
python swe_smith/critic_warmup/prepare_checkout.py --checkout /data/agl-example/critic-code --apply
cd /data/agl-example/critic-code
python -m pytest tests/verl/test_critic_atomic_save.py tests/verl/test_critic_checkpoint.py tests/verl/test_tracking_config.py -q
python examples/swe_smith/run_critic_lowmem.py --help
```

检查器验证精确基线、干净工作区、补丁可应用性及46个overlay文件SHA256。更完整的测试范围和真实GPU验收边界见 [critic包说明](../swe_smith/critic_warmup/README.md)。某些回归依赖训练环境，skip不代表对应GPU能力已被本机验证。

launcher要求显式模型/数据/job-template/parser/chat-template路径及AGL密钥文件，SwanLab密钥经环境传入。保持actor禁用、attention冻结；用独立supervisor监测奖励供给、有限梯度和磁盘余量。最近checkpoint约22.2GiB，原子发布需临时容纳新旧两份；当前实验data准入64GiB、运行保护32GiB。不足时停止，不转移到根分区。

2026-10-16修正：所有运行产物写入数据盘后，根分区余量只作观测，不再作为额外启动/停训阈值。CUDA与FlashInfer缓存也显式指向数据盘。见[续训验收记录](critic-resume-20261016.md)。

## 路径B：历史GRPO与GSM8K

历史框架补丁为 [patches/agentlightning-core.diff](../patches/agentlightning-core.diff)，应用到另一个干净的上游218f1f7c checkout。步骤及verl补丁见 [patches/README.md](../patches/README.md)。不要在路径A的checkout上重复套用。

- GSM8K沿用上游agent/trainer；本仓库补充启动、全量评测和曲线脚本，见 [GSM8K说明](../gsm8k/README.md)。
- SWE-smith历史脚本按训练配置、镜像、ConfigMap与评测fleet组织，见 [SWE-smith说明](../swe_smith/README.md)。
- `training/lineages` 和 `opencode_lineage` 保留历史路径、协议与环境假设，不作为当前critic启动入口。

训练/运维脚本可能创建容器、安装依赖或操作服务。代码阅读与补丁验证可以先独立完成；重跑历史训练前逐项核对路径、GPU、端口和数据，不能仅凭文件名直接执行。

## 验证实验，而非只验证启动

保存上游及本项目commit、模型起点、数据/模板版本、评测预算、随机采样参数与有效样本数。报告应同时给出分子分母、基础设施错误、排除规则、对照和配对统计。具体记录格式见 [维护约定](../CONTRIBUTING.md)。
