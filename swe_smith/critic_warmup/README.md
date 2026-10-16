# MiniCPM5 SAO/PPO critic-only warmup

这是独立于历史 GRPO 的 critic 预热线：固定 actor、冻结 critic attention，
使用 GAE。工程链路已跑通，value 质量仍未达标，禁止自动开启 actor。

## 目录与应用

- `../../patches/critic-warmup-20261015.diff`：框架改动，基线为上游
  `218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4`。
- `overlay/`：任务脚本、模板、结构化评分和回归测试，保留其在上游 checkout
  中的相对路径。`overlay_manifest.json` 记录每个文件的 SHA256。
- `prepare_checkout.py`：默认只检查基线、干净工作区、补丁与文件校验；
  显式 `--apply` 才写入独立 checkout。

这份 critic 补丁直接应用到上述干净基线，**不要叠加旧的
`patches/agentlightning-core.diff`**。旧补丁与历史训练脚本继续保留供复现。
本目录是源码与补丁包，不是已安装的 Agent Lightning 环境。

```bash
git clone https://github.com/microsoft/agent-lightning.git /data/agl-example/critic-code
git -C /data/agl-example/critic-code checkout 218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4
python swe_smith/critic_warmup/prepare_checkout.py --checkout /data/agl-example/critic-code
python swe_smith/critic_warmup/prepare_checkout.py --checkout /data/agl-example/critic-code --apply
cd /data/agl-example/critic-code
python examples/swe_smith/run_critic_lowmem.py --help
python -m pytest tests/verl/test_critic_atomic_save.py tests/verl/test_critic_checkpoint.py tests/verl/test_tracking_config.py -q
```

依赖仍沿用项目已验证的训练环境；这里不安装或升级依赖。真实 GPU 运行还需要
匹配的 MiniCPM runtime/parser/chat template、AGL 服务、任务镜像和 ConfigMap，
portable launcher 要求显式传入这些路径，不会自动创建集群或下载模型。
模板中的 `<LAN_IP>` 必须替换为容器可访问的宿主机地址；模板不包含实验服务器的地址。

## 已完成与未完成

- 采样 token ID 与 behavior logprob 对齐；缺失奖励、基础设施失败和不完整
  轨迹不能被伪装成零奖励用于训练。
- critic-only 原生恢复 model/optimizer/RNG/scheduler/dataloader，实际 GPU
  恢复后更新并保存已验收；不加载 actor checkpoint。
- 全分片与 data.pt 非空检查后发布目录，再原子更新 tracker，最后清理旧版。
  稳定态只保留最新一份，写新期间必须允许临时双份。
- SwanLab 配置递归脱敏。密钥、权重、原始轨迹和私人运维审计不入库。
- 所有宿主机训练产物使用 `/data`，空间不足停训，不迁移到根分区。
  本包个人路径替换为 `/data/agl-example`，使用前改为自己的数据盘目录。
- 低显存双卡配置、保存间隔及最新质量结果见
  [2026-10-15 实验记录](../../docs/critic-warmup-20261015.md)。

2026-10-16 已从完整 step40 续训并自然完成到 step200，新增155次有效更新，
唯一完整 checkpoint 已原子替换为 step200。训练仍使用32 train / 1 validation，
独立 value 评估正在进行，尚未达到可开 actor 的质量门槛，也没有宣称 VAL474 提升。
完整的工程验收与质量边界见 [step200最终记录](../../docs/critic-final-20261016.md)。

## 发布包验证

本包在干净的指定上游 checkout 中实际完成 `prepare_checkout.py --apply`。
迁移后运行 checkpoint/atomic-save/freeze/tracking、GAE、storage、loop guard、
reward/training-phase、behavior token、OpenCode 合约与结构化 grader 的相关测试：
**175 passed，4 skipped**。跳过项受本地可用依赖/平台能力限制。
这验证了补丁和 overlay 的依赖及路径，未将 portable launcher 宣称为新的 GPU 验收。
