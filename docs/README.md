# 文档索引

## 建议阅读顺序

1. [项目首页](../README.md)：项目目标、代表性结果与工程工作。
2. [结果与口径](results.md)：区分40/100轮、官方SFT/ep3/原版基座，以及蒸馏与RL。
3. [复现与代码验证](reproduction.md)：选择补丁线，检查外部依赖和已验证边界。
4. [critic工程验收](critic-warmup-20261015.md)：恢复、原子保存、显存配置与质量限制。
5. [历史实验笔记](pitfalls-and-lessons.md)：按日期保留假设、故障与修复证据。

## 按问题找入口

- 单轮GRPO是否有效：[GSM8K结果与日志](../gsm8k/README.md)。
- 多轮训练为什么可能退化：[SWE-smith实验线](../swe_smith/README.md)与[诊断工具](../swe_smith/probes/README.md)。
- 训练与评测脚本如何分工：[训练导航](../swe_smith/training/README.md)与[评测导航](../swe_smith/harness/README.md)。
- 40轮蒸馏与100轮重评：[结果页](results.md)，原始记录在笔记§84–§90。
- 轨迹采集、token/logprob对齐与训练闸门：[补丁说明](../patches/README.md)。
- critic-only恢复与完整保存：[critic源码包](../swe_smith/critic_warmup/README.md)。
- 新实验应如何记录：[维护约定](../CONTRIBUTING.md)。

## 当前说明与历史记录

首页、结果页和各实验线README承担当前导航；带日期的验收文档描述当时验证状态。历史笔记保留原始推断与后续修正，不应把早期章节的“还在评测”或机制判断直接当作最新结论。对外介绍时优先使用有明确起点、预算和证据的读数。
