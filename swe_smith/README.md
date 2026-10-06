# SWE-smith：多轮 coding-agent RL（尚未训出真实增益）

任务是 AGL 官方示例 `examples/swe_smith`：agent 在装好目标仓库的 docker 容器里修 bug，奖励来自隐藏测试
（F2P = 本应从失败变通过的测试，P2P = 本应保持通过的测试）。学生模型 MiniCPM5-2B（2.5B，Llama 架构，
原生 131k 上下文，这里用 51200）。协议是官方 smith harness：每轮一个 `THOUGHT` + 一个 bash 代码块，最多 40 轮。

所有评测都在**固定 474 题验证集**上做，temperature 0.6，一题一次作答，配对检验（`harness/paired_cmp.py`）。
4 题的镜像没有 python，任何模型都做不出，真实上限 470。

## 血统表（2026-10-02 起，官方 harness）

| 血统 | 起点 | 奖励 | 与上一条的差别 | 探针（474 题） | 结论 / 根因 |
|---|---|---|---|---|---|
| s1 | 基座 | 二值 resolved | lr 2e-6，n=4，batch 16 | 91 步；48 题小探针八点均值 7.63/48，全在 ±1σ | 训练奖励 0.31→0.45 不转化为验证集增益；奖励上涨主要是抽题噪声；多轮合并此时**还没生效**（见下） |
| s2 | SFT ep3 | 二值 | lr 提到 5e-6 | step 5：**50/474**（基线 136，p≈0） | 4 步坍塌。机理：少写 token 自强化 |
| s3 | SFT ep3 | **F2P 通过比例（稠密）+ P2P 硬闸**，700 道筛过的题 | lr 回 2e-6 | step 5 坍塌 | 根因在反分词：vLLM `skip_special_tokens` 把模型原生工具调用的特殊 token 剥掉，残文回流对话，格式错烧光 40 轮；零方差组 41%→15% 证明稠密奖励本身有效 |
| 合并修复 | — | — | 代理透传 `chat_template_kwargs`（89% 的合并失败）+ 模板按正文 `</think>` 重切（13.3% 的轮次） | — | 此前每条 episode 被打碎成「一轮一行」，前向浪费 18.3×；修后 `MAXRESP=43008`、`PPO_MINI=4`、40 轮不砍 |
| s4 | SFT ep3 | s3 奖励 + **轮数罚**（T0=32，λ=0.1）+ 上下文罚 | 合并修复后首条 | step 10：145（p=0.185，不显著）→ step 20：**117（p=0.030，显著跌）**，停 | 轮数罚是组内**唯一**的稳定信号（同样做对的样本 t=−35），模型学到的是「早交卷」（提交率 71%）而不是「修 bug」 |
| s5 | SFT ep3 | = s4 去掉轮数罚与上下文罚 | 其余逐字相同（pod 模板已核对） | step 10：132（p=0.90）；step 20 进行中 | 每 10 步全量探针，显著下降即停（`training/s5_cycle.sh`） |

基线说明：s4/s5 的对照基线是 SFT ep3 在当前模板下的重测 **134/474**；更早记录的 127 是旧模板读数，+7 题来自模板修复而不是 RL。

## 为什么还没训出来（已排除与已确认）

- **不是奖励稀疏、不是组太小。** s3 的稠密奖励把零方差组从 41% 压到 15%；`rollout.n` 4→8 只多出 ×1.25 的有效梯度信号，不值双倍 rollout 时间；筛题只 ×1.24。
- **确认的根因是 credit assignment。** 一条 episode 20–30 轮、几万 token，奖励是序列级标量；同样做对的两条轨迹在几千个 token 上不同，组内优势在 token 上互相抵消，Adam 救不了。PPO 在 2B 上更差。
- **长度通道。** 任何让「更短」与奖励相关的项（轮数罚、上下文罚、把未提交记 0）都会成为组内最强信号，起点越强通道越宽（≈p⁴）。s4 是实证。
- **步时的大头是 rollout 长尾**，不是算力：一道题的 pytest 能拖 6 分钟且零方差；异步方案与动态采样互斥，只省 2–6%。
- 训练前必须核对三件事：pod 模板里的 env（不是代码默认值）、ConfigMap 已刷新（否则跑的是旧 agent 代码）、`critic/score/mean` 不是奖励（bf16 + 行加权 + 过滤后偏差，s2 时偏高 0.20，曾让坍塌「看起来奖励没报警」）。

## 蒸馏与 SFT

- **教师**：deepseek-v4-flash，经 OpenAI 兼容代理（`distillation/teacher_proxy_smith.py`）接入官方 harness。代理做归一化：35% 的回合在 bash 块后拖伪 XML 结束标签、4% 幻觉续写下一步、1.2% content 为空（答案在 reasoning_content）、1.7% 原生 DSML 标记，归一化后格式错误率 4.5%→1.6%，有效命令零改动。
- **拒绝采样**：只留 `resolved` 且已提交的轨迹；剪掉（格式错的 assistant 轮，`Format error:` 的 user 轮）对；assistant 归一化为 THOUGHT + 单 bash 块（`distillation/assemble_smith_teacher.py`）。
- **批次**：b5 = 1000 道训练题，100 并发 47 分钟，822 resolved，清洗后留 707；b6 = 其余 5338 道训练题，150 并发，采集中。更早的 b1–b4 用的是 OpenCode harness 格式，不能复用（已重采）。验证集的教师轨迹只用来评测，不进 SFT。
- **SFT**（`sft/`）：全参，4 卡 FSDP + Ulysses sp4，lr 1e-5，动态打包；`swe_sft_dataset.py` 渲染出的序列与推理时 vLLM 用的模板**逐字相同**（`enable_thinking=False`），loss 只盖 assistant 轮（占 23.4% token）。SFT ep3 来自 b1–b4 的 858 行、3 个 epoch 78 步；`run_sft_b5.sh` 从 ep3 续 1 epoch。

## 评测 harness（`harness/`）

`smith_fleet.sh` 在多卡上起 vLLM 实例（端口 18000–18009，容器的 iptables 只放行这些端口），`run_smith_sweep.py` 按分片并发跑官方 agent 循环，每题落 `instance/result/status.json`；
`accuracy_smith.py` 汇总，`paired_cmp.py` 做两次 sweep 的配对检验，`eta_smith.py` 估剩余时间。474 题全量约 1 小时（4 卡各一个 vLLM 实例）。

## 探针（`probes/`）

`s3_watch.py` 是 Tier-1 看门狗，直接读 trainer 日志：零方差组比例、回复长度、熵、`training/reward` 相对门槛、格式错误率，任一触发就写刹车文件；回放验收过（s2 第 4 步开火，s1 91 步不开火）。
`s4_trend.py` 对探针序列做趋势显著性；`s4_reward_audit.py` 逐条比 reward 与原始测试结果，就是它找出了 s4 的长度通道；`weight_drift.py` 算相对基座的权重漂移（OpenCode 血统 v4 的 RL 0.17% vs SFT 0.33%，判欠训）。

## 容器与 k3s（`containers/`、`training/`）

rollout 以 k3s Job 跑，每题一个容器，agent 代码通过 ConfigMap 下发（`refresh_agent_configmap.sh`，**起训前必刷**）。
`k3s_unlatch.sh` 处理 kubelet DiskPressure 误闩（`evictionMinimumReclaim` 10% 让解锁要几百 GB，重启 k3s 5 秒清）；
Job 模板加了 toleration 与 critical 优先级，闩着也不影响训练。`pull_images.py` 按仓库规划并预拉镜像；镜像 GC 阈值要与磁盘水位对齐，否则「边拉边删」。

## 更早的 OpenCode 血统（`opencode_lineage/`）

2026-09 用 OpenCode 作为 harness 训了 v2–v4（含采集率 8.1% 的 bug 发现与修复、步数预算、上下文压缩切分），
step 62 全量 136/474 vs 基座 139，无增益；随后转教师蒸馏，再转官方 smith harness。代码原样保留供对照，不再维护。
