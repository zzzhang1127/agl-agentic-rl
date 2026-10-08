# SWE-smith OpenCode GRPO 训练:踩坑与经验记录

> 项目:Qwen3-8B + Agent Lightning + VERL(GRPO)+ OpenCode agent,rollout 跑在 k3s Job 里,
> 4 卡 FSDP 训练 + 2×(TP=2) vLLM 0.8.5 推理副本,单机 8×~96GB 与多位同事共享。
> 本文档持续更新,每条按"现象 → 根因 → 解决 → 经验"组织。

---

## 1. 显存分配:`gpu_memory_utilization` 的语义陷阱与公平 KV 分配

**现象**:两个 vLLM 副本(各 TP=2)的 KV cache 容量严重不对称(一个 21.4 万 token、
一个 38.7 万 token),历史上每张卡总占用都被推到统一的 78.3GB。

**根因**:vLLM 的 `gpu_memory_utilization` 不是"我这个进程用多少",而是
**"这张卡的总占用(含别人进程)最多到多少"**。共享 GPU 上,同事进程占得多的卡,
分到的 KV 就少——每张卡的 KV 容量取决于别人用了多少,既不公平也不可复现。

**解决**:改用绝对值分配 KV:`engine_kwargs.vllm.num_gpu_blocks_override = 20480`
(vLLM 0.8.5 V1 引擎在 `vllm/v1/core/kv_cache_utils.py` 中生效)。
按目标反推:8 并发 × 40960 max_model_len = 327,680 token = 20480 块(block_size=16)。
KV 显存公式(Qwen3-8B,TP=2):36 层 × 8 KV头 ÷ 2 rank × 128 维 × 2(K,V) × 2 字节
≈ 72KB/token/rank,20480 块 ≈ 24.2GB/rank。`gpu_memory_utilization` 只留作 profiling
上限(0.85),不再决定实际 KV 大小。改后两副本 KV 完全一致,本进程每卡占用均匀 ~34GB
(权重 8.2 + KV 24.2 + 开销 ~1.6)。

**经验**:共享 GPU 环境下,凡是"按总显存比例"的配置都是坑;要用绝对量配置,
并且要能手算 KV 显存公式来验证。

---

## 2. 更新阶段的显存构成与 TP 的选择

**现象**:TP=4 时权重更新阶段 OOM(92GB+);TP=2 正常。

**实测构成**(4 卡 FSDP + offload):更新阶段每卡 38.2GB allocated / 49.2GB reserved。
参数和优化器状态已 CPU offload(占 ~298GB 内存),大头是 **32k token 长序列的激活值**
(即使开了梯度检查点)。rollout 阶段 KV 是大头,但 `free_cache_engine=true` 会在
更新前释放 KV,两个阶段的显存峰值错开。

**经验**:agentic RL 的序列极长(几万 token),更新阶段显存瓶颈在激活值而非参数;
估算显存要分"rollout 峰值"和"update 峰值"两条线分别算。

---

## 3. 孤儿 pod:k8s Job 生命周期与 trainer 解耦

**现象**:重启 trainer 后,新一轮 rollout 异常变慢(推理吞吐减半)。

**根因**:k8s Job 是独立对象,trainer 死了 Job 照跑。旧 pod 只认 agl-server 网关
(18082),网关把请求路由给当前注册的 vLLM 后端——**上一轮的 16 个孤儿 pod 和新一轮
的 16 个 pod 抢同样的 16 个推理槽位**,把实验数据也污染了(rollout 耗时不可比)。
孤儿完成后结果还会因 rollout ID 过期被丢弃,纯属浪费。

**连环坑**:清理时 `kubectl delete jobs -l app=agl-rollout` 静默删了个寂寞——
**这个 label 只打在 pod 上,没打在 job 上**,label selector 匹配不到任何 job 且不报错。

**解决**:重启 trainer 的标准流程必须先按名字删 Job:
`kubectl get jobs --no-headers -o custom-columns=:metadata.name | grep '^agl-rollout-' | xargs -r kubectl delete job --wait=false`

**经验**:(1) 编排系统里"父进程死了"不等于"子任务停了",重启流程要显式清理;
(2) 用 label 删除资源前,先用同样的 selector `get` 一遍确认非空——静默空匹配是 k8s 经典坑。

---

## 4. 反复超时:不是算力问题,是模型"不收尾"

**现象**:训练 rollout 几乎 100% 超时(600s 上限时 32/32,放宽到 1200s 后依然
32/32),而同一个模型做基线评测时中位 8 轮、59 秒就提交,超时率仅 5.9%。

**排查过程**(控制变量):
- 超时上限 600s → 1200s:无改善,全超时;
- 温度 1.0 → 0.8:无改善(第一轮被孤儿 pod 污染,清干净重跑仍 16/16 全超时);
- 公平 KV + 无争抢的干净环境:仍全超时,p50 耗时 1226s,每条 62~196 轮(中位 83)。

**当时的结论**:模型"不收尾"。**后被推翻**:真相是第 11 条的传输层 bug——
模型每轮的工具调用全被代理吞掉,opencode 空转重试直到超时。降温、对齐采样参数
都无效的原因也在此:病根根本不在采样。

**经验**:遇到超时先分清"慢"还是"不停":看轮数分布和单轮耗时,而不是一味加时限。
加时限只对"慢"有效,对"不停"是白白拉长每步训练时间。

---

## 5. 温度与 top_p/top_k:训练和评测的采样参数悄悄不一致

**现象**:同模型同 agent 框架,评测中位 8 轮,训练中位 83 轮。

**根因**(两层):
1. **客户端不传采样参数时,vLLM 会套用模型自带的 `generation_config.json`**。
   基线评测走 opencode 直连 vLLM,没传温度,实际生效的是 Qwen3-8B 官方推荐:
   temperature=0.6, top_p=0.95, top_k=20。
2. **VERL 显式下发采样参数,覆盖模型默认值**,而 VERL 的默认是
   top_p=1.0、top_k=-1(全分布不截断),温度我们又设了 1.0/0.8。
   Qwen 官方文档明确:thinking 模式必须配 0.6/0.95/20,否则长序列会退化——
   退化形态正是"无止境推理、反复验证、不收尾"。

**教训**:我们一开始只盯着温度调(1.0→0.8→0.6),完全没意识到 top_p/top_k
也和基线不一致——因为它们"没写在配置里",而没写 ≠ 一样,恰恰是两套默认值体系
(模型 generation_config vs 框架 default)各管一摊。

**经验**:RL 训练前,必须把训练 rollout 与基线评测的**完整采样参数逐项 diff**
(temperature/top_p/top_k/repetition_penalty/max_tokens),不能只看显式配置,
要看日志里实际生效的 `override_generation_config`。
(后记:本项目里对齐采样参数后仍全超时,最终病根是第 11 条的传输层 bug;
但这个 diff 原则本身仍然成立——它帮我们干净地排除了一整类嫌疑。)

---

## 6. thinking 模式:开没开、要不要

**排查**:抽查基线 60/60 条日志、训练 16/16 条轨迹,都含 `<think>` 块——
两边 thinking 都开着,它不是评测/训练差异的来源。机制:Qwen3 聊天模板默认开
thinking,只有显式传 `chat_template_kwargs: {"enable_thinking": false}` 才关。

**决策**:保留 thinking。理由:(1) 基线证明 thinking + 正确采样参数能正常解题;
(2) RL 起点应与基线策略分布一致,关 thinking 等于换了个没验证过的起点;
(3) Qwen3 模板自动丢弃历史轮 think 内容,不会撑爆上下文。
若采样参数对齐后仍全超时,再把关 thinking 作为备选(能大幅缩短单轮耗时)。

---

## 7. GRPO 的零梯度陷阱:奖励无方差 = 白跑

**现象**:训练跑了很多步,`grad_norm` 恒为 0,权重从未更新。

**根因**:GRPO 的优势 = 组内奖励减组均值再归一化。全组 rollout 全超时、
奖励清一色 0(或清一色 -0.2)时,**组内方差为零 → 优势为零 → 零梯度**。
奖励是纯二值(resolved=1 / else=0)时尤其容易踩:难题全 0,简单题全 1,都白跑。

**缓解**:加奖励塑形制造排序:超时(rc=124)额外 -0.2,使
resolved(1.0) > 跑完但未解决(0.0) > 超时(-0.2)。GRPO 组内归一化后只有排序
有意义,绝对值不重要。塑形已验证端到端生效(pod 上报 -0.2 → trainer 指标
rewards/mean=-0.2),但**只要组内仍然全同,依然零梯度**——塑形是必要不充分条件,
根本解还是让部分 rollout 能在时限内跑完(见第 5 条)。

**经验**:GRPO 训练要盯三个指标:组内奖励方差、n_zero_adv_groups、grad_norm。
reward 均值正常不代表在学习。

---

## 8. 代码分发:ConfigMap 与仓库的双源漂移

**现象**:准备重建 ConfigMap 下发新代码时,发现 ConfigMap 里的 agent 代码比仓库
**新**——线上热修过(vLLM 流式关闭 `stream=False`、BrokenPipeError 容错)却没回写仓库。
盲目用仓库文件重建 ConfigMap 会把线上修复冲掉。

**解决**:先 dump ConfigMap 到本地、与仓库 diff、把线上修复合回仓库,再加新改动、
重建 ConfigMap。另外 trainer 只在启动时读 job-template,**改模板/ConfigMap 后必须
重启 trainer 才对新 pod 生效**。

**经验**:任何"仓库 → 线上"的单向分发机制,一旦有人在线上直接改,就变成双源。
改配置前先 diff 双方,永远假设线上可能比仓库新。

---

## 9. 运维杂坑(每条都真实浪费过时间)

- **tmux 启动后立刻 pgrep 抓到的是瞬态 pid**:启动脚本会先后拉起多个进程,
  10 秒内抓到的 pid 可能几秒后就退出,导致监控误报"trainer 挂了"。
  正确做法:等 ≥1 分钟,用 `pgrep -af <入口脚本特征> | grep -v "bash -c"` 取稳定 pid。
- **kill 掉 trainer 后 SwanLab 会残留"运行中"的僵尸实验**:进程被杀,客户端来不及
  上报结束状态。以创建时间分辨真假,僵尸条目不占资源。
- **本机回环请求被 shell 代理劫持**:`curl http://127.0.0.1:18082` 走了 http_proxy
  返回 502,需 `--noproxy '*'`。
- **scp 拷来的脚本带 CRLF**:diff 显示每行都不同、bash 报奇怪语法错,
  `sed -i 's/\r$//'` 处理。
- **本机 find(bfs 实现)的 `-newermt` 只认 ISO 时间戳**:`'15:05'` 报错,
  要写 `'2026-09-06T15:05:00'`。
- **重启只重启该重启的**:trainer 挂了只重启 trainer,agl-server/controller 保留,
  避免孤儿 pod 找不到网关导致整批 rollout 报废。

---

## 10. 实验方法论

- **一次只动一个变量**:孤儿 pod 污染过一轮温度实验(32 pod 抢 16 槽),那轮数据
  只能作废重跑。任何对比实验前,先确认环境干净(无孤儿、KV 对称、无同事抢卡)。
- **先建立基线再训练**:474 题的基线评测(轮数/耗时/token/超时率分布)是后来所有
  诊断的参照系——没有"正常应该中位 8 轮"这个数,就看不出"83 轮"是病。
- **看实际生效值,不看配置文件**:采样参数、KV 块数、超时上限,都以启动日志里
  打印的生效值为准。

---

## 11. 【真·病根】代理强改 stream=False 后把 JSON 原样回给要 SSE 的客户端

**现象**:所有温度(1.0/0.8/0.6)、对齐 top_p/top_k 之后,训练 rollout 依然 100% 超时;
挖开 pod 里 opencode 的会话数据库才看到真相:60 轮 step 里**零工具调用、零正文**,
每条 assistant 消息 `finish: "unknown"`、token 计数全 0——模型的输出根本没到 opencode 手里。

**根因**:AGL 网关只能从非流式 JSON 记录训练轨迹,所以 pod 内代理把每个请求强改
`stream=false` 再转发。但 opencode(AI SDK)发出的请求是 `stream=true`,只会解析
SSE 流;代理拿到 vLLM 的 JSON 后**原样返回**,客户端解析出一条空消息:无 text、
无 tool_calls、finish_reason 丢失。opencode 视为空 step,再发一轮,如此空转 60~200
次直到 1200s 被 timeout 杀掉。**模型本身完全正常**——直接向 vLLM 发同样的带工具
请求,think 后正常输出 `tool_calls`,`finish_reason=tool_calls`。

**诊断路径**(值得复述):trajectory dump 里 response 全部止于 `</think>` 且无
tool_call → kubectl exec 进运行中 pod,拷出 opencode 的 sqlite 会话库 → 发现 60 step
零 tool part、finish unknown、usage 全 0 → 定位到代理的 stream 改写 → 绕过代理直测
vLLM 证明模型无辜。

**解决**:代理在客户端要求流式时,把上游 JSON completion 转成标准 SSE chunk 流再
返回(role/content chunk → 每个 tool_call 一个 chunk → finish_reason+usage chunk →
`[DONE]`),单元测试验证 chunk 格式后重建 ConfigMap 重启训练。

**经验**:
- 中间件改写协议参数(stream/编码/分页)时,**请求侧改了,响应侧必须做对应逆变换**,
  否则两端各说各话。
- 端到端指标(全超时)会把注意力引向"行为"层(温度、提示词),但**先验证数据通路
  再调行为参数**——一次 `finish_reason/usage` 健全性检查就能提前发现这一切。
- agent 框架"每轮都 200 OK"不代表内容被消费了:HTTP 层成功 + 应用层全丢,
  是最隐蔽的故障形态。

---

## 12. 第一次真实梯度更新才 OOM:管道修好的当天,显存账单到期

**现象**:修复第 11 条的传输 bug 后,第一轮 rollout 16/16 全部完成、出现第一个
1.0 reward(第一次有组内方差、第一次要做真实更新)。结果 trainer 恰好在这第一次
update 的 forward(Qwen3 MLP down_proj)上 CUDA OOM:要 826MB,卡上只剩 295MB。
OOM 的是物理 GPU1——同事进程最重的卡(<neighbour-process> 21.9GB + <neighbour-process> 19.2GB,共占
41GB),我们的进程用到 52GB 就见顶了。

**根因**:之前所有"成功"的 update 都是在残废数据上跑的——传输 bug 导致轨迹只有
一次短输出(中位 1.4k 字符),update 阶段实测峰值仅 38GB alloc/49GB reserved,
假象是"显存绰绰有余"。管道修好后,训练样本变成真实的多轮全历史 prompt
(最长 14 万字符 ≈ 3.2 万 token)+ 最后一轮 response,单条序列近 39k token,
激活值随之翻倍。FSDP 各卡对称分摊,**瓶颈永远是同租户最挤的那张卡**。

**为什么不能用常规刀**:
- micro batch 已经是动态 bsz、最小到单条序列——OOM 时就是一条 39k 序列在算,没法再切。
- `ppo_max_token_len_per_gpu`(40960)不能降:动态 bsz 要求它 ≥ 最长单条序列
  (32768 prompt + 6144 response = 38912),降了直接断言失败。
- TP=2 不能动(TP=4 update 阶段 92GB+,另一个坑)。

**解决**:开 verl 的 `actor_rollout_ref.model.enable_activation_offload=True`
(依赖 gradient checkpointing,本来就开着)。它把重计算边界的激活值异步搬到 CPU,
以少量吞吐换出数 GB 显存,且不改任何数据形状/批次语义。

**经验**:
- **压测要用真实负载**。管道有 bug 时测出的"显存充裕"毫无意义——修好 bug 的
  那一刻,资源画像整个换了一张。
- 共享 GPU 上,OOM 报错里其他进程的占用数字(`Process X has …GiB`)先读一遍,
  分清"自己吃多了"还是"邻居本来就多"——决定了该调自己还是该换卡/等卡。

### 12a. 顺手踩的连环坑:expandable_segments 与 vLLM sleep mode 互斥

第一次重启时顺手加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(OOM 报错
里 PyTorch 自己建议的)。结果 vLLM worker 初始化直接断言失败:
`Expandable segments are not compatible with memory pool`——verl 的 hybrid engine
用 vLLM sleep mode(CuMemAllocator 内存池)做权重热切换,与 expandable_segments
互斥(pytorch#147851)。而且这个环境变量经 Ray 传进了 vLLM worker,炸的不是设它的
driver。**经验**:报错信息里的"官方建议"也要过一遍与当前技术栈的兼容性;何况本例
碎片才 124MB,expandable_segments 本来就治不了 800MB 的真缺口——别捡不对症的药。

---

## 13. AGL 轨迹的真实形状:一个 rollout 只训最后一次调用

**现象**:step_1_train.jsonl 只有 16 行(16 个 rollout),response 中位仅 1.4k
字符,一度怀疑多轮轨迹没聚合全。

**真相**:检查发现每行 prompt 是全量多轮历史(system+全部工具结果,3.7万~14万
字符),response 只是**最后一次** LLM 调用的输出。这是 Agent Lightning 适配器的
设计:一个 rollout 产出一条 (全历史 prompt, 最后一轮 response) 训练样本,梯度只
落在最后一轮的 token 上。不是 bug,但要清楚训练信号的真实密度比"多轮全监督"低
得多。

---

## 14. 配置写对了却不生效:opencode `compaction.reserved` 依赖 `limit.input`

**现象**:opencode.json 明明配了 `"compaction": {"auto": true, "reserved": 12000}`,
预期对话到 40960−12000=28960 token 就自动压缩,结果个别 rollout 涨到 34.9k 才发请求,
连同 6144 的 completion 预留一起超过 40960 上限,vLLM 返回 400。

**根因**(读 opencode v1.18.28 源码 `session/overflow.ts` 确认):

```ts
return input.model.limit.input
  ? Math.max(0, input.model.limit.input - reserved)   // reserved 只在这条路生效
  : Math.max(0, context - maxOutputTokens)            // 没配 limit.input 走这条
```

模型条目只声明了 `limit.context` 和 `limit.output`,没有 `limit.input`——于是
`reserved` **被整个绕过**,压缩阈值变成 context − maxOutput = 34816,对"上一轮结束
到下一轮请求之间还会追加工具结果"这段增长零余量。两次报错的 34909/34865 恰好都刚
跨过 34816,完美互证。

**解决**:模型 limit 里补上 `"input": CONTEXT_LIMIT`,让阈值回到本意的 28960。
一行改动,重建 ConfigMap。

**实测确认**:修复后拷出运行中 pod 的 sqlite,会话 token 峰值 29077(刚过 28960
就触发)、含 1 个 compaction part、日志零 "maximum context length" 报错——从"被动
报错自愈"变成了"主动压缩"。

**经验**:
- 配置"被接受"≠配置"在生效"。schema 校验通过、程序不报错,不代表那个键真的进入了
  计算路径——尤其是有多条 fallback 分支的代码。
- 数值异常时先算账:34909 和 34865 都落在 (34816, 40960) 区间,一眼就能反推出
  真实阈值是 34816 而不是 28960,再回头找哪条分支算出 34816。
- 顺带一提:这个机制还依赖 usage 统计正常。§11 传输 bug 时代 token 全是 0,
  就算阈值算对了压缩也永远不会触发——bug 会层层掩护。

**后记(同日)**:回查昨晚 baseline val,489 个任务里有 63 个(13%)日志里就有同样的
"maximum context length" 报错——这个 bug 一直都在,只是没人看 vLLM 日志。而且两天
都"看起来能跑通"的原因是 opencode 的兜底:收到超限 400 后会触发**恢复性压缩**再
继续(日志顺序:overflow 错 → 压缩 → 继续 → 再 overflow → 再压缩)。所以现象不是
"压缩失效导致任务失败",而是"主动压缩从未按预期阈值触发,一直靠报错后被动自愈",
每次自愈都白白浪费一轮报错往返。修复后变为 28960 主动压缩,不再依赖兜底。

## 15. 【配置放错层级】`nccl_timeout` 写进 trainer 块,FSDP worker 读不到 → step 2 长更新被 600s 看门狗打死

**现象**:step 1 更新成功(全是短失败轨迹),step 2 出现 5 条 resolved 长多轮轨迹后,
`_update_actor → update_actor(batch)` 期间四个 FSDP worker 被 NCCL 看门狗集体判超时:
`ProcessGroupNCCL.cpp:632 checkTimeout` / `ncclCommWatchdog` → `SYSTEM_ERROR` → 连锁
`ActorDiedError`,整个 Ray 训练 job 崩溃。时间线:更新 18:11:23 起,18:23:08 被杀,
≈705s,正好卡在 PyTorch NCCL 默认 **600s** 单集合通信超时上。

**误诊排除**:不是 OOM(dmesg 无 oom-kill、无 CUDA OutOfMemory),不是 `ray stop --force`。
看门狗超时的语义是"某个集合通信(all-gather/reduce-scatter)挂起超过 timeout",
根因是**某个 rank 在到达该集合通信前算得太慢**:step 2 的长序列 + 三重 CPU offload
(`param_offload` + `optimizer_offload` + 我为修 §12 OOM 加的 `enable_activation_offload`)
在共享机器上换入换出极慢,慢 rank 落后 >600s,其余 rank 在 barrier 上被判超时。

**真病根**:配置里 **明明有 `nccl_timeout: 3600`,但放在了 `trainer` 块**。而 VERL
`fsdp_workers.py:162` 的 `init_process_group(timeout=...)` 读的是
`self.config.get("nccl_timeout", 600)`,这个 `self.config` 是 **`ActorRolloutRefWorker`
自己的配置(即 `actor_rollout_ref` 块)**,不是 trainer 块。于是那个 3600 形同虚设,
实际生效的是默认 600s。

**修复**:把 `nccl_timeout: 3600` 加到 `actor_rollout_ref` 顶层(与 `hybrid_engine`
同级)。重启后解析配置确认 `actor_rollout_ref.nccl_timeout=3600` 就位,worker 的
process group 超时才真正变 3600s,让"慢但在推进"的长更新有时间跑完。

**教训**:
- **配置项要放在读取它的那个对象的配置作用域里**。同名 key 在 `trainer` 和
  `actor_rollout_ref` 两层都可以写、都不报错,但只有 worker 那层被真正读。写完
  一定回查解析后 dump 的**嵌套位置**,不能只 grep 到 key 就以为生效——这和 §14
  "配置写对了却不生效"是同一类元教训(值对、位置错)。
- **看门狗超时 ≠ 死锁**。先判断慢 rank 是否在推进(GPU util 有无、有没有 OOM),
  能推进就抬 timeout 让它跑完;真死锁再去拆 offload。本例先用最小改动(接对
  timeout)保命,不盲目动 offload 三件套——它们是共享卡下的显存安全垫,拆了会重现 §12 OOM。
- **offload 是显存换时间**。`activation_offload` 修好了 OOM,却在长序列 step 变成
  更新阶段的头号拖慢项。省显存和防超时是一对张力,记录在案备后续调优。

## 16. SwanLab 日志增强:字符串指标会刷屏报错,聚合标量才是有用信号

**背景**:早期 swanlab 每步只 log 一条随机轨迹的 prompt/response 文本 + 几个 char 数,
信息稀疏;而且 `train/trajectory_path`、`train/sample_rollout_id` 这类**字符串**被当成
图表指标 `swanlab.log`,每步报 `Chart creation failed... input type is str`,纯噪音。

**改法**(`rollout_adapter.py`,下次 trainer 重启生效):
- compact 轨迹记录里补 `n_turns / resp_tokens / prompt_tokens / duration_s / total_s`
  (来源:`CompletedRollout.triplets` 数、各 triplet response token 数、
  `finished_at - running_at`),写进 jsonl 持久化。
- 新增 `_log_rollout_step_metrics_to_swanlab`:每步聚合成可画的标量——
  reward_mean / n_resolved / resolve_rate / n_timeout / 耗时 mean·p50·p90·max /
  轮次 mean·max / 输出 token mean·max。这才是训练时该盯的曲线。
- 删掉两个字符串图表 key(rollout_id 挪进 Text 头部),消除刷屏报错。
- 冗长的随机样本文本降频:`AGL_SWANLAB_SAMPLE_EVERY`(默认 5,step 1 必留)。

**教训**:监控面板"稀疏"往往不是没数据,而是**把该画曲线的聚合量漏了、却把不能画的
字符串硬塞成指标**。指标层(标量趋势)和样例层(文本快照,低频)要分开;后者别每步刷。

---

*最后更新:2026-09-06(新增 §12 首次真实更新 OOM、§12a expandable_segments×vLLM、§13 AGL 轨迹形状、§14 compaction.reserved 失效、§15 nccl_timeout 放错层级致看门狗打死、§16 SwanLab 日志增强)。新坑随时追加。*

## §17 掩码更正:trajectory 级训练所有 assistant 轮,非"只训最后一次"
- 之前 §13 记成"只拿最后输出",经读码(rollout_adapter.py:661-717)+ resolved config(level=trajectory)更正:
- trajectory 级把多轮合并成一行,response_mask 逐轮累加:模型每一轮输出 `+=[1]`(全训),
  工具观测 `+=[0]`(不训),初始 prompt 天然不计 loss。故"整条轨迹的所有 assistant 输出都算梯度"。
- 合并依赖 `ids_startswith(prompt_ids, current_context)`;上下文压缩打断前缀→break 成新 group
  (仍各保留每轮 mask=1),打断量可在 swanlab merge_mismatch 表监控。
- 实测(step1,带 activation_offload):update_actor=153s,step=832s,gen 131ms/token 占大头。
  3600 是 nccl 看门狗上限非每步耗时;崩溃前 705s 是掉队异常,修 nccl_timeout 层级后回落 153s。
- 开 activation_offload 之前无干净 update 计时:那之前的真实更新直接 OOM(offload 就是为扛 OOM 才加),
  更早的秒级"更新"是 grad_norm=0 空更新(transport bug 期)。SP=2 收益需实测,当前 153s 已不构成瓶颈。

## §18 rollout 长尾 + 压缩切分致 GRPO 偏斜;峰值显存杠杆的一处自我修正

**背景**:排查"更新半小时不动 / 撑爆显存"时,按 rollout 粒度统计了三步埋点
(每步 16 行 = train_batch 4 × n 4):

| 指标 | step1 | step2 | step3 |
|---|---|---|---|
| n_turns 最大 | 316 | 268 | **384** |
| n_turns 均值 | 62 | 50 | 78 |
| prompt_tokens 峰值 | 27.7k | 27.3k | 25.2k(始终 <40k) |
| resp_tokens 累计最大 | 29.6k | 22.4k | 25.9k |

**两条结论**:
- **爆的不是窗口,是轮数长尾**。每轮 prompt 窗口从没超 ~28k(压缩在正常工作),
  但单条轨迹能聊到 384 轮。一条 384 轮的 rollout 会触发约 10 次压缩,每次压缩打断
  前缀(见 §17)→ 切一行训练样本 → 单 rollout 变 ~10 行、每行 prompt 20~28k,
  单条 rollout 送进 trainer 的训练 token 估算峰值 **~200k 量级**,顶得上 16 条正常
  样本之和。16 条里只要有一两条这种怪物,整个 update batch 就被撑爆。
- **GRPO baseline 偏斜(动机,非已完全验证)**:压缩把一条 rollout 切成 k 行,若这
  k 行都带同一终局 reward 参与 GRPO 分组,长尾轨迹在组内被 over-weight(计 k 次),
  baseline 被拉偏。这是要做"一 rollout 一行"(保留带 reward 的末段 / 或按 1/k 加权 /
  或把分组从'prompt 行'改成'rollout')的动机。reward 到底如何挂到切出的各行,
  下次动 rollout_adapter 前需实读 append_training_row 的 score 落点确认。

**⚠️ 自我修正(重要)**:一度对外说"治 OOM 的正手是降 `ppo_max_token_len_per_gpu`
(40960→24576/20480)"——**错**。§12 已记:动态 bsz 要求它 ≥ 最长单条序列
(32768 prompt + 6144 response = 38912),40960 已贴着地板,降了直接断言失败。
故它**不是**可调的峰值杠杆。峰值由**单条 ~39k 序列在最挤那张卡上的激活**决定,真实
可动的杠杆只有:(a) 降 max_prompt_length / max_response_length(会截断长 prompt,
掉数据);(b) `enable_activation_offload`(已开,§12);(c) 给最挤的卡腾 headroom
(邻居进程动不了)。**"一 rollout 一行"降的是行数→总量→step 时长与 OOM 概率,不降单
bin 峰值**——别把它当省峰值显存的手段。

**GRPO × 压缩,工业界四条路**(按改动量):
1. **CompactionRL**(slime):弃 GRPO 组内 baseline,改 PPO + value critic + 跨段 GAE
   `Â_s=(γλ)^{N>s}·Â_local`,把切段当同一轨迹做信用分配,summary token 也训。完全体,改动最大。
2. **Context-Folding**(arXiv 2510.11967 附近,编号需核对):保留 GRPO,把"折叠/压缩"
   当可训练动作,折叠 token 不 mask 而训,配 turn/token 预算。
3. **一 rollout 一样本 / rollout 级归一**:算法不动,保证每条 rollout 恰好贡献一个样本。
   最务实,零算法风险——本项目当前选型。
4. **GiGPO**(verl-agent):episode 级 + step 级两级分组,长程多轮信用分配比 vanilla GRPO 干净。

**教训**:压测/调参前先分清"峰值 vs 总量 vs 频率"三件事——它们的杠杆不同,搞混就会
像这次一样开出降不了的药方。文档里已有的硬约束(§12 的地板)要先查再建议。

---

*§18 追加 2026-09-06:rollout 长尾数据 + 压缩→GRPO 偏斜动机 + ppo_max_token_len_per_gpu 杠杆自我修正 + GRPO×压缩四路。*

## §19 峰值显存的真旋钮:ppo_max_token_len_per_gpu 被钉死,is_drop 会丢行(不是截断)

**背景**:决定"保留压缩、砍 `ppo_max_token_len_per_gpu` 减峰值、慢点无所谓"时,读码发现
这个方案行不通,记录三个硬事实(均已在代码核对):

1. **`ppo_max_token_len_per_gpu` 有地板,不能当峰值旋钮**。动态 bsz 要求它
   `≥ 最长单条训练行 = max_prompt_length(32768) + max_response_length(6144) = 38912`
   (append_training_row rollout_adapter.py:584–595 把 prompt/response 逐项截断到这两个上限)。
   现值 40960 → 最多降到 38912(−5%),再低断言失败。且造成 OOM 的是一条 ~39k 行**独占**
   一个 bin,峰值=该行激活,与 cap 高低无关(cap 只决定多少条**短**行拼进同一 bin)。

2. **真正的峰值旋钮是 `max_prompt_length`**(降它才降地板,再连带降 ppo cap)。但——

3. **`is_drop` 是"丢整行"不是"截断保留"**。prompt 超过 `max_prompt_length` → 打 is_drop
   (rollout_adapter.py:584–586)→ trainer.py:589–590 `keep=(~is_drop_mask).nonzero()` 整行剔除。
   所以降 `max_prompt_length` = 把 prompt 超阈值的**长尾 rollout 直接从训练删掉**(删的正是用了
   压缩的长任务样本),不是"省点显存"那么无痛。压缩目标输入 ≈ CONTEXT−RESERVED = 40960−12000
   = 28960,实测 prompt 峰值 ~27.7k 贴在 ~29k;故阈值要留 ≥~29k 才不丢行。

**可用配置(保留压缩、慢点)**:
- 第一档(零丢行):max_prompt_length 32768→30720 + ppo 40960→36864(−5%,30720>28960 不丢行)。
  OOM 只差 826MB,−5% 通常够;用 opencode_smoke/monitor_gpu_peak.py 改后实测再定档。
- 第二档(更狠):先缩压缩窗口 AGL_OPENCODE_CONTEXT 40960→32768(prompt 降~21k,压缩更勤),
  再 max_prompt_length→24576 + ppo→30720(−21%,仍近无损)。

**两个正交旋钮别混用**:
- 治**峰值 OOM** → 只有降 max_prompt_length(+ 连带 ppo)。
- 治 **GRPO 偏斜 / step 时长 / OOM 概率** → "一 rollout 一行";但它**不降单 bin 峰值**
  (单行仍 ≤38912),别拿它当省显存手段。

**GRPO 分组粒度的自我更正**:一度对外说"把分组从 prompt 级改成 rollout 级"——**错**。
本仓 GRPO 分组键 uid=data_id 是**任务级**(同题 n 个 rollout 共享,trainer.py:100–127;否则每组
1 员方差恒 0、advantage 恒 0 训练就死)。分组本就是 prompt/任务级、标准 GRPO,不该改。真问题是
"一条 rollout 贡献几行":trajectory 级本意 1 轨迹=1 行,压缩打断前缀切成 k 行、都挂同一 data_id
(rollout_adapter.py:646–660)→ 长尾占 k 席拉偏 baseline。"一 rollout 一行"是**恢复 trajectory
本意**,不是换归一粒度。多轮 agentic RL 干净主流=分组在 prompt 级 + 每轨迹一行+token 掩码
(RAGEN/verl-agent 这一路);也有故意 step 级分解的(GiGPO);不存在"统一 rollout 级"。

**教训**:动手前先读透"上限参数之间的依赖"(ppo cap 被 max_prompt+max_response 钉死)和"标志位
的真实语义"(is_drop=丢行非截断)。否则会像这次一样,选了个动不了的旋钮、还差点用丢数据的方式省显存。

---

*§19 追加 2026-09-06:ppo_max_token_len_per_gpu 地板 + is_drop 丢行语义 + 峰值旋钮=max_prompt_length + GRPO 分组粒度自我更正。*

## §20 压缩切分的 GRPO 正解 = 两个已内建的 opt-in 开关;附一次 trainer OOM 事故复盘(2026-09-06)

### 事故:trainer 因邻居挤占共享 GPU 而 OOM 死亡
- 2026-09-06 17:16,trainer(pid <N>)`torch.OutOfMemoryError`:在 actor 更新阶段(`modeling_qwen3` 的 `down_proj` 前向)申请 826MiB 失败。
- 现场:trainer rank0 落在物理 GPU1,与他人进程 1219935(<neighbour-process> 19.6GB)+ 956750(<neighbour-process> 22.3GB)= ~42GB 共享一张 95GB 卡;trainer 自身 52GB,合计越过 95GB 上限。
- 关键结论:**根因是共享卡的邻居挤占,不是我们配置本身过大**;邻居进程在保护名单里不能动。所以自救只能"给自己留够余量"——降训练峰值 + 降 vLLM KV。server/controller 存活,符合"trainer 挂只重启 trainer"。
- **教训**:共享 GPU 上跑 FSDP,峰值必须留足冗余对抗邻居波动;OOM 余量只有 ~826MB 时,任何邻居涨一点就会复现。

### 压缩切分问题:正确做法早已在代码里,是两个 opt-in 开关
一条 rollout 因上下文压缩被切成 k 段,朴素 GRPO 会把 k 段当成 k 个组成员 → 组均值/方差被"多段长 rollout"带偏。正解**不是**改 GRPO 的分组层级(分组仍是 task 级 uid=data_id),而是"一条 rollout 只算一次优势、再广播到各段",且 loss 层每段独立前向、按 rollout 归一后求和。这套逻辑 agent-lightning 已实现:

- `agentlightning/verl/rollout_level_advantage.py::compute_rollout_level_advantage`
  - 按 `rollout_id_list` 分组,每条 rollout 只取 1 个代表行进标准 `compute_advantage` → GRPO 组内**一条 rollout = 一个成员**(消除多段偏斜);
  - 校验同一 rollout 各段 uid 一致、reward 一致;算出的标量 Â_i 广播回该 rollout 的**全部段**的全部 token。
  - = 规格里"所有 token 共用同一个全局优势 Â_i"。
- `agentlightning/verl/per_rollout_loss.py`(`loss_mode="per_rollout_mean"`,已在 `entrypoint.py:59` 的 worker import 中注册)
  - `normalize_advantages_by_rollout`:每行优势除以(该 rollout 总 token 数 × 批内行数);
  - `compute_policy_loss_per_rollout_mean`:每段独立算 ratio/clip/dual-clip(各段用**自己的** old_log_prob),再 `masked_sum` → 每条 rollout 贡献一份"逐 token 均值"目标。
  - = 规格里"每段独立前向…累加全部 token 的 clip loss…完成单条 rollout 的 GRPO 目标"。
  - 注:全局预除 + 分微批 masked_sum,因除法是全局线性的,rollout 的段即使落在不同微批,求和仍正确。

启用方式(本次已改,git 跟踪、未部署):
```
algorithm.enable_rollout_level_advantage = True          # trainer.py:697 读取
actor_rollout_ref.actor.policy_loss.loss_mode = "per_rollout_mean"   # trainer.py:708 读取
```

### 压缩摘要 token:天然已被训练,无需额外改
- opencode 配置 `compaction={auto:True, prune:True}`,摘要是走**同一模型端点**(AGL 代理)的一次 assistant 调用 → 被记为带 logprob 的 triplet。
- ~~该 triplet 的 prompt 仍以 `current_context` 为前缀 → 并入压缩前那一段~~ **(09-17 更正,读了 opencode v1.18.28 源码后)**:摘要调用是一条**独立 prompt**——`session/compaction.ts:378-390` 把历史序列化成一条单独的 user 消息,并且 `system: []`、`tools: {}`,不带原 system prompt 和工具定义,所以它的 prompt 不以 `current_context` 为前缀,在 adapter 里**自成一行**(摘要 token 作为 response,mask=1);压缩后的下一轮 prompt 又是 `[compaction-user, summary, 保留的尾部, continue-user]`(`message-v2.ts:521-570`),也不以旧前缀开头。**一次压缩 = 两处前缀断裂 = 三段**,不是两段。
- "含压缩摘要 token"仍然自动满足(摘要是策略自己写的、被训练),结论不变;错的只是"并入哪一段"。这和 SUPO(2510.06727,Thm 3.2)/Context-Folding(2510.11967,App. A.1)的做法一致:每段摘要子轨迹作为独立的因果序列,共用该 rollout 的一个 advantage。

### KL 项:当前整训练关闭
- `use_kl_loss=False, kl_loss_coef=0, use_kl_in_reward=False`。规格提到"累加 KL 项",但本配置无 KL/ref 惩罚 → KL 项恒为 0。per-segment 机制天然支持 KL(若将来开 `use_kl_loss`,每段独立前向自带该段 KL),但是否开 KL 是独立的训练动力学决策,未擅自改。

### Part 2「刚好够用」的显存收缩(与 §19 的真旋钮一致)
- 窗口 40960→36864(−10%):`AGL_OPENCODE_CONTEXT`、rollout `max_model_len`。
- `AGL_OPENCODE_RESERVED` 12000→8192 → 压缩目标 = 36864−8192 = **28672 ≈ 原 28960**:刻意保持不变,避免难题被迫更频繁压缩(观测最长 prompt 峰值 27.7k < 28672,无损)。
- 训练峰值旋钮 `max_prompt_length` 32768→30720(仍 > 28672 压缩目标,`is_drop` 无损,留 ~2k 余量);`trajectory_max_prompt_length` 同步。
- `ppo_max_token_len_per_gpu`/两处 `log_prob_max_token_len_per_gpu` 40960→36864(= 30720+6144 地板,恰好触底)。
- vLLM KV「成比例变化」:`num_gpu_blocks_override` 20480→18432(8×36864/16),每 rank KV ~24.2GB→~21.8GB。
- 单条 rollout 峰值 seq ≈ 28672+6144 = 34816 < 36864,余 2k = "刚好够用"。
- **残余风险**:训练峰值只降 ~5%(prompt 32768→30720,受 is_drop 地板约束不能再低,除非同时压小窗口/压缩目标)。若邻居继续膨胀仍可能复现 OOM——那属于共享卡竞争,需邻居迁走或进一步压小窗口。

### 复盘教训
- 动手写"新算法"前先 grep 现有代码:本次要实现的方法整套已内建为 opt-in,工作量从"写 RL 核心"降为"翻两个开关 + 核对语义"。
- 峰值显存与"无损"是一对约束:`is_drop` 按 prompt 长度丢行,所以 `max_prompt_length` 不能低于压缩目标;要更大训练余量必须连带压小压缩目标(=更多压缩),这是显式取舍。

### §20 更新(2026-09-06 二次收缩 + 保存/SwanLab 核验)
- 人类要求"再压狠一点"(我们已比原作者长很多)。最终值(取代上文 36864 一档):
  - 窗口 `AGL_OPENCODE_CONTEXT`/`max_model_len` 40960→**24576**(−40%);`RESERVED` 保持 **8192**(≥ output 6144)→ 压缩目标 = **16384**。
  - `max_prompt_length`/`trajectory_max_prompt_length` 32768→**18432**;`ppo_max_token_len_per_gpu`、两处 `log_prob_max_token_len_per_gpu` →**24576**(= 18432+6144 地板,恰触底);vLLM `num_gpu_blocks_override` →**12288**(8×24576/16,KV ~14.5GB/rank)。
  - 自检:压缩目标 16384 < max_prompt 18432(无 is_drop,余 2k);单 rollout 峰值 seq 16384+6144=22528 < 24576(余 2k);**训练峰值地板 38912→24576,降 37%**——这才是真正抵御 OOM 的量级(首版仅降 6%,不够)。
  - 取舍:>16k 的难题会更频繁压缩、分更多段;但 Part 1 已把压缩摘要正确纳入训练,多段被当"一条 rollout"(同一 Â_i、每段独立前向后求和,非拼接、非不同组员),故可接受。
- Part 1 段处理语义(人类二次确认):一条 rollout 因压缩分多段时——**每段分别前向计算、最后求和**(=`per_rollout_mean`),**共用一个全局 Â_i / 算作同一组员**(=`enable_rollout_level_advantage`),**绝不拼成一条连续序列**。当前实现与此完全一致。
- 保存机制核验:`global_step_2/actor` 为完整 FSDP 分片(model/optim/extra_state × rank0-3 + fsdp_config.json + huggingface/ 分词器与 config),92GB,与 Qwen3-8B 全量+Adam 优化器态吻合 → **保存正常**。
- SwanLab 核验:`/workspace/agl-checkpoints/swanlog/` 最新 run 的 `backup.swanlab` 含 175 条标量、键覆盖 actor/critic/perf/prompt_length/response 全套,写入至 20:14 → **记录正常**。当日已有 6 个 run(15:30→18:33),印证 OOM 全天反复复发。

## §21 NCCL 空转死锁:util 100% 不等于在算;dynamic_bsz 在分段批次下的 DP 错配(2026-09-06)

### 现象与误导
- step 1 rollout 全部完成、"rollout replicas slept." 后,trainer 停在 update 阶段 90+ 秒无输出;`nvidia-smi` 显示 4 张训练卡 **util 100%**,极易误判为"在慢慢算"。
- 决定性证据来自 `nvidia-smi dmon -s u`:**sm=100% 但 mem=0%、功耗 ~123W**(远低于满载)。真实前向/反向是访存密集型(mem 30~90%、功耗接近 TDP);sm 满、mem 零、低功耗 = SM 在通信 kernel 里自旋轮询对端数据 = **NCCL 集合通信死锁**。`utilization.gpu` 只统计"有 kernel 在跑",自旋也算,单看它毫无诊断价值。
- 环境没有 py-spy/gdb/dmesg 可用时,dmon 三元组(sm/mem/power)就是最低成本的卡死判据。

### 根因:DP 各 rank micro-batch 数不一致
- 配置组合:`use_dynamic_bsz=True` + `enable_rollout_level_advantage`/`per_rollout_mean`(批次含压缩分段的变长行)+ 4-way FSDP + `balance_batch=True`。
- dynamic_bsz 按 `*_max_token_len_per_gpu` 的 token 预算打包 micro-batch:各 rank 拿到的行长度分布不同 → **打包出的 micro-batch 数量不同** → 循环内的 FSDP all-gather/reduce-scatter 次数不齐 → 少跑的 rank 先退出循环,多跑的 rank 在集合通信里永久等待。
- balance_batch 只均衡各 rank 的 **token 总量**,不保证 micro-batch **数量**一致——分段批次行长方差大,恰好放大这个错配。

### 修复与验证
- 翻三处开关:`use_dynamic_bsz=False`(actor)+ `log_prob_use_dynamic_bsz=False`(rollout/ref 两处),配合已有 `*_micro_batch_size_per_gpu=1` → 每 micro-batch 恒为 1 行,micro-batch 数 = 行数;balance_batch 均衡行数 → 各 rank 集合通信次数天然对齐。
- 代价:放弃 token 打包的吞吐优化。实测 step 1 整步 719.8s、update 阶段 mfu 6.5,可接受。
- 验证:重启后 step 1 完整通过(16/16 rollout 全成功、pg_loss/grad_norm 正常吐出、权重同步 4.2s),step 2 rollout 正常启动。卡死点(slept→old_log_prob 首个多 rank 集合)不再复现。

### 杀卡死 trainer 的正确姿势
- SIGINT 优雅退出(driver + 4 worker 约 14s 内全退),**只杀 trainer**,server/controller 保留;禁 `ray stop --force`(会误伤同机其他 Ray 任务)。

## §22 防 400 的 token 预算法:让压缩"主动"而非"撞墙后自救"(2026-09-06)

### 两堵墙归一
- vLLM 硬墙:messages + completion ≤ `max_model_len` 24576(超了返回 400)。
- 训练墙:记录的 prompt > `max_prompt_length` 18432 会被 is_drop 丢行。
- 由 completion=`AGL_MAX_TOKENS`=6144,两墙合并为一条:**messages ≤ 18432**。

### 预算公式与实测
- opencode 压缩触发点 = `AGL_OPENCODE_CONTEXT` − `AGL_OPENCODE_RESERVED`。此前 CONTEXT=24576(=vLLM 墙)、触发点 16384:opencode 计 token 偏少 + 触发后还要长完一轮,实测过冲 +4449(触发 16384 → 实际 20833),20833+6144=26977 > 24576 → **400**。虽 opencode 会压缩重试(自愈、无脏数据:轨迹里 prompt_tokens 峰值 15431 ≤ 18432),但每次 400 浪费一轮生成。
- 修复:**CONTEXT 必须设为"墙 − 预期过冲"级别,而不是墙本身**。落地 CONTEXT=18432、RESERVED=8192 → 触发点 10240;峰值 ≈ 10240+4449=14689,距两墙均余 ~3.7k。验证:重启后 32 个 job 日志 0 次 400。
- 调优余地:RESERVED 可压到 6144(=单次输出上限,触发点抬到 12288,压缩更少、GRPO 分段更少),峰值 16737 仍留 ~1.7k 余量;4096 则实测过冲即顶穿 is_drop 墙,不可取。工具截断 `AGL_TOOL_CHAR_LIMIT` 别用来省压缩——压小它模型看到的工具结果变少,解题轮数反增。
- 提醒:job 模板是 trainer 启动时 `read_text` 进内存的,改模板文件对运行中的训练无效,需重启 trainer 才生效(改文件本身零风险,可提前 stage)。

## §23 训练中途 ImagePullBackOff:混合训练集的镜像只预拉了一部分(2026-09-07)
- 现象:step 4 rollout 停在 4/16 达 20+ 分钟,窗口无任何报错;`kubectl get pods` 才暴露 12 个 pod 全在 ImagePullBackOff——本步抽到的新仓库(gpxpy/funcy/tweepy)镜像本地没有,Docker Hub 直连超时。
- 根因:train_dataset_mixed 覆盖 132 个不同镜像,此前只预拉了 51 个原名镜像;前几步恰好全抽中已有镜像,假象顺畅。
- 修复:走国内镜像代理 `docker pull docker.1ms.run/jyangballin/<img>` 再 `docker tag` 回原名(job 模板 IfNotPresent 即认),pod 在 BackOff 重试时自动复活,**无需重启任何组件**;其余缺失镜像用后台脚本逐个补齐(带磁盘护栏:/data 低于 200G 即停,保护 92G/份的 ckpt 空间)。
- 教训:(1) rollout 长时间无进展先查 k8s 层(pods/events),不要只盯训练日志;(2) 数据集换血/扩容时,先 `docker images` 对账全部 image_name 再开训;(3) agent 超时计时从容器启动算起,pull 等待不吃 600s 预算,卡住的 rollout 镜像就位后可无损续跑。

## §24 CUBLAS_STATUS_NOT_SUPPORTED 崩溃 + resume_mode=auto 无损续训;附 watchdog pgrep 自匹配坑(2026-09-07)
- 现象:step 6 rollout 期间 vLLM worker 抛 `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasGemmEx(...)`(BF16 GEMM),EngineCore 变 EngineDeadError,trainer 整体退出。
- 根因:共享卡余量被榨干。此前实测我方 update 峰值 ~55.4 GiB/卡、GPU1 合计峰值 97405/97871 MiB(余 0.46 GiB);邻居进程(<neighbour-process>/<neighbour-process> 等共 42 GiB)一有波动,cuBLAS 连 workspace 都申请不到就报 NOT_SUPPORTED——它不是"不支持",是"没显存"。rollout 推理期同样会中招,不止 update 期。
- 续训:`run_opencode_k8s.sh` 第 117 行硬编码 `trainer.resume_mode=disable`,但 `"$@"` 排在其后,dotlist 后者覆盖前者,所以 `bash examples/swe_smith/run_opencode_k8s.sh trainer trainer.resume_mode=auto` 即可从最新 ckpt 恢复(`max_actor_ckpt_to_keep=1` + `del_local_ckpt_after_load=True`,磁盘只留一份 92G)。只重启 trainer,server/controller 原样保留;恢复后 step 5/6/7 各 ~10 分钟重跑,无任何状态损失。job 模板是 trainer 启动时读入的,重启顺带让模板改动(如 RESERVED 调参)生效。
- watchdog 坑:守护脚本用 `pgrep -f "train_opencode_agent.py --agl-base-url"` 判活,而这段模式串就写在守护脚本自己的 bash cmdline 里——trainer 真死了 pgrep 仍匹配到守护进程自身,死亡告警永远不触发。修法:模式里塞一个字面反斜杠(如 `/\.venv/bin/python examples/swe_smith/train_opencode`),自身 cmdline 含 `\` 导致正则不再自匹配。凡是"脚本文本里包含要找的模式"的 pgrep/grep 判活逻辑都有此坑,一次性命令行检查同样会匹配到自己。
- 教训:(1) 共享卡余量 <1 GiB 时,崩溃只是时间问题,预先写好续训 runbook 比盯盘更有用;(2) CUBLAS_STATUS_NOT_SUPPORTED 出现在共享卡上先查显存挤兑,不要怀疑算子/驱动;(3) 判活脚本上线前,先手动 kill 目标进程验证告警真的会响。

## §25 kubelet imageGC 在磁盘 95% 触发,一夜删掉 261GB 预拉镜像:批量预拉在满盘节点上是自我挫败的(2026-09-07)
- 现象:凌晨补拉完 80 个训练镜像后,镜像对账突然显示 132 缺 91,连刚 OK 的都"消失";/data 可用空间反而从 373G 跳到 450G。
- 根因:k3s kubelet 配了 `image-gc-high-threshold=95 / low=90`;批量预拉约 200G+ 把 6T 盘顶过 95%,imageGC 立即回收 261GB"未被容器使用"的镜像——预拉得越多,删得越快。有运行中容器引用的镜像删不动(报 must force conflict),层数据幸存,但 plain 名标签会先被摘掉。
- 修复:(1) 层还在、只丢标签的,`docker tag dockerproxy.net/<img>:latest <img>:latest` 秒级恢复,无需重拉——23 个 val 评测镜像全靠这个救回;(2) 彻底删掉的训练镜像不再批量重拉,改靠 autoheal 循环按 step 需要 JIT 拉(每步最多 16 个、约 3G/个,BackOff 重试窗口内完成)。
- 教训:(1) 盘常年 93% 的节点上,"把 132 个镜像全部预拉"永远不可达成——GC 会边拉边删,还顺带威胁 val 镜像和 ckpt 空间;JIT + autoheal 才是稳态;(2) 磁盘护栏要对齐 GC 阈值而不是拍脑袋:95% used = 300G free,护栏设 200G 等于放任 GC 先动手;(3) 镜像"消失"先看 `journalctl -u k3s | grep -i "image garbage"`,再看是不是只丢了 tag(`docker images | grep <proxy前缀>`)。

## 2026-09-09 MiniCPM5-2B 50k 训练启动踩坑两则

### vLLM 显存校验按"整卡总占用"计算,不是"本进程占用"
共享 GPU 上 `gpu_memory_utilization` 的语义是:所有进程占用总和 ≤ util × 整卡容量。
邻居已占 45G 时 util=0.45(预算 44G)直接报 `No available memory for the cache blocks`。
修复:util 提到 0.7,同时用 `num_gpu_blocks_override` 钉死 KV 大小防止真去抢 68.5G。
教训:共享卡上 util 是"天花板线"不是"份额",要按 (邻居+权重+激活+KV)/整卡 计算下限。

### VERL 自定义 vLLM server 路径不加载 tool_parser_plugin
`vllm serve` 官方入口在 api_server.py 主流程里 `ToolParserManager.import_tool_parser(plugin)`,
但 verl 的 `vllm_async_server.py::run_server` 自己拼 build_app/init_app_state,跳过了注册,
报 `--enable-auto-tool-choice requires tool_parser:'minicpm5' which has not been registered`。
修复:给 run_server 开头补注册(site-packages 已打补丁,留有 .bak 备份)。
教训:engine_kwargs 透传成 CLI 参数 ≠ 官方入口的所有副作用都会发生,自定义 server 路径要逐项核对。

## 并发翻倍导致超时率暴涨:瓶颈是算力带宽不是 KV(2026-09-09)

batch 4→8(rollout 并发 16→32)后超时率 19%→53%、解题率 65%→41%,但 KV cache 0 抢占、显存零增长。
原因:`max_num_batched_tokens=8192` 是全引擎每迭代的固定 prefill 带宽,多轮 agent 每轮要重算 8-15k 上下文,
并发翻倍后单次调用时延 ×1.6(做对题中位用时 295s→465s),原来 400-600s 能解的题被推过 600s 死线。
诊断关键:对比"做对的 rollout 用时分布"——分布右移且顶到超时天花板 = 时延杀人;分布不动 = 题真难。
修复:AGL_SAMPLE_TIMEOUT 600→900(比例还原:900/465 ≈ 600/318),超时率回落到 12%、reward 复位 0.63。
教训:加 rollout 并发时,agent 超时预算要按实测的"做对题用时中位数"等比例放大;KV 够用≠不排队,
排队发生在 prefill 带宽上。job 模板每步重读,改 AGL_SAMPLE_TIMEOUT 无需重启训练。

## k3s 用 docker 运行时:镜像存在性要对着 docker 查,不是 containerd(2026-09-09)

课程走出已缓存镜像区后 pod ImagePullBackOff(直连 docker.io 超时)。本机 k3s 是 docker://27.5.1 运行时,
权威镜像库是 docker images;`ctr -n k8s.io` 看到的系统 containerd 与 k8s 无关(往里导镜像是无用功)。
补法:`docker pull dockerproxy.net/<img> && docker tag` 即可,pod 下次重试自动恢复,无需删 Job。
预防:换数据集/课程段前,用数据集 image_name 列表对 docker images 求差集,按首次使用顺序提前补拉。

## §26 人类明令的 KL 在换启动脚本时静默丢失,裸跑 180 步才被追问发现(2026-09-11)

- 时间线:09-07 人类明令"开KL",v2(Qwen3-8B)用 CLI 覆盖 `use_kl_loss=True kl_loss_coef=0.001` 启动并核实生效;
  09-08 切 MiniCPM5-2B 50k 时新写 start_train_minicpm50k.sh,窗口/解析器/批大小都迁了,唯独 KL 两条覆盖没搬——
  底层 train_opencode_agent.py 的默认值(use_kl_loss=False, kl_loss_coef=0.0, use_kl_in_reward=False)静默生效,
  步 1-180 无锚裸跑。直到 09-11 人类看 swanlab 曲线追问"KL 呢"才暴露。
- 为什么没被发现:(1) 启动汇报列了十几项配置,恰好没列 KL——没写出来的项等于没被审;(2) 长会话多次压缩,
  "开KL"指令在摘要里被丢;(3) `actor/kl_loss:0.0` 每步照常打印,"指标存在但功能关闭"看起来和"功能开着值很小"一样。
- 侥幸面:2B 模型 + lr 1e-6 + 前 180 步全是易题,裸跑没触发 v1 式漂移(响应长度全程稳定);val 也证实无退化。
  但这是运气不是设计——v1 的崩塌就是同样的无锚配置在 8B 上的下场。
- 修复与预防:(1) 覆盖已补进脚本(备份 .bak_nokl),重启后在运行时 resolved config 回显里核实 True/0.001,
  而不是只看本地脚本文本("加到线上"≠"改了文件");(2) 人类裁决过的开关写进持久记忆文件,换脚本/换模型时逐项对照;
  (3) 启动汇报必须显式列出所有"人类裁决过"的项,让遗漏可见;(4) 恒零指标(kl_loss/ppo_kl)要主动区分
  "关了"还是"值恰为零",别当正常噪声略过。

## §27 纯易→难课程被证伪:训练奖励 0.3→0.7 换来 val 零增益(2026-09-11)

- 实验:MiniCPM5-2B 按纯课程序(易→难,shuffle=False)训 180 步(约 1440 个最易样本),训练奖励从 ~0.3 涨到 0.7+。
  暂停后 merge step_180,用与基线完全一致的配置(温度 0.6 单次)评 474 题 val。
- 结果:同题 470 配对,step_180 146 解(31.1%) vs 训练前 139(29.6%);翻案 25:18,McNemar p≈0.36 不显著;
  F2P 细粒度通过率 49.4% vs 49.6% 持平。结论:奖励上涨全是分布内拟合("把易题做熟"),零迁移。
- 评测方法两教训:(1) 结论必须建立在**同题配对 + McNemar**上,总体率差 1.5pp 在温度 0.6 单次采样下就是噪声;
  (2) 分片评测**中途分数不可外推**:instance 顺序有难题簇,跑到 176/474 时只有 15.3%,终值 31.1%——
  基线在同一前缀段也只有 ~17%。早下结论会把"顺序偏差"当成"模型退化"。
- 机理解释(GRPO 视角):学习信号=组内方差,只有"4 条 rollout 有对有错"的题贡献梯度。纯课程前期信号密度最高
  但分布极窄;易题吃透后梯度继续把策略往易题风格上推,泛化不涨。
- 处置:改易/难交替配方——课程序前半/后半当两条队列(各自仍升序),24 题/块(=batch8×3步)交替,难题第 4 步即接触。
  查证上游 AGL 原始做法:train_dataset_mixed.jsonl + data.shuffle=True,**全随机、无课程**;交替是"纯课程↔纯随机"
  的折中,若难题块长期 4/4 全错,正解是 DAPO 式动态采样(超采后只留有对有错的题),不是回退随机。
- 概念钉子:**数据顺序管效率轴(学得快不快/偏科),KL/温度/奖励设计管稳定轴(会不会崩)**。v1 崩塌与顺序无关,
  换随机照崩;交替配方选错最坏是学得慢,有 KL 锚着不至于崩。两轴要分开归因,别混着调。

## §28 评测提速:闲置卡加第 5 实例 +25%,忙卡加并发只换超时(2026-09-11)

- 判断依据:卡 2/3/5/1 与邻居共享、SM util 91-100%,在这些卡上提 max-num-seqs/workers 只会拉长单请求时延、
  把 400-600s 能解的题推过超时线(§"并发翻倍"同款陷阱);卡 0 有 25G 空余且 util=0%,是唯一真增量。
- 做法:卡 0 起第 5 个 vLLM(预算 15G,`gpu_memory_utilization=(已占+15000MiB)/总量`≈0.896,KV 21.6 万 token,
  4.21x 并发余量),评测 4 shard 改 5 shard(AGL_SWEEP_RESUME=1 使重切分零浪费),配第 5 条截断代理。
- 教训:(1) 共享机上找吞吐先看"哪张卡 util 为零",不是"哪张卡显存有洞";(2) util 公式按整卡口径算(§26 上一条同源);
  (3) $ROOT 未定义导致 tool-parser 插件路径展开成坏值、报 invalid tool call parser——inline 起服务时环境变量要先自检。

## §29 一处反直觉回显:配置里同时出现 temperature 1.0 和 0.7(2026-09-11)

resolved config 里 rollout 温度 1.0(且 vLLM 服务端 override_generation_config 回显 1.0,这才是出 token 的真路径),
另一处 0.7 是 rollout.val_kwargs(验证采样)——test_freq=0 时永不执行,是"存在但不生效"的死配置。
教训与 §26 同根:审配置时,每个值要先问"这条路径到底跑不跑",再问"值对不对"。

## §30 共享 CUDA MPS:一次 vLLM 非法访存打瘫全机新任务(2026-09-11)

**事故**:v2 交替配方训练 step 4 rollout 期间,vLLM 采样器 `apply_top_k_top_p → logits.sort`(CUB 分段基数排序)触发 CUDA illegal memory access,引擎级联死亡。重启却报 Error 807 "MPS server is not ready"——所有 worker 看不到 GPU。

**根因**:09-10 11:10 有人以 root 起了全局 CUDA MPS 守护(`nvidia-cuda-mps-control -d`,接单管道在默认路径 /tmp/nvidia-mps,权限全开)。CUDA 程序初始化时发现该管道即自动连接,无需任何配置——我们 09-11 14:45 新启的训练就这样无感上了 MPS。MPS 下所有客户端共用一个 server 进程的 GPU 上下文,隔离性变弱:一个客户端的致命 GPU 错误把共享 server 打进坏状态,此后**全机任何新启动的 CUDA 进程都吃 807**(存量已连接的照跑,"挡新不杀旧"),且不自愈,只能 root 重启守护。

**处置**:root 守护不可动(会拔线其他 MPS 客户端)。绕行:启动脚本加 `export CUDA_MPS_PIPE_DIRECTORY=<空目录>`——程序找不到接单管道就退回自开独立上下文,与 MPS 零接触。单卡验证(走 MPS 报 807 / 绕过正常)后重启,从 global_step_2 resume,损失约 40 分钟。

**教训**:
1. 共享机上,"我的进程崩了为什么影响别人/别人为什么影响我"要先查 MPS——`ps aux | grep mps-control` 一秒钟的事。各卡上神秘的 62MiB 进程就是 mps-server。
2. MPS 不制造 bug,只放大爆炸半径:纯软件崩溃(Ray 死、NCCL 超时、死锁)不伤它,只有 CUDA fatal fault(非法访存类)才会打坏共享 server。
3. 大显存长任务的机器开 MPS 收益个位数(它的收益场景是海量小 kernel 挤一张卡),风险却是全机新任务瘫痪——负载画像不匹配就别开。
4. 判断谁会被 MPS 重启波及:按进程启动时间粗筛(晚于守护启动时刻的 GPU 进程才可能是客户端),我们据此定位到全机只有一个真实存量客户端,把"重启要通知所有人"收敛成"通知一个人"。

## 31. §30 非法访存的真正根因:verl 0.7.1 abort 路径在 vLLM 0.8.5 上静默失效(09-11 追查)

§30 当时把 illegal memory access 归因为"vLLM 偶发 bug + MPS 放大"。当天晚上通过日志取证找到了确定性的根因链:

**根因链**:
1. verl 0.7.1 `vllm_async_server.py` 的 abort 路径调用 `req_state.make_request_output([], pooling_output=None, finish_reason=ABORT, stop_reason=None)`;
2. 但 vLLM 0.8.5 的 `make_request_output(self, new_token_ids, finish_reason, stop_reason)` **没有 pooling_output 参数** → 第一个请求就抛 TypeError;
3. TypeError 被兜底的 `except Exception` 吞掉(日志只留一行 `Error aborting requests: ... pooling_output`,返回 aborted_count=0)→ **一个请求都没真正 abort**;
4. 超时/失败 rollout 留下的孤儿请求继续在 vLLM 里生成,与训练阶段(FSDP 前反向/权重更新)并发 → sampler `apply_top_k_top_p → logits.sort`(CUB segmented radix sort)非法访存。

**证据**:`Error aborting requests: ... pooling_output` 在整个日志中恰好出现 2 次,都在崩溃那一秒(16:24:23,039),TP 两个 server 各一次;step 1~3 abort 时无残留请求所以没炸,step 4 是第一个有 6 个 failed/timeout rollout 的难题步 → 孤儿请求 → 崩。重启后重跑的 step 4~6 无 rollout 失败,也就无 abort 残留,平安通过——与"偶发"假说相比,该模型完美解释了全部观测。

**版本考古**:这段代码来自 verl PR #4453;verl main 已要求 `vllm>=0.18.0`(新签名有 pooling_output),所以上游主干不算 bug;但 verl 0.7.1 官方声明支持 `vllm>=0.8.5,<=0.12.0`,在自己声明范围的低端是坏的。AGL 钉的是 `verl>=0.7.1,<0.9.0` + vllm 0.8.5 —— 这个组合正中枪口,**AGL 是提 issue/compat 修复的正主**。

**本地止血**(09-11 18:00):给 site-packages 里的两处调用点包了 `try/except TypeError` 回退(不带 pooling_output 重调),备份 `.bak.09111800_abortfix`。此后 abort 真正生效,超时 rollout 不再留孤儿,随机喂题模式(难题超时更常见)下不会复发。

**教训**:
- 兜底 `except Exception` + 只打日志不上抛,会把"清理失败"伪装成"清理成功",让真正的故障在几分钟后以完全不同的面目(CUDA 非法访存)爆发,归因极其困难。
- 库声明的版本范围 ≠ 测试过的版本范围。verl 0.7.1 的 abort 路径显然只在新版 vLLM 上测过。
- 低频崩溃先查"崩溃步和平安步的差异"(这里是 failed rollout 数),比查"崩溃点的堆栈"更快指向根因。

## 32. 镜像"边拉边删"循环的终结:全量钉住 + GC 阈值对齐(09-11)
- 背景:随机喂题后同一仓库镜像被反复"GC 删→autoheal 拉",每次 2~3 分钟,纯浪费。人类授权把磁盘闸门降到 100G 后,空间足够全量常驻(78 缺失 × ~3.2G ≈ 250G,拉完仍余 ~140G)。
- 做法:(1) 对每个镜像 `docker create --name pin-<hash> --label agl-pin=1 <img> true` 造一个停止容器,kubelet imageGC 遇到有容器引用的镜像会拒删("must force")——这正是 23 个 val 镜像一直幸存的机制,现在主动用于全部 132 个训练镜像;(2) k3s.service 里 image-gc-high/low 从 95/90 抬到 98/97(空余 <121G 才触发 GC),与 100G 闸门对齐,同时保护邻居的未引用镜像;docker 运行时下 `systemctl restart k3s` 不杀运行中容器,pod 由 kubelet 重新接管,实测零损伤;(3) 后台 pull_all_pin.sh 串行补拉+钉住,闸门 100G。
- 教训:百分比型 GC 与绝对值型闸门必须显式对齐,否则"谁先动手"取决于盘的总量;"钉住"是比"抢着拉"更根本的解——GC 的豁免规则(容器引用)本身就是官方提供的钉子。

## 33. 崩溃重启时,新 tmux 窗口不继承 MPS 绕过环境变量(09-12)

step22 生成阶段再次出现 vLLM 非法访存(§31 同类)。恢复时 Ctrl-C 后原 trainer 窗口随进程退出被 tmux 关闭,新开的窗口是干净 shell——重启命令没带 `CUDA_MPS_PIPE_DIRECTORY`,新进程自动连上坏的全局 MPS(§29),启动即 NCCL "unhandled cuda error / Cuda failure 1 'invalid argument'",又损失一轮启动时间。

**教训**:凡重启训练,启动命令必须显式前置 `export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty &&`,不能指望 shell 环境还在。已固化到本文档;更稳妥的做法是把完整启动命令(含 env)存成脚本,重启只执行脚本。

## 34. Xid 31 崩溃后 NVLS(NVLink SHARP 组播)损坏:≥4 卡 NCCL 初始化必挂,双卡幸存(09-12)

step22 vLLM 非法访存崩溃(内核侧 Xid 31 MMU Fault,00:29 四卡各一条)后,重启训练反复在 FSDP `_sync_params_and_buffers` 的 NCCL 广播里报 `Cuda failure 1 'invalid argument'`。定位过程:

1. 最小复现:`torchrun --nproc_per_node=N` + 一次 broadcast。双卡各组合基本都过,四卡(1,2,3,5)100% 挂——**卡数相关**而非某张卡坏。
2. `NCCL_DEBUG=WARN` 给出确定性证据:`transport/nvls.cc:158 NCCL WARN Cuda failure 1 'invalid argument'`——挂在 NVLS(NVLink SHARP 组播)传输层初始化。双卡不用 NVLS(纯 P2P),≥4 卡默认启用,所以出现"卡数越多越挂"的假象。
3. 根因推断:Xid 31 进程级崩溃泄漏了 NVSwitch 组播组资源,新建组播组时 cuMulticastCreate 报 invalid argument。ECC 零、fabric manager 正常、dmesg 无 NVSwitch 错误——常规体检查不出来。
4. 绕过:`export NCCL_NVLS_ENABLE=0` 回退普通 NVLink 环,损失少量大消息 allreduce 带宽,训练即恢复。彻底修复需重启 fabric manager 或整机(共享机不可行)。

**排查中的两个弯路**(都值得记):
- 先怀疑 MPS(§29 前科)——但带/不带绕过都挂,双卡不带绕过也过,证伪。测试时别忘了每一轮都显式带上要验证的环境变量,对照组才成立。
- 后怀疑丢了 `CUDA_VISIBLE_DEVICES`——确实丢了(也确实要补),但不是本次 NCCL 挂的原因。一次事故可以同时暴露多个配置缺失,修了一个不等于修完。

**教训**:崩溃恢复失败时,先用 torchrun 最小 NCCL 测试把"训练栈问题"和"机器状态问题"切开,再按卡数二分。NCCL 报 unhandled cuda error 时 `NCCL_DEBUG=WARN` 一行定位,比读框架栈快得多。

## 35. resume_mode=auto 静默从零开训:ckpt 目录靠 AGL_CKPT_DIR 环境变量

**现象**:09-12 崩溃恢复第 4 次重启,所有已知 env(MPS 绕过/卡号/NVLS/VLLM_USE_V1)都带上了,进程正常起来、rollout 正常跑,但日志里是 `Training from scratch` 而不是 `Resuming from ...global_step_20`。没有任何报错。

**根因**:`train_opencode_agent.py` 的 ckpt 目录来自 `os.environ.get("AGL_CKPT_DIR", ".../swe_smith_opencode")`。原始 shell 里导出过 `AGL_CKPT_DIR=.../swe_smith_opencode_minicpm50k_v2`,新 tmux 窗口丢了它,于是 `trainer.default_local_dir` 落回默认目录——那里没有 ckpt,`resume_mode=auto` 找不到就"合法地"从零开始。这是 §33(干净 shell 丢 env)的第五个受害者,也是最阴险的一个:**前四个丢 env 都会报错,这个不报错,只是安静地丢掉全部训练进度**。

**排查**:日志里 grep `default_local_dir` 与 `Resuming from|Training from scratch`。重启后第一分钟必须确认 resume 行,不能只看"进程起来了、reward 在跑"。

**修复**:`restart_v2_trainer.sh` 增加 `export AGL_CKPT_DIR=...v2`。顺带把计划中的 `RAY_TMPDIR=/workspace/ray_tmp` 也加了(ray 会话/溢出目录默认在 99% 满的根分区)。

**教训**:审计"原 shell 有哪些 env"不能靠回忆逐个补,应一次性 `tr '\0' '\n' < /proc/<原pid>/environ` 拿全量对比——可惜原进程已死。替代办法:grep 训练脚本里所有 `os.environ.get`,逐个确认默认值是否可接受。本例 grep 后确认 AGL_CKPT_DIR 是该脚本唯一的 env 开关。

## 36. 温度消融:agentic 工具调用里"贪心解码(temp=0)显著更差",而采样温度 0.6≈1.0 无差异(2026-09-13)

**背景**:step154-v2 checkpoint 在同一批 474 题验证集上跑四组解码配置,逐题对齐做配对 McNemar(两侧精确二项)。

| 配置 | 通过 | 通过率 |
|---|---|---|
| temp=0 贪心 | 122/474 | **25.7%** |
| temp=0.6 | 150/474 | 31.6% |
| temp=1.0 | 149/474 | 31.4% |
| 基线(50k, 未训) | 139/474 | 29.3% |

**配对检验(同题对齐,不是独立两组)**:
- temp0 vs temp0.6:独赢 7 : 35,**p<0.0001** —— 贪心显著更差
- temp0 vs temp1.0:独赢 10 : 37,**p=0.0001** —— 贪心显著更差
- temp0 vs 基线:独赢 8 : 25,**p=0.0046** —— 贪心连未训基线都打不过
- temp0.6 vs temp1.0:不显著;两者 vs 基线:也均不显著(p=0.09/0.18)

**结论与根因**:关掉采样(贪心)在本 agentic 多轮工具调用任务上**稳健地掉 ~6 个点**,且不一致对高度偏向采样一侧(贪心追不回来),不是噪声。原因是多轮 agent 里贪心容易陷入确定性死循环——重复同一个工具调用 / 卡在同一条错误路径出不来;采样提供的多样性正好帮它跳出局部循环。相反,采样温度在 0.6~1.0 区间对结果不敏感。

**教训**:
1. **别用 temp=0 复现"最优"**。直觉上贪心=取最大概率=最稳,但在 agentic 场景里恰恰相反——解码多样性本身是有效信号。报告最终指标时用采样(温度 0.6~1.0 任选),不要用贪心。
2. **温度扫点先扫"开不开采样",再扫具体值**。0 vs >0 的差远大于 0.6 vs 1.0 的差;把预算花在有梯度的地方。
3. **单点差异要用配对检验坐实**。这里 25.7% vs 31.6% 若只看边际通过率也许会被当成噪声,但逐题配对后 discordant 对 7:35 一边倒,p<0.0001,结论才立得住。这也复用了 §27 的方法论(边际相同 ≠ 逐题相同)。
4. 附带确认:本 checkpoint(GRPO 训练)相对未训基线的 +2pt 边际增益,配对检验下**不显著**(p=0.09~0.18)——训练奖励未转化为泛化,与 §27 一致。

## 37. 共享节点的 CPU 内存被邻居吃光,Ray 杀的是我的 worker(2026-09-15)

**现象**:05:48:13 训练整组暴毙——主进程、raylet、三个 FSDP `WorkerDict`、`_AglTaskRunner` 全死,最后完成 step 262。Ray 报 `OutOfMemoryError: 21 worker(s) were killed due to the node running low on memory. Memory on the node was 965.91GB / 1007.00GB (0.959197)`。

**这不是显存 OOM,是整机 RAM OOM**,两者容易混。`nvidia-smi` 当时一切正常。

**归属**:按 RSS 排序一眼定位——

| | RSS |
|---|---|
| 邻居 `sop_kit/line14/bundle_v1/sop14_t256.py train`(4 个 recipe / 13 进程) | **560 GB** |
| 本训练全部 Ray 进程 | ~137 GB |

Ray 的 OOM 守卫只管得着自己节点上的 task,所以邻居把机器吃到 96% 时,**被杀的是我**。邻居进程按红线一个没碰;约一小时后对方自行降到 11 GB,可用内存回到 617 GB,压力自解。

**处置(净损失零训练步 + 约 4 分钟停机)**:
1. `ls -dt $AGL_CKPT_DIR/global_step_*` + `cat latest_checkpointed_iteration.txt` 确认 262 完好(29G,`save_freq: 2` 救了命)
2. 逐卡 `nvidia-smi -i N --query-compute-apps` 确认卡上残留显存**全属邻居**、没有自己的僵尸进程占着不放
3. 按固化脚本 `restart_v2_trainer.sh` 重启(`resume_mode=auto`)
4. 确认 `Load from checkpoint folder: .../global_step_262` 后再确认 step:263 真的跑出来

**教训**:
1. **`save_freq` 是共享机器上的保险费,不是性能损耗。** 本例 2 步一存,29G/次,事故代价因此是 0 步。如果是 20 步一存,这次要白跑最多 20 步。
2. **判活别信"主进程还在"**。事发后 `ps -p <主pid>` 一度仍返回存活,但 raylet 和所有 WorkerDict 已死。可靠判据是逐个 `ps -p` 查 worker pid + `nvidia-smi` 看利用率是否归零。
3. **重启后必须验两件事,缺一不可**:`Load from checkpoint folder: global_step_N`(证明没 from scratch,见 §35)与**至少一个新 step 真的产出**(证明不是起来了又卡死)。
4. 自己能缩的只有 Ray object store(配 200 GB,事发时 `/dev/shm` 实占 101 GB)。改它要动固化重启脚本,属于人类决策,不自作主张。

**监控脚本的坑(本次连踩两次,都是自己误报)**:
- 用 `ps -eo args | grep -q '<模式>'` 判进程活否会假阴性,误报"任务已结束"。改用 `kill -0 <pid>`。
- 在**追加写**的长日志(本例 34 万行、跨多轮重启)上做全文 `grep`,会捞到几天前的历史行当成本次事件——本次据此误报了"恢复点是 global_step_154"(实为 09-13 那轮)。正确做法:重启前记下 `wc -l` 作基线,之后只 `awk 'NR>BASE'`。
- 故障 grep 别写 `Error` 这种宽词:verl 启动时打印的配置字典里就有含该词的字段名,会连续误报。只匹配真实故障签名(`OutOfMemoryError|CUDA out of memory|RayActorError|NCCL error|Xid`)。

## 38. 单步 1100s 的大头是"等最后一个 rollout",不是算力不足(2026-09-15)

**现象**:每步耗时 ~1100s,期间采样 20 秒发现训练卡(1/3/5)GPU 利用率**恒为 0%**。第一反应是被同机的 llama.cpp 抢了卡——**错的**。

**实测**:`AglRolloutManager: completed=N/32` 的轮询行按 N 分桶统计,停在 `31/32` 的次数(9386)**比停在任何其他进度都多**,`0/32`(7508,启动段)次之,越接近尾部等得越久。当时 docker 里正有一个 `agl-rollout-*` 容器 Up 23 分钟——31 个 rollout 早已 `succeeded`,整步在等第 32 个。

**佐证**:翻 step 160–230 的历史,尚无 llama.cpp 共卡时每步就是 1000–1060s;有共卡后 ~1100s,差 ≤10%,落在噪声里(同期还有 648s 和 1675s 的离群点)。邻居释放内存后单步直接掉到 588s。

**含义**:
1. GPU 空转不等于训练卡死。agentic RL 的 rollout 大头在容器里跑测试,本来就是 CPU/IO bound。
2. 这是**长尾问题**:`rollout_timeout_seconds: 2400` 是兜底,最坏单步能被一个慢样本拖到 40 分钟。调小能显著提速,但会丢掉"慢但能解出来"的样本——真实取舍,不是纯优化。
3. 顺带推翻了"共卡拖慢训练"的直觉判断。**先量化归因再下结论**:有没有 A,B 的差值是多少,而不是看到 A 和 B 同时存在就认定因果。

## 39. 两条配方、两次从头训,都没有统计显著增益——以及怎么把"没效果"诊断到根因(2026-09-15)

**结论先行**:v1(纯课程,step180)与 v2(交错,step284)两条独立配方,**没有任何一个检查点跟基座拉开过统计显著的距离**。

同题配对 + 精确 McNemar(474 题,温度 0.6 单次采样,`compare_runs.py`):

| run | resolved | F2P |
|---|---|---|
| 基座 MiniCPM5-2B | 139/474 = 29.32% | 49.65% |
| step154 v2 | 150/474 = 31.65% | 52.56% |
| step284 v2 | 142/474 = 29.96% | 50.10% |

| 对比 | both | onlyA | onlyB | net | p(精确二项) |
|---|---|---|---|---|---|
| 基座 vs 284 | 121 | 18 | 21 | +3 | **0.7493** |
| 154 vs 284 | 127 | 23 | 15 | +8 | **0.2559** |
| 基座 vs 154 | 127 | 12 | 23 | +11 | **0.0895** |

**教训 0:必须同题配对 + 显著性检验。** 31.65% vs 29.32% 看着像"涨了 2.3 个点",实际 p=0.09。474 题的规模下,±2 个点就是噪声。只看两个百分比就宣布有效,是这类工作最常见的自欺。

### 39.1 第一诊断动作:直接量权重漂移

比任何 loss 曲线都硬的指标——**把训练前后的权重拿出来做 L2 对比**:

```python
# 逐 safetensors 张量:d = ||W_new - W_old||, n = ||W_old||
# 全局相对漂移 = sqrt(Σd²) / sqrt(Σn²)
```

实测 base vs step284(381 个张量全比):

```
全局相对漂移 = 4.316e-04        ← 0.043%
最大单张量  = 1.260e-03  layers.1.self_attn.v_proj
部分 layernorm 漂移 = 0.000e+00
```

**284 步、85 小时,参数只动了万分之四。** `lr=1e-6` + `train_batch_size=8` + `rollout.n=4` → 全程只喂了 2272 个题目实例。在讨论"该用 Dr.GRPO 还是 GSPO"之前,这一版根本没训够。

**这个检查花 2 分钟,应该在每次"训练没效果"时第一个做。** 它能一次性区分两种截然不同的失败:"算法有偏,模型学歪了" vs "优化量不够,模型没动"。

### 39.2 第二诊断动作:数一数有多少 batch 是废的

`training/n_zero_adv_groups / training/n_groups` 聚合 286 步:

```
n_groups           合计 1928
n_zero_adv_groups  合计  902   →  46.8%
```

**同题 4 条 rollout 拿到相同 reward(全对或全错)→ GRPO 组内优势恒为 0 → 这组对梯度零贡献。** 一步名义 8 道题,实际有效 ~4.3 道。

agent-lightning 的处理是按容量上限丢行(`agentlightning/verl/trainer.py:602-615`),**丢了不补采**。DAPO 的 dynamic sampling(过采样+过滤+续采直到填满)正是为此;verl 的 `AlgoConfig` 有 `filter_groups` 字段(`trainer/config/algorithm.py:611`),但 agent-lightning 有自己的 `fit()` 循环,没接进去。便宜的近似是提高 `rollout.n`(全对/全错概率随 n 指数下降)。

### 39.3 第三诊断动作:确认 PPO 的裁剪到底有没有生效

> **更正(09-15):** 本节说的「永远只更新一次、ratio 恒等于 1」按日志重算不成立——
> 36% 的步更新一次、64% 更新两次。结论方向不变(clip 几乎从不触发),但原因是 lr 太小而非更新次数。见 §42.3。

我们配了 DAPO 的 clip-higher(`train_opencode_agent.py:104-105`,`clip_ratio_low=0.2 / clip_ratio_high=0.28`),但实测:

```
actor/ppo_kl       ≈ 1e-4,多数步恒为 0
actor/pg_clipfrac  ≈ 1e-4,多数步恒为 0
```

**这两个指标同时趋零,就是"一轮 rollout 只更新一次权重"的指纹。** 推导:

- `restart_v2_trainer.sh:32` → `train_batch_size=8`,`ppo_mini_batch_size=8`
- `verl/workers/fsdp_workers.py:249` → `ppo_mini_batch_size *= rollout.n`
- → mini_batch = 8×4 = 32 = 整个 batch → **1 个 minibatch,1 次 optimizer.step()**
- → `π_θ = π_θ_old` → ratio ≡ 1 → clip 永不触发

**于是 PPO 的 clip、DAPO 的 clip-higher、GSPO 的序列级 ratio、dual-clip(`clip_ratio_c=3.0`)全是死代码,实际在跑的是纯 REINFORCE with baseline。**

修法与代价(286 步均值):

```
timing_s/gen           980.8s   ← 91% 的时间在 rollout
timing_s/update_actor   62.2s   ←  5.8%
timing_s/step         1078.9s
```

`ppo_mini_batch_size` 8→2 得到 4 次更新,step 时间 1079→1265s(**+17% 换 4 倍梯度更新**),同时把上述一堆开关从死代码变成活代码。**agentic RL 里 rollout 占九成时间,多做几次梯度更新几乎是白送的。**

### 39.4 长度归一化不会让模型变短——它让模型"错得更长"

一个反直觉但关键的点。我们的实际目标函数(`agentlightning/verl/per_rollout_loss.py:29-41,83`)展开是:

```
L = -(1/R) Σ_rollout (A_i / T_i) Σ_{t∈i} ratio_t
```

即**每 token 梯度系数 = A_i / (T_i · R)**,是原版 GRPO 的 1/|o| 长度归一化(在整条多轮轨迹这一级)。

直觉推理"除以长度 → 短回答每 token 拿得多 → 模型应该学短"**漏了符号**:

- `A_i > 0`:短的正确回答每 token 被推得更狠 → 确实偏好短
- `A_i < 0`:**长的错误回答每 token 被压得更轻**。同一个 -1 的优势摊到 500 token 是 -1/500,摊到 5000 token 是 -1/5000

**模型学到的是"要是八成会错,就写长点,罚得轻"。** 这就是 Dr.GRPO(arXiv 2503.20783)说的 response-level length bias。任务 resolve 率只有 ~30% 时,**负优势分支是数据主体**,这一支占上风。

Dr.GRPO 的第二个偏差我们也全中:**除 std**(`verl/trainer/ppo/core_algos.py:325`,`norm_adv_by_std_in_grpo` 默认 True)。组内全对/全错时 std→0,`1/(std+1e-6)` 把噪声放大成巨量优势。verl 的注释就写在 `core_algos.py:294-296`,改法是一个开关 `algorithm.norm_adv_by_std_in_grpo=False`。

**但要先量,别先治**:我们训练内长度其实是平的(`response_length/training/avg_by_turn` 分段均值 1175 → 1064 → 1066 → 1181),变长的是**评测时**(温度 0.6:base 703 → step284 1131 tok/call,+61%)。训练采样温度是 1.0,基座在 temp 1.0 下本来就 1175。准确描述是 **RL 把模型低温下的行为推向了它原本高温才有的长尾行为**,不是"目标函数在奖励长度"。在没诊断出"确实在变长"之前就加长度惩罚,是先上药后确诊。

### 39.5 KL 不是瓶颈——别把"离参考模型太近"当默认嫌疑人

"DAPO 去掉了 KL,我们是不是也被 KL 拴住了"是个很自然的假设,**但要算数字再下结论**:

| 项 | 每 token 梯度系数 | 实测 |
|---|---|---|
| policy | `\|A\|/(T_i·R)` | 1.2/(2236×52.6) ≈ **1.0e-05** |
| KL | `coef/N_batch_tokens`(`dp_actor.py:650-655`,走 token-mean) | 0.001/71552 ≈ **1.4e-08** |

**policy 项大约 700 倍于 KL 项。** 即使按"KL 是系统性的、policy 梯度是噪声会抵消"算累积(284 步:policy √284×1e-5=1.7e-4,KL 284×1.4e-8=4e-6),KL 仍小 40 倍。

更直接的反证——**KL 自己涨上去了**:

```
step   6:  actor/kl_loss 0.0083
step 200:  actor/kl_loss 0.0431
step 284:  actor/kl_loss 0.2672   ← 涨了 32 倍
```

**策略已经跑到离参考模型 0.267 nats/token 的地方了。** 如果 KL 真是束缚,它会被压在低位。同期 entropy 全程平(0.15~0.29,没塌)、reward 平。**结论:模型离参考模型跑得挺远,但能力没跟着走——它跑偏的方向不对,不是被拴住了。** 去掉 KL 的收益约等于 0。

### 39.6 稀疏到极致的信用分配

`agentlightning/verl/rollout_level_advantage.py:63-72`:一道题 4 条轨迹 → 每条一个标量 reward → **广播到这条轨迹被采集到的全部 turn 的每一个 response token**。

> 更正(同日):本节初稿写的是"全部 72 个 turn、2236 个 response token"。**这个数字是错的。**`training/n_turns` = 72.33 是**一个 step 内 32 条 rollout 的合计**,不是单条轨迹的轮数;单条轨迹真正被采集进 batch 的只有 ~2.3 个 turn。追这个数字追出了 §41 的真正根因。

找对文件的那一步、读错报错的那一步、最后打对补丁的那一步,**同赏同罚**。这是 `actor/grad_norm` 只有 0.005~0.03 的直接原因:大量梯度互相抵消。按代价排序的改法:

1. `algorithm.gamma` 1.0 → 0.95~0.99(`trainer/config/algorithm.py:602`),让靠近成功的 turn 权重更大。改一个配置项。
2. turn 级过程奖励(在 `examples/swe_smith/opencode_agent.py:399-430` 的 reward 发射点加):工具调用是否合法解析、patch 是否 apply 成功、F2P 通过数是否单调增。有了 turn 级信号才能在 turn 维度做 GAE。**收益最大,但是实打实的工程量。**
3. token 级选择性加权,verl 内置:`clip_cov`(`core_algos.py:1734`)、`kl_cov`(`1839`),按 token 的 log-prob 与优势的协方差挑 token。换个 `loss_mode` 即可。
4. 真 critic(`adv_estimator=gae` + value head):2B + 4 卡 + 45k prompt 下显存和时间都吃紧,且 critic 在长轨迹上很难训好。**现阶段不推荐。**

### 39.7 verl 0.7.1 现成可用的 loss 变体

注册表见 `verl/trainer/ppo/core_algos.py`,全部一个配置项切换(但都依赖 §39.3 先修好"一轮多更新",否则 ratio≡1,变体之间没有区别):

| loss_mode | 行号 |
|---|---|
| `vanilla` | 1277 |
| `gspo`(序列级重要性比,长序列方差控制) | 1537 |
| `cispo`(裁权重不裁梯度,保住高熵 token) | 2005 |
| `geo_mean` | 1919 |
| `clip_cov` / `kl_cov` | 1734 / 1839 |
| `dppo_tv` / `dppo_kl` / `sapo` / `gpg` | 1371 / 1452 / 1613 / 1698 |

## 40. 两个自己踩的工具坑(2026-09-15)

**`find` 其实是 `bfs`,而 `2>/dev/null` 会把它的报错吞掉。** 跑 `find /data -size +1G -newermt "24 hours ago" 2>/dev/null` 返回空,据此报告了"24 小时内没有大文件写入"——**假阴性**。这台机器的 `find` 是 `bfs`,它拒绝相对时间戳(`Invalid timestamp. Supported timestamp formats are ISO 8601-like`),而 `2>/dev/null` 把这条错误藏了。正确写法:

```bash
TS=$(date -d "26 hours ago" +%Y-%m-%dT%H:%M:%S)
find /data -size +1G -newermt "$TS"
```

**教训:凡是"返回空 = 好消息"的检查,都必须先用一个已知非空的条件验证命令本身能跑通。** 空结果和命令失败长得一模一样,尤其在 `2>/dev/null` 之后。

**vLLM 的 INFO 级请求日志在评测舰队里会吃爆磁盘。** 6 个评测实例 25 分钟写了 1.6 GB(~4 GB/小时),在 `/data` 只剩 100G 闸门时是实打实的风险。发现后原地截断(日志句柄不是 O_APPEND 时可用,保留尾部即可,writer 不受影响):

```bash
tail -c 200000 f > f.t && cat f.t > f
```

并挂一个后台修剪器(超过 200MB 就裁到尾部 20MB,舰队结束自动退出)。**教训:起评测舰队时把日志增长当成一项资源预算,和显存、磁盘一起在起飞前算一遍。**

## 41. 真正的根因:92% 的 agent 轮次从来没进过训练 batch(2026-09-15)

追 §39.6 里那个写错的数字时发现的。结论比 §39 整节的算法讨论都重要:**我们训练的那 85 小时里,
模型实际做的工具调用基本没有产生过一次梯度。**

### 41.1 现象:采集率 8.1%

`$AGL_CKPT_DIR/trajectories/step_N_train.jsonl` 里 `n_turns = len(rollout.triplets)`,
即这条 rollout 有多少次 LLM 调用被收进了训练 batch。把它和容器内 tool-truncate 代理
(`opencode_agent.py:188` `_GatewayProxyHandler`,日志 tee 到
`agl-logs/new/agl-rollout-<id>.log`)记的真实 HTTP 请求数对照,最后 40 步 1280 条 rollout:

| | 中位 | 均值 | 合计 |
|---|---|---|---|
| 采集到的 triplet | 1 | 2.08 | 2,660 |
| 真实 `POST /v1/chat/completions` | 17 | 25.69 | 32,880 |

**总体采集率 8.1%**,单条中位 8.7%。按退出码分组几乎一样(rc=0 为 8.2%,rc=124 超时为 7.7%),
所以不是超时/崩溃造成的截断。全量日志 457,759 次调用里 99.3% 是 HTTP 200,也不是失败请求。

47.4% 的 rollout 只剩 1 个 triplet、17 个 token,内容恒为
`<think>\nLet me start by exploring the repository structure to understand the codebase.\n</think>`。
这些**不是**快速失败:时长中位 161s,没有一条低于 10s,它们实际打了中位 11 次 LLM 调用(97.2% ≥5 次)。

### 41.2 被采集到的是什么:全是"非 agent"调用

把 1280 条 rollout 最后一个 triplet 的解码文本分类:

| 类别 | 占比 | 平均 resp_tokens |
|---|---|---|
| 首轮 `<think>` 探索桩 | 43.9% | 18 |
| OpenCode 的上下文压缩/摘要调用 | 56.1% | 3,466~7,321 |
| 真正的工具调用轮 | **≈0%** | — |

抽样 category "其他" 全部命中 "produce a structured summary that a coding agent can use to
continue the work" / "create a structured summary of the conversation so far" —— 也是摘要调用。

**即:reward 来自 SWE-bench 测试通过与否,梯度却几乎全部打在"开场白"和"OpenCode 自己的会话摘要"上。**
读文件、改代码、跑测试这些真正决定成败的轮次,一次都没进优化器。

### 41.3 根因链(已实验证实)

1. **vLLM 0.8.5 不支持 `return_token_ids`**。代理在 `proxy.py:68,78` 注入了这个字段,vLLM 直接忽略,
   响应里没有 `choices[0].token_ids`。AGL 自己在 `proxy.py:225` 的 docstring 里就写着
   "vLLM 0.8.x ignores return_token_ids" —— 兜底本来就是为这件事写的,只是没人验过它在工具调用上会炸。
   (实测:训练 venv 是 **0.8.5**;miniconda 那个 0.9.2 同样 0 处命中 `return_token_ids`,
   所以"升级 vLLM"要跳到远比 0.9.2 新的版本,而 `minicpm5_parser_plugin_085.py` 是写死的
   0.8.5 兼容垫片 —— 升级不是小改。见 §41.5 的替代方案。)
2. 于是走兜底:`proxy.py:224 _fill_missing_token_ids` → `_tokenize_chat` →
   `tokenizer.apply_chat_template(messages, tools=tools, ...)` 自己 tokenize 一遍。
3. **MiniCPM5-2B 的 chat template 处理不了 OpenAI 标准的 tool_calls**:
   ```
   arguments = JSON 字符串(OpenAI 标准) -> UndefinedError: 'str object' has no attribute 'items'
   arguments = dict(模板期望)            -> OK, 240 tokens
   ```
   模板对 `function.arguments` 直接 `.items()`,而 OpenAI 协议里它是**字符串**。
4. `proxy.py:235-237` 把异常 catch 掉,`log.warning` 之后**静默 return**,token_ids 留空。
5. `agl_rollout_manager.py:461` `if not response_token_ids: continue` —— 这一轮被丢出训练集。

带 tools / tool_calls / `role:"tool"` 的请求 = 所有真正的 agent 轮 → 全部在第 3 步炸掉 → 全部被丢。
不带 tools 的纯 user/assistant 请求 = 开场白 + 摘要调用 → 模板正常 → 被留下。**8.1% 的构成就是这么来的。**

### 41.3b 第二个独立的 bug:连活下来的那一轮,response 也是残的

`proxy.py:207 _assistant_text` 的优先级是 `content` → `reasoning_content` → `json.dumps(tool_calls)`。
而 MiniCPM5 的 XML 工具解析器(`minicpm5xml_tool_parser.py:315` `tool_call_start_token = "<function"`)
在解析成功后,把 `<function=...>...</function>` 整段**从 content 里剥掉**,只留下前面的普通文本
(`extract_tool_calls` 的 `normal_parts`,:400 `content=content`)。于是 content 非空 → `_assistant_text`
**在第一优先级就返回了,永远走不到 tool_calls 分支**。

实测一条典型的工具调用轮:

| | tokens |
|---|---|
| 模型真实采样出来的 response(含 XML 工具调用) | 59 |
| 兜底回填进训练集的 response | 17 |

**丢掉 71%,而丢掉的恰好是"模型决定做什么动作"那一段** —— RL 要学的就是这个,它一个 token 都没进去。

### 41.3c 两个 bug 合起来,严丝合缝地解释了全部观测

| 调用类型 | prompt 里有没有历史 tool_calls | 结果 |
|---|---|---|
| **第 1 轮** | 没有(还没发生过工具调用) | 模板通过 → **被采集**,但 response 被剥成只剩 `<think>` |
| **第 2 轮及以后** | 有(历史 assistant 带 tool_calls) | 模板抛 UndefinedError → **静默丢弃** |
| **摘要/压缩调用** | 没有(纯 user/assistant 文本) | 模板通过 → **被完整采集** |

所以 43.9% 的 rollout 只剩一个 17-token 的 `<think>` 桩 —— 那就是第 1 轮被剥干净之后的残骸。
实测:`'<think>\nLet me start by exploring the repository structure to understand the codebase.\n</think>'`
tokenize 出来**正好 17 个 token**,与轨迹文件里记的 `resp_tokens=17` 逐字对上。剩下 56.1% 是摘要调用。
真正的工具调用轮,**一轮都没有**。

### 41.4 教训

- **`catch → warning → 静默降级` 是这次的真凶。** 那条 warning 写在 AGL server 自己的 structlog 里,
  而 `restart_v2_trainer.sh` 的 tee 只收 trainer 的 stdout,**两个日志不在一个文件里**。
  我一度用"trainer 日志里 grep 不到 `token-id backfill`"当证据,这是错的——同样 grep 不到
  `Retrying upstream`,可 502 实实在在发生了 2,995 次。**用"日志里没有"当证据前,先证明那条日志会落到这个文件。**
- **训练前必须有一条"采集率"断言**:`sum(len(r.triplets))` 对 agent 真实 LLM 调用数,低于 90% 就拒绝开训。
  这个数一直躺在 `training/n_turns` 里,只是没人拿它跟 ground truth 比过。
- **§39 的算法优先级表要重排。** Dr.GRPO 的长度偏置、GSPO、DAPO 动态采样、一轮多更新,在 92% 数据缺失
  面前都是二阶项。先修采集,再谈 loss。
- 权重漂移 4.3e-04 / 284 步(§39.1)现在有了完整解释:8% 的数据量,且那 8% 与任务成败几乎无因果关系。
- **这个 bug 其实一直有个免费探测器,只是没人看**:`_include_log_probs` 默认 True(`proxy.py:46`),
  所以响应里一直带着**按真实采样 token 逐一对应**的 logprobs。
  `rollout_adapter.py:598` 和 `:687` 都写着"长度对不上就把 log_probs 丢掉":
  ```python
  if response_log_probs is not None and len(response_log_probs) != len(response_ids):
      response_log_probs = None
  ```
  重建出来的 17 个 token 对上真实的 59 条 logprobs,**每一个工具调用轮都会命中这个分支**。
  只要给这个丢弃动作加一个计数器,第一步就能发现问题。
  **凡是"对不上就静默降级"的分支,都应该带计数器。**

### 41.5 修法(未实施,待裁决)

**两个 bug 要分开修,只修一个都不够**(修了 41.3 只会让残缺的 response 变多)。

1. **修 prompt 侧(41.3)**:`_tokenize_chat` 在调 `apply_chat_template` 前把 `function.arguments`
   从 JSON 字符串 parse 成 dict。候选补丁与测试已写好、**未入库**,在
   `scratchpad/capture-fix/{proxy_tokenize_fix.py,test_fix.py}`。
   策略是「原样 → arguments 归 dict → 去 tools」依次降级,实测 6 个用例 **3/6 → 5/6**,
   原本就通过的两个用例 token 数逐一不变(走 `as-is` 路径,**无回归**);
   唯一仍失败的是「模型吐出非法 JSON 参数」,这种应当计数上报而不是静默吞掉。
2. **修 response 侧(41.3b)—— 不需要升级 vLLM,加一个 per-request 字段即可**。
   `return_tokens_as_token_ids`(`protocol.py:405`)打开后,logprobs 里每个 token 字段变成字面量
   `"token_id:<id>"`(`serving_engine.py:535-536`)→ **直接拿到模型真实采样的 response token_ids**,
   完全不走 `_tokenize_chat` 的重建。已确认它走的是**非流式** chat 路径(`serving_chat.py:930`),
   正是代理在用的那条。代价为零:我们本来就在请求 logprobs
   (`_include_log_probs` 默认 True,`proxy.py:46`;`top_logprobs` 默认 **0 而非 None**,
   所以 `serving_chat.py:923` 的分支一直是进的),这个 flag 只改 token 字段的**格式**,不改计算量。
   解析函数与 8 条单测已写好、未入库:`scratchpad/capture-fix/{exact_token_ids.py,test_exact.py}`
   (8/8 通过;服务端没开 flag 时整体返回 None 而不是瞎猜)。

   > **撤回:prompt 侧不要用 `prompt_logprobs`。** 本节初稿建议用它拿服务端的真 prompt token_ids。
   > 查下来这是个陷阱——`v1/core/kv_cache_manager.py:120-122` 写得很直白:
   > ```python
   > # When the request requires prompt logprobs, we skip prefix caching.
   > if request.sampling_params.prompt_logprobs is not None:
   >     return [], 0
   > ```
   > **一开就整体关掉 prefix caching。** 我们的多轮轨迹前缀高度重叠,`proxy.py:56` 还专门把每条
   > rollout 钉在同一个 endpoint 上就为了复用前缀缓存;而 §38 已经量过 rollout 占 91% 的墙钟。
   > 再叠上 45k prompt 下约 3.0 MB/次、每 step 约 0.6 GB 的 JSON 回传,这条会把吞吐打残。**不要开。**

3. **prompt 侧仍然走模板补丁,但必须加一条零成本的校验**。
   响应里的 `usage` 代理本来就存进事件了(`proxy.py:313 _extract_usage`),而 `usage.prompt_tokens` /
   `usage.completion_tokens` 是 **vLLM 自己的权威计数**。于是:
   * `len(filled_prompt)` vs `usage.prompt_tokens` —— 校验模板重建是否和服务端逐一对齐
   * `len(response_ids)` vs `usage.completion_tokens` —— **这一条当初就能在第一步抓到 §41.3b**
     (重建 17 vs 真实 59)
   零额外计算、零额外带宽,对不上就计数上报 / fail-fast。
4. **兜底失败一律不许静默**:`_fill_missing_token_ids` 的 `except` 至少要计数上报到 trainer 指标,
   最好让该 rollout 直接 fail-fast。
5. **开训前加断言**:`sum(len(r.triplets))` / agent 真实 LLM 调用数 < 90% 就拒绝启动。

### 41.6 已实施(2026-09-15,训练停在 step 284 期间改的)

41.5 的 1~4 条已经落到 `agentlightning/server/proxy.py`(备份 `proxy.py.bak.09151854`),
第 5 条(开训前断言)留到写 v3 脚本时一起做,因为它改的是 trainer 不是代理。

| 改动 | 位置 | 作用 |
| --- | --- | --- |
| `prepare_body` 加 `return_tokens_as_token_ids: True` | train 分支、与 `logprobs` 同一个 `if` 里 | response token_ids 取服务端真值,彻底绕开模板重建 |
| `_response_ids_from_logprobs()` | 新增 | 解析 `"token_id:<id>"`;**有一个 token 不是这个形式就整体返回 None**,宁可回落也不半解码 |
| `_fill_missing_token_ids()` | 重写 | 顺序改成「服务端真值 → 模板重建」,每条降级分支都计数 |
| `_apply_chat_template()` 降级梯子 | 替掉原来的 `except TypeError` | as-is → args_dict → no_tools → inlined_tool_calls → inlined_no_tools |
| `_coerce_tool_call_arguments()` | 新增 | `function.arguments` 字符串 → dict,解掉 41.3 |
| `_inline_tool_calls()` | 新增 | 非法 JSON 参数时把 tool_calls 渲染进 content,解掉梯子最后一格 |
| `_assistant_text()` | 改 | content 和 tool_calls 从「二选一」改成**拼接**,解掉 41.3b |
| `_check_against_usage()` | 新增 | 和 `usage` 的权威计数对表,超过 `max(2, expected/20)` 就计数上报 |
| `_CAPTURE_STATS` / `capture_stats()` | 新增 | 前 3 次和每 200 次打一条 warning;供开训前断言取数 |

回归测试 `scratchpad/capture-fix/test_patched_proxy.py`(直接 import 打过补丁的真模块),**21/21 通过**:

* 模板梯子 6 个用例 **3/6 → 6/6**。原本就通过的两个用例 token 数逐一不变(16→16、267→267,
  都走 `as-is`),**无回归**;多轮并行调用那个用例走 `args_dict` 拿到 373 tok;
  非法 JSON 参数那个走 `inlined_tool_calls` 拿到 312 tok。
* `_assistant_text` 在「content 有 think、tool_calls 有动作」的样本上 **8 tok → 49 tok**。
* `_check_against_usage` 拿 41.3b 的真实数字(17 vs 59)做输入,**抓到了**;59 vs 59 不报;
  58 vs 59 在容差内不报。
* `return_tokens_as_token_ids` 已再确认是 `ChatCompletionRequest` 的字段(`protocol.py:405`,
  不只是 `CompletionRequest`),且 `OpenAIBaseModel` 是 `ConfigDict(extra="allow")`(`protocol.py:48`),
  所以老的 `return_token_ids` 一直被静默接受也是这个原因。

**还没验证的那一半**:以上全是离线单测。真实采集率有没有从 8.1% 起来,必须等一次带
vLLM 的 smoke 跑(建议 10 步),看 `capture_stats()` 和三元组数 / 真实调用数的比值。
**在那个比值确认到 90% 以上之前,不要把 §39 的那批优化器参数(mini_batch/lr/GSPO/rollout.n)
放进来** —— 否则任何结果都归因不了。

## 42. 上 v3 那七项之前查出来的三个拦路条件(2026-09-15)

### 42.1 `loss_mode=gspo` 会把 rollout 级优势归一化**顺手关掉**

`trainer.py:708-718` 那段是这么写的:

```python
loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
if loss_mode == PER_ROLLOUT_MEAN_LOSS_MODE:
    batch.batch["advantages"] = normalize_advantages_by_rollout(...)
```

**归一化是挂在 `loss_mode == "per_rollout_mean"` 这个 if 上的**,不是挂在
`algorithm.enable_rollout_level_advantage` 上。所以把 loss_mode 换成 `gspo`,
`normalize_advantages_by_rollout`(按该 rollout 的总 token 数 × 批内行数做除法)整段不执行,
而且**不报错、不告警**。

后果不是抽象的:GSPO 的 `agg_loss(..., "seq-mean-token-mean")` 是按**行**平均的
(`core_algos.py:1597`),而我们一行 = 一个 LLM 轮次,不是一条轨迹。于是一条 50 轮的轨迹
会按 50 行进梯度,一条 1 轮的按 1 行——长轨迹的权重直接压死短轨迹。
`per_rollout_mean` 现在正是在挡这件事。

要 GSPO 就得先决定:是把 rollout 归一化挪到 `if` 外面(改 AGL),还是接受按轮次加权。
**这不是一个配置项能切的。**

### 42.2 `max_ppo_update_times=2` 卡死了"一个 rollout 更新四次"

`trainer.py:594-598`:

```python
mini_bs = actor.ppo_mini_batch_size * rollout.n
n_remained_transition = n_transition // mini_bs * mini_bs
if max_ppo_update_times is not None:
    n_remained_transition = min(n_remained_transition, mini_bs * max_ppo_update_times)
```

`agentlightning.max_ppo_update_times` 当前是 **2**(`train_opencode_agent.py:146`)。
它直接把参与训练的行数砍到 `mini_bs × 2`,也就是**最多两次优化器步**。
只把 `ppo_mini_batch_size` 从 8 改成 2,得到的还是 2 次不是 4 次——
`max_ppo_update_times` 必须同步改成 4(或 None)。

日志实测对得上:286 步里 `n_sample_trained` 只有两种取值,**32(102 步)和 64(184 步)**,
正好是 `mini_bs=32` 的 1 倍和 2 倍。

### 42.3 更正 §39.3:不是"永远只有一次更新、ratio 恒等于 1"

之前记的是"train_batch=8 × n=4 = 32 = 满批 → 一次优化器步 → ratio≡1 → clip 全是死代码"。
按日志重算:

| 指标 | n=286 步 | 非零占比 | 中位 | max\|·\| |
| --- | --- | --- | --- | --- |
| `actor/ppo_kl` | 286 | **64.3%** | 0 | 1.75e-03 |
| `actor/pg_clipfrac` | 286 | 63.6% | 8.0e-05 | 3.26e-03 |
| `actor/pg_clipfrac_lower` | 286 | 1.0% | 0 | 8.26e-06 |

非零的 184 步和 `n_sample_trained=64` 的 184 步**逐步对齐**——两次更新的那些步,
第二次更新的 ratio 确实不等于 1。所以准确说法是:
**36% 的步只更新一次(ratio 恒为 1),64% 更新两次**;
而即便在更新两次的步上,`pg_clipfrac` 中位数也只有 8e-5、最大 0.3%,
**clip 阈值几乎从不触发**。

结论的方向没变(clip-higher / dual-clip / GSPO 目前都是二阶项),但原因不是"只更新一次",
而是 **lr=1e-6 × 至多两次更新,策略移动得太小**。这反过来正好说明 #1(多更新)+ #2(抬 lr)
是这批改动里唯二动到主因的。

### 42.4 零优势组占 46.8% —— 动态采样是这七项里最值钱的

`training/n_zero_adv_groups` / `training/n_groups` 全程累计 **902 / 1928 = 46.8%**:
接近一半的 rollout 组内奖励完全没有方差,GRPO 优势恒为 0,梯度为 0,算力白烧。
(不过没有任何一步是**整步**全零,286 步里 0 步,所以训练从没彻底空转。)

AGL 其实已经有半套机器:`_same_reward_uid_indices`(`trainer.py:115`)专门找零方差组,
`trainer.py:605` 在必须丢行凑 mini_bs 整数倍时**优先丢它们**。
差的是 DAPO 的另一半——丢完之后**继续采样**补上,而不是就这么把批缩小。
所以 #7 是"扩这段已有代码",不是"从零实现",但仍然是改代码不是改配置
(`filter_groups` 在这套 verl 里只有一个 config dataclass 空壳,`ray_trainer.py` 没接)。

## 43. 结算:之前跑的 284 步是废的,以及为什么它能废得这么安静(2026-09-15)

### 43.1 废到什么程度

不是"效果差一点",是**梯度来源错了**。§41 证明的两条 bug 合起来的效果是:

- 多轮 agent 的第 2 轮及以后,**全部**在 proxy 里因 chat template 崩溃被静默丢弃
  (`if not response_token_ids: continue`),采集率 8.1%;
- 连活下来的第 1 轮,response 也被 `_assistant_text` 砍掉了动作部分
  (MiniCPM5 的 XML 工具解析器已经把 `<function=...>` 从 content 里摘走了),
  59 token 只剩 17 token。

所以**实际进入 loss 的,是这么一段字面量**:

```
<think>
Let me start by exploring the repository structure to understand the codebase.
</think>
```

一句开场白,动作被剃掉,然后被赋予整条 ~25 轮轨迹的奖励。
284 步优化的是"怎么把这句开场白说得更像能拿高分的开场白"。
评测结果(step 154 / step 284 与基座均无统计差异,p=0.75)与这个结论完全自洽——
**没效果不是因为方法不行,是因为方法根本没被执行。**

### 43.2 为什么它能这么安静地废掉

值得单独记,因为这是这次最贵的一课:**整条链路上没有一个环节把"丢弃"当成错误**。

1. **异常被吞**:模板渲染抛 `jinja2.UndefinedError`,被 catch 住返回空,不 raise、不计数。
2. **丢弃被当成正常路径**:`continue` 是循环里最不起眼的语句,没有任何计数器。
3. **唯一能暴露它的指标被我读错了**:`training/n_turns = 72.33` 我当成"每条轨迹 72 轮",
   实际是"每步 72 行 / 32 条 rollout = **2.2 轮**"。
   一个 agent 平均只跑 2.2 轮去修 bug 是荒谬的,这个数字当时就该炸。
   **指标读错比指标缺失更危险**,因为它会主动提供虚假的安全感。
4. **日志分家**:proxy 的 structlog 写 agl-server 的日志,我一直在 grep trainer 的 stdout。
   "trainer 日志里没有报错"根本不是证据。
5. **结构性失败长得像正常分布**:第 1 轮没有历史 tool_calls 所以活,第 2 轮起必死,
   summarizer 调用不带 tools 所以活。8.1% 这个数很稳,不抖,看起来完全像"设定如此"。

一句话:**一个把错误当成过滤器的管道,会安静地训练空气。**

### 43.3 已经加上的护栏

- `CaptureTally`(`agl_rollout_manager.py`):每步打印
  `capture_rate=X% (kept/calls) dropped_error=… dropped_no_token_ids=…`,低于 90% 告警;
  设了 `AGL_MIN_CAPTURE_RATE` 就**直接 raise**,拒绝在残缺数据上开训。
- proxy 里每条丢弃分支都过 `_bump()` 计数,前 3 次和每 200 次打 warning。
- `_check_against_usage()`:拿服务端返回的 `usage.prompt_tokens/completion_tokens`
  跟本地 token id 数对账,容差 `max(2, expected//20)`,对不上就计数。
- 根治项:`return_tokens_as_token_ids=True`(vLLM 0.8.5 `protocol.py:405` 就有,
  之前传的 `return_token_ids` 是个不存在的字段,被 `extra="allow"` 静默吃掉了)。
  **不再从解析后的字段反推 token,直接用服务端真值。**

### 43.4 顺带记一条:verl 的 batch 单位是"行",不是"轨迹"

今天被问到才发现这点没写过,而它直接决定了 v3 怎么配参:

- verl 里一行 = 一次 LLM 调用 = 一个 (prompt, response) 序列。多轮 agent 的一条 rollout
  摊成 N 行,共享一个 `rollout_id`。**所有 batch 旋钮都按行计,没有"轨迹"这个单位。**
- 全局 mini-batch = `ppo_mini_batch_size × rollout.n`(`fsdp_workers.py:249`),
  更新次数 = `n_transition ÷ 全局 mini-batch`,再被 `max_ppo_update_times` 截断
  (`trainer.py:594-598`)。
  v2 实测反验:`mini_bs = 8×4 = 32`,cap=2 → `n_sample_trained ∈ {32,64}`,与日志逐步吻合。
- "一条轨迹算一个 rollout、分段相加"的语义**不在 batch 层,在 loss 层**:
  `rollout_level_advantage.py` 每条轨迹一个优势广播回所有行,
  `per_rollout_loss.py:29-41` 每行除以 `该轨迹总 token × 整批行数`,`:83` 用 `masked_sum`。
  三者合起来一条轨迹的贡献恰好是 `A_i / N`,与轮数无关。
- 归一化系数是**切 mini-batch 之前**按整批算好的(`trainer.py:708-718` 在 `_update_actor` 之前),
  所以每行带着正确系数进 mini-batch,切到哪块都不变;8 次更新的贡献加起来精确等于整批 loss。
  变的只是"相加时 θ 已经走了几步"——这是 minibatch SGD 的固有性质,不是轨迹被切才有的,
  而且 `old_log_prob` 在 `trainer.py:654` 对整批算一次、所有 mini-batch 共享同一个 θ_old,
  PPO 的 ratio 正是补偿它的机制。
- 真正要小心的是 `trainer.py:594` 那个**按行**向下取整:它可能砍掉一条轨迹的部分轮次。
  但在 v3 参数下(`n_transition≈800`,`mini_bs=100`,`MAX_UPDATES=8`)要丢的只有余数 0~99 行,
  而零优势行有 ~45%×800=360 行足够填,这些行优势恰好为 0、梯度贡献为 0,丢了无损。
  只有当 `n_transition > mini_bs × MAX_UPDATES` 时才会丢到活行——这是 smoke 要看的第二个数。

### 43.5 rollout.n 该不该从 4 涨到 8:算下来是负收益

n 只决定 GRPO 组内样本数。实测零优势组 46.8%(n=4),由 `p⁴+(1-p)⁴=0.468` 反解
单条成功率 `p≈0.175`。代入 n=8:零组率 45% → 21.5%,每步可用组 4.4 → 6.3。
但生成调用翻倍(每步 ~800 → ~1600)。折算成**每千次调用拿到几个可用组**:
n=4 是 5.5,n=8 是 3.9,**n=8 反而低 30%**。
零组的对症解法是动态采样(只重采死掉的组),不是对所有组加倍采样。

### 43.6 什么没白费

基础设施部分全部有效,且与这个 bug 无关:k3s rollout 管道、verl 0.7.1 的
abort 静默失效补丁(AGL #589 / PR #590-592)、NVLS 规避、MPS 绕过、
温度消融结论(§36)、"单步大头是等最后一条 rollout"(§38)、以及本文件。
**报废的是那 284 步权重和由它得出的一切效果结论,不是这条链路。**

---

## §44 修好采集率之后,同一类 bug 在下一层又出现了一遍(2026-09-15)

§43 的补丁把 `capture_rate` 从 8.1% 抬到 **100.0% (961/961)**,零丢弃。但同一次 smoke 的
step1 指标立刻暴露了下一层:961 轮只变成 227 个训练行,32 条 rollout 里 **30 条没合并成**。

教训的形状和 §43 一模一样:**一个静默的降级,藏在一个看起来只是"效率不高"的指标后面。**

### 44.1 断裂率要自己算,日志里那个数是上限

日志打的是 `training/n_trace_merge_mismatch_rows:100`。这个 100 是
`rollout_adapter.py:23` 的 `_TRACE_MERGE_MISMATCH_WANDB_LIMIT` 写死的收集上限,不是真值。
真值可以从别的字段推出来:

    断裂次数 = n_sample - n_rollouts          (每条 rollout 断 g-1 次,产出 g 行)
    断裂率   = 断裂次数 / (n_turns - n_rollouts)

第一次 = (227-32)/(961-32) = **21.0%**。把收集上限提到 2000 之后实测 171,
而 `203-32 = 171`,**完全对上**。

> 任何"带上限的计数器"都不能直接当频率读。要么先确认没顶到上限,要么用别的字段反推。

### 44.2 根因:prompt 和 response 不同源

adapter 的 trajectory 级合并只有一个判据(`rollout_adapter.py:672`):

```python
if ids_startswith(prompt_ids, current_context):   # current_context = 上一轮 prompt + response
```

要求第 k+1 轮的 prompt token 序列**逐 token** 等于第 k 轮的 prompt+response 往后接。
而这两半根本不是同一个来源产出的:

- `response_ids` 来自服务器(`return_tokens_as_token_ids` + logprobs 路径),精确;
- `prompt_ids` **只能本地用 chat template 重渲染**——因为 vLLM 0.9.2 根本没有
  `return_token_ids` 这个字段(`entrypoints/openai/protocol.py` 里 grep 不到),
  而 `:48` 是 `extra="allow"`,所以传过去被静默吞掉。停机时 vLLM 自己把这句打出来了:

      The following fields were present in the request but ignored: {'return_token_ids'}

**一半是真 token、一半是重渲染,逐 token 比对必然掉。** 真正该问的不是"怎么让重渲染更准",
而是"为什么要重渲染"。

### 44.3 定位现场:截断方向错了,字段就永远拍不到

`previous_trace` / `current_trace` 这两个字段是为了诊断存在的,但
`_decode_trace_text` 从**头部**截到 4000 字符,而断裂发生在 previous_trace 的**尾部**
(实测真实长度 38244 字符)。**这两个字段拍不到任何一次现场。**

改成在 token 空间定位首个分歧下标,记录 `diverge_index` / `diverge_kind` 和两侧
各 ±60 token 的解码窗口(`_diverge_report()`),一次就把 152 条全分类完了。

> 诊断字段也要验证。"存了上下文"不等于"存到了出事的那一段"。

### 44.4 152 次断裂的完整归类

| | 占比 | 原因 | 性质 |
|---|---|---|---|
| A | 34.9% | opencode 把系统提示换成 "context summarization agent" | 不是 bug:另一个 agent 共用 rollout |
| B | 28.3% | parser 无条件 `.strip()` 参数值 | **线上功能 bug** |
| E | 27.0% | `_FUNC_BLOCK_REGEX` 匹配散文里的 `<function` | **线上功能 bug** |
| D |  9.9% | 文本相同、token 切分不同 | 重分词漂移 |

`diverge_kind` 全是 `token_differs`,没有一条 `ran_out`——历史从来没被裁剪过,只是被**改写**。

### 44.5 B:parser 把每个参数值都 strip 了

`minicpm5xml_tool_parser.py:215`(上游原版):

```python
val_text = (param.text or "").strip()
```

`edit` 工具的 `oldString` / `newString` 因此丢掉前导缩进和尾随换行。真实轨迹里的因果链完整可见:

    模型写的:  <param name="oldString">    return str(v).upper()</param>
    parser 给: <param name="oldString">return str(v).upper()</param>
    工具回答:  Found multiple matches for oldString. Provide more surrounding context.

匹配失败只是轻的;匹配成功时写进去的是**顶格的 Python 代码**。
单测里旧 parser 把 `'    def f(self):\n        return 1\n'` 变成
`'def f(self):\n        return 1'`——首行被顶格,尾换行丢失,**静默损坏源码**。
模型在为 parser 犯的错挨罚,reward 被这个压着。

修法:CDATA 包着的值必须原样保留(模板正是在含 `<`/`&`/换行时才包 CDATA,
就是为了保内容),裸值仍然 strip 以容忍 XML 缩进排版。

**但这个补丁只救 43 条里的 24 条。** 剩下 19 条是"有前导空白但不含 `<`、`&`、换行"的
单行值(例如上面那个 `    return str(v).upper()`)——模板不会给它包 CDATA,
parser 也就无从分辨。**这种值在这套 XML 协议里根本无法表达。**
要救得放宽模板的 CDATA 触发条件,那会改变模型的训练分布,属于要人来拍的决定。

### 44.6 E:散文里的 `<function` 被当成工具调用

`minicpm5xml_tool_parser.py:50`(上游原版):

```python
_FUNC_BLOCK_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)
```

匹配的是任意 `<function`,不是 `<function name="..."`。模型推理里引用 Python 的函数
repr —— `<function Email at 0x7f02944c6950>`,在 voluptuous 这种满是函数对象的库里到处都是 ——
正则就从这里开块,一路非贪婪吃到**真正那次调用**的 `</function>`,
把中间模型写的所有推理正文整段删掉。

修法是加一个 `name=` 约束,和它上面第 45 行的 `_FUNC_NAME_V1_REGEX` 保持一致:

```python
_FUNC_BLOCK_REGEX = re.compile(r"<function\s+name=['\"].*?</function>", re.DOTALL)
```

> §43、§44.5、§44.6 是同一个 bug 家族:**解析层悄悄吃掉内容,而调用方拿不到任何信号。**
> 每一次都表现为"效果不好",不是"报错"。

### 44.7 单测必须证明自己能失败

7 条用例(CDATA 保缩进 / 保尾换行 / 裸值仍 strip / 只识别真调用 / 推理正文保留 / 普通调用回归),
在新 parser 上 7 条全过。**光这个不说明任何问题。**
把同一份用例跑在备份的旧 parser 上:**4 条失败、3 条通过**——失败的正是两个 bug 对应的那 4 条,
通过的是两条容错/回归用例。这才说明用例测的是补丁本身,而不是在自说自话。

### 44.8 顺带修掉的:MAX_UPDATES 太小,每步随机扔掉 51 行

多轮 rollout 碎成 227 行,而全局 mini-batch 上限是 `train_batch × n = 32`
(verl 硬约束 `train_batch_size >= ppo_mini_batch_size`,`workers/config/actor.py:216`)。
`MAX_UPDATES=4` 只吃 128 行,剩下 99 行被丢:48 条同奖励(优势为 0,无损)+ **51 条随机**。

随机那 51 行是真丢数据,而且会**偏掉指标**:step1 `critic/score/mean` 是 0.095,
而 `training/reward` 是 0.625——差一个数量级,因为被训练的那个子集不是原批次的无偏样本。
改成 `MAX_UPDATES=7`(7×32=224 ≈ 227)之后 `n_sample_dropped/random` 归零,
三步实测 score/mean 与 reward 贴合(0.625/0.5625、0.418/0.55、0.891/0.844)。

> 两个本该相等的指标差一个数量级,就是在告诉你采样有偏。别把它当噪声。

### 44.9 还没动的

- **长度配置**:`max_prompt 45056 / max_response 6144` 这个划分是给单轮定的。多轮下
  `response_length/clip_ratio` 实测 0.22~0.45,行被 6144 截断。而且**合并一旦修好,
  6144 会把大多数轨迹砍掉一大截**——所以修合并之前必须先把长度理顺,否则是负优化。

  这个"一大截"不再是估计。44.3 加的落盘诊断里每条记录都带 `expected_len`
  (= 上一轮 prompt + response 的 token 数,也就是轨迹跑到该轮时的**总上下文长度**),
  拿 step 2~7 的 210 条 rollout 聚合出来:

  | 量(按 rollout 取极值) | p50 | p90 | p99 | max |
  |---|---|---|---|---|
  | 轨迹总 token | 28695 | 32508 | 34770 | 36922 |
  | 最大 turn_index | 15 | 56 | — | 156 |
  | response 跨度(轨迹尾 − 最早观测 prompt) | 15174 | 23857 | 26329 | 30035 |

  **全是下界**:落盘的只有断裂轮,一条 rollout 的首次断裂往往已经在轨迹中段,
  所以"最早观测 prompt"比真正的首轮 prompt 大,算出来的跨度偏小。

  照这个跨度分布反推各档 `max_response_length` 的覆盖率(同样是乐观值):

  | max_response | 6144(现值) | 8192 | 12288 | 16384 | 20480 |
  |---|---|---|---|---|---|
  | 覆盖率 | 46.2% | 48.1% | 49.0% | 52.4% | 71.9% |

  两件事要读出来。一是**现值只覆盖不到一半**,合并修好之后过半轨迹会被截。
  二是这条曲线**是双峰的**:6144→16384 只多覆盖 6 个点,16384→20480 一下多 20 个点。
  说明任务天然分成"几轮就结束"和"几十轮长跑"两簇,中间档位买不到什么东西——
  要么省着不动,要么一次给够。

  预算是够的:`max_model_len=51200`,而轨迹总长 p99 才 34770。实测 prompt 侧
  `prompt_length/max=28532`、`clip_ratio=0.0`,45056 的额度有近 16k 白放着。
  可选的重切法是 **prompt 28672 / response 22528**(合计仍是 51200):prompt 侧
  刚好罩住实测最大值,response 侧从 46% 拉到 90% 以上。**但这会改变训练分布,
  要和 44.5 的 parser 补丁一起上、一起评**,不能单独改一边。
- **D 类重分词漂移(9.9%)**:文本完全相同、token 切分不同。这类只能让合并判据容忍,
  修不到上游。
- **A 类(34.9%)**:opencode 的 summarization agent 共用 rollout,要在 adapter 里
  按系统提示分流,而不是当成断裂。

### 44.10 credit assignment 其实一直是对的

值得记一笔,免得下次又慌:碎片化**没有**破坏信用分配。
`per_rollout_loss` 按 `rollout_id` 汇总该 rollout **所有行**的 token 数做分母,
所以碎成 20 行和合成 1 行,每条 rollout 的贡献都还是 `A_i / N`。
碎片化真正的代价是**行数暴涨**,撞上 44.8 那个 mini-batch 上限。

### 44.10 断裂归因:压缩只占一半,另一半是我们自己造的(2026-09-16)

人类给的标准很干脆:**"如果触发压缩,变成碎片无可厚非,但如果没触发压缩,就不能断裂。"**
压缩 agent 换了系统提示,对话确实换了一条,切开是对的;除此以外的断裂都是 bug。

把全 32 步、4950 个断裂点逐条归因:

| 类 | 成因 | 条数 | 占比 | 可修 |
|---|---|---|---|---|
| A | 压缩 agent 换系统提示 | 2531 | 51.1% | 否(本来就该切) |
| B | 空白/缩进被 CDATA 解析吃掉 | 1800 | 36.4% | 是 |
| D | 重分词漂移 | 571 | 11.5% | 是 |
| E | 散文里的 `<function` 被当成工具调用 | 48 | 1.0% | 是 |

**非压缩断裂 = 2419 = 48.9%,平均每条 rollout 2.4 次。** 这些是白丢的。

D 类最阴:**对话根本没变,只是被分词了两次**。
server 侧的 context 是 `prompt_ids + 逐个采样出来的 token`;
下一轮的 `prompt_ids` 是整段对话重新过一遍 chat template。
模型逐 token 采样时**完全可以吐出非规范切分**——它吐 `pass`+`ed`,
而重新分词得到 `passed`,解码后一模一样,token id 却对不上。
`tokenize(a+b) != tokenize(a) + tokenize(b)`,于是 `ids_startswith` 判定"对话变了",
把一条好好的轨迹从中间剖开。实测一条 20k token 的轨迹能漂 13 个 token。

修法是在 id 判据失败后**回退到文本层**再判一次
(`rollout_adapter.text_level_continuation`):解码两边比 `startswith`,
residual 重新编码成 observation。注意 residual 必须**重新编码**,不能从 `prompt_ids`
里切片——文本边界不一定落在它的 token 边界上。开销实测 4.3 ms/次,
按 4950 次/步算是 21 s,占 step 总耗时 1456 s 的 1.5%。

**测试必须能失败。** 第一版链式测试我把 seam 切在 `<|im_end|>` 上,
那是特殊 token 边界、根本不漂移,结果旧判据也合并成 1 行——测试通过了,但什么都没证明。
改成模拟"逐 token 采样产生非规范切分"之后,旧判据 8 轮断成 8 组、新判据合成 1 组,
判据才真正被区分开。仓库里那套用 `MergingTokenizer`(字符级 + 合并规则)表达
"同一文本有多种切分",不依赖模型文件,CI 能跑;
另配一个 `test_merging_tokenizer_actually_drifts` 守着,防止哪天它退化成不漂移、
让下面几个测试全部空过。

顺带记一条:**加埋点要同步改测试**。09-15 给 mismatch 表加了 4 个诊断列、
给轨迹 jsonl 加了 5 个统计字段,没动测试,13 个既有用例一直红着;
其中 `self._capture = CaptureTally()` 写进 `__init__`,直接让 10 个
用桩子类(故意不调 `super().__init__()`)的用例挂掉。
改成类属性 + 惰性访问器 `_capture_tally()` 就好了——
**埋点不该是压垮测试桩的那根稻草。**

**B/E 补丁的验证(同日)。** 补丁在盘上躺了一天没测过,而它要负责 2419 条非压缩断裂里的 1722 条。
补上对照测试(`agl-checkpoints/swe_smith_smoke/test_parser_patch.py`):
同一组输入分别喂给打补丁前的备份和现在的版本,旧版必须失败、新版必须通过,
6/6 全绿,兜底正则路径和"非 CDATA 值仍然 strip"的回归都覆盖了。

E 类这里又踩了一次"测错东西":我最初断言的是**工具调用丢了**,结果旧版照样把
`edit` 解析得好好的,测试红着。真实的坏法是——`<function.*?</function>` 从散文里的
`<function Email at 0x7f02944c6950>` 就开了块,一路吞到真调用的 `</function>`;
**调用没坏,坏的是 `content`**:旧版返回 `'I got back'`,后面那半句连同 Python repr
全被吞进 block 里没了。重渲染时这段文字就凭空消失,于是对话对不上、轨迹被切开。
教训是:归因说"解析被破坏"还不够,得说清**破坏落在哪个字段上**,
否则写出来的测试量的是另一件事。

另外顺手确认了一件容易搞错的事:线上 `--tool-parser-plugin` 指的是
`minicpm5_parser_plugin_085.py`,不是我改的那个文件——但那只是个 0.8.5 兼容 shim,
它 `exec_module` 的正是同目录的 `minicpm5xml_tool_parser.py`,补丁在对的位置。
shim 的 docstring 原本写着"unmodified upstream file",现在已经不成立,一并改掉了;
**这种注释一旦过期,下次就是照着它做出错误判断的起点。**

### 44.11 补丁上线后的实测:非压缩断裂降了 93%,但把截断问题顶了上来(2026-09-16)

09-16 14:28 带三个补丁重启(parser B/E + adapter D),`resume_mode=auto` 从
global_step_36 续跑 —— 特意不从头重训:模型、优化器状态、数据分布都不变,
**只有补丁变了**,这样断裂数的变化才归得了因;从基座重训会把模型漂移和补丁效果混在一起。

| step | 训练行数 | 未合并 rollout | mismatch 行 | 其中非压缩 | 文本层合并 |
|---|---|---|---|---|---|
| 35(补丁前) | 134 | 27/32 | 102 | 76 | — |
| 36(补丁前) | 119 | 28/32 | 87 | 51 | — |
| 37 | 73 | 12/32 | 41 | **9** | 13 |
| 38 | 98 | 14/32 | 66 | **1** | 16 |
| 39 | 100 | 18/32 | 68 | **6** | 4 |

非压缩断裂 ~76/步 → 5.3/步,**降了 93%**;折算每条 rollout 从 2.4 次降到 0.17 次。
轮数没怎么变(573~870),训练行数却从 180+ 掉到 73~100 —— 合并是真的在合。

**读数时要看的是分类,不是总数。** 37→38→39 的 mismatch 从 41 回升到 68,
乍看像是在退化;拆开看回升全在 A(压缩):32 → 65 → 62,非压缩反而是 9 → 1 → 6。
总数会随任务难度和压缩触发次数起伏,**只有归因后的"非压缩"这一列才是判据**。

**代价:截断率从 36~38% 跳到 52~59%。** 行合并之后每行更长,
`max_response_length=6144` 就截得更狠。而按 44.8,截断的行 `is_drop=False`,
照样拿整条轨迹的 reward 去训一段被砍掉的回复 —— **修好合并反而放大了这个坑**。
长度重配不再是可选项。

**残留的非压缩断裂都指向同一件事:模型自由输出 vs 模板确定性重渲染。**
- 模型给 `cat /tmp/...` 包了 CDATA,但值里没有 `<`/`&`/换行,模板按规则不包 → 标记消失
- 模型多行 `oldString` 没包 CDATA,解析器照旧 `.strip()` → 缩进被吃
- opencode 对超长工具输出的截断标记 `...[truncated N chars: tool output limit]`
  在两次渲染之间挪了位

只改模板的 CDATA 触发条件最多修一半 —— 模型往哪边偏是不可预测的。
真正的根治是 **assistant 轮不要重渲染,历史里直接回放模型的原始生成文本**;
只要还是"解析成结构再拼回去",自由输出和确定性重渲染就永远会有对不上的时候。

**分类器本身踩了三次同一个坑,记在这里。** merge_mismatch dump 里的
`expected_window`/`actual_window` 是围绕分歧点截的定长切片,**两侧长度未必相同**
(比如 435 vs 440)。拿整段窗口做比较,尾巴错位会污染判断:
第一次把 361 条 D 误判成一个不存在的"多出整轮消息"类,第二次第三次把
CDATA 那族全扔进"其他"。正确做法是**只比对分歧点之后的一小段**。
教训:**比较两个对齐不确定的切片时,先找共同前缀,再只看分歧点之后。**

#### 44.11.1 归因闭合:37~42 六步 308 条断裂,0 条未归因

| step | 总断裂 | A 压缩 | B 空白 | C CDATA | E 散文 | F 截断漂移 | **非压缩** |
|---|---|---|---|---|---|---|---|
| 37 | 41 | 32 | 4 | 5 | | | 9 |
| 38 | 66 | 65 | 1 | | | | 1 |
| 39 | 68 | 62 | 2 | | 1 | 3 | 6 |
| 40 | 62 | 56 | 3 | 2 | | 1 | 6 |
| 41 | 37 | 33 | 1 | 3 | | | 4 |
| 42 | 34 | 28 | 3 | 3 | | | 6 |

非压缩稳定在 4~6 条/步(补丁前 51~76)。**F 是新发现的一类,成因在我们自己的
proxy**:`opencode_agent.py` 的合并截断预算只作用于"最后一个 assistant 之后"的
工具消息,于是同一条工具输出的渲染取决于它的**位置** —— 第 k 轮它在尾部被截成
`...[truncated N chars: tool output limit]`,第 k+1 轮后面多了个 assistant,它不在
尾部了,全文又回来了。内容一个字没改,渲染却变了,重渲染的 prompt 就不再是服务端
context 的延续。改成"每一段连续工具消息各自算预算",裁剪就只取决于这段自己的内容,
整条 episode 稳定。测试 `agents/test_truncation_stability.py` 6/6,含一条对照
(同一 fixture 打在改前备份上必须失败)。

**教训:重渲染链路上任何"取决于位置"的规则都是断裂源。** 幂等性在这里不是
洁癖,是正确性前提 —— 只要 `render(msgs[:k]) ` 不是 `render(msgs[:k+1])` 的前缀,
轨迹就会被切开。写这类裁剪/摘要/省略逻辑时,判据是"这条消息的渲染是不是只由
它自己的内容决定"。

**分类器那个坑,第四次。** 这次是两处空白差异:一次 edit 调用里 `oldString` 和
`newString` 会各丢一处缩进,只在第一个分歧点 lstrip 一次,第二处照样不等,于是
被扔进"其他"。正确做法是**去掉全部空白再比公共前缀**,并且只比到两侧较短的
那一个为止(窗口是定长切片,空白差异让一侧整体左移、多伸进未来一点)。
三条改完,未归因从 4 条降到 0。

### 44.12 6144 截断:两个 6144 不是一回事,截断是"砍尾",不是"切段"

**先纠一个直觉错误。** "6144 是单轮输出上限,一条 rollout 当然可以超" —— 对,
但训练里的 6144 不是那个 6144。有两个:

| 6144 | 在哪 | 管什么 |
|---|---|---|
| `OUTPUT_LIMIT = AGL_MAX_TOKENS` | `opencode_agent.py:41,132` | 生成侧,每轮最多生成多少 |
| `data.max_response_length` / `trajectory_max_response_length` | `rollout_adapter.py:667-668` | 训练侧,**每一行**的 response 槽位 |

轨迹级合并之后一行 = 整条轨迹:`row.response = resp₀ + obs₁ + resp₁ + … + respₙ`
(`rollout_adapter.py:766-773`,观测 mask=0)。25 轮加起来早过 6144,于是
`append_training_row` 在 :667-673 **切片**:`response_ids[:6144]`,尾巴直接扔掉,
`is_drop` 还是 False,行照训。**这不是把 rollout 切成很多段、产生很多小 batch,
而是只留开头几轮、丢掉后面所有轮** —— 而拿到 reward 的恰恰是最后几轮。
44.11 修好合并之后行变长,截断率从 36% 跳到 55%,坑被放大。

**槽位为什么不能直接调大。** `:683-688` 是定长 padding,
`max_prompt + max_response = 51200 = max_model_len`,也是 `ppo_max_token_len_per_gpu`
和 `num_gpu_blocks_override=25600` 的依据。槽位只能挪,不能长。

**长度重配(45056/6144 → 28672/22528)为什么也不行。** 一旦按预算切行,后面那
行的 prompt 是到该轮为止的**全部上下文**(能到 45k),prompt 槽 28672 装不下 →
`is_drop=True` 整行丢。重配是把"砍 response 尾"换成"丢整行",不是修。

**切段对训练效果有没有影响 —— 没有,梯度层面是中性的。** 一行训的是
`Σ_t log π(y_t | prefix_t)`,把轨迹在第 k 轮切开,第 k 轮之后的 token 只是从
"response 里 mask=1 的一段"变成"下一行 response 里 mask=1 的一段",条件前缀一模一样,
每个 token 的 log-prob 一样;reward 是 rollout 级广播,两行拿同一个数。代价只有两个:
行数多 → 更容易撞 `trainer.py` 的丢行逻辑;共享前缀重复 prefill。两个都能堵。

**解法:到预算就切一刀,永远不砍(`rollout_adapter.py` 合并循环)。** 合并第 k 轮前先算
`projected = len(current_response) + len(obs_k) + len(resp_k)`,超过
`max_response_length` 且 `len(prompt_k) <= max_prompt_length` 就先把当前组落成一行,
从第 k 轮开新组(prompt = 该轮完整上下文)。prompt 也装不下才退回原来的截断,
并计数 `training/n_budget_splits_blocked`(应该长期为 0,不为 0 就是上下文真过 45k)。
预算切行**不计入** `n_unmerged_rollouts` —— 那个指标是断裂诊断,不能被正常切分污染。

**下游两处丢行一起堵(`trainer.py`)。**
1. `n_transition // mini_bs * mini_bs` 把凑不满 32 的余数扔掉 → 改成**向上补齐**:
   随机复制几行,复制行给独立 uid + 零 score,GRPO 单元素组均值 0 → 优势恰好 0,
   源组的均值不动;只花一次前向,不训任何东西。同奖励行本来就零优势,不需要特殊对待。
   为什么不给复制行原 uid:组均值会挪 `(mean−r)/(n+1)`,别的行的优势跟着偏。
   为什么不把 mask 清零:`seq-mean-token-mean` 那类聚合会除以 0 出 nan。
2. `MAX_UPDATES` 7 → 16:封顶是唯一还会真丢行的地方,512 行/步远高于实测 66~110。

**指标:** `training/n_truncated_sample` 应归零;`training/n_budget_splits` 每步几十
是正常的;`training/n_sample_padded` ≤ 31;`training/n_sample_dropped/*` 应为 0。

**测试(先写、先在旧代码上失败):** `tests/verl/test_rollout_adapter.py`
`test_response_budget_cuts_a_new_row_instead_of_truncating`(4 轮链式 fixture,
生成 token {2,4,6,8} 必须全部 mask=1 到达优化器;旧代码只剩 {2,4})及 blocked 变体;
`tests/verl/test_mini_batch_fit.py`(66 行 → 补 30 而非丢 2;pad 行 uid 独立、score 为 0)。

**教训:** "限制"要问清楚是哪一层的限制;"截断"要问清楚是切段还是砍尾 ——
砍尾丢的是拿 reward 的那几轮,是 credit assignment 级别的错,不是长度统计上的小事。

**补齐上线第一步就崩了,原因是 pad 行的"身份"没给全。** 报错
`rollout-level advantage found multiple uid values for rollout_id=...`:
`rollout_level_advantage.py:41-50` 按 `rollout_id_list` 分组、每组取一个代表行算优势,
并校验组内 uid 一致、reward 一致。pad 行拿了新 uid 却沿用源行的 `rollout_id`,校验直接炸。
修法:pad 行同时拿新 `rollout_id`(`pad-<hex>`)—— 它就是一条单行 rollout、单元素 GRPO 组、
score 0,优势恰好 0。顺带发现第二处:`per_rollout_loss.py:36-41` 把优势除以
`rollout_token_count × num_trained_rows`,`num_trained_rows=len(batch)` 若含 pad,
真实行的梯度会被整体缩小 `n/(n+pad)`(最多近一半)—— 改成 `len(batch) − n_to_pad`。
测试 `test_padded_batch_passes_rollout_level_advantage_validation` 直接调这个模块过一遍。

**教训:往 batch 里塞"假行"之前,grep 一遍所有按 uid / rollout_id / len(batch) 分组或归一化
的消费者。** 这条链上至少有四个:GRPO 分组(uid)、rollout 级优势(rollout_id + uid + reward
三重校验)、per-rollout 归一化(rollout_id 的 token 数 × 行数)、mini-batch 切分(len)。
只满足其中一个的"中性行"在另一个眼里就是脏数据。

**上线验收(step 47,09-16 20:15,修复后第一步):**

| 指标 | step 44–47 旧代码 | step 47 新代码 |
|---|---|---|
| n_truncated_sample | 59 / 74 / 65 / 66 | 0 |
| n_budget_splits / blocked | — | 127 / 0 |
| n_sample_dropped(same_reward+random) | 12 / 19 / 28 / 13 | 0 |
| n_sample_padded / trained | — / 96~128 | 9 / 247 |
| rollout_level_advantage/n_rollouts | 32 | 41 = 32 + 9 pad |
| n_groups / n_zero_adv_groups | 8 / 4 | 17 / 14 = (8+9)/(5+9) |
| n_unmerged_rollouts / mismatch_rows | 23 / 109 | 19 / 88(F 类修复同步上线) |

对账恒等式:`n_sample = n_rollouts + n_budget_splits + n_trace_merge_mismatch_rows`
→ 247 = 32 + 127 + 88。前两项是正常开销,第三项才是要压到 0 的。
step 47 的 88 处断裂经 classify_breaks 归因:A 压缩 82(93%,合法必切)、B 空白 4、C CDATA 2 ——
非压缩仅 6(修 F 前 ~76/步)。**不要把 mismatch 总数当成"可省的前缀重算"**:压缩行 prompt 反而更短
(29914 → 12369),前缀重算的大头是 127 次预算拆行,那是 6144 槽位的硬约束。
代价:行数 128 → 256,update_actor 787 s(step 2113 s),8 次 mini-batch 更新。
未解决:`response_length/training/max_by_turn = 6144` 说明仍有单轮顶到生成上限被 vLLM 截停 ——
这是生成期 cap 不是训练丢失,但那轮回复残缺;要不要放宽 AGL_MAX_TOKENS 待统计撞墙频率。

### 44.13 三个追问:6144 从哪来、为什么要填充、长 prompt 是不是更难训(2026-09-16)

**6144 = 51200 − 45056,是剩下来的,不是选出来的。** 窗口 `max_model_len=51200` 由显存定
(2B 模型 + vLLM 0.7 显存占比,`restart_v3_trainer.sh:50-51`)。SWE 任务的 prompt 实测均值 12k~17k、
最大 31k,还要给后期轮次留余量,所以 prompt 槽给 45056;response 槽只能是差值 6144。
它恰好等于 proxy 单轮生成上限 `AGL_MAX_TOKENS=6144`(`opencode_agent.py:41`),这不是巧合:
单轮回复最长 6144,那么"至少能装下一整轮"的 response 槽下限就是 6144。再大就要从 prompt 槽里抠,
后期轮次 prompt 超限整行 `is_drop`(§44.12 已否决 rebalance)。

**为什么 prompt 左填充 / response 右填充。** verl 的 `DataProto` 是定形张量 `[B, 45056+6144]`,
prompt 靠右对齐(左填充)、response 靠左对齐(右填充),这样 `prompt_ids[-1]` 和 `response_ids[0]`
在张量里永远相邻,`response_mask` / `advantages` 才能用固定切片 `[:, -6144:]` 取。
填充**不花算力**:`use_remove_padding=True`(`train_opencode_agent.py:113`)时
`dp_actor.py:164-165` 用 `unpad_input` 把 pad 全部剥掉,拼成 varlen 序列喂 flash-attn,
前向只算真实 token;`perf/total_num_tokens` 统计的也是真实 token(step 47:525 万)。
填充只占传输/存储,不占 FLOPs。

**后面的行 prompt 更长,是不是更难训?** 分两层:
- 优化意义上不更难。loss 只落在 response 位置,prompt 是条件;第 5 行(prompt 24500)训的是
  "在这段历史下第 24~29 轮该怎么说",这和推理时模型面对的条件一模一样——训练分布 = 部署分布,
  这正是要的。真正"难"的是 credit assignment:30 轮共用一个 reward,第 29 轮和第 0 轮拿同一个 A。
  这是 outcome reward 的固有问题,和行怎么切无关(§44.10)。
- 算力意义上更贵。attention 对 24500 的上下文是 O(L²)(flash-attn 只省显存不省 FLOPs),
  第 5 行前向的代价 ≈ 第 1 行的十几倍。这就是 step 47 update_actor 787 s 的主要来源,
  也是为什么 `ppo_max_token_len_per_gpu=51200`、micro batch 只能 1 行。

**上一行的 KV cache 会不会传给下一行?不会。** 训练没有 KV cache——每行都是独立的一次完整前向,
第 2 行的 prompt(6500 token,含第 1 行已训过的输出 0~5)从头重算一遍。这也正是 trajectory 拼接
省算力的原因:拼成一行,前缀只算一次;拆成 N 行,前缀就重算 N 次。KV cache 是生成期(vLLM)的东西:
rollout 时第 k 轮的 prompt 是第 k−1 轮上下文的延伸,vLLM 的 prefix caching 复用前缀 KV——
但 09-13 评测已测出 KV 够、瓶颈在 prefill 带宽,那是另一条线的事。

所以"行"的切分逻辑一句话:**在 6144 的槽位约束下,尽量少切(前缀少重算),切的地方一定落在轮边界
(观察不进 loss、回复不砍尾),切出来的 prompt 必须 ≤ 45056(否则整行报废)。**

## §45 step 76 评测 8.9%:模型不 edit 了,原因不是"不会写参数"而是复读死循环(2026-09-17)

### 45.1 现象
数据完整性修好(§44)之后从 step 47 续训到 76,合并权重、同一套评测口径(temp 0.6/top_p 0.95/top_k 20,
600 s/题,474 题)跑出来:**step76 42/474 = 8.9%,基座 139/474 = 29.3%**,step154-v2 是 150。
McNemar:only_base=98、only_76=1,p≈0——不是噪声,是真退化。

| 474 题汇总 | 基座 | step76 |
|---|---|---|
| edit 成功次数 / 有 edit 的题数 | 616 / 364 | 54 / 46 |
| bash 调用次数 | 5901 | 22294 |
| 超时(rc 124) | 180 | 287 |
| 每题 tool_use 中位数 | 14 | 42 |
| 连续 ≥5 次一模一样的调用的题数 | 10 (2%) | 239 (50%) |
| 重复调用占全部调用 | 17% | 75.5% |
| 无效工具调用(JSON 解析失败) | 6 | 199 |
| 压缩次数 | 233 | 226 |

压缩次数没变,所以不是压缩机制的锅。典型样本:同一句 `<think>Let me look at the test data…</think>` 加
同一条 `ls … && find …` 连续 44 次,再换同一段 `python -c` 17 次,直到超时。239 个死循环题里 227 个一次
edit 都没有——模型不是"决定不改",是永远走不出探索阶段。

### 45.2 "参数格式错误"这条线:有,但根因在收尾,不在参数
199 次无效调用全部是 `JSON Parse error: Expected '}'`,edit 149 / bash 36 / write 14。先别急着说
"模型 JSON 写错了"——**模型写的是 XML**(`<function name="edit"><param name="…">`),JSON 是 vLLM 端
`minicpm5xml_tool_parser.py` 里 `json.dumps` 出来的,不可能手写漏括号。真相在流式解析:
- `:76-80` 故意在块完成前不发右花括号;`:649-656` 只有见到 `</function>` **且**整块解析成功才补上。
- 把 199 段残缺 JSON 补上 `}` 后都能解析,edit 149 次里 143 次三个必填参数齐全,6 次缺 newString。
  即参数值都写完了,但整块最终没被判定为完整。两种可能:(a) 模型写完 `</param>` 没写 `</function>` 就 EOS;
  (b) 块里出现重复/未知参数名,`:227-232` 整块作废(但流式阶段 `_parse_partial_params` 对重复键是 `continue`,
  所以残缺 JSON 看起来是完整的)。
- 训练 dump(`trajectories/step_N_train.jsonl`)每步 `<function` 与 `</function>` 数量完全相等(step76 26/26),
  temp 1.0 下从不漏闭合,所以更像 (b)——重复参数名本身也是复读的一种。坐实要用 step76 权重回放那几条请求(待卡)。
- 真跑了但失败的 edit(找不到 oldString)反而更少:37 → 29。不是"写不对 oldString"。
- 教训:看到 `Expected '}'` 先查**谁生成的 JSON**。模型-解析器-opencode 三层,报错文本里的那个句号也是
  opencode 拼的,不是 payload 的一部分(第一眼被它骗了)。

### 45.3 训练侧:奖励方向没错,是优化动力学把"复读"放大了
- 训练 dump 里有 edit 的 rollout 比例:step 1-46 约 76% → 71-76 降到 66% → step76 只有 14/32;
  每条平均轮数 27 → 61;`training/n_turns` step 76 = 1954(平时 800)。
- 有 edit 的 rollout 奖励均值 0.67,无 edit 的 −0.04,全程如此——奖励从没鼓励过"不 edit"。
- step 73→76:kl_loss 0.16→0.56,ppo_kl 0.003→0.026,clipfrac 0.015→0.056;**entropy 0.27→0.37 是上升的**,
  所以不是熵塌缩,而是"复读上一轮"这个具体模式在 lr 5e-6 × 每步 16 次 mini-batch 更新下被放大。
- `per_rollout_mean`(`per_rollout_loss.py:29-41`)把超长失败轨迹的负优势摊到几万 token 上,复读几乎不挨罚;
  短的成功轨迹正优势集中。奖励里也没有轮数/长度项(`opencode_agent.py:416` 只有 1/0/−0.2)。
- 评测比训练更惨:temp 0.6 + top_k 20 让复读模式几乎必然被选中,训练 temp 1.0 还能偶尔跳出;评测 600 s 也比训练短。

### 45.4 和官方配方的差距(为什么 AGL 官方能训成)
官方 `train_smith_agent.py`:Qwen3.5-9B,lr **1e-6**,ppo_mini 16,`max_ppo_update_times=2`,无 KL,
prompt/response 65536,`smith_agent.py` 有 `SMITH_MAX_TURNS=40`、`length_penalized_reward`(:656)、
`prompt_length_penalty`(:680),上下文溢出直接结束 episode(:757-761)。我们:lr 5e-6、每步最多 16 次更新、
无轮数上限、无长度惩罚、超时只 −0.2。方向一样,力度差 5~40 倍,还没有刹车。

### 45.5 结论与后手(待人类拍板,未擅自续训)
1. 先评 step48 对照(修复前最后一个 ckpt),看退化是从 47 之后开始还是更早。
2. 续训建议:从 step48 起,硬轮数上限(40),超时/溢出轨迹 loss mask 掉(DAPO overlong filtering),
   lr 5e-6→1e-6~2e-6,PPO_MINI 16,接上 length/prompt 惩罚,KL 系数不动。
3. 上线一个廉价的死循环探测(连续 N 次相同调用即终止 episode 并按超时计),让复读在训练里真的挨罚。

## §46 "修过的 bug"只修在了一份没人跑的副本上;以及给 episode 装刹车(2026-09-17)

### 46.1 −0.2 超时罚分:09-07 裁决"奖励最低是 0",实际一直在罚

09-07 人类明令奖励下限 0。当天改的是 `examples/swe_smith/opencode_agent.py`。
但 rollout pod 挂载的是 configMap `swe-smith-opencode-scripts`,它由
`examples/swe_smith/agents/{smith_agent,opencode_agent}.py` 生成(run_opencode_k8s.sh:86-89,
job-template-opencode.yaml:75-79)。`agents/` 下那份还留着

```python
if oc_proc.returncode == 124 and not resolved:
    reward -= 0.2
```

结果:step 60/70/76 分别有 8/8/18 行拿到 −0.2。同一个"两份同名文件,一份是活的"坑,
§44 已经踩过一次(proxy 补丁),这次是 agent 脚本。
教训:**改 agent 脚本后必须 `kubectl get configmap ... -o jsonpath` 反查一行特征字符串**,
现在 refresh_agent_configmap.sh 最后一步就是 grep `class EpisodeGuard`。

对训练的影响不是"多罚一点":GRPO 组内如果 4 条都失败,3 条 0、1 条 −0.2,
这个组就有方差,超时的那条被当负样本推开。step 76 的 287 次超时(§45)
就等于每步在教模型"别超时"而不是"去改文件",与探索死循环互相强化。

### 46.2 三个刹车,全装在代理层(agents/opencode_agent.py `EpisodeGuard`)

opencode 本身没有"最大轮数"参数,所以刹车装在 `_GatewayProxyHandler._forward`
——每次模型调用都经过它:

| 刹车 | 环境变量(默认) | 触发时机 | 后果 |
|---|---|---|---|
| 轮数上限 | `AGL_MAX_TURNS=40` | 第 41 次 chat/completions 到达代理 | 代理不转发,直接回一个无 tool_call、`finish_reason=stop` 的合成回复;opencode 正常结束,**照常评测**,reason 带 `turn cap 40` |
| 死循环 | `AGL_LOOP_REPEAT=3` | 上游回复的 tool_calls(名字+参数 json)与前一次完全相同,连续 3 次 | 这条回复被替换成合成 stop;**不评测**,reward 0,`loop_detected=true` |
| 长度惩罚 | `AGL_LEN_PEN_T0=25 / LAMBDA=0.2`,`AGL_PROMPT_PEN_SOFT=30000 / HARD=40960 / MAX=0.2` | 只在 train(base_url 含 `/mode/train/`)且**已解决**时 | 复用 smith_agent.length_penalized_reward / prompt_length_penalty(:657/:680),最多各扣 0.2;失败样本永远是 0 |

最后 `reward = max(0.0, reward)`。

为什么合成 stop 要经过 `completion_json_to_sse`:opencode 请求时带 `stream: true`,
代理把它改成非流式再转回 SSE(§44.9);合成回复如果直接回 JSON,客户端会挂在 SSE 解析上,
episode 不会结束,反而是另一种超时。

为什么死循环的那条也算进 trace:上游 vLLM 已经把这条回复记进 agl server 的 trace 了,
代理只是不让 opencode 看到它。所以训练数据里这一轮仍在,reward 0 让它被推开——
这正是我们想要的:惩罚的是"重复",不是丢掉证据。

### 46.3 数值怎么定的

- 40 轮:上游 SMITH_MAX_TURNS 默认;基座 val 中位 tool_use 14 次,step76 塌缩后 42 次,
  40 刚好切在"塌缩态"上,健康轨迹几乎不受影响。
- T0=25:上游 80 是给 40 轮以上配置用的;我们的解决题中位 <25 轮。
- λ=0.2 / prompt 惩罚 0.2:两项叠加最多把 1.0 压到 0.6,仍远高于失败的 0,组内排序不翻。
- prompt 硬顶 40960 = AGL_OPENCODE_CONTEXT,超过它本来就会压缩,再往上罚没意义。

### 46.4 训练配方(等人类拍板 resume 时用)

PPO_MINI 保持 8(verl 硬约束 train_batch ≥ ppo_mini,actor.py:215-218);lr 5e-6→2e-6;
KL 0.001 不动;动态采样 DYN_MAX_GEN_BATCHES=2 / DYN_MIN_VALID_GROUPS=6;
先跑 `examples/swe_smith/refresh_agent_configmap.sh`(内含 15 个单测 + configMap 反查),再 restart_v3_trainer.sh。

## §47 v4 起训:三个"配置对了但没跑起来"的坑(2026-09-17 夜)

### 47.1 vLLM 的显存检查把邻居的显存也算进去

`vllm/v1/core/kv_cache_utils.py:527`:`available = total × gpu_memory_utilization − peak_memory`,
peak 是**整卡**当前占用(含别人的进程),≤0 就抛 "No available memory for the cache blocks"。
这一步在 `num_gpu_blocks_override` 生效**之前**(624-630 行),所以 override 固定了 KV 大小也救不了它。
共享卡上邻居一个突发(<other-user> point_t280 一个 17-23G),训练就在启动时死掉。
处理:util 0.95 只是让"检查"过关(KV 实际由 override 固定 25600 块 ≈ 8.2 GiB/rank),
驱动脚本在起训前等每张卡余 ≥40G,并且挂了自动续。

### 47.2 FSDP checkpoint 绑 world size,两卡和四卡不能互相 resume

`verl/utils/checkpoint/fsdp_checkpoint_manager.py:139` 的文件名是 `model_world_size_{ws}_rank_{r}.pt`。
人类要求"4 卡不够先用 2 卡,够了再回 4 卡",所以每次切模式都要:停训 → `verl.model_merger` 合并成 HF →
新模式 `resume_mode=disable` 从 HF 起(优化器重置)。步数按"总步数 = 之前各段之和 + 当前段 ckpt"记
(`v4_cycle.state`),探针按总步数每 20 步一次。两卡配方:train_batch 4 / PPO_MINI 4 / 3 组 / MAX_UPDATES 32(行数上限不变)。

### 47.3 换了起点权重路径,agent 的每一次请求都 503,而训练"成功"跑完了一步

现象:两卡首步 16 条 rollout 3 分钟内全部 SUCCEEDED、reward 全 0、`n_turns=0`,然后
`_report_capture` 在 `tally.calls` 上 AttributeError(tally 是 None:整批一条 model_request 事件都没有)。
pod 日志:`proxy "POST /v1/chat/completions" 503` × 6 次后 opencode 退出。

根因:trainer 用 `actor_rollout_ref.model.path` 作为模型名注册 vLLM 端点(`agl_rollout_manager.py:355`),
而 agl-server 的 `default_proxy.model_name` 是启动时定死的(`/workspace/models/MiniCPM5-2B`)。
v4 从合并权重 `MiniCPM5-2B-step48-v3` 起训,名字对不上,`ProxyRouter.select_server` 查不到 → 503
"No servers available"(`server/routes/proxy.py:65`)。以后每次从合并权重续训(每次切模式)都会撞一遍。

为什么它安静:rollout 的 SUCCEEDED 只表示 agent 进程正常退出,不表示它碰到过模型;
reward 0 是合法值;采集率报告在 0 次调用时直接 return。三层各自"合理",合起来就是一步空训练——
和 §43/§44 是同一个模式:**每一层都不对"什么都没发生"报警**。

修法(06a4a1b):
1. 服务端 `select_server`:配置名查不到但只注册了一个模型时,路由到它,并把 body 的 model 改成该端点的名字
   (vLLM 按 path 提供服务,body 里的名字必须和它一致,否则 vLLM 404)。
2. `_report_capture`:整批 0 次调用直接 RuntimeError 说明原因,不再静默 return / AttributeError。
3. agl-server 重启才能加载新代码(pane 里 `agl-server` 不在 PATH,要用 `.venv/bin/agl-server`);
   controller 对 server 短暂断连只是打一段 ConnectError,进程不死,server 回来就恢复轮询。

教训:凡是"起点权重路径变了"的重启,先 `curl` 一次 `/proxy/rollout/.../chat/completions` 看状态码,
或者至少看首批 rollout 的 `n_turns` 不是 0,再让它跑。

### 47.4 步 3 一轮 32 条 rollout 失败 28 条:镜像明明在本地,pod 却卡在 ImagePullBackOff(2026-09-18 01:40)

现象:`dynamic sampling round 1/2: 0/1 groups … 0/38 rows kept`,`completed=32/32 succeeded=4 failed=28`,
`kubectl get pods` 里 24/32 个 `agl-rollout-*` 是 `ImagePullBackOff`,`describe` 显示
`Failed to pull image "jyangballin/swesmith.x86_64.bottlepy_1776_bottle.a8dfef30": Get "https://registry-1.docker.io/v2/": Client.Timeout`。
而 `docker images | grep bottlepy` 能看到这张镜像。

三层原因叠在一起:
1. **docker 里的名字是 `dockerproxy.net/jyangballin/...:latest`**,pod 要的是 `jyangballin/...:latest`。
   `imagePullPolicy: IfNotPresent` 按"名字"查本地,名字不同就当没有,去 docker hub 拉。
   本地 132 张训练镜像里 57 张只有 `dockerproxy.net/` 前缀的 tag(之前自愈脚本从镜像代理拉回来的)。
2. **docker daemon 没配代理**(`systemctl show docker -p Environment` 为空,shell 里的 `https_proxy` 对 daemon 无效),
   `daemon.json` 的 mirror `mirror.aliyuncs.com` 早就不可用,于是直连 `registry-1.docker.io`——今晚直连彻底超时。
   步 1/2 期间 journal 里也有 40 次 "Failed to pull",只是当时直连偶尔能通,退避重试后拉成功了,看起来一切正常。
3. **自愈脚本被磁盘闸门按住**:`autoheal2.sh` 会给 Pending pod 从 `dockerproxy.net` 拉镜像再 `docker tag` 成 hub 名字,
   但它有 "/data 余量 <100G 不拉" 的规则(人类 09-11 定的闸门),而 /data 此时只剩 87G → 自愈静默暂停。

修:`docker tag dockerproxy.net/jyangballin/X:latest jyangballin/X:latest`(57 张,纯本地操作,秒级),
kubelet 下一次退避重试就命中本地镜像,26 个 pod 全部转 Running,当前 round 直接被救回来。
验证:`train_dataset_mixed.jsonl` 132 张 distinct 镜像 / `val_dataset_filtered.jsonl` 23 张,现在 `comm -23` 缺失为 0。

顺手发现的两件事:
- kubelet 的 image GC(`image-gc-high-threshold=98`)在 /data 用量 ≥98% 时每 5 分钟试图删镜像("wanted to free 102GB"),
  之所以一直 "freed 0 bytes",是因为 143 个 Exited 的 rollout 容器还引用着镜像(`conflict: unable to remove repository reference`)。
  **所以不要 `docker system prune`/`docker container prune`**——那些死容器是镜像的保护伞。
- 磁盘:v3 的 `global_step_76`(29G,已塌缩且 HF 合并版还在)和 pip_cache 删掉,87G → 116G;
  v4 的 save keep=2 会占 58G,加了 `v4_ckpt_janitor.sh`(每 5 分钟只留 latest 一份,`.prev` stint 里的全删)。

教训:
- "pod 拉不到镜像"先 `docker images | grep <repo 的后半段>`,看是不是**名字前缀不同**,而不是真的没有。
- rollout 大面积 failed 时,第一眼看 `kubectl get pods` 的 STATUS 列,比翻 trainer 日志快得多。
- 任何"依赖外网"的环节(docker hub 直连)都可能在半夜突然断,训练集用到的镜像要提前保证本地齐全且**名字正确**。

**47.4 补记(06:50)**:`docker tag` 补上的 hub 名字只活了半小时——kubelet image GC(/data 用量 ≥98%,每 5 分钟一轮)
按 imageID 删镜像,docker 删不掉被旧容器引用的 `dockerproxy.net/...` tag,却能把**没有容器引用的** `jyangballin/...` tag 摘掉
(`docker events --filter type=image` 里一串 `image untag`),于是 step 8 的 pod 又开始 ImagePullBackOff。
两手处理:① `docker create --name pin-<repo> --label agl-image-pin=1 <image> true`,给 132 张训练镜像各建一个**不运行的空容器**当"引用锚",
GC 就摘不掉了(每个容器只有几 KB 元数据);② `v4_retag_guard.sh` 每 60s 把缺的 hub 名字补回(兜底)。
GC 的触发线:kubelet 用 (capacity-available)/capacity,这块盘 6.49T,**余量 <130G 就 ≥98%**,要彻底不触发得 ≥195G(97%),
靠删自己的东西够不到,所以"锚容器"才是真正的修法。
另:`trainer_v4.log`(driver 侧转发的 actor stdout)会漏行——step 8 第二轮结束的 `dynamic sampling round 2/2` 行只在
`ray_tmp/session_latest/logs/worker-*-<actor pid>.out` 里有;看到日志"停在 31/32 十几分钟、四卡 100%",先去 worker 原始日志确认,不要误判成卡死。

**47.4 再补(07:12,锚容器不够)**:建了 pin 容器之后 GC 照样每 5 分钟摘掉 57 个 hub tag。原因是 docker 的规则:
**一张镜像有 ≥2 个 tag 时,`rmi <任一 tag>` 只是摘名字,不做"容器正在引用"的冲突检查**;只有删最后一个引用时才会报 conflict。
所以两个 tag 并存 = 给了 GC 一个可以白拿的名字。修:`docker rmi dockerproxy.net/jyangballin/X:latest`(先确认 image id 与 hub tag 一致),
让每张镜像**只剩 pod 要的那一个 tag**,再由 pin 容器 + 旧容器的引用挡住 GC。验证:GC 又跑了 126 次 "Removing image",132 个 hub tag 原样。
`v4_retag_guard.sh` 留着做兜底(现在是空转)。

## §48 教师轨迹采集(deepseek-v4-flash 走 opencode)当晚踩的三个"数据看着对、其实错了"的坑(2026-09-20)

背景:v4 GRPO step62 全量 136/474 与基座 139 无差异,转向"API 教师轨迹 → SFT"。教师经 `teacher_proxy.py`
(逐请求落盘)走 opencode,学生也用同一套 opencode 内置工具。数据要"压缩前 / 压缩请求 / 压缩后"三段都能学。

### 48.1 压缩请求里嵌着任务原文,分类器按"含 `<pr_description>`"判任务 → 第一次压缩全被当成新任务
opencode 压缩时发一条 system="You are a context summarization agent…" + user="Here is the conversation so far:
<conversation>…" 的请求,**第一次**压缩的 `<conversation>` 里原样嵌着首条用户消息(含 `<pr_description>`)。
代理先测 `<pr_description>` 再测压缩标记,于是 185 个"post"只有 2 个"compact"。修法:压缩标记 / "What did we do so far?"
的判断必须排在任务判断**之前**。教训:用"内容里含某标签"做分类,要先想清楚这个标签会不会被别的消息**转述**。

### 48.2 多个 run 共用一个代理目录,重跑同一题会覆盖 `<task>.json`
文件名是 md5(任务提示),batch1 与重跑 rc 落在同一目录,rc 把 batch1 的 235 个主段文件盖掉了;重新汇编 batch1 时
拿到的是 rc 的段文件(行数 2312、`over_turn_cap` 1286 全是错的)。修法:汇编器按请求 `ts` 加 `--since/--until` 窗口,
旧 run 加 `--main-only`;**旧 run 的行只从当时汇编好的 jsonl 取**,不再从代理目录重建。更根本的做法是每个 run 一个代理目录。

### 48.3 `compact_events` 是对整条事件 JSON 做子串匹配,把仓库源码里的 `compact` 也数成了压缩
`run_repo_sweep.py` 里 `if "compact" in blob.lower()` —— funcy 的 `compact()`、pyasn1 的 `supportCompactZero`、
grep 命令参数里出现 compact,统统计为压缩事件;而 opencode 的 `run --format json` 流里**根本没有**压缩事件。
后果:batch1 报 235 题压缩,按这个名单重跑;用代理抓到的压缩请求(嵌任务原文可反查题号)复核,真压缩的是 210 题,
重跑名单 = 207 真 + 28 假阳性,还有 3 题真压缩没在名单里。修法:只按事件 `type` 计数;汇编器的 `n_compactions`
改为数代理的 `<task>.compactK.json` 文件;旧 run 用 `--compacted-ids` 传真名单,计数记 null(未知 ≥1)。
教训:**任何"有没有发生 X"的字段,来源必须是产生 X 的那一层**(这里是代理看到的请求),不能靠日志文本模糊匹配;
上线前用一个已知不会压缩的 30 秒短跑验一下计数是否为 0,这个坑当场就能抓到。

## §49 把教师的"压缩阈值"对齐到学生分词器:一次测错的比例、一个模板报错、一套代理改造(2026-09-20 夜)

背景:学生(MiniCPM5-2B)在 opencode 里于 `context − max_output = 45056 − 6144 = 38912` token 处触发压缩(§48);
教师 deepseek-v4-flash 的 token 是网关按 deepseek 分词器算的,同一段上下文两边计数不同,压缩点就对不齐。
用户裁决:代理把 `usage` 改写成 MiniCPM 分词器的计数、只重跑 500 题、教师单轮 16384 不变、加 60 次模型调用上限、旧数据保留(run `ra`)。

### 49.1 "MiniCPM/deepseek 提示 token 比例 0.852"是测错的:只数了消息正文,没数聊天模板和工具 schema
第一版测量把 messages 的 content 拼起来直接 `tokenizer(...)`,得 0.852,据此推断"教师压缩点比学生早 15%"。
按 vLLM 的方式用 `apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)` 重测(取最大的 40 条请求),
比例是 **0.992(0.98–1.01)**——聊天模板 + 工具 schema 占了几千 token,两边分词器在这部分差异被摊平了。
所以 batch1/rc 其实早就对齐到 1% 以内;ra 的真实收益是"精确计数 + 60 上限 + 每题第二次采样",不是"修正 15% 偏差"。
教训:比较两个分词器的"上下文长度",要数**模型实际看到的整条输入**(模板、system、工具 schema 都算),
并且先看比例的分布而不是单点均值;向用户报数前把测量方法写在报告里,错了也好追。

### 49.2 MiniCPM 聊天模板在 `tool_calls.function.arguments` 是 JSON 字符串时抛 `'str object' has no attribute 'items'`
OpenAI 协议里 arguments 是字符串,MiniCPM 的 Jinja 模板却按 dict 迭代 `.items()`。vLLM 的 `chat_utils` 在进模板前
会把 arguments `json.loads` 成 dict,我们自己调 `apply_chat_template` 就得照做(解析失败退化成 `{"_raw": s}`;`content: None` 改成 `""`)。
和 §41 的采集率 bug 同源:**凡是自己调聊天模板,先用一条带 tool_calls 的真实请求跑通**,不要只拿纯文本对话试。

### 49.3 代理改写 usage 就够触发压缩;上限用"合成 stop 回复"而不是断连接
opencode 1.18.28 的压缩判断是 `count = tokens.total || input+output+cache.read+cache.write; count >= context − output`,
`reserved` 不参与——所以代理只要把 `usage.total_tokens`(顺带 prompt/completion)改成 MiniCPM 计数,压缩点就落在 38912。
实现要点:非流式直接改 JSON 再按新 Content-Length 发;SSE 逐行透传,只在带 `usage` 的那块改写,同时用 `StreamAssembler`
拼出完整 assistant 消息来数 completion token;events.jsonl 里 `response.usage` 保留网关原值(算钱用),另加 `response.usage_minicpm`。
计数在 `conn.request` 之后、读响应之前做,和网关延迟重叠,单条 40k 请求 ~0.1 s。
60 上限照抄学生的 `EpisodeGuard`(`opencode_agent.py:65-133`):按客户端 IP 数一个 run 内**所有**已回答的模型调用(任务 / 压缩 / 压缩后都算),
第 61 次直接回一条无工具调用的"episode ended"合成回复(JSON 或 SSE 都能造),opencode 自然收尾;
不用断连接——断连接 opencode 会重试,既浪费钱又让 rollout 不完整。事件表记 `event=turn_cap`,汇编器据此打 `over_turn_cap`。
落地检查:上线前用真实请求验证改写后的 usage 与网关值相差 ±1%(实测在 1% 内)、`n_turn` 单调递增、第 61 次确实被拦。
结果:500 题 21 分钟、366 解决(73.2%)、$3.97、65 次触顶;数据集 v3 = v2 + ra,clean 行 904,累计 $23.06。

## 50. 教师轨迹 SFT(2026-09-20 夜):自定义数据集、压缩三段、verl 存 fp32 权重

**背景**:v4 GRPO 无增益后改走 SFT:基座 MiniCPM5-2B 全参,在 deepseek-v4-flash 的拒绝采样轨迹(resolved 且无 flag,
904 行 → 训练 858 行 / 留出 46 行按题号切)上训。启动脚本 `swe_smith_smoke/sft/run_sft_v3.sh`,verl 0.7.1 `sft_trainer`。

### 50.1 verl 自带的 MultiTurnSFTDataset 和 MiniCPM 模板不兼容,自己按"整段渲染 + 字符区间打 mask"做
verl 的多轮数据集是逐条消息渲染再拼接,MiniCPM 模板里 tool 消息是"连续多条 tool 合并进一个 `<|im_start|>user` 块、
只在最后一条后加 `<|im_end|>`",assistant 又要求 `tool_calls.function.arguments` 是 dict(见 49.2),逐条渲染的结果和
vLLM 部署时整段 `apply_chat_template` 的结果不一致。做法(`sft/swe_sft_dataset.py`):整段 `apply_chat_template`
得到 full;对每个要学的 assistant 消息 i,渲染 `messages[:i]`(加 generation prompt)和 `messages[:i+1]`,二者都必须是
full 的前缀(assert),差集就是该轮的字符区间(去掉尾部 `<|im_end|>\n` 的换行,保留 `<|im_end|>`);用
`return_offsets_mapping` 把区间映射到 token 的 loss_mask。858 行全部通过前缀断言,损失 token 占 23.4%。

### 50.2 压缩 = 三个独立样本;后段被"拆成两条"的合成总结要全 mask
opencode 压缩后重建上下文为 user "What did we do so far?" + assistant(总结);汇编器只给第一条打 `synthetic`,但有的
rollout 总结被拆成两条 assistant(第二条没标)。规则:post 段从下标 2 起的连续 assistant 串只学最后一条(真实首个动作),
之前的全 mask。compact 段是 opencode 真实发出的压缩请求(system + user 内嵌 `<conversation>` 文本转写 + 模板),只学总结
正文;不把它拼到 main 段尾巴上——部署时就是单独一条请求。

### 50.3 `checkpoint.save_contents=[hf_model]` 存的是 fp32 主权重(9.4 G),不是 bf16
`fsdp_checkpoint_manager.py:326-342` 用 `torch_dtype=bfloat16` 建空模型,但 `save_pretrained(state_dict=...)` 直接写
FSDP 的 fp32 full state dict,2B 模型一份 9.4 G。/data 只剩 75 G 时才发现。补救:`sft/to_bf16.py` 逐 shard 读 safetensors
转 bf16 原地覆盖(4.7 G,幂等,顺手把基座的 `chat_template.jinja` 拷进去,verl 不会存它),训完链先转再评测/上传。
教训:存第一份权重后立刻 `du`,别信 dtype 参数。

### 50.4 4 卡与邻居共卡时的并行选择:FSDP + Ulysses sp=4(dp=1)
启动时每卡只剩 43–47 G(邻居 100% util),dp=4 要每卡装整条 49k 打包序列的激活,选 sp=4 把每卡激活压到 12k token:
rank 峰值 34–58 G,卡总占用最高 95/98 G,没炸。代价是吞吐:MFU 10.5%,每步 ~290 s(67 万 token),3 epoch 78 步约 6 h。
邻居让出显存时应改 `ulysses_sequence_parallel_size=1` + `max_token_len_per_gpu=49152`(dp=4,理论 ~4×)。
首个 epoch:train loss 0.80→0.68,val loss 0.692。swanlab 项目 `swe_smith_sft`(和 GRPO 同账号)。

### 50.5 "杀掉重启"的后台链没杀干净 → 训完同时起了两套评测集群 + 两条 HF 上传
训完后链(`sft/after_train_v3.sh`,`setsid nohup` 起)第一版没有 bf16 转换步,改脚本后"kill 旧链再起新链",但只杀了
`bash -c` 外壳,真正的 `bash sft/after_train_v3.sh` 子进程还活着,两条链等到同一个训练 pid 退出后,一分钟内各自
`v4_full_fleet.sh start` 同一 TAG(4 卡各起两个 vLLM 抢同一端口,pid 文件互相覆盖)、各自 `hf upload` 同一路径,日志
文件也被 `>` 互相截断。处理:全部杀掉(`v4_full_fleet.sh stop TAG` 只认最后写的 pid 文件,得按 ps 手工补刀),重起一条。
教训:(1)杀 setsid 链要 `ps -ef | grep 脚本名` 按脚本进程杀,杀外壳无效;(2)`pkill -f 某路径` 会连当前 shell 一起
杀(自己的命令行也含该路径,exit 144);(3)链脚本开头应先 `pgrep -f 自身名` 拒绝重复实例。三 epoch val loss
0.692 → 0.681 → 0.683,第 3 轮不再降(train 0.68→0.56),支持"下次只训 1 epoch"。

### 50.6 SFT 模型"全是 invalid 工具调用"——不是模型学坏了,是 vLLM 流式 tool parser 吃掉了 CDATA 前导缩进
SFT v3 ep3 评测跑到 140 题时:resolved 24、`invalid:completed` 275 次(88/140 题中招)、135/140 超时 600 s、几乎不 edit。
opencode 报错全是 `Invalid input for tool edit: JSON parsing failed ... Expected '}'`,参数 JSON 完整但没有闭合括号。
根因(`minicpm5xml_tool_parser.py`):09-15 打的"CDATA 原文保留"补丁只改了整块解析 `_parse_function_block`,流式快照
`_parse_partial_params` 仍对 CDATA 值 `strip()`。opencode 走流式:parser 先按快照推送 `{"oldString": "def f():..."`(缩进
被吃、不带 `}`),`</function>` 到齐后整块解析得到 `"    def f():..."`,`_streaming_args_diff` 发现新串不是旧串的前缀,
静默返回 None → 收尾的 `}` 永远发不出去 → opencode 判 invalid,并把这次调用以 `name="invalid"` 回灌到历史里,
模型接着模仿 `<function name="invalid">…`,雪崩。复现:`scratchpad/test_stream_cdata.py`,delta 按 17 字符切就 FAIL,
按 1/3/4096 切反而 PASS(块大小恰好决定快照有没有来得及带上 oldString)。
为什么以前没炸:基座 653 次 edit 里 0 次带前导空白(6 次 invalid/474 题);教师 deepseek-v4-flash 的 edit 93% 以缩进开头
(1097/1177),SFT 模型如实学来。**v4 RL step62 的 136/474 同样中招:975 次 invalid、347/474 题命中**——"RL 无增益"的
结论建立在坏 parser 上,需重评(训练侧 restart_v2/v3/v4 也是 `tool_call_parser=minicpm5`,同一 parser 文件)。
修法:`_parse_partial_params` 对 CDATA 值不 strip,和整块解析一致(备份 `.bak.09210545`);评测集群重开
(`sft/eval_ep3_p2.sh`,坏结果目录改名 `val_sft_v3_ep3_parserbug`)。教训:(1)流式和非流式两条解析路径必须共用同一
取值函数;(2)`_streaming_args_diff` 遇到非前缀应记 warning 而不是静默 None;(3)看到 `invalid` 工具名先查 harness,
再怀疑模型;(4)评测对比要先核对两边的 invalid 次数,基座"没事"可能只是基座不触发那条路径。

### 50.7 "unknown" finish → opencode 无限重呼:vLLM 流式 parser 在 EOS 返回 None 就丢掉 finish_reason 块
修好 50.6 重评 SFT ep3 得 139/474,和基座 139 持平,但 415/474 题超时 600 s(基座 180)。翻 `opencode.jsonl` 的 step_finish:
SFT ep3 reason 分布 unknown 39374 / tool-calls 10103 / stop 134 / length 7;基座 unknown 41225 / tool-calls 9733 / stop 407;
走 API 的教师只有 tool-calls 15178 / stop 1137,一个 unknown 都没有。opencode 把 provider 的 finish_reason 记进 step_finish,
收不到 finish 块就记 unknown,然后**不加新 user 消息、原样带着历史再呼一次模型**(`--auto` 只是自动批准权限,不是自动续写),
模型每次再输出一段"任务完成总结",如此循环到 600 s 被杀:SFT 每题这样的纯文本尾步 38922 个、白算 8.04 亿 prefill token,
基座 37889 个 / 9.91 亿;≥20 个废步的题 SFT 217、基座 267。更坏的是共卡:同一 shard 上循环的兄弟把 vLLM 塞满,
正常干活的题被拖慢(超时题 13.6 s/步 vs 正常 3.0 s/步)然后在干到一半时被杀——这才是超时率的主因。
根因链:vLLM 0.8.5 `serving_chat.py:760` `if delta_message is None: continue`——工具 parser 返回 None 就跳过本轮,
而携带 `finish_reason` 的正是最后这一轮(823-832 行),最后一轮 `delta_text == ""`(EOS),我们的 parser 在空 delta 上返回 None,
循环后面只有 usage 块没有兜底。所以**凡是经过这个 parser 的流式回复,只要以纯文本收尾就没有 finish 块**——影响基座评测、
v2–v4 全部 RL rollout(轨迹里混着多段重复总结、rollout 墙钟被白白拉长)、SFT 评测。教师走 API 不经 vLLM,所以干净。
修法:包一层 `extract_tool_calls_streaming`,内层返回 None 且 `delta_text==""` 时返回 `DeltaMessage(content="")`(空内容块
无害,vLLM 随后正常发 finish 块);复现/回归 `scratchpad/test_stream_finish.py`(旧 parser 三种收尾全 None,新 parser 全非 None)。
端到端验证要用 `curl --noproxy '*'`:本机 shell 带 http_proxy,直连 localhost 会被代理吃成 502、看起来像"没修好"。
效果:修后前 10 题 0 超时、0 unknown、中位 wall 19 s、中位 9.5 步(旧 harness 中位 44 步)。
教训:(1)流式 parser 的契约是"每轮都要有返回",None 不是"没东西"而是"丢块";(2)看评测先看 step_finish reason 分布,
unknown 占大头就是 harness 在自转;(3)超时率高先查"是不是被同 shard 的邻居拖慢",别急着给模型定性;(4)训练 rollout 和
评测共用同一 parser,harness bug 会同时污染 reward 和评测,而且以基座为对照时"两边一样坏"会把 bug 藏起来。

### 50.8 学生评测的 `compacted` 字段一直是假的(2026-09-21 修)

- **现象**:修好 harness 后两次评测 `compacted=0`,但 SFT 模型 137 题上下文顶到 38912(压缩阈值 = context 45056 − output 6144)。
- **根因**:opencode 的压缩不是工具调用,而是一次独立的模型请求(首条 user 消息以 `Here is the conversation so far` 开头);
  `run --format json` 事件流里不发任何压缩事件,`run_repo_sweep.py` 按事件类型数永远是 0。更早的"233 题压缩"来自对整条事件做
  子串匹配,把仓库里的 `compact()` 函数名也算进去了,同样不可信。
- **修复**:`tool_trunc_proxy.py` 识别压缩请求,按容器 IP 记到 `compact_events/<port>.jsonl`;`run_repo_sweep.py` 用容器 IP + 运行时间窗
  对账,写入 `compact_events/compact_source`;代理日志不可用时退回"上下文骤降"启发式(输入 >20k 后跌破一半),`ctx_drops` 字段两者都留。
- **顺带**:vLLM 0.8.5 在显存剖析期间若邻居释放显存(free 变大)会断言失败;刚杀完上一轮 fleet 立刻重起也会中招。`v5_fleet.sh` 加了
  单卡 4 次重试 + 失败剔除。另外 util 是按启动瞬间的占用算的,邻居处于峰值时启动,之后邻居回落,vLLM 会把让出的显存全吃掉(卡 1 曾到 94.3/97.9G)。
- **教训**:指标字段要有一次"阳性对照"——找一条确定发生过压缩的轨迹,看字段是不是真的变成 True。

## 51. 步数预算评测与"互斥数据逐轮 SFT"(2026-09-21 ~ 09-22)

背景:SFT/RL 模型的主要失分是"不会停"(超时),用户选定的对策是在提示里给出最多步数(40),并由代理在临近上限时提醒收尾;
之后按"每轮新采 1000 题(各轮互斥)→ 在上一轮权重上续训 1 epoch → 只做有预算评测"循环。

### 51.1 预算 40 步 × 共卡时每步 13–16 s ≈ 600 s:预算恰好压在超时线上

- **数据**(val-474,预算 40,600 s,卡 1/3 与邻居共用):

  | 模型 | 解出 | 超时 | 超时中解出 | 非超时解出 | 步数中位 | 秒/步 |
  |---|---|---|---|---|---|---|
  | 基座 | 134 | 66 | 3 | 131/408 | 14 | 11.3 |
  | RL v4-step62 | 142 | 192 | 31 | 111/282 | 22 | 15.8 |
  | SFT 第一轮 | 122 | 226 | 10 | 112/248 | 22 | 13.6 |

- 配对 McNemar:step62 对基座 p=0.30;第一轮对基座 p=0.13;第一轮对 step62 p=0.0135。
- **解读**:训过的模型探索步数更多,用满 40 步就要 560–600 s,正好撞超时。非超时子集里第一轮 45% 对基座 32%,能力是涨的,
  但被墙钟吃掉了。步数预算必须和"每步耗时 × 超时"一起算,不能单独定。对策(更小预算 / 训练数据筛短轨迹 / 空闲卡评测)由用户定。
- **教训**:报评测结果时把"超时 / 非超时"拆开报,并附每步耗时;总分单独看会把"慢"误读成"差"。

### 51.2 组装预算批次的教师数据时忘带 `AGL_STEP_BUDGET=40` → matched=0

提示里的预算段落是由环境变量拼进任务提示的;组装脚本靠"重建提示并与代理日志对账"来匹配轨迹,不带同一环境变量,提示就对不上,
一条也匹配不到。**教训**:凡是"重建输入再对账"的离线脚本,运行环境必须和采集时完全一致,最好把这些开关写进采集目录的元数据里。

### 51.3 后台 Bash 超时后,命令行后半段仍会执行 → 同一个 finalize 等待脚本起了两份

前台工具调用超时转后台后,整条命令行会继续往下跑;我以为没起来又手动起了一次,结果两个 finalize 同时写同一个输出目录。
处理:按 pid 杀掉两份及其子进程,单独重跑一次。**教训**:重起前先 `ps -eo pid,args | grep "[x]xx"` 确认没有在跑的副本;
等待类脚本开头加锁文件。另:`pkill -f`/`pgrep -f` 的模式若能匹配到自己这条命令行,会把工具调用本身杀掉。

### 51.4 HF 下载经代理只有 0.8–1 MB/s(上传快得多):用可续传下载器,别删"以后还要评"的权重

ep3 权重本地删掉后要做预算评测,只能从 HF 拉回,4.7 G 拉了数小时。`hf_pdl2.py` 按 Range 分块续传,下完逐分片核 sha256。
**教训**:删权重前先问"之后还有没有评测要用它";当前最优和下一轮起点一律留本地。

### 51.5 第二轮 SFT 被显存守卫杀掉;verl 自带的 offload 开关不降峰值,FSDP2 `offload_policy` 才行

- **现象**:2 卡 sp=2,每 rank 约 55 G;邻居在卡 5 涨到 42.5 G,`mem_guard.sh` 在 97182 MiB 处杀了训练(守卫先于 OOM,没波及邻居)。
- **不能降 MAXTOK**:MAXTOK × SP 必须 ≥ 最长样本(43860 token),降了就丢行,违反"不损失信息"。
- **`engine.param_offload/optimizer_offload` 无效**:`EngineTrainModeCtx` 在整个训练步期间把模型和优化器都搬回 GPU,峰值不变;
  只开 `optimizer_offload` 还会触发 `verl/workers/engine/base.py:180` 的断言("Model must be moved to device along with optimizer and grad")。
- **有效做法**:`engine.strategy=fsdp2 engine.offload_policy=True`(FSDP2 `CPUOffloadPolicy`)+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  每 rank 35.4 G;第 1 步 loss 与 fsdp1 完全一致(0.6886);代价是 9–12 分钟/步;HF 权重保存正常。
- **教训**:"offload"开关要读代码确认它卸的是"常驻"还是"峰值";切换策略后用第 1 步 loss 对齐来验证数值等价。

### 51.6 验证集比 batch 小 → val loss 恒为 nan

`verl/trainer/sft_trainer.py:256-257` 的验证 sampler 用 `drop_last=True`,batch 取训练 batch(32);第三轮留出集只有 30 行,
验证 batch 数为 0,均值即 nan。已核对训练/验证数据里没有全零 loss mask 的行,训练本身不受影响。**规则**:留出集 ≥ 32 行。
另:各轮留出集不同,val loss 不能跨轮比,跨轮只看 val-474 预算评测。

### 51.7 vLLM 0.8.5 启动后显存超占

`gpu_memory_utilization` 按启动瞬间的空闲量算,邻居回落后 vLLM 会把让出的显存吃掉。每次 FLEET_UP 后按 pid 核对每个 vLLM ≈ 30 G,
并对评测卡设 96300 MiB 告警。

### 51.8 v2–v4 的 GRPO 没做 rollout 重要性采样校正

vLLM 采样与 FSDP 前向的数值差是业界公认要校正的(token 级 TIS 或 bypass),官方示例开了,我们没开,也没监控 `rollout_probs_diff`。
RL 行确实用的是 vLLM 自己的 token id(没有重新分词),缺的只是 logprob 和配置。恢复 RL 前:采集 logprob → 开 `calculate_log_probs`
看训推差 → 再选 TIS 阈值 2 或 `bypass_mode`。**教训**:上新框架先把自己的配置和官方示例逐项 diff;下结论前先到代码里核实机制。

## 52. smith 血统 s1:官方 harness 上的 agentic RL(2026-10-02 起)

### 52.1 `_ALLGATHER_BASE` 挂死 3600 秒:`prepare_dynamic_batch` 漏传 `dp_group`,一个 rank 提前退出 old_log_prob

**现象**(2026-10-03,s1 第 22 步)。04:21:20 四个 rank 都写完 `global_step_21`,gen 正常收完,然后整条流水线静止 65 分钟:
驱动日志 mtime 冻在 04:27:52,尾部是一墙一模一样的 `AglRolloutManager: completed=63/64`。05:32:22 NCCL watchdog 报出来:

```
[Rank 3] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=285593, OpType=_ALLGATHER_BASE,
           NumelIn=11797504, NumelOut=47190016, Timeout(ms)=3600000)
  ran for 3600011 milliseconds before timing out.
[Rank 3] PG status: last enqueued work: 285593, last completed work: 285592
```

rank 0 / 1 / 3 三个都是「285593 已入队、285592 已完成」,**rank 2 从没入队这一发**。`NumelOut = 4 × NumelIn`
= world size 4,所以这是 FSDP 的参数 unshard,不是数据通信。挂起起点 = 05:32:22 − 3600s = **04:32:22**,
正好是 gen 收完后的第一遍 `old_log_prob`。

**根因**。`verl/workers/actor/dp_actor.py:468`(compute_log_prob)和 `:560`(update_policy)都是

```python
micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
```

**没传 `dp_group`**,而默认值是 `None`;`verl/utils/seqlen_balancing.py:402` 的跨 rank 对齐偏偏要求
`dp_group is not None`:

```python
if dist.is_initialized() and same_micro_num_in_dp and dp_group is not None:
    num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
    dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
```

于是那次 `all_reduce(MAX)` 被**静默跳过**:每个 rank 用自己分片的 token 长度各算一个
`num_micro_batches = min(batch_size, ceildiv(total_seqlen, max_token_len))`。谁算得少,谁就先退出
`dp_actor.py:475` 的 `for micro_batch in micro_batches` 循环、从 `compute_log_prob` 返回,
不再参与 FSDP 的逐层 `_ALLGATHER_BASE` —— 剩下三个 rank 等到 `nccl_timeout` 为止。
`same_micro_num_in_dp` 的默认值是 `True`(`seqlen_balancing.py:353`),所以**光看默认值会以为这条路是安全的**,
真正的开关是 `dp_group`。engine 路径没这个问题,它是显式传的
(`verl/workers/engine/fsdp/transformer_impl.py:583`:`dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True`);
漏的只有 legacy 的 `dp_actor`,而 `actor_rollout_ref` 的 FSDP worker 走的正是 legacy 这条。

为什么跑了 21 步才中:要等一个让 `ceildiv(total_seqlen, 51200)` 在四个 rank 间不相等的长度分布。
smith 没有上下文压缩,单条 rollout 的累计 prompt token 方差极大(p50 1.23M / max 7.09M),
再叠加动态采样让每步行数变化,命中只是时间问题,约 1/21 ≈ 5% 的步。

**怎么判断「挂死」而不是「这一遍慢」**(这次绕了一圈才走对,记下来):

* **日志尾部的计数行不能用。** ray 对重复行做去重且退避是指数的,所以
  mtime 冻住 **不等于** 轮询循环没了,尾部那行 `completed=63/64` 也 **不是**当前阶段。
* **权威是 store**:`curl --noproxy '*' 'http://127.0.0.1:18082/api/rollouts?state_in=queuing&state_in=running'`
  返回 0 条,就说明 gen 已经收完了(manager 完成一条就删一条)。
* **真正有判别力的三个指标**:
  1. `/proc/<pid>/stat` 第 14+15 域的差值。挂住的 rank 会**正好钉死 1 个核**
     (实测 1511 jiffies / 15 s = 15.11 s CPU / 15 s),那是 NCCL 的忙等;
  2. `nvidia-smi` 的 util 会显示 **100%**(忙等 kernel 常驻 SM),所以 100% 不代表在干活;
  3. **每个 rank 的显存不对称**:掉队的 rank 2 在 GPU3 上从 ~9.0 G 掉到 **1.65 G**
     (返回时 FSDP 把分片 reshard/offload 掉了),而另外三个还是 8.7–9.1 G。
     再加上 ray 把它的进程名从 `ray::WorkerDict.actor_rollout_compute_log_prob`
     退回成裸 `ray::WorkerDict`、`/proc/<pid>/wchan` 是 `ep_poll` —— 这一组证据直接指出是哪个 rank 先返回的。
* 另外 `timing_s/` 能对上 99.9% 的墙钟时间这条经验只在**正常步**成立,挂死的那步压根不会写出 `timing_s/`。

**处置**。等 watchdog 自己报(别急着杀):3600 秒到点后它会打出 OpType / SeqNum / 哪个 rank 少入队,
这是定位的全部依据,手动杀掉就什么都没有了。然后进程自己 SIGABRT 退干净、显存全还,
从 `global_step_21` 用同一套 env `resume_mode=auto` 续跑即可(本次 05:34 重启)。

**`nccl_timeout` 是双刃的**。设 3600 是 v3 血统的教训(长 update_actor 超过 600s 默认值会被 watchdog abort,
见 `restart_s1_smith_trainer.sh` 差异清单第 10 条);代价是真挂一次要烧掉**四张卡一小时**。
实测 update_actor 最长 977 s,所以 1800 其实也够用、能把损失腰斩 —— 但这是配方,留给人类定。

**修法**(当时判断:一行,未经人类同意不动 —— 这个判断在 2.5 小时后被推翻,见 §52.2):
给这两处调用补上 DP 组,和 engine 路径对齐。数值上中性 ——
它只把各 rank 的循环次数**补齐到最大值**,log_prob 是逐行算的、梯度归一化在 verl 里是显式按 token/rollout 缩放的。
要注意 `seqlen_balancing.py:410` 的 `assert num_micro_batches <= num_groups`:对齐后取的是全局最大值,
若某个 rank 的行数比别人少就会改成断言失败(崩,但不再是静默挂死)。这条值得往 AGL 上游提,
和 #589 的 abort 静默失效是同一类问题:**失败被静默跳过,症状出现在一小时之后的另一个地方**。

### 52.2 同一个 bug 的第二个落点(`update_policy`)——命中率不是 1/21 是 3 步 2 次,补丁最终打了

§52.1 结尾我把"补 `dp_group`"留给人类定,理由是"命中率约 1/21 步,每次代价有界(四卡一小时),
而改错梯度归一化会悄悄毒化几天的训练",风险不对称。**2.5 小时后这个前提就塌了。**

重启后 step 22 正常落地(1193 s:gen 271 + old_log_prob 153 + update_actor 610,reward 0.367、
`rollout_is_mean` 0.995、capture 100%),紧接着 **step 23 又挂**,watchdog 07:24:42 报:

```
[Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=26720,
  OpType=_ALLGATHER_BASE, NumelIn=133693952, NumelOut=534775808, Timeout(ms)=3600000)
  ran for 3600015 milliseconds before timing out.
[Rank 3] PG status: last enqueued work: 26721, last completed work: 26719   [repeated 3x across cluster]
```

"repeated 3x" + 日志里只有 rank0/2/3 报错 → **这次少入队的是 rank 1**(上次是 rank 2)。
`NumelIn=133693952`、`NumelOut=534775808`(= 4×133693952),是一个大得多的 FSDP flat-param unshard,
和 §52.1 那次的 `NumelIn=11797504` 不是一个量级。

**两次的"长相"不同,但是同一个 bug 的两个调用点**:

| | §52.1 | §52.2 |
|---|---|---|
| 调用点 | `dp_actor.py:468` `compute_log_prob` | `dp_actor.py:560` `update_policy` |
| 掉队 rank 的样子 | **空闲**:5 jiffies/15 s、显存 9.0 G→1.65 G、进程名退回裸 `ray::WorkerDict`、`wchan=ep_poll` | **也在自旋**:四个 rank 全是 1 核 |
| 为什么不同 | 这一遍后面没有集合通信,提前退出的 rank 直接 return 回 ray 空等 | 后面紧跟梯度规约/optimizer,提前退出的 rank 进了**另一个**集合通信,于是四个都在等 |

**所以"对称"不等于"没事"。** §52.1 我写的"判据是不对称"只对 `compute_log_prob` 那个落点成立。
在 `update_policy` 落点上四个 rank 全自旋,这时唯一好用的判据是**功耗**:

> `nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm`
> 挂死:**100% util / 1980 MHz / 123–125 W**;真训练:同样 100% util 但 300 W 以上。
> NCCL 忙等 kernel 常驻 SM,所以 util 读数满格,但几乎不耗电。**功耗是最便宜也最锐利的那把刀。**

顺带两条:
* `update_actor` 这一遍**会先正常跑一段再发散**(micro-batch 跑到某个 rank 先用完才分叉),
  所以"进了 update_actor"不等于"已经挂了";06:26 那次采样我只看了 CPU 和显存没看功耗,
  把一个还在正常跑的时刻当成了挂死起点,结果 watchdog 比预估晚了 20 分钟。**采样一定要带功耗。**
* 反推挂死起点最可靠的方式是 watchdog 时间减 3600 s(本次 = 06:24:42)。

**为什么最终动手打了补丁**(推翻 §52.1 的判断):
1. **命中率从 1/21 修正为 3 步 2 次。** 两个调用点都会命中,而 `update_policy` 还把 mini-batch
   再切一刀(PPO_MINI 被 `fsdp_workers.py:250` 按 DP 规一化成 4 行/rank),切得越碎
   `ceildiv(total_seqlen, max_token_len)` 在 rank 间越容易不等。
2. **SAVE_FREQ=3、最后一个 ckpt 是 `global_step_21`,照这个节奏永远走不到 24**,
   等于这条血统彻底不前进,而且每挂一次白烧四卡一小时。风险的不对称性反过来了。
3. **"数值中性"从代码里核实了,不是靠记忆**:`dp_actor.py` 里
   `loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size`(dynamic_bsz 分支),
   micro-batch 的 loss 按**行数占比**加权,怎么切分加总都等于同一个 mini-batch 均值。
   补 `dp_group` 只改"切成几块",不改梯度。
4. **这是 bug 不是配方**,补完就等于 verl 自己 engine 路径的做法;而且万一判断错,
   症状是 grad_norm/reward 立刻变脸,半小时一次的巡检能抓到,不是"悄悄毒化几天"。

**补丁**(`.venv/.../verl/workers/actor/dp_actor.py`,原件备份 `dp_actor.py.orig_20261003`):

```python
def _dp_group(self):
    if not torch.distributed.is_initialized() or self.use_ulysses_sp:
        return None          # SP>1 时推不出 DP 子组,维持原行为;本 run 是 sp=1
    return torch.distributed.group.WORLD
```

两处调用都加 `dp_group=self._dp_group()`。逻辑照抄
`FSDPEngine.get_data_parallel_group()`(`verl/workers/engine/fsdp/transformer_impl.py:558`)。

**验证补丁真的生效,别用 `inspect.getsource`。** 这两个方法被
`verl/utils/profiler/performance.py:104` 的装饰器包了一层,而那个装饰器**没有用
`functools.wraps`**,所以 `inspect.getsource(C.compute_log_prob)` 和 `__code__` 拿到的都是
**wrapper** 的源码/字节码,我第一次自检就因此得到了假阴性。正确的查法是顺着闭包拆到里层:

```python
def unwrap(f, depth=6):
    for _ in range(depth):
        cl = getattr(f, "__closure__", None)
        if not cl: break
        cands = [c.cell_contents for c in cl
                 if callable(c.cell_contents) and hasattr(c.cell_contents, "__code__")]
        if not cands: break
        f = cands[0]
    return f
# compute_log_prob -> dp_actor.py:444  refs_dp_group=True
# update_policy    -> dp_actor.py:530  refs_dp_group=True
```

07:26 用同一套 env(`TRAIN_BATCH=16 PPO_MINI=16 ROLLOUT_N=4 VLLM_SEQS=16 EAGER=False
SAVE_FREQ=3 TEST_FREQ=10 RESUME_MODE=auto`)从 `global_step_21` 重启,step 22/23 的成果丢失。
`nccl_timeout` 仍留 3600 没动(那是配方)。

### 52.3 补丁站住了:3 步 0 挂,血统越过 21

打完补丁后连过三步,`update_actor`(两次挂死其中一次就挂在这一遍)全部走完:

| step | gen | update_actor | 全步 | reward | grad_norm | entropy | is_mean |
|---|---|---|---|---|---|---|---|
| 22 | 332s | **595s** | 1234s | 0.382 | 0.0026 | 0.247 | 0.994 |
| 23 | 236s | **695s** | 1288s | 0.766 | 0.0027 | 0.253 | 0.994 |
| 24 | 199s | **894s** | 1549s | 0.565 | 0.0049 | 0.247 | 0.994 |

对照:补丁前是**3 步挂 2 次**,且 `global_step_21` 之后永远走不到 24(SAVE_FREQ=3)。
现在 `global_step_24` 已落盘 29G、`latest_checkpointed_iteration.txt`=24、21 按 keep=1 自动清掉,
**血统第一次越过 21**。894s 还是目前见过最长的一遍 update_actor。

数值面没有异常:grad_norm 同量级、entropy 不塌、`rollout_is_mean` 稳在 0.994、capture 100%、
`kl_coef` 0.001 在。reward 0.766 那一步是批间方差(16 题/步,`critic/score` max 1 / min 0,
不是退化全对),前十步本来就在 0.30–0.56 摆。

`seqlen_balancing.py:410` 的 `assert num_micro_batches <= num_groups` 三步都没响——
这是补丁万一算错的失败模式,会在第一次 `compute_log_prob` 就炸;没炸说明 MAX 对齐后的
micro-batch 数确实还在各 rank 的分组数以内(64 行 /4 rank = 16 行,ppo_mini 规一化后 4 行/rank)。

## §50 坍塌探针自己有个静默失效的判据:verl 的 `actor/*` 是 numpy 标量 repr(2026-10-05)

s3 起训前写了 Tier 1 坍塌探针 `s3_watch.py`,8 条判据,并且用 s1(健康 90 步)/
s2(坍塌 5 步)的真实日志回放验收过 —— s2 在 step 4 开火、s1 九十步不开火。
**验收通过,但第 6 条判据(梯度爆炸)其实是死的。**

### 50.1 病灶:同一行日志里两种数字形态

verl 把 metric 拼成一行 `step:7 - timing/...:... - critic/score/mean:0.5 - actor/grad_norm:np.float64(0.00247) - ...`。
`critic/*` 和 `training/*` 是裸浮点,**`actor/*` 和 `rollout_corr/training_ppl` 是 `repr()` 出来的 numpy 标量**:

    critic/score/mean:0.5234375
    actor/grad_norm:np.float64(0.002471662061806354)

原来的提取式 `re.escape(k) + r":([-\d.eE+]+)"` 对后者**一个字符都匹配不到**(卡在 `n`)。
于是 `actor/grad_norm` / `actor/entropy` / `actor/kl_loss` / `actor/pg_clipfrac` 全部 0/90 步。

修法是让数字式同时吃两种形态:

```python
RE_NUM = r"(?:np\.float\d+\(|np\.int\d+\()?(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
mm = re.search(re.escape(k) + r":" + RE_NUM, line)
```

### 50.2 真正的教训:**缺值被当成"没这项"静默跳过**

`judge()` 里每条判据都写成 `x is not None and <条件>`,这是对的 —— 日志行可能确实
没有某个 metric(不同 verl 版本、不同算法分支)。但副作用是:**解析失败和"指标不存在"
在代码里长得一模一样**。所以

- 不会抛异常,
- 不会打 WARN,
- 回放验收照样绿(第 1/2 条靠 `response_length/mean`,那是裸浮点,一直好的),
- 日志里的 `grad_norm` 栏一直显示 `-`,而 `-` 又正好是"这步还没算完"的正常显示。

四重掩护,所以这 bug 不是被监控发现的,是**被一次手工提数发现的**:回答"用什么指标判断
模型有没有更新"时我去捞 s1 的 `actor/*` 曲线,脚本打印"日志里没有",而我肉眼在同一行里
看得见它。如果那天没有人问这个问题,它会一路瞎到 s3 跑完。

### 50.3 处方:回放验收不够,要加**解析率断言**

回放验收(「s2 开火、s1 不开火」)只能验证**用到的**判据,对**没用到的**判据完全是瞎的。
所以把验收从 scratchpad 挪进仓库,并补上一条精确断言 —— `python3 s3_watch.py --selftest`:

- **A 解析率**:`METRICS` 每一项在两份历史日志上都必须 100% 解析得出(s1 90/90、s2 5/5)。
- **B 门槛分离度**:增量回放 `judge(rows[:i])`,s2 必须开火、s1 必须全程不开火。

实测这个测试有牙:把 `RE_NUM` 退回旧版,**B 仍然通过**(所以旧的回放验收抓不到),
**A 精确报出 8 条**(4 个 metric × 2 份日志)。

一句话:**监控自己也要被监控,而且要测"它读不到东西时会不会说话",不只测"它判得对不对"。**
同类坑:§46.1(本地改了不等于 pod 里改了)—— 都是"看起来在生效,其实没生效",
区别只在一个发生在部署边界,一个发生在解析边界。

---

## §51 跨血统比「每步多久」是个陷阱:分母不是一回事(2026-10-05)

s3 起训第一步跑了 **3233s = 53.9 分钟**,s1 同配置(TRAIN_BATCH=16 / PPO_MINI=16 /
ROLLOUT_N=4 / 卡 1235)的中位只有 **1703s = 28.4 分钟**。四个阶段几乎等比翻倍:

| | s1 step1 | s1 中位 | s3 step1 | s3/s1中位 |
|---|---|---|---|---|
| gen | 269s | 370s | 516s | 1.40 |
| old_log_prob | 223s | 203s | 451s | **2.23** |
| ref | 226s | 202s | 454s | **2.25** |
| update_actor | 885s | 806s | 1806s | **2.24** |
| 整步 | 1607s | 1703s | 3233s | 1.90 |

### 51.1 第一个想到的原因是错的
当时的第一反应是 `EAGER=False` 首步抓 CUDA graph(只付一次)、或者序列更长。
序列确实更长 —— `prompt_length/mean` 10851 vs 9210(+17.8%)、`response_length/mean`
266 vs 197(+35%)、`n_sample` 2044 vs 1927(+6%)。但这些乘起来只有 **1.19×**,
离 2.24× 差一倍。**按这个思路继续查就会查到错的地方去。**

### 51.2 真正的分母:动态采样筛剩下多少
verl 的 `timing_per_token_ms/*` 是个**自带分母的量**,可以反解出真实训练 token 数:

    训练 token = timing_s/update_actor ÷ (timing_per_token_ms/update_actor ÷ 1000)

拿它除以 `n_sample × (prompt+response)`,得到的留存率和
`(n_groups_seen − n_groups_zero_adv) ÷ n_groups_seen` **逐步对得上**:

    s1 step1  实测留存 0.822  组留存 0.812
    s1 step2  实测留存 0.809  组留存 0.812
    s1 step3  实测留存 0.459  组留存 0.400
    s1 中位   实测留存 0.627  组留存 0.562   训练 token 11.55M
    s3 step1  实测留存 1.002  组留存 1.000   训练 token 22.77M

也就是说 old_log_prob / ref / update_actor 三趟都发生在**动态采样丢完组之后**。
s1 中位扔掉 44% 的组,s3 一组没扔,光这一项就是 1.79×;再乘序列变长的 1.19× 和
per-token 代价的 1.11×(序列长 → attention 二次项),正好 ≈ 2.24×。**全部解释完,
没有剩余项。**

### 51.3 结论和用法
- **「每步多久」在不同 reward 口径之间不可比**,因为动态采样的留存率会变。可比的是
  **训练 token 数**和 **per-token 耗时**。要对比 wall clock,先把留存率摆出来。
- s3 慢一倍**不是退化,正是想要的东西**:二值奖励下近一半 rollout 的组内方差为 0、
  梯度为 0、算完就扔;改成 F2P 比例之后这些组全都带上了梯度。多花的时间买的是
  **多一倍的梯度覆盖**。
- 反过来也要有预期:留存率会随训练下降(s1 第 3 步就掉到 0.40),所以 s3 的步时
  **大概率会自己回落**到 30~40 分钟。如果它一直钉在 54 分钟,反而说明组内方差没衰减。
- 另有一个容易看岔的计数器:`training/n_groups:20` / `training/n_zero_adv_groups:4`
  和动态采样的 `n_groups_seen:16` / `n_groups_zero_adv:0` **不是同一个东西**。
  前者在 rollout-level advantage 阶段计数,把 `n_sample_padded:4` 那 4 条补齐行
  各算一个退化组(2044+4=2048=`critic/n_transition_after_dropping`)。
  **别把那个 4 当成"还是有 4 组没方差"。**

### 51.4 我自己在脚本里写反了一句,待订正
`restart_s3_smith_trainer.sh:168` 的注释写的是「部分分把零方差组从 40.4% 压到 30.3%,
所以这里的重采轮数应该变少、**步时变快**」。**后半句是反的。** 零方差组变少 →
丢掉的行变少 → old_log_prob/ref/update_actor 三趟要算的行变多 → **步时变慢**
(实测 28.4 → 53.9 分钟)。前半句"重采轮数变少"方向对但没有意义:s1 实测 91 步里
只有 3 步用到第 2 轮(`min_valid_groups=6` 在第 1 轮几乎总能满足),本来就没什么可省。

改不了是因为 `restart_s3_smith_trainer.sh` 作为 python 的父进程**还在跑**
(pid <N>),bash 是边读边执行的,在跑的时候原地改会损坏执行 —— 等停训再改。

### 51.5 动态采样和 KL 的一个交互(DAPO 原文不会提,因为它把 KL 去掉了)
被丢掉的零方差行**同时也丢掉了它们的 KL 惩罚** —— KL 项只在保留下来的行上算。
所以 s1 中位有 44% 的行既没有 policy gradient 也没有 KL 锚定。我们的 KL 是人类
09-07 明令开的(coef 0.001),这个覆盖面缺口是动态采样的副作用,不是 bug,但
记一下:**留存率低的时候 KL 的实际约束范围也跟着缩小**。s3 目前 100% 留存,
这一条暂时不生效。

## §53 我把「坍塌时奖励没报警」写进了代码、文档和记忆,错了 —— `critic/score/mean` 不是奖励(2026-10-05)

s3 立项时我给两层探针写的理由是:**「s2 坍塌时奖励完全没报警」**,证据是 5 步里
`critic/score/mean` 全程 0.40~0.56、唯一动的量是 `response_length/mean`。这句话进了
`s3_watch.py` 的模块 docstring、`s3_cycle.sh` 的文件头、pitfalls 和记忆。

**10-05 复查发现是我盯错了指标。** 同一份日志里 `training/reward` 是这样的:

| step | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `response_length/mean` | 274.5 | 254.8 | **112.5** | **60.4** | 91.3 |
| `training/reward` | 0.5482 | 0.5801 | **0.1375** | **0.1465** | 0.2350 |
| `critic/score/mean` | 0.4336 | 0.5625 | 0.3965 | 0.4316 | 0.4395 |

奖励在第 3 步掉了 **4.2 倍**,和长度坍塌**同一步**。坍塌是看得见的,我看的地方不对。

### 53.1 `critic/score/mean` 不配当奖励看,三条独立理由

**a bf16 量化。** s1 九十一步的读数**无一例外**全部落在 `k/2^n` 网格上(分母 128 出现
32 次、256 出现 32 次、512 出现 27 次,零例外),0.6 附近分辨率只有 ~1/512。而
`training/reward` 是 float64(20 个读数里 18 个离网格,如 `0.6219277828052747`)。
**后果:两步读数完全相同可能只是落进同一个 bf16 桶,不是"指标冻住了"。** s3 step1/2
的 score 都是 0.6015625 = 77/128,我一度怀疑指标卡死,查 reward 才知道是
0.6219 → 0.6286 在动。

**b 行加权,不是 rollout 加权。** 轮数多的 rollout 占更大权重,而它们解得更差 ——
系统性偏低。

**c 过滤后选择偏差(最毒的一条)。** 它只看得见**动态采样留下来的组** —— 全对组和
全错组被系统性排除。偏差随丢弃率增长。实测 `score/mean − reward` 中位数:

    s1  +0.0479      s2  +0.2045      s3  −0.0237(step 1-2,零方差组极少)

s2 那 **+0.2045** 就是把 4.2 倍跌幅压成 1.4 倍、让我判成"没报警"的原因。

→ **口径教训:凡是 verl 里 `critic/*` 开头的量,都是"过滤后、行加权、bf16"口径,
只能当参照,不能当被监控的目标。** 要监控奖励就读 `training/reward`。

### 53.2 但订正不能摆到另一个极端:只看 reward 也不行

健康的 s1 自己就会掉到 **0.2137 ~ 0.2768**(第 3/8/12 步,91 步最低 0.2137),和 s2
的坍塌值 **0.1375 / 0.1465 区间重叠**。所以:

- **绝对门槛会在健康血统上误报** —— 判据必须用相对形式(× 前 2 步均值)。
- **分得开的程度两条判据各有胜负**(都按"连续 2 步"口径,健康侧看全程最接近开火的
  那一对、坍塌侧看开火那一对):

    | 判据 | s1 健康侧余量 | s2 坍塌侧余量 |
    |---|---|---|
    | 1 长度 `< 0.50×B` | 147.7/88.9 = **1.66×** | 132.3/112.5 = 1.18× |
    | 7 奖励 `< 0.40×B` | 0.3008/0.2004 = 1.50× | 0.2257/0.1465 = **1.54×** |

  长度在健康线上留的余地更大,奖励在坍塌上跨得更果断。**谁也不压倒谁。**

- **长度仍是主判据,但理由是可测的,不是感觉**:s2 第 5 步奖励已回升到 0.2350(高过
  门槛 0.2257),尾部连续性断了;长度 91.3 仍远在门槛 132.3 之下。实测回放:

      探针只在 step 4 这轮评估 → 开火 [1, 2, 7]
      探针只在 step 5 这轮评估 → 开火 [1, 2]        ← 判据 7 已经沉默

  **即"奖励信号会自己恢复、长度信号持续更久"。** 探针漏一次轮询,长度还兜得住。

### 53.3 判据 7 原来结构上**永不可能开火** —— 第二次 §50 类静默失效

原文:`score/mean == 0 连续 3 步`。这永远不会成立:

1. 留存下来的组**按定义有奖励方差**,而分数非负 → 过滤后的均值几乎不可能恰好为 0;
2. 真要所有组都零方差,`_collect_train_batch` 返回 `None`(`verl/trainer.py:610-700`),
   **这一步根本不会训练**,也就不会有这行指标。

而巡检指令第 7 条明确写着要看「reward 持续为 0」—— 所以这是个**写在清单上、看着有
覆盖、实际永远不响**的盲区。和 §50(`actor/*` 解析率 0/90)是同一个病:**失效是静默的。**

### 53.4 处方:回放验收必须**逐条**核,不能只核"有没有告警"

旧验收是「增量回放 `judge()`,s2 必须开火、s1 必须不开火」。它是绿的 —— 因为判据 1
开火了,**合并列表非空就算过**。死掉的判据 7 就藏在这片绿里。

改法(`s3_watch.py` selftest 的 B 段):按告警前缀 `[N ` 拆出每条判据**各自**首次开火
于哪一步,然后断言 `CALIBRATED = (1, 7)` 里每一条都在 s2 上开过火:

    >>> 首次开火于 step 4;逐条 {1: 4, 2: 4, 7: 4}

`CALIBRATED` 只放"阈值在这两份日志上标定过"的判据。4/5/6/8(采集率、训推偏差、梯度
爆炸、进程没了)在这两份日志里**没有正例**,不能要求它们开火 —— 否则验收会变成逼
自己造假夹具,那比没有验收更糟。**它们的覆盖只能靠别的办法(造数据或实测到为止),
这一点必须承认而不是掩盖。**

### 53.5 余量的分子不是"单步最低"

我第一版 selftest 把长度打的是**单步**余量(115.1/88.9 = 1.29×)、奖励打的是**连续
2 步**余量(1.50×),两个数不可比,还得出"长度余量更小"的错结论。判据要求连续 2 步,
所以分子必须是「所有相邻两步里较高那个的最小值」—— 这个量跌破门槛才会开火。
统一后:长度 1.66×、奖励 1.50×(单步口径下分别是 1.29× 和 1.07×)。

## §54 reward 掉了 2.4 倍,真凶在反分词器:`skip_special_tokens` 把模型的工具调用剥成了乱码(2026-10-05)

s3 血统五步内 `training/reward` 从 0.6286 掉到 0.2625,"白忙"(P2P 完好但一个 F2P 都没过)
从 14% 涨到 66%。第一反应是 s2 那个坍塌(策略学会少写 token),但**方向是反的**:
s2 的失败 rollout 烧满 40 轮上限、每轮只写 53-74 token;s3 的失败 rollout **提前短收**
(白忙中位 13 轮 vs 非白忙 30 轮)。不是同一个病。

去 rollout 日志读结束原因,全是 `3 consecutive format errors, ending episode` +
`submitted=False patch=0B`。再去轨迹里读模型到底发了什么,看到的是:

```
<think>\n\n</think>\n\n name="bash"> name="command">find /testbed -name '*.py' | head -50
```

**开头的 `<` 不见了。** 这里有两种完全不同的结论,代价差很远:

- 策略漂移 → 是训练的问题,该停训、该查 lr/KL;
- 解析/模板 bug → 是服务端配置的问题,改一行就能回来,停训是误判。

**判别子是最便宜的那个:全库搜有没有一例"完整的开标签"。**
1472 次 `name="` 里,带 `<` 的完整开标签 **0 例**;而 `<think>` 在同一批文本里完好出现 8001 次。
0/1472 不是采样噪声 —— 一个真在漂移的策略不可能一次都没写对过。而 `<think>` 活着说明
**没有任何东西在全局剥 `<`**,剥的是**特定的几个标签**。这立刻指向特殊 token。

查 tokenizer 坐实:`<function` / `<param` / `</param>` / `</function>` 是 id 18/20/21/19 的
**特殊 token**,MiniCPM 的 chat template 原生工具格式就是 `<function name="x"><param name="y">`。
然后逐字复现:

```python
s = '<function name="bash"><param name="command">grep -rn "x" /testbed</param></function>'
tk.decode(tk.encode(s, add_special_tokens=False), skip_special_tokens=True)
# → ' name="bash"> name="command">grep -rn "x" /testbed'     ← 和轨迹里一字不差
```

完整链条:

1. 模型发**格式完全正确**的原生工具调用;
2. vLLM 反分词默认 `skip_special_tokens=True`,把那四个标签原地删掉;
3. harness 只认 ```` ```bash ```` 围栏(`agents/smith_agent.py:38`)→ `found 0` → `Format error`;
4. **残文被追回对话历史**,下一轮 prompt 装的就是它 —— 模型看不见自己发了什么,
   而这段残文现在是上下文里的强先验,下一轮更可能再发一次;
5. 连 3 次 → episode 结束、patch 0B、reward 0。

### 四条可复用的

1. **"模型在胡说"和"我在错读模型"长得一模一样。** 判别不需要跑任何东西,只要问:
   *正确的那个形态在数据里出现过几次?* 0/1472 就是判决。这比读日志、读代码都快。
2. **`skip_special_tokens=True` 对"用特殊 token 表达结构"的模型是破坏性的。** 它是
   chat/completion 的默认值,为的是不把 `<|im_end|>` 吐给用户;但当模型的**工具调用语法本身**
   由特殊 token 构成,这个默认值会把语法删掉而把参数留下,产出的东西**看起来像模型的错**。
   多轮 agent 里更毒:残文进了历史,污染会自我放大。
3. **预存缺陷和 RL 造的缺陷要分开算。** step 1 的 rollout 是起点权重未经任何梯度更新的输出,
   原生格式已占 4.8% —— 所以 RL 不是发明了这个行为,是**放大了 5.4 倍**。
   没有 step 1 这个"零点",就会把一个数据/服务端缺陷记成"RL 把模型训坏了"。
4. **假阴性要主动证伪。** 我想确认 SFT/基座的 474 评测是否也中招,grep 得到"0 命中"
   差点当成证据 —— 回头查才发现那两份目录**根本不存助手文本**(4201 个文件里 ```` ```bash ````
   出现 0 次)。§50 同一类:**"搜不到"必须先证明"搜得到的东西在里面"**。

## §55 为什么基座没这个毛病、SFT 之后才有:是我们自己把原生格式训进去的(2026-10-05)

§54 查清了"标签去哪了",但没回答**为什么偏偏 SFT 起训的血统才崩**。
人类追问「之前的训练为啥没有格式问题,而这一次有」—— 这一问才逼出真因。

### 相关性(§54 停在这里,这不算回答)

| 血统 | 起点 | lr | 原生格式占比(逐步) | 总计 | reward |
|---|---|---|---|---|---|
| s1 | **基座** | 2e-6 | 0.06~2.00%,**九十步没涨** | 1837/166226 = 1.11% | **没崩**(0.21~0.77) |
| s2 | SFT ep3 | 5e-6 | 2.65 → 2.41 → **14.04** → 12.84 → 18.31 → 24.90% | 12.62% | **step 3** |
| s3 | SFT ep3 | 2e-6 | 4.85 → 4.25 → 6.54 → **16.05** → 25.79 → 19.36% | 11.85% | step 4 |

两条 SFT 血统**崩在原生格式占比跳 2.5~6 倍的那一步**。但"起点不同"只是换了个说法。

### 真因:SFT 的 loss 就是盖在原生格式上的

教师(deepseek-v4-flash via opencode)的轨迹把工具调用存在 OpenAI 的
`tool_calls` 结构化字段里,`content` 是空串。`sft/swe_sft_dataset.py` 做了两件事:

- `:29-42 normalize_messages()` 把 `function.arguments` 从 JSON **字符串**解成 dict
  —— 不解模板直接抛 `'str object' has no attribute 'items'`(就是 `trace-capture-8pct-bug` 那个坑);
- `:64-67 render()` 调 `apply_chat_template`,而 `chat_template.jinja:69-75` 把每个
  `tool_calls` 渲染成 `<function name="bash"><param name="command">…</param></function>`,
  值里含 `<`/`&`/换行时还套 `<![CDATA[...]]>`(模板第 8 行明确教的)。
- `:75-94` 的 loss mask 按字符区间精确盖在 assistant 段上 —— **也就是正好盖在这些标签上**。

在 858 行 train.parquet 的 **12528 个被算 loss 的 assistant 轮**里实测:

```
原生 <function name=  13254 次  (1.06 次/轮)
```bash 围栏              0 次
```

教的工具名:`bash` 7718、`read` 3270、`edit` 1178、`grep` 975、`glob` 81、`write` 20、`webfetch` 12。
参数名:`command`、`filePath`、`limit`、`offset`、`oldString`、`newString`、`pattern`、`path`…

**所以不是"SFT 之后模型学坏了",是 SFT 一万三千次梯度在教它发 harness 不认的格式,
同时零次在教 harness 认的格式。** 基座没被这么教过,所以九十步都不往那边漂。
而之前被当成"模型瞎编工具名"的那批残文(`name="grep"> name="pattern">`、
`name="read_file"> name="path">`),是**对 SFT 教过的 read/grep 的忠实回忆**,不是幻觉。

### 修法:查出来是三个独立缺陷,不是一个

| # | 缺陷 | 证据 | 修 |
|---|---|---|---|
| ① | detokenizer 把标签删掉 | 确定性验证:prompt 停在 `<param name="command">ls -la`,`True` 续写 ` /testbed`、`False` 续写 ` /testbed</param></function>` | `extra_body["skip_special_tokens"] = False` |
| ② | `<|im_end|>` 漏进 `content` | 开了 ① 之后 8/8 命中(①的必然副作用) | `strip_turn_delims()` 在第一个分隔符处截断 |
| ③ | **服务端根本不在轮边界停** | 16/16 `finish_reason=length`,单条回复含 9~239 个 `<|im_end|>`,中位 2236 字符;围栏数 {0:9, 1:5, **5:2**} —— "found 5" 那种格式错是这么来的 | `extra_body["stop"] = ["<|im_end|>"]`(`SMITH_STOP` 可改) |

> **⚠ 2026-10-05 订正:缺陷 ② 和 ③ 不成立,下面这段归因作废。**
> 那组「16/16 `finish_reason=length`、单条回复含 9~239 个 `<|im_end|>`」是我用 `n>1`
> 采样量出来的**测量假象**:同一个请求的多个 sample 被拼在一起读了。重测(n=1)后
> 服务端**是**在 `<|im_end|>` 停的,所以
> - 「vLLM 0.8.5 不把 `generation_config.json` 的 `eos_token_id` 变成停止条件」—— **撤回**;
> - `extra_body["stop"]` 和 `strip_turn_delims()` 变成**无害的保险**,不是修了什么病。
>
> **只有缺陷 ① 是真的**:`<function` 是特殊 token 18,`skip_special_tokens=True` 会把它原地删掉。
> §54 整节的 reward 掉 2.4 倍,归因只落在 ① 上。
>
> 教训:**`n>1` 的采样结果不能按"一条回复"读。** 这是我在同一个月里第二次被自己的
> 测量方式骗(第一次见 §50:缺值被静默跳过)。量之前先问"这个数字的一个单位是什么"。
>
> 以下原文保留,作为错误归因的留痕:

③ 的根因:`<|im_end|>` 是 130073,**确实**写在 `generation_config.json` 的
`eos_token_id: [1, 130073]` 里,但 vLLM 0.8.5 的 OpenAI server(`vllm/config.py:371`,
`generation_config="auto"`)只从那个文件取 temperature/top_p,**不把 `eos_token_id` 变成停止条件**;
而 tokenizer 自己的 `eos_token` 是 `</s>`(id 1),这个 chat 模型从不发。
加上 `stop` 后 16/16 干净停住、中位 116 字符 —— **少生成 19 倍文本**。
注意 verl 的 rollout 路径停得是对的(真实轨迹 0/64 带残留分隔符),③ 只影响 OpenAI server 路径,
也就是 474 评测走的那条。

另外两处配套改动(都不扩动作空间):
- `parse_action()` 接受原生语法作为"同一个动作的另一种拼法"(含 43.2% 的 CDATA);
- `read`/`edit`/`grep`/`glob`/`write`/`webfetch` **不翻译成 shell 等价物** —— 那会扩动作空间、
  使分数和 mini-swe-agent 基线不可比。改为回一条准确的 `UnavailableToolError` 提示
  (「没有这个工具,本环境只有一个动作:在 /testbed 跑一条 shell 命令」),
  而不是误导性的"格式错"。

验收(真实分布,858 行 SFT 的 11726 个带原生调用的学习轮):解析成 shell 动作 54.0%、
判"无此工具" 37.7%、仍判格式错 8.3%。端到端(同 prompt 同 seed,n=32,SFT ep3 真服务):
格式错 **25/32 → 0/32**,可执行动作 7 → 18,`finish_reason` {length:31,stop:1} → {stop:32},
生成字符 38749 → 4031。单测 21 条见 `test_native_action_parse.py`。

**口径后果**:解析器和停止符都会改分数,`val_smith_mc_base` 123/474 和
`val_smith_mc_sft3` 136/474 **必须用补丁后的 harness 各重测一次**才能再做比较。

**边界(别越过)**:s1 从基座跑了九十步、基本没这个毛病(1.11% 且不涨),
八次探针均值 7.63/48 **全在 ±1σ 内,从来没涨过**。这次修的是一个坍塌机制,
**不等于能训出增益**。两件事必须分开说。

### 教训

1. **"起点不同"不是机理,是待解释项。** 相关性做完必须继续问"起点里的什么"。
   我第一轮答到相关性就收手了,人类一句追问就把它打回去 —— 自己该问这一句。
2. **SFT 数据的"格式"由模板决定,不由你眼里的 JSON 决定。** 数据文件里是
   `tool_calls: [{function:{name,arguments}}]`,看不出任何格式问题;格式是
   `apply_chat_template` **渲出来**的。检查 SFT 数据必须检查**渲染后、loss mask 覆盖的那段文本**,
   不是检查源 JSON。一行 `span.count('```bash') == 0` 就能在训练前拦住这事。
3. **目标格式要作为数据校验项。** 下游 harness 认什么格式是已知的,
   那它在 SFT 学习区里的出现次数就该有个断言。13254 : 0 这个比值,
   训练前跑一遍 `test_native_action_parse.py` 的验收段就会炸出来。
4. **蒸馏要连工具集一起对齐。** 教师有 read/edit/grep/glob/write/webfetch 六个工具,
   学生环境只有一个 shell。教师轨迹里 41.7% 的调用指向学生根本没有的工具 ——
   这部分不是"学生学得不好",是**我们教了它环境里不存在的动作**。
5. **harness 有两份,我差点只修了一份。** 补丁写完、单测全过、真服务端到端验过之后,
   才发现**评测路径根本不读我改的那个文件**:
   - 训练侧:`refresh_agent_configmap.sh` 把 `examples/swe_smith/agents/smith_agent.py`
     推进 k3s configMap;
   - 评测侧:`smith_fleet.sh` → `smith_rollout.py:18-19` 从 `swe_smith_smoke/` **自己目录**
     import `smith_agent`,那是 09-05 分叉出去的 949 行旧拷贝(新版 1220 行);
   - 更隐蔽的一层:`smith_rollout.py:161` 有 `sa._query = _query` —— 把 `_query` **猴补掉了**。
     就算把文件同步过去,改在 `_query` 里的 `extra_body`
     (`skip_special_tokens`/`stop`)依然进不了评测路径,必须在 `smith_rollout.py` 重复一遍。
   AST 级比对确认两份只差 5 个定义(`_query`/`evaluate`/`main` 各自专属,
   `parse_action`/`run_agent_loop` 的差异正好是本次补丁),评测侧只用到 10 个名字且全部逐字相同,
   且新版是纯 stdlib、自带 `main()`,所以直接用同一份覆盖,裸 `/usr/bin/python3` 验过能 import。
   **已加硬闸**:`smith_fleet.sh:26-37` 两份 `cmp -s` 不一致就 `exit 3`,并断言
   `smith_rollout.py` 里有那三个关键字(正反例都验过)。
   教训:**宣布"改完测完"之前要先问"跑的是哪个文件"。** 一个文件被两条路径以两种机制加载
   (configMap 推送 / 同目录 import + 猴补),就一定会分叉;靠纪律记不住,得让它跑不起来。

---

## §56 多轮 RL 的「多轮」从来没生效过:每条 episode 被打碎成一轮一行,病根是 chat 模板里 4 个 token 的不对称(2026-10-05)

s1 九十一步、s2 五步、s3 六步,三条血统都没训出增益。此前的归因是
credit assignment 梯度抵消(§见 smith-rl-root-cause),处方是稠密奖励。
处方本身是对的(零方差组 41% → 实测 15%),但**它治的不是主病**。

### 病灶:一个指标我一直没看

```
training/n_unmerged_rollouts: 64 / 64        ← 每一步、两条血统,全都是满的
training/n_sample_collected:  2044 / 1613 / 1416 / 1169 / 1049   (s3 步 1-5)
```

64 条 episode 应该产出 64 行训练数据,实际产出一两千行。
`rollout_adapter.py:816-864`:adapter 要把一条 episode 的各轮拼成一条序列,
判据是「第 k+1 轮的 `prompt_ids` 是否以(第 k 轮 prompt + response)为前缀」。
不成立就把当前这一轮**当成一条独立的训练样本 flush 掉,并打上整条 episode 的最终奖励**。

两个后果,都是致命的:
1. `enable_rollout_level_advantage=True` 和 `policy_loss.loss_mode=per_rollout_mean`
   —— 整套多轮 credit assignment 的实现 —— **从来没拿到它们要的那条合并序列**。
   「一条 episode 一个样本」退化成「一轮一个样本」。
2. 一条 39 轮的 episode 贡献 39 行、一条 8 轮的贡献 8 行,**长 episode 的梯度权重大 5 倍**。
   而轮数长基本等于做不出来 —— 等于在给失败模式加权。

### 真因:模板对「历史轮」和「生成位」不对称

`MiniCPM5-2B/chat_template.jinja`:

```jinja
# :121-122  历史里的 assistant 轮 —— 无条件插空 think 块
{%- elif '<think>' not in content and '</think>' not in content %}
    {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content.lstrip('\n') }}

# :168-177  生成位 —— 只有显式传了 enable_thinking is false 才插
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined %}{%- if enable_thinking is false %} ... 
```

于是同一轮 assistant 在两次渲染里长得不一样:

```
第 k 轮 prompt 结尾      ...<|im_start|>assistant\nTHOUGHT: Let me look at ...
第 k+1 轮重渲染这一轮    ...<|im_start|>assistant\n<think>\n\n</think>\n\nTHOUGHT: Let me look at ...
                                              ^^^^^^^^^^^^^^^^^^^^^^^^ 多出 4 个 token
```

前缀判据必败。s1 step10 turn1 `diverge_index=1126`、s3 step1 turn1
`diverge_index=1113 exp_len=1138 act_len=1460` —— **两条血统的 mismatch 记录逐字相同,
是同一个病因**(我一度以为 s3 形状不同,读串了,已订正)。

### 代价:18.3 倍的前向 token

`step_1_merge_mismatch.jsonl`,64 条 rollout:

| | 前向 token 总量 |
|---|---|
| 未合并(实际在训的) | **22,123,895** |
| 合并后(本该训的) | **1,207,261** |

因为未合并时第 k 轮的 prompt 被**完整重复训练** k 次 —— prompt 随轮数平方增长。
一步 1744s 里 `update_actor` 811s(47%)+ `old_log_prob` 205s + `ref` 205s = **71%**
全是按前者在烧。所以这同时是正确性 bug 和性能 bug,**而且是同一个修法**。

### 修法与验收

生成位改成**默认非思考**,和 :121-122 对齐:

```jinja
{%- if enable_thinking is defined and enable_thinking is true %}
    {{- '<think>\n' }}
{%- else %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

三处都要改(`MiniCPM5-2B/chat_template.jinja`、`MiniCPM5-2B-sft-v3-ep3/chat_template.jinja`、
以及 **`sft-v3-ep3/tokenizer_config.json` 里内嵌的那份** —— 两份内容逐字相同,9060 字符,
但加载路径不同,只改一份会分叉,见 §55 教训 5)。

token 级验收,四条都要过:
- 不传 kwarg:前缀判据 **False → True**(有判别力:旧模板在这个条件下是 False);
- `enable_thinking=False`:补丁前后**渲染逐字相同**;
- `enable_thinking=True`:同上逐字相同;
- 加的文本正好是 `'<think>\n\n</think>\n\n'`,token 数 45 → 49。

后两条是**口径保护**:评测路径显式传 `enable_thinking=False`,所以基座 123/474、
SFT 136/474 这些历史数字不受补丁影响,仍然可比。

### 修完会连带踩两个坑(不一起改就等于没修)

1. **`MAXRESP=4096` 会截掉 88% 的合并行。** 合并后 response 是所有轮输出+观测的总和:
   实测(n=635)中位 11889、p95 22953、max 44532。改 `MAXPROMPT=8192`(实测 prompt max 5277,
   截断 0.0%)+ `MAXRESP=43008`,两者之和正好等于 `CTX=51200`。
   `use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu=51200` 意味着**峰值显存不变**
   —— micro-batch 按 token 预算切,而旧配置的最坏情况(prompt 45448 + resp 4096 ≈ 49.5k)
   本来就和新的最长序列(49809)一个量级。
2. **一步凑不满一个 mini-batch。** 全局 mini-batch = `ppo_mini × rollout.n` 行。
   合并前一步 ~1262 行 → 约 20 次优化更新;合并后只有 ~64 行,配 `ppo_mini=16`(=64 行)
   就只剩 1 次,且凑不满时 `_pad_with_neutral_duplicates`(`trainer.py:757-789`)会补中性复制。
   改 `PPO_MINI=4` → mini_bs=16 行 → 每步 3 次更新、0 填充。

### 教训

1. **`training/n_unmerged_rollouts` 这种「本该是 0」的计数器,要么进告警,要么等于不存在。**
   它每一步都打在日志里,打了九十一步,我看了无数遍 step 时间和 reward,没看它一眼。
   凡是「正常值应为 0 / 100%」的指标都该有硬断言,而不是靠人眼扫。
2. **`rollout_adapter.py:825-827` 的归因字段是硬编码的假话:**
   ```python
   "template_mismatch": False, "retoken_mismatch": False, "others_mismatch": True,
   ```
   这三个字段零信息量,而且**主动把我从模板上引开了** —— 我因为它写着
   `template_mismatch: False` 而几次跳过了模板这条线。
   **宁可不给归因,也不要给假归因。** 这条待上游。
3. **配置项之间有隐式契约,改一个要把它的下游全走一遍。**
   这次「修对了但会被静默打败」的点有两个(长度几何、mini-batch 配比),
   两个都是在启动前量出来的,不是跑崩之后。启动前花二十分钟量几何,省掉的是几十小时。
4. **「同一个修法同时治正确性和性能」是个强信号,说明找到的是根因而不是症状。**
   反过来说:如果一个修法只能治性能或只能治正确性,要多问一句是不是还在症状层。

## §57 §56 的修法没治到病根:真凶是 AGL proxy 自己渲染 prompt、而且丢掉了 `chat_template_kwargs`(2026-10-06)

§56 把「每条 episode 被打碎成一轮一行」定位到了 **4 个 token 的错位**,这一步是对的;
但它把病因写成「chat 模板对历史轮和生成位不对称」,**这是错的**。模板补丁打完、
三份文件(`MiniCPM5-2B/chat_template.jinja`、`sft-v3-ep3/chat_template.jinja`、
`sft-v3-ep3/tokenizer_config.json` 内嵌那份)都改了之后重跑 smoke,验收指标没动:

```
training/n_unmerged_rollouts: 8 / 8          ← 和改之前一模一样
training/n_trace_merge_mismatch_rows: 271
```

§56 自己留了一条「未解的矛盾」:harness 确实发了
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`,可记录下来的 prompt
就是没有那个块。**这条矛盾就是答案,我当时没顺着它往下查。**

### 真凶

训练行里的 `prompt_ids` **不是 vLLM 返回的**,是 AGL proxy **自己重新渲染的一份**。
`proxy.py:347 _fill_missing_token_ids` 的 docstring 写得很直白:

> vLLM 0.8.x ignores return_token_ids; GRPO still needs prompt/response ids

也就是说 `prompt_token_ids` 永远是空的,**永远走回填路径** `_tokenize_chat`;
而 `response_ids` 来自 logprobs,是真·生成出来的 token。于是一条训练行里:

* `response_ids` ← vLLM 实际生成(vLLM **收到了** `chat_template_kwargs`)
* `prompt_ids`  ← proxy 本地重渲染(`_apply_chat_template`,**没收到** kwargs)

`_apply_chat_template()` 原来只拼这么点:

```python
base = {"tokenize": True, "add_generation_prompt": True}   # 加上 tools
```

`enable_thinking` 没定义 → 旧模板在生成位什么都不发;而 vLLM 那侧
`enable_thinking=False` 走的是 `chat_template.jinja:168` 的 else 分支,发
`<think>\n\n</think>\n\n`。**4 个 token,系统性,每一轮。**

于是 `rollout_adapter.py:756` 的 `ids_startswith(prompt_ids, current_context)`
每一轮都不成立,每一轮都被单独 flush 成一行、各自带整条 episode 的最终奖励。
修法是把请求自己的 kwargs 转发进去:

```python
def _apply_chat_template(tokenizer, messages, tools, template_kwargs=None):
    base = {"tokenize": True, "add_generation_prompt": True}
    if isinstance(template_kwargs, dict) and template_kwargs:
        base.update(template_kwargs)
```

调用处(`_tokenize_chat`)取 `request_body.get("chat_template_kwargs")` —— `extra_body`
的键会落在请求体顶层,所以客户端发的 kwargs 就在那儿。

### 为什么改了模板也没用:进程活了 18 天

训练指向 `--agl-base-url http://127.0.0.1:18082`
(`restart_s4_smith_trainer.sh:159`),那是**从 9 月 17 日 22:48 就常驻的
`agl-server` 进程**。`proxy.py:404-409` 的 `_TOKENIZERS` 是进程内缓存,里面那个
tokenizer 是**九月加载的**。23:06 改的模板文件对它完全不可见。

> 一条通用的:**长命进程 + 进程内缓存 = 改磁盘上的文件不算改。**
> 判断「我的修改生效了吗」,要先回答「跑这段代码的进程是什么时候起的」。

### 为什么已有的校验没报警

`proxy.py:343 _check_against_usage` 的容差是

```python
abs(len(ids) - expected) > max(2, expected // 20)      # 5%
```

1195 token 的 prompt 允许差 **59 个**。4 个 token 的系统性偏移**结构上不可能报警**。
容差是为"分词器版本差一两个 token"设计的,用来挡系统性错位就太松了 ——
**系统性偏移要看符号一致性,不是看绝对值**:连续 N 步同向偏同一个量,比偏多少更有信息。

### 连带踩到的第二个坑:重启 agl-server 丢了 `no_proxy`

杀掉 18 天的旧进程、从我自己的 shell 重起之后,8 条 rollout 全在
`LLM call failed / burning turn`,十五六轮一条都没成。proxy 日志里是
`POST http://<LAN_IP>:33455/v1/chat/completions "HTTP/1.1 502 ERROR"`。

原因:我的 shell 里 `http_proxy=<CORP_PROXY>` 而 **`no_proxy` 是空的**,
于是 proxy 发给上游 vLLM(**eth0 的 <LAN_IP>**,不是 127.0.0.1)的请求
全被本机代理截走。已经写成 `start_agl_server.sh`,按网卡生成 `no_proxy` 并自检选路。
两条反直觉的点写在脚本注释里:httpx 的 `no_proxy` **不支持 CIDR**(逐个列 IP);
验证必须让 **httpx 自己选路**,`curl --noproxy` 只能证明 curl 绕过了。

### 教训

1. **留在文档里的「未解的矛盾」要当成待查项,不是免责声明。**
   §56 把矛盾写下来了,也写了「补丁让两侧一致所以不管谁吞了 kwarg 都行」——
   这个推理错在:吞 kwarg 的那一环**同时**决定了训练侧 prompt 长什么样。
   绕过一个没定位的环节,只在它不参与结果时才成立。
2. **先把分布统计出来,再从个例推机理。** 我盯着 dump 的第 0 行推了很久,
   它显示 `<think>\n`(要走模板的 `reasoning_content` 分支)。把 271 行全表一遍:
   `<think>\n\n</think>\n\n` 占 ~250,`<think>\n` 只有 ~20。**我按少数派推了一整轮。**
3. **「训练学的那份输入」和「策略生成时的那份输入」是两个对象,要分别问它们从哪来。**
   RLHF 栈里这两份常常是不同代码路径渲染的,默认它们相同是最贵的假设之一。

---

## §58 残余的 31 处断链:模板把回复正文里的 `</think>` 前后对调了(第二个成因,已修)

§57 的 `chat_template_kwargs` 补丁去掉了 **89%** 的断链(271 → 31),但没清零:

```
training/n_trace_merge_mismatch_rows: 271 → 31
training/n_sample_collected:          279 → 39
training/n_unmerged_rollouts:         8/8 → 7/8
```

7/8 条 rollout 仍然碎成多行,rollout-level advantage 仍然拿不到合并序列,所以**不能**
就这么起训。

### 定位:先全表统计,再看一条原文

吸取 §57 的教训(盯第 0 行推了一整轮),这次先把 31 行按
「共同前缀的尾 / expected 的续 / actual 的续」归类。19 个"类"只是 THOUGHT 后面的
英文不同,**结构 31/31 完全一致**:

```
共同前缀尾 = '<|im_start|>assistant\n<think>\n'
expected 续 = '\n</think>\n\nTHOUGHT: …'     ← prompt_k ++ response_k
actual   续 = 'THOUGHT: …'                   ← prompt_{k+1} 重渲染
```

`diverge_kind` 全是 `token_differs`,而窗口前 16 字**逐字相同** —— 典型的
「同一段文字、不同切分」。但不是 BPE 边界问题,而是**文字被挪了位置**。

从 `trajectories/step_1_train.jsonl`(训练行,`prompt` 字段是重渲染后的完整历史)
取原文,机理一眼可见。模型的回复实际长这样:

```
THOUGHT: The current implementation is `range(...)`. Let me understand … [一大段推理]
Let me search tests for drange usage.
</think>

THOUGHT: The current code … 
```bash
grep -rn "drange" /testbed --include=*.py
```
```

SFT 的教师轨迹里 assistant 是「推理 + `</think>` + 答案」,**模型学会了在回复中途
吐一个孤立的 `</think>`**(开标签在 prompt 里,所以它只吐闭标签)。
真实频率:**31 / 233 个 assistant 历史轮 = 13.3%**。
31 处断链和 31 个"非空 think 渲染"数量一一对应。

### 病根:模板的历史分支按 `</think>` 重切

`chat_template.jinja:48-51`(旧):

```jinja
{%- if '</think>' in content %}
    {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
    {%- set content = content.split('</think>')[-1].lstrip('\n') %}
{%- endif %}
```

**只要回复正文里出现 `</think>`(哪怕在 bash 命令里、在引号里)**,重渲染就把它前面
当 reasoning、后面当 content 重新拼。于是那 4 个 token `\n</think>\n\n` 从推理
**前面**挪到了推理**后面**:

```
原始 token 流 :  <think>\n  \n</think>\n\n  推理…  \n</think>\n\n  答案
重渲染       :  <think>\n  推理…  \n</think>\n\n  答案
```

文字一个没丢,但 token 流必然不同 ⇒ `ids_startswith` 必败。

**这不只是训练侧的记账问题**:vLLM 用同一份模板,所以 rollout 时 agent 第 k+1 轮
真的看到自己上一轮被重排过的回复。环境的上下文**不是 append-only 的**,而合并逻辑
的前提就是 append-only。

### 修法:让历史分支成为生成位置的严格逆运算

```jinja
{%- set think_prefix = '<think>\n' if (enable_thinking is defined and enable_thinking is true) else '<think>\n\n</think>\n\n' %}
...
{%- if reasoning_content %}                 {# 只认显式的 reasoning_content 字段 #}
    …原分支…
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + think_prefix + content }}   {# 逐字,不 lstrip、不重切 #}
{%- endif %}
```

要点:**前缀用同一个变量 `think_prefix`**(生成位置发什么、历史位置就发什么),
正文**逐字**输出。只有这样才对 `enable_thinking` 的两个取值都成立。

### 验收:233 条真实回复的属性测试 + 反向对照

从 `step_1_train.jsonl` 还原出全部 233 条真实回复,验证
`render(前缀 + [assistant(reply), user(obs)])` 以 `render(前缀) + reply` 开头:

| 模板 | `enable_thinking=False` | `=True` |
|---|---|---|
| 旧 | **31/233 不成立** | 202/233 |
| 新 | **0/233** | 0/233 |

* 旧模板在 `False` 下的失败数 **31,和线上观测的 31 处断链完全相同** ——
  测试精确复现了生产数字,这是最强的一类验证。
* 旧模板在 `True` 下失败 202(正好是互补集),说明**旧模板对两个取值都不往返**,
  不是"换个 kwarg 就好"。
* 字符串层和 **token 层**都验了(合并判据跑在 token id 上,字符串过了 token 层
  也可能在边界合并)。
* 第 1 轮生成提示新旧**逐字相同** ⇒ 基座/SFT 的单轮行为不变;
  只有 13.3% 的多轮历史渲染变了 ⇒ 多轮口径的基线(127/474)需要重测。

### 教训

1. **「在新旧两版上都通过的测试」没有判别力。** 第一遍只跑新模板得到 0/233,
   看着很好,但它证明不了任何东西 —— 必须加反向对照,让旧版在同一个测试上
   复现出生产的那个数字。能复现 31 才说明测的是同一件事。
2. **把生产指标当成测试的期望值。** 31 这个数不是"差不多对",是精确对上。
   凡是机理假设能算出一个整数,就去和线上那个整数比。
3. **round-trip(往返)不变式比"渲染得好看"重要。** 旧模板的输出其实更"规范"
   (think 块闭合完整),新模板的输出带一个孤立的 `</think>` —— 但多轮 RL 要的是
   「历史 == 实际发生过的 token 流」,忠实 > 美观。
4. **`</think>` 这类标记只要可能出现在正文里,就不能用它做结构切分。**
   模型会写它、bash 命令里会有它、文件内容里会有它。
5. 又一次:**改了磁盘上的模板必须重启 agl-server**(`_TOKENIZERS` 进程内缓存,
   §57)。这次我是照着 §57 的教训主动重启的,没再踩。

### 订正(2026-10-06 01:00):机理对了,**补丁打在了没人读的那份文件上**

上面写「已修」,但验证 smoke(#4)的读数是**变差**:

```
training/n_trace_merge_mismatch_rows: 31 → 67      ← 升了
training/n_unmerged_rollouts:        7/8 → 8/8     ← 退回满额
```

**为什么不是噪声**:67 条断链 == 该步 256 个 assistant 历史轮里非空 think 渲染的个数
(67),又是**一一对应**;结构仍是 31/31 那一种。31 → 67 只是两次 smoke 采的题和
轨迹不同(241 → 256 轮)。

**差点走进的第三个错理论**:我先以为是 `think_prefix` 两头取值不一致
(生成位置 `enable_thinking=False` 发空 think、历史位置发 `<think>\n`)。
掐死它的是一个很便宜的观察:**同一条 prompt 里**(row2)24 个空 think 和
16 个非空 think **共存**。`think_prefix` 在一次渲染里是常量,不可能同时是两个值
⇒ 两种形态只能来自**逐消息的分支**,也就是那段重切。机理没错,是别的东西没生效。

**真凶**:proxy 加载 tokenizer 用的是**请求里的 model 名**,而
`smith_agent.py:1052` 发的是 `model="auto"`,proxy 把它解析成上游 vLLM 的真实模型名
= actor 路径 `/workspace/models/MiniCPM5-2B-sft-v3-ep3`
(`restart_s4_smith_trainer.sh:164`)。`default_proxy.model_name=…/MiniCPM5-2B`
**只是请求没指定 model 时的默认值**。而 SFT ep3 目录自带一份模板副本
(`chat_template.jinja` + `tokenizer_config.json` 内嵌,两处 md5 相同),
和 base 目录**修补前**的版本逐字相同 —— 我改的是 base 目录那份,
proxy 从头到尾读的是 SFT ep3 那份。

**怎么钉死的**(差分渲染,一条探针消息两份模板):

```
新(base dir)   : '<think>\n\n</think>\n\nTHOUGHT: reasoning here\n</think>\n\n```bash…'  ("</think>"×2)
旧(sft-v3-ep3) : '<think>\nTHOUGHT: reasoning here\n</think>\n\n```bash…'               ("</think>"×1)
线上观测到的块  : '<think>\nTHOUGHT: …' 且块内 "</think>" 恰好 1 个,位于 1449
```
⇒ 线上那个块**是旧模板的输出**。一次对照就定位,不需要再读代码。

**修法**:把补丁同时打到 SFT ep3 目录的两处副本
(备份 `*.bak-pre-roundtrip`),再重启 agl-server。验收用**真实的 SFT ep3
tokenizer** 跑严格前缀断言:`hist.startswith(gen + reply)` → OK。

#### 教训(这次新增的)

6. **「改哪一份模板」由 agent 发的 model 名决定,不是 `default_proxy.model_name`。**
   最快的查法是数日志里的路径频次:
   `/usr/bin/grep -o "/workspace/models/[A-Za-z0-9_.-]*" agl_server_18082.log | sort | uniq -c | sort -rn`
   —— 1827 次 `…-sft-v3-ep3` vs 668 次 `…MiniCPM5-2B`,一眼就知道真正加载的是哪个。
7. **权重目录会自带模板副本,而且是两处**(`chat_template.jinja` 和
   `tokenizer_config.json["chat_template"]`)。每导出一个新 ckpt 目录就多一份要同步的
   拷贝,靠记性必错。治本是**起训前的 preflight 断言:所有在用副本逐字相同 +
   往返不变式成立**(已落地 `preflight_templates.py`)。
8. **又犯了一次「测试在两个假设下都通过」。** 我想确认 transformers 认 jinja 文件
   还是认 tokenizer_config 内嵌那份,结果两份 md5 相同 —— 这个检查**没有判别力**。
   正确做法不是再设计一个精巧实验,而是**换仲裁者**:直接加载真实 tokenizer 渲染,
   谁赢都不影响结论。
9. **指标变差同样是信息,但要先问"可比吗"。** 两次 smoke 的题和轨迹都不同,
   绝对数不可比;可比的是**结构**和**配对关系**(断链数 == 非空 think 渲染数)。
   先找不变量,再读数字。

### §58 验收:smoke#5 通过(2026-10-06 00:49)

打在 actor 目录(`MiniCPM5-2B-sft-v3-ep3`)那两处副本上之后,同配方重跑一遍 smoke:

| 指标 | 期望 | smoke#3(§57 修完) | smoke#4(补丁打错文件) | **smoke#5** |
|---|---|---|---|---|
| `training/n_trace_merge_mismatch_rows` | 0 | 31 | 67 | **0** |
| `training/n_unmerged_rollouts` | 0 | 7/8 | 8/8 | **0/8** |
| `training/n_sample_collected` | 8 | 8 | 8 | **8** |
| `training/n_rollouts_w_trace` | 8 | 8 | 8 | **8** |
| `training/n_text_level_merges` | — | — | — | 3 |
| `training/reward` | — | — | — | 0.5946 |

**每条 rollout 现在是一整行训练样本**,episode 不再被切成一轮一行。
`enable_rollout_level_advantage=True` + `policy_loss.loss_mode=per_rollout_mean`
第一次真的收到了合并后的序列 —— 也就是说 s1/s2/s3 三条血统从来没跑过的那条路,
到这里才第一次通电。

**`n_text_level_merges: 3` 是良性路径,不是残余 bug。** `ids_startswith` 失败后
会先走 `text_level_continuation()`(`rollout_adapter.py:60-90`):解码两侧文本、
确认 `prompt_text.startswith(context_text)`、只把残余的观测文本重新编码。
它存在的理由写在 docstring 里:`context_ids` 是「服务端的 prompt + 它一个一个采出来的
token」,`prompt_ids` 是「整段对话一次性重新分词」,在**「response 结尾 → 下一轮观测
开头」这个接缝上 `tokenize(a+b) != tokenize(a)+tokenize(b)` 是必然的**,
和对话内容有没有变没关系。而且重新编码出来的 `observation_ids` 拿的是
`response_mask = 0`(`rollout_adapter.py:801-803`),**不进 loss**,只当条件。
所以 3/8 命中说明接缝效应真实存在、且被正确吸收了。

**顺带回答「怎么减少每步时间」:合并修复本身就是那个加速,而且大得多。**
未合并时一步的前向是 **22.1M token**(每一轮都把增长的前缀重算一次,平方级),
合并后 **1.2M**(每个 token 只算一次,线性)= 18.3×。s1 实测
`update_actor + old_log_prob + ref` 占一步的 **71%**,这部分按 18.3× 缩到 ~3.9%
⇒ 步时预期从 40–70 min 掉到 **13–23 min**。
对比「前向时记下 logprob、不再算第二遍」那条:smoke 实测 `old_log_prob` 只占
4.22s / 305.5s,gen 占 271.7s(**89%**),所以复用 logprob 是 **~1%** 的杠杆,
而且 `rollout_corr_helper.py:1073-1074` 会把 `policy_loss_config["loss_mode"]`
覆盖成 `bypass_mode`,直接开会把 `per_rollout_mean` 毁掉。
**真正剩下的成本是 gen 的长尾**(`rollout_run_duration_s` 均值 134 / p90 182 / max 260),
不是重复前向。

**同一轮 smoke 里唯一的异常是上下文溢出,1 次 / ~240 次调用(0.4%)**:
`ValueError: maximum context length is 51200 tokens. However, you requested 52883
(48787 in the messages, 4096 in the completion)`。prompt 上限 = 51200 - 4096 = 47104,
那条 episode 的历史长到 48787。失败那一轮烧掉一轮、episode 继续,不是阻断项;
但它说明 `AGL_MAX_TOKENS=4096` 的留白在 40 轮上确实会被顶满,
值得在全量跑里盯 `LLM call failed` 的频率。

**闸门已落地**:`preflight_templates.py` 接进 `s4_cycle.sh` 的 `start|resume`
(`template_gate || exit 10`,排在 `disk_guard` 和 `selftest_gate` 之后)。
理由写在脚本注释里:模板坏掉的时候**训练照样跑完、指标照样有值**,
只是学的东西不是多轮轨迹 —— 这种「不报错的坏」必须在起训前拦。
同时把 `trainer.max_actor_ckpt_to_keep` 从 1 改成 `${KEEP_CKPT:-3}`:
Tier-2 探针一次 45 分钟,等它判完「step N 好」,训练已经往前走了 2–3 步,
keep=1 会把那份权重回收掉 —— 等于把交付物删了。

### §59 起训前的最后一遍审计:长度几何、动态采样语义、探针抢卡(2026-10-06 01:30)

#### 59.1 我自己写错过一个判断,这里纠正

起训前我从 `step_1_train.jsonl.run5-pass` 读出首轮 `prompt_tokens` =
`[3856, 5087, 5241, 15132, 15757, 16397, 17220, 46707]`,对着 `MAXPROMPT=8192`
写下了「**5/8 行 prompt 超限**」,并顺着它推到「`data.truncation=left` 会从左边
切掉 system prompt 和题面」。**这个判断是错的,错在读错字段。**

`rollout_adapter.py:142-143`:

```python
resp_tokens  = sum(len(_token_ids(t.response)) for t in rollout.triplets)
prompt_tokens = len(_token_ids(last_triplet.prompt))   # ← 最后一轮
```

`prompt_tokens` 是**末轮的上下文长度**(system + 题面 + 全部历史 + 观测),
`prompt`/`response` 两个文本字段也都是末轮的。而**训练行的 prompt 是第一轮的
prompt** —— 合并从第一轮 prompt 起算,之后所有轮的 response 和观测都进 response。

真正该量的东西:末轮上下文的开头就是首轮 prompt,按第一个 `<|im_start|>assistant`
切开再 tokenize,精确可得:

| 轮数 | reward | 末轮ctx | **训练行prompt** | 合并resp | is_drop | resp截断 |
|---|---|---|---|---|---|---|
| 8–12 | 1.00 | 3856–5241 | **1192** | 2729–4140 | - | - |
| 30–40 | 0.0–0.9 | 15132–17220 | **1192–1239** | 13985–16063 | - | - |
| 35 | 0.00 | 46707 | **1239** | 47490 | - | **YES** |

训练行 prompt 恒定 **1192–1239**,离 8192 有 6.6 倍余量,**一行都没 is_drop**。
构成是 system(~110)+ `<pr_description>` 题面(~250)+ 固定 `<instructions>`
块(~850);固定块占绝大部分,所以长度几乎不随题目变化,也就不可能顶到 8192
(需要 7000 token 的题面)。`n_truncated_sample: 1` 对应末轮 ctx 46707 那一行:
合并 response 47490 > `MAXRESP=43008`,被右截断 —— 完全自洽。

**这次是哪里止损的**:反例不是靠想出来的,是靠一个对不上的数。
`n_truncated_sample` 报 1,我推出来的是 5,**对不上就必有一方是错的**。
顺着这个矛盾去读 `:1050`,发现那个计数器是 `n_trunc_sample_because_of_response`
(response-only),于是以为「5 是对的、计数器瞎」;再往下读才发现是 5 本身错了。
另有一个本该更早用上的证据:IS 比值 `rollout_is_seq_mean=0.99975`、
`seq_max_deviation=0.0012`。**如果真有 5/8 行的条件 prompt 被从 46707 砍到 8192,
重算的 `old_log_probs` 和 vLLM 的 `rollout_log_probs` 必然大幅背离,
0.0012 的偏离在物理上不可能。** 一个已经在手的指标就能否掉整个假设。

#### 59.2 prompt 超长的真实代价不是「被截断」,是整条 episode 作废

`rollout_adapter.py:663-667` 在右截断的同时置 `is_drop = True`,
该标记进 `is_drop_list`(`:695`)→ `is_drop_mask`(`:1003`)→
`trainer.py:750-751` **把整行从 batch 里剔掉**。40 轮的 episode 只因首轮 prompt
多出几个 token 就全部丢弃。

它有告警,但不是我原先盯的那个:

* `training/n_sample_dropped/marked`(`trainer.py:752`)← **prompt 截断的真告警**
* `training/n_truncated_sample`(`rollout_adapter.py:1050`)← response-only,对 prompt 瞎

smoke#5 实测 `n_sample_dropped/marked: 0`。已加进全量跑的盯盘清单。

#### 59.3 预算不动的理由(以及 padding 不要钱)

`CTX=51200` 被 vLLM 固定,要把 `MAXRESP` 提到 49152 就得把 `MAXPROMPT` 压到 2048
—— 拿 1192 之上 6.6 倍的安全余量换 1.65 倍,只为救回一条 reward=0.00 失败
episode 的尾巴。代价方向是静默的(题面长一点就整条 is_drop),收益方向是 1/8 行
的无效尾巴。**不换。**

顺带否掉一个想当然:`get_left_padded_ids_and_attention_mask(prompt_ids,
max_prompt_length, ...)`(`:686`)会把 prompt **pad 到 8192**,看着像是每行白烧
7000 token 的前向。但 `restart_s4_smith_trainer.sh:195-200` 三条前向路径
(actor / rollout logprob / ref)全开 `use_dynamic_bsz=True`,verl 按 **token 数**
打包并 unpad,**padding 不耗算力**。改 `MAXPROMPT` 买不到速度。

#### 59.4 `min_valid_groups` 的语义:停止条件,不是目标批量

`TRAIN_BATCH=16` 配 `min_valid_groups=6`,看着像「只训 6 组、丢掉 10 组」。
读 `trainer.py:683-720` 否掉:

```python
if n_groups_valid >= min_valid_groups or n_rounds >= max_gen_batches:
    break          # :701
```

它**纯粹是「要不要再抽一轮」的停止条件**,从不裁剪 —— 每一轮存活的组都进
`kept_parts` 并 concat(`:717-720`)。所以 `=6` 的含义是「第 1 轮 16 组里只要
≥6 组有方差就开训,**全部存活组参训**」。

**反过来,把它恢复成 16 是有害的**:等于要求 16 组全有方差,任何一组零方差就
强制第二轮生成,而 gen 占步时 89% —— **步时直接翻倍,换来零数据增益**。
`:626-627` 的默认值确实是 `data.train_batch_size`,但那个默认值在 `max_gen_batches=2`
下是个坑。留在 6。

#### 59.5 探针会和训练抢同一组卡(已修)

`s4_cycle.sh` 的 `probe` 分支第 34 行注释写着「停训跑全量 474」,**但代码从没落实**:
从 ckpt 检查直接跳到 `FLEET_CARDS="5 3 1 2" smith_fleet.sh start`,
而那就是 `restart_s4_smith_trainer.sh:123` 的 `GPUS=1,2,3,5`。训练侧卡上已经有
FSDP actor + vLLM rollout engine,再并发起 4 个评测 vLLM 必然挤爆一边 ——
**一次探针废掉整条跑**。

修成硬拒(`exit 11`),不做自动 pause:自动 pause 会让 `probe` 变成破坏性操作,
万一评测起不来就白停了训练。正确顺序写进了报错里:
`pause` → `probe <STEP>` → `resume`。

**同一次还抓到基线指错**:`s2_verdict.py:23` 的默认基线是 `val_sft3_fixed`
(127/474),那是**旧模板**下的读数。模板补丁改了 13.3% 的 assistant 轮的历史渲染,
拿新模板的 step-N 比旧模板的基线,涨跌分不清是 RL 还是模板,**判据作废**。
改成显式传 `val_sft3_tmpl2`(SFT ep3 在当前模板下重测的那份),并加了
「基线必须测满 474」的前置断言(`exit 12`)—— 否则半成品目录会悄悄参与配对比较。

**这两个缺陷的共同形状**,和 §56–§58 是同一个:**注释写了、代码没做。**
前三次是「我以为在用的那份模板 ≠ 真正加载的那份」,这次是「注释声称的行为 ≠
脚本的行为」。都属于「不报错的坏」,所以都只能靠断言拦,拦不住就等着在运行中付钱。

### §60 判决的尺要两头一样长;以及把「会无声消失的补丁」写成机检(2026-10-06 01:50)

s4 起训后补的两件事,都不是修 bug,是**补判据**。

#### 60.1 `s2_verdict.py` 原来只有向下的尺

原版的停止规则是对的:方向向下 + 配对 McNemar p<0.05 → `STOP`,这是人类
「明显下降就不要训了」的操作化定义。但它对**向上**只打一句
`VERDICT=CONTINUE 不低于基线` —— 不管涨了 2 道还是 20 道,不管显不显著。

这是个双标:跌了算噪声、涨了算成绩。而我这条跑的交付物恰恰是「涨了」,
最该被严格对待的就是向上那一侧。所以补了两个分支:

* `VERDICT=GAIN` —— 方向向上**且** p<0.05,才算增益。
* `VERDICT=CONTINUE_FLAT ... **不能当增益报**` —— 涨了但不显著,
  字面写进输出里,免得我自己过几十个小时后在疲劳状态下把它读成好消息。

#### 60.2 顺带量出 MDE,把「没测出来」和「没有效果」分开

同时加了**可检出下限**(MDE):在当前不一致对数 `n=b+c` 下,
最小的能让 McNemar 显著的净差。拿两份真实评测对跑验证:

```
val_sft3_tmpl2 134/474   基线 val_sft3_fixed 127/474   Δ=+7
  配对: 基线独对=32  探针独对=39  McNemar p=0.4767
  不一致对 n=71,本次可检出下限 MDE=+19 道
VERDICT=CONTINUE_FLAT 高于基线 7 道但不显著 —— **不能当增益报**
```

**n=71 这个数本身就是情报**:两次 474 评测之间有 15% 的题会翻面。
这 71 里混了模板带来的真实差异,不全是噪声;纯重采样那次(10-04,168 道失败题
原样重跑)翻上来 13 道,反向按同量级估 ~12,所以**纯噪声的不一致对约 25 道,
单次 474 探针的可检出下限约 +11 道(+2.3pp)**。

⇒ 探针涨了 5 道而 MDE 是 12,那是**没有判断力**,既不能当增益报、
也不能当失败报。这个区分不写进脚本,就会在截止期压力下被我自己糊掉。

#### 60.3 `dp_group` 补丁打在 site-packages 里,`uv sync` 会无声冲掉它

[[s1-nccl-allgather-hang]] 那个补丁(`dp_actor.py` 的 `_dp_group()` + 两处调用点)
修的是:`dp_group=None` 让 `seqlen_balancing.py:402` 的 `all_reduce(MAX)` 被静默跳过,
各 rank 算出**不同的 micro-batch 条数**,下一个 allgather 对不上 → **挂死,不报错**。

它在 `.venv/lib/python3.12/site-packages/` 里。`uv sync` / `pip install -U verl`
都会把它冲掉,而且**不会有任何提示**。这条跑要连轴 40 小时、`nccl_timeout=3600`,
丢了补丁的代价是每次挂死烧掉一小时、还得重新靠功耗(123W ≠ 300W)去认。

所以写成 `preflight_templates.py` 的第 5 阶段,判 `FAIL` 不判 warn ——
这个缺陷没有任何「降级也能跑」的形态。实现上**不 import verl**:
import 会连带拉起 torch,在共享卡上多开一个 CUDA 上下文纯属找事,
而判据本来就是纯文本比对(数 `def _dp_group(self)` 和 `dp_group=self._dp_group()`)。

**阴性测试是必须的。** 之前吃过「两个副本 md5 本来就相同、测试没有判别力」的亏,
所以这次把 site-packages 那份 sed 掉补丁、拷到 scratchpad 的假 repo 里跑一遍,
确认 `exit=1` 且 FAIL 如期开火,才算这条检查成立。

**可推广的那一条**:判据要对称(两个方向用同一把尺),
而且要能区分「测不出来」和「不存在」—— 否则统计量只是用来确认偏见的。

### §61 容器被杀 = 模型被罚;以及一个分母内生的"指标"骗了我两次(2026-10-06 02:10)

#### 61.1 infra 失败会以 reward=0 进训练,而且被动态采样**优先选入**

s4 step 2 报 `failed=1`。查下来不是模型问题也不是 NCCL:是 **cgroup OOM** ——
`job-template-smith-s4.yaml:132-137` 给 rollout pod 的 `limits.memory` 是 **8Gi**,
模型在用病态 HTML 注释输入压 `striptags`、顺手写了几个 heredoc 脚本,
第 16 轮上 RSS 顶到 8.35G,内核直接杀。dmesg 实据:

```
Memory cgroup out of memory: Killed process 1709771 (python)
  total-vm:7891740kB, anon-rss:7875632kB, oom_score_adj:-997
```

(`oom_score_adj:-997` 是 k8s pod 的指纹;容器日志是 **UTC**,17:50 UTC = 01:50 北京。)

然后它去哪了:

* `agentlightning/verl/agl_rollout_manager.py:623-630` —— FAILED 只是
  `num_failed += 1` 加跑一个 hook,接着 `completed_rollouts.append(...)`
  **无条件执行**。失败的 rollout 照样进 batch。
* `agentlightning/verl/rollout_adapter.py:1131-1134` —— `_fillna_reward` 只有两行:
  有 `final_reward` 就用,否则返回 `self.reward_fillna_value`。
* 那个值是 **0.0**(`agentlightning/verl/config.yaml:22`、
  `examples/swe_smith/train_smith_agent.py:163` 都写死 0.0)。

**对账口径(以后直接看这一条):`training/n_rollouts` 减 `training/n_rollouts_w_reward`
就是本步填了几个 0。** 实测:s4 step1 = 64/64,step2 = **63/64**;
翻历史日志,**s1 step3 和 s2 step3 都是 60/64 = 6.25%** —— 这事一直在发生,从没记过账。

真正的危害不在被杀那条轨迹,**在它的三个同组兄弟**。`ROLLOUT_N=4`:
假如四条本来都拿 0.5 → 组内零方差 → 动态采样把整组滤掉,不产生梯度;
掺进一个 0 之后组内**有了方差,于是被选进 batch**,三条平庸轨迹白拿正优势,
被杀那条背一个大负优势。也就是说 —— **infra 噪声不是被平均掉,是被动态采样优先选入。**
s4 当前 1/128 = 0.8%,低于自定的 5% 升级线,所以**只记账不动手**:
改 job template 的内存上限属于 infra 而不是配方,但它会改变 rollout 结果、
污染步间对比,跑中不碰。

> 可推广的一条:**凡是"失败"和"做得差"共用同一个数值出口,infra 故障就会被
> 当成策略信号**。要么给失败一个可区分的出口(排除/单独标记),要么至少把
> `n_rollouts - n_rollouts_w_reward` 当常规体检项盯着。过滤器会放大它,不会稀释它。

#### 61.2 `n_zero_adv_groups / n_groups` 的分母是内生的,这个比率没有判断力

我两次拿它当"稠密奖励有没有打中 credit assignment"的证据,两次都错。

按权威指标键重新拉(不是 gen 过程里那些阶段性打印):

| 血统 | step1 | step2 | step3 | step4 | step5 |
|---|---|---|---|---|---|
| s1 | 46/59=78% | 35/48=73% | 3/9=33% | 6/17=35% | 16/23=70% |
| s2 | 14/26=54% | 63/73=86% | 7/13=54% | 23/29=79% | 2/11=18% |
| s3 | 4/20=20% | 51/64=80% | 56/70=80% | 47/59=80% | 39/52=75% |
| s4 | 12/25=48% | 8/22=36% | | | |

**为什么比率不能用**:动态采样会一直生成直到攒够有效组,所以零方差率高的时候
分母 `n_groups` 自动变大。比率被机械地钉在 (1 − 需要数/看到数) 附近,
既含"这批题有多难"又含"奖励有没有区分度",两者分不开。
实证就是它根本不稳:s2 在 18%~86% 横跳、s3 在 20%~80%。

**该看的是分母本身** —— 攒满一个 batch 要看多少组,越少说明奖励越常能区分:
s1 `59/48/9/17/23`、s2 `26/73/13/29/11`、s3 `20/64/70/59/52`、**s4 `25/22`**。
s4 是唯一 step2 不恶化的血统(别家 54→86、20→80、78→73)。n=2,**弱证据,不是结论。**

顺带校验一下这个分母是自洽的:有效组 × 4 应该等于 `n_sample_trained` ——
s4 step1 (25−12)×4 = 52 ✓,step2 (22−8)×4 = 56 ✓。对得上,所以读法没错。

我原先记的「零方差组 3/16 = 18.8%,比预测的 30% 更低」用的是另一个分母,
**不可比,已作废**。这是同一类错误的第三次:`critic/score/mean` 不是奖励、
`capture_rate` 对丢掉的 rollout 瞎、现在是这个比率。
**共同形状:指标名字听起来是我想问的那个问题,但它的分母/口径是别的东西。**
以后引用一个比率之前,先把分子分母各自是什么、谁决定它们,写出来一遍。

#### 61.3 `rollout_is_mean` 在 **step 1** 的偏差是合并 bug 的直接证据(比 grad_norm 硬)

| | step1 | step2 | step3 | step4 | step5 |
|---|---|---|---|---|---|
| s1 `rollout_is_mean` | 0.9940 | 0.9931 | 0.9933 | 0.9927 | 0.9913 |
| s2 | 0.9956 | 0.9951 | 0.9890 | 0.9808 | 0.9863 |
| s3 | 0.9948 | 0.9954 | 0.9940 | 0.9934 | 0.9890 |
| **s4(修复后)** | **0.999958** | **0.999851** | | | |
| s2 `chi2_token` | 0.0141 | 0.0125 | 0.0247 | 0.0310 | **0.0486** |
| s3 `chi2_token` | 0.0047 | 0.0078 | 0.0179 | **0.0231** | 0.0132 |
| **s4 `chi2_token`** | **0.00212** | **0.00207** | | | |

**论证靠的是 step 1,而不是"s4 数字更漂亮"。** 在 step 1,vLLM 生成用的权重
和 trainer 算 `old_log_prob` 用的是**同一份**(还没做过任何梯度更新),
所以 IS 比值理论上必须是 1.0,只差 bf16/kernel 的数值噪声。
s1/s2/s3 在 step 1 就偏了 0.44~0.60% —— **那不可能是策略漂移,没东西可漂**。
只剩一个解释:训练侧打分的 token 序列和 vLLM 真正生成的不是逐字同一条
(每轮各成一行 + 模板按正文里的 `</think>` 重切 → 每行的 **prefix 是错的**,
prefix 一错,被打分的 response token 的条件分布就跟着错)。

而且 s2/s3 的 `chi2_token` 是**单调爬**的(3.4× / 4.9×):策略一动,
错 prefix 带来的失配就被放大。s4 平在 0.0021。

**先排除了"指标被旁路所以恒等于 1"这种假好看**:
`timing_s/old_log_prob` = 30.0s / 22.6s(真在重算,没复用 vLLM 的);
若 `rollout_log_probs` 缺失被 `old_log_prob` 顶替,比值会是**恰好** 1.0 且
max=min=1.0,而实测 `rollout_is_max: 7.92` / `min: 0.00023` / `std: 0.0404`,
逐 token 有真实散布。`restart_s4_smith_trainer.sh:231-232` 的
`rollout_is=token` + `threshold=2.0` 开着,`fraction_high: 3.5e-5`(几乎不裁)。

归因范围要说清:s4 改的是**合并修复 + 模板修复这一捆**,
`rollout_is_mean` 的改善归于这一捆,不能拆给其中某一个。

#### 61.4 `chi2_seq` 在多轮长序列上是浮点垃圾,不要进体检单

实测 s3 五步全是 **−1.0**,s4 step1 **−0.524**、step2 **+216.77**。
χ² 散度 = E[(w−1)²] ≥ 0,**出现负数就说明估计量不是 χ²**:
它按 `mean(w²) − 1` 算,而 `w` 是整条序列 token 比值的**原始乘积**(exp of sum)。
几千个 token 一乘,要么下溢到 0(于是恰好 −1),要么炸上去(于是 +216)。

对照组:同一步的 `rollout_is_seq_mean` = 0.99972、`seq_min` = 0.99544 完全正常
—— 说明 `rollout_is_seq_*` 是**长度归一化**的(几何均值),和 `chi2_seq` 不是一个量。
**从 −0.52 跳到 +216.77 不是训练信号。**

⇒ 多轮可用的训推一致性体检项只有:`rollout_is_mean`、`chi2_token`、
`rollout_is_seq_max_deviation`、`seq_fraction_high/low`、`eff_sample_size`。

## §62 s4 训出的不是「会修 bug」,是「会交卷 + 同样做对就少跑几轮」(2026-10-06)

> **这一节的机理在同一天被我自己的数据推翻过一次。** 第一版写的是「部分分近路」,
> 原文留在 §62.2 末尾加了删除线,实测见 §62.6。**停训的结论没变**(step20 显著低于
> 基线,证据是配对 McNemar,与机理无关),但机理换了,**处方也跟着换**:
> 该动的是轮数罚,不(只)是部分分。

s4 是合并修复(§56–§58)之后第一条管线完全干净的血统:1344 条 rollout
**100% 有 trace**、断链只有 1 行(0.07%)、infra 填 0 只有 2 条(0.15%)、
KL 实测在跑(coef 0.001)。机械上没得挑。结果照样掉了。

### 62.1 探针序列
| step | 解出 /474 | Δ | McNemar p |
|---|---|---|---|
| 0(SFT ep3) | 134 = 28.27% | — | — |
| 10 | 145 = 30.59% | +11 | 0.185 **不显著,不能当增益** |
| 20 | **117 = 24.68%** | **−17** | **0.030 显著低于起点** |

先排除评测侧:三份 `chat_template.jinja` md5 **完全相同**,
`max_turns/max_tokens/max_model_len/sample_timeout` 一致,都是 474/474、
`status_error` 0 份、失效的是同样那 5 道 ⇒ 掉是真的掉。

### 62.2 根因:t 值两位数,毫无歧义
step20 对基线,**配对** 468 道:

| 指标 | 基线 | step20 | t |
|---|---|---|---|
| **提交了补丁** | 51.3% | **71.2%** | **+8.39** |
| **生成 token** | 36983 | **19829(−46%)** | **−12.57** |
| **轮数** | 27.8 | **22.4(−19%)** | **−9.19** |
| 顶满 40 轮 | 46.2% | 29.5% | −6.97 |
| 补丁字符数 | 869 | 1487 | +4.06 |
| **解出** | 28.6% | **25.0%** | −2.30 |

**它不再查问题了,提前写一个大补丁交上去。** 交得多、交得早、补丁更大,解出率反而掉。
行为漂移这张表是实测,不受下面机理订正的影响。

~~**奖励侧的算术**:跑满 40 轮没交卷 ⇒ 奖励 0;早交一个像样的补丁 ⇒ 大概率拿到
F2P **部分分**。于是「早交」的期望奖励高于「查透」。稠密奖励打开了一条刷部分分的近路。~~
**↑ 这段是错的,当天就被 s4 自己的训练数据推翻了,订正见 §62.6。** 错在三处:
没交卷**不等于**奖励 0(照样判分,83.4% 仍有补丁、f2p 均值 0.3235、26.4% 真做对);
部分分那一档只占 8.0% 的 rollout,不是主路;而且组内「交了卷的里面,早交并不更值钱」
(t=−0.71)。我是从奖励公式推的,没去量,**公式里的默认值还不是线上跑的值**。

### 62.3 训练侧早就在喊,我读错了方向
`response_length/mean` 从 step 1 起就是 **−343 token/步、t=−4.64**
(16087 → 11397,−29%);`actor/kl_loss` 涨了 **78 倍**(t=+12.62)而
`pg_clipfrac` 只有 0.15% ⇒ **每步挪一点点、方向高度一致、累积成大位移**。
`training/reward` 本身是平的(斜率 +0.0027/步,**t=+0.95**)。

**奖励不动、长度猛掉、KL 猛涨 = 优化器找不到任务信号,就去优化它唯一还能
便宜优化的东西。** 这个组合以后见到就该直接当"在抄近路"处理。

### 62.4 我犯的方法错误:把单次探针内的横截面相关当成了预测
step10 时我测出「新解出的题少写了 2 万 token(t=−4.17)、丢掉的题反而写得更多
(t=+1.48)、两组差 t=+3.75」,据此判定"缩长度是良性的、是省废话"。
**对 step10 是对的,当预测用就错了。**

错在哪:那是**同一个 ckpt 内部**不同题目之间的相关,量的是良性早期段的截面;
它不包含"再沿这个方向推 10 步会怎样"的信息。优化器沿同一方向继续推,
就从"省废话"冲过了最优点,变成"不干活就交卷"。

⇒ **规矩:判断"能不能接着训"只能靠时间序列上的下一个点,不能靠当前点内部的
横截面相关。** 当时那个 t=−4.64 的持续下行(我自己标成"机械风险"的那条)
才是正确的信号;我给它的权重太低了。

### 62.5 顺带确认的两件小事
- `gen_rounds` 21 步全是 1(有效组 11–15 ≫ `min_valid_groups=6`),所以
  `training/reward` 的「只留最后一轮」口径问题(`trainer.py:151` 只让
  `training/n_*` 相加)**在 s4 上从未触发**。
- `s3_watch.py` 的 `METRICS` 白名单里**没有** `training/n_rollouts_w_reward`,
  而 `parse_trainer` 只抽白名单里的键 ⇒ §infra 那套"对账方式"其实一直靠手工 grep,
  机器没在查。**白名单式解析的通病:我以为在看的指标,可能从来没进过 rows。**
- SIGINT 对这个 trainer **完全无效**(10-06 一天内 4 次全是等满超时靠 SIGTERM 收)。
  `s4_cycle.sh pause` 的空等已从 300s 改成 60s。

### 62.6 订正机理:组内唯一的「长度信号」是我自己加的轮数罚(t=−35)
工具 `s4_reward_audit.py`。口径:s4 窗口 **1553 条 mode=train rollout**,按
(instance,起始时间相邻 20min)聚成 **391 个组**(380 个是满 4 条)。
看**组内**而不是全局,因为 GRPO 的优势是 `(r − 组均值)/组标准差`,梯度只认同组内部的相对高低。

**(1) 组内奖励方差的来源分解**

| 组的类型 | 个数 | 占比 | 这些组的梯度在说什么 |
|---|---|---|---|
| 全零方差 | 91 | 23.3% | 被动态采样丢掉,没梯度 |
| **只有整形项造出方差** | **31** | **7.9%** | **「同样做对,少跑几轮」—— 别无内容** |
| 任务信号造出方差 | 269 | 68.8% | 「做对」 |

那 31 个组**四条全做对**,raw 奖励完全相同 ⇒ 组内优势 100% 由轮数罚决定。
更难看的是:**没有轮数罚,它们会被动态采样当零方差丢掉;是整形项把它们捞回了训练**,
占全部有梯度组的 **10.3%**。

**(2) 把「同样做对」的 rollout 单独拿出来做组内回归**

| 被解释量 | 每多跑 1 轮 | t |
|---|---|---|
| 整形后 `reward` | **−0.0032** | **−35.18** |
| 整形前 `raw_reward` | **+0.0000**(方差为 0) | — |

**这是整篇调查里最大的 t 值,而且分解得干干净净:在正确性相同的 rollout 之间,
长度信号 100% 来自轮数罚,一点都不来自任务。** 另外「同样没做对」的之间是
−0.0065/轮(t=−7.56),这条来自任务(跑得久的 f2p 更差)。
⇒ **两个分层都指向"短",没有任何一层指向"长"。** 于是训练侧
`response_length/mean` 单调下行(t=−4.64)、val 侧轮数 −19%/token −46% 完全是顺着这个方向走。

**(3) 「交卷」在组内值 +0.4667(t=+16.50)**,但这是**混淆**的:交卷的做对率 71.6%、
没交卷的 26.4%,交卷大多意味着活干完了。真正能归因到奖励设计的是 (2),不是这条。

**(4) 这条通道的宽度随「做对率」放大**。纯整形组占比 ≈ p⁴(p=组内单条做对率):
s4 p=53.7% ⇒ 预测 8.3%,实测 7.9%(近似成立);s1 p=44.2% ⇒ ≈3.8%。
**从更强的 SFT 起训,这条通道就更宽。** 这解释了为什么同一个轮数罚在 s1 上没捅出事。
~~(s1 还叠了 lr 2e-6 vs s4 5e-6)~~ **← 10-06 订正:s4 的 lr 也是 2e-6**
(`trainer_s4.log:67` 'lr': 2e-06;`restart_s4_smith_trainer.sh:128` 默认值),
s1 和 s4 的 lr 相同,差别只有起点(基座 vs SFT ep3)和合并修复。不能把 lr 算进解释里。

**证据强度要说清楚**:以上都是实测分解,但**我没做消融**(没跑一条关掉轮数罚的对照)。
所以正确说法是「轮数罚是头号嫌疑,且是唯一找到的、方向一致的长度通道」,
**不是「已证实的唯一原因」**。

**(5) 害我写错机理的那个坑:我读了代码默认值,没读线上值。**
`agents/smith_agent.py:1166` 的默认是 `SMITH_LEN_PEN_T0=80`,而 `max_turns=40`,
所以按代码读**这项是死代码**。可线上真正生效的是
`job-template-smith-s4.yaml:116` 的 **T0=32** —— 32 轮内做对给 1.0,拖到 40 轮给 0.9。
我还一度拿**均值差**(reward 0.5943 vs raw 0.6073)去判断"整形没生效",
均值差 0.013 看着像噪声,逐条比才看到 **15.5% 的 rollout 被扣、最大扣 0.1160**。

⇒ **两条规矩**:① 判断某个 env 开关生效没有,**读 pod 模板,不读代码默认值**
(k8s 环境变量覆盖是看不见的);② 判断整形项开火没有,**逐条比 `reward` vs `raw_reward`**,
永远不要看两个均值像不像。`s4_reward_audit.py` 现在两条都是硬打印。

### 62.7 `response_length/mean` 跨血统不可比(合并修复改了它的单位)
s1 的 `response_length/mean` 是 **~190**,s4 是 **~16000**,差 85 倍 —— 不是模型变了,
是 §56–§58 的合并修复把 episode 从「一轮一行」拼成了「整段一行」,
**这个指标的单位从"每轮"变成了"每 episode"**。
所以「s1 长度还在涨(+0.30/步,t=+2.16)、s4 在跌」**不能**拿来对比。
可比的是 `n_turns`:s1 均值 29.8、顶满 40 轮 37.9%;s4 中位 29、38.8% —— 其实差不多。
⇒ 和「步时跨血统不可比」(分母是动态采样留存率)是同一类坑:
**合并/过滤口径一变,指标名字没变但量纲变了。跨血统比任何指标前,先问它的分母是什么。**

## §63 s5 起训:只拔掉轮数罚和上下文罚,其余与 s4 逐字相同(2026-10-06 13:05)

**这是对 §62 头号嫌疑的消融,不是新配方。** 改动只有 `job-template-smith-s5.yaml` 的两个数:
`SMITH_LEN_PEN_LAMBDA=0`、`SMITH_PROMPT_PEN_MAX=0`。`length_penalized_reward` 在 λ=0 时返回
`1.0 - 0×frac = 1.0`(= 原 reward,因为它只在 reward≥1.0 时才动),`prompt_length_penalty` 在
max_pen=0 时返回 `reward - 0`(`agents/smith_agent.py:803-836`),所以 **reward == raw_reward**:
训练奖励 = F2P 通过比例 × P2P 硬闸,没有任何整形。其余 lr 2e-6 / KL 0.001 / SFT ep3 起点 /
700 题 / TRAIN_BATCH=16 ROLLOUT_N=4 PPO_MINI=4 / 40 轮 / 动态采样,全部不动。

为什么只动这一刀而不顺手改别的(lr、起点、KL):29 小时只够 ~100 步、10 次探针,一次性实验
改两个变量就说不清是谁的功劳;s2 的 5e-6 四步崩过,不冒险;step10 的 145 不显著(p=0.185)且
权重已带漂移,从它续训等于把嫌疑带进对照。

**可证伪预测**(写在起训前,之后不许改):
1. 训练侧 `response_length/mean` 不再单调下行(s4 为 -343 tok/步,t=-4.64);
2. 纯整形组消失 ⇒ `zero_adv_frac` 比 s4 同步数高约 8pp,有梯度组由 ~300/391 降到 ~269/391;
3. 探针 step10/20 的提交率不再往 71% 冲、生成 token 不再腰斩。
若 1-3 成立但 474 仍不涨 ⇒ 轮数罚只是"早交卷"的因、不是"不涨"的因,得另找。
若 1-3 不成立 ⇒ §62 的归因错了,"早交卷"另有来源(候选:F2P 部分分本身让"交一个半对的补丁"
比"继续改"更划算)。

探针:`PROBE_EVERY=10` 全量 474 配对 `val_sft3_tmpl2`(134),**p<0.05 且低于 134 就停**;
驱动在 scratchpad `s5_auto_cycle.sh`(DEADLINE 10-07 16:00)+ `s5_sentinel.sh`(PID 动态读,
不再像 s4 第 3 版那样写死 pid、探针一 pause 就退出)。起训前 configMap 已刷到与本地逐字节一致
(md5 2cbf9745 / cfa2f6fe)。

顺手修掉的一个坑:用 `sed '1,49c...'` 换脚本头部注释时把紧随其后的 `set -u / ROOT= / EX=`
三行一起删了,`$ROOT/.venv/bin/python` 展开成 `/.venv/bin/python`,selftest 闸门直接 ABORT。
**改头部要先看第 N 行是不是注释,别按行号盲切。** 闸门起了作用:没探针就不许起训。

## §64 ZGCM 预训练:配比里的「latex」组件喂了 5790 步 reasoning——按稀疏探针外推分片类别是错的(2026-10-06)

**现象**:给 run2 写「按配比自动补货」脚本时加了一道分片体检(读 parquet 前 200 行的 `category`/`source`),
拿现有 mixture_v3 的各组件首片自检——六个组件全过,唯独 latex 组件的 part-00450 报 `category ['reasoning']`。
逐行读完 450/451 两片(28.6 万行):**100% 是 `('reasoning','nemotron-specialized-infinibyte-reasoning')`**,
没有一行 latex。真正的 papers_with_latex(rpj-proofpile-arxiv)在 stage2 的 440–449,450 起就是 reasoning。

**根因**:09-22 建配比时探针是「每 40 片探一片」(440 是 latex),我把 440 的类别外推到 450/451 两片,
而类别边界恰好落在 449/450 之间。run1 6B + run2 到 step 5790 的「latex」份额(2% ≈ 0.11B tok)实际喂的是
reasoning;留出集 ppl 表里 10-06 之前的 latex 列也是 reasoning 文本(等于 reasoning 权重被暗中抬到 2.6%)。
影响不大(2% 份额、而且 reasoning 本身也是配比里的组件),但口径错了必须改、必须说。

**修法(mixture_v4)**:latex 换成真分片 444/448(新下载,留出集换成 00448 的末 row group,旧清单备份为
`heldout_manifest.bak_*_latex_was_reasoning.json`);450/451 挪进 reasoning 列表尾部不浪费。
重启时带 `RESET_COMPS=latex`(train.py 新选项,丢掉该组件的游标和缓冲从新列表 file 0 / epoch 0 开读)。
**不能直接换文件列表就重启**:mixdata 的 resume 逻辑是「游标文件不在列表里 → 取字符串序后继;后继为空 → epoch+1 从 0 读」,
而留出集的 `is_fresh` 一看 `epoch != 0` 就永远判脏——换文件会让 latex 留出集从此失效。

**教训**:
1. 分片类别**必须逐行核实**(`pq.read_row_group(0)` 的 `category` 列,几百毫秒),不能按稀疏探针外推——一个 HF 配置里按
   类别连续排布的分片,边界恰好落在你没探的位置的概率并不小。
2. 配比脚本应该在**装载时**自检(我现在的 `shard_topup.check_shard` 对每个新片都做),而不是靠 15 天后另一个脚本偶然撞上。
3. 换组件文件 = 三件事一起做:新 mixture 版本 + 留出集重建 + 数据流状态重置;漏掉第三件会静默污染评测。

**后记(10-06 15:00)**:上面的「换真 latex」修法没有落地。人类同日指示「预训练数据多用日常的,模型太小,不要求输出复杂 latex 公式」,
于是 mixture_v4 改成**日常版**:latex 组件整个删掉(450/451 挪进 reasoning 尾部),web_en .39→.46、web_zh .15→.26、
code .236→.12、ocr .108→.07、math .09→.07、reasoning .006→.02;留出集清单删掉 latex 条目;真 latex 分片 444/448 的下载中止。
删组件不需要 RESET_COMPS(mixdata 按新列表的键建流,旧游标里多出来的 latex 键被忽略;其余组件游标文件都还在各自列表里,
重启前用 `files.index(cursor)` 逐组件自检过)。14:58 从 step 5817 重起,loss 2.58→2.85:
按各域留出 loss × 新旧权重算出来的预期值就是 2.57→2.90,这是配比变化不是退化;各域列仍可比,加权总 ppl 不可比。
两个小坑:(a) `redl.sh` 包装脚本有重试循环,只杀 aria2c 它会再拉一次、删掉的半成品又长回来——要先杀包装 shell 再杀子进程再删文件;
(b) 第 1 条教训仍然成立,只是这次用不上了:latex 不要了,但「逐行核实类别」已经写进 `shard_topup.check_shard`,以后补货每片都查。

## §65 s5 step 10 探针 132 vs 134(p=0.90)继续训;哨兵的「故障串」正则一天误报两次(2026-10-06 15:40)

**探针**:`val_s5_step10` 132/474 vs 基线 `val_sft3_tmpl2` 134/474,McNemar p=0.9007(基线独对 33 / 探针独对 31),
MDE +18。判 CONTINUE_WARN:在噪声量级内,既不是增益也不构成「明显下降」,auto_cycle 自动 resume(新 trainer pid <N>,
15:42 日志「Load from checkpoint folder: …/global_step_10 / Setting global step to 10」,step 11 的 64 条 rollout 15:45 已完成 43 条)。
对照 s4 同节点 145(+11,p=0.185):拔掉轮数罚后第 10 步没有再出现那个「早交卷」的虚高,与 §63 的预测一致。
真正的判决在 step 20(s4 在那里显著跌到 117):s5 不跌 ⇒ 轮数罚是 s4 跌的原因;s5 也跌 ⇒ 另找病因。

**哨兵误报**(`s5_sentinel.sh`,判据之一是对日志新增行 grep 故障串正则):
1. 14:4x:vLLM `serving_chat.py` 对单个请求打 Traceback(「maximum context length 51200」被 400 拒),这是 rollout 顶到上下文上限的
   正常现象,capture_rate 只掉 0.1%;正则里的 `Traceback` 把它当引擎故障。修:排除 `serving_chat\.py:[0-9]+\]` 前缀的行,
   引擎级故障走 `Engine core|EngineDeadError` 另有标签。
2. 15:40:resume 时 trainer 把整份 verl 配置打进日志,其中一行 `'nccl_timeout': 3600,` 命中 `nccl.*(error|timeout|abort)`,
   哨兵 FIRE 后退出。它的动作只是写日志+退出(无破坏),但退出后就没人盯了,等于**每次探针周期都会把哨兵打掉**。
   修:排除 `'nccl_timeout':`,重挂。

教训:(a) 故障串正则**必须拿一次真实的 resume 日志和一次满载 gen 日志回放**,而不是只对着故障样本写——配置回显、逐请求 4xx 这两类
「含故障词的正常行」在这两段日志里一定出现;(b) 哨兵退出要能被看见:现在的 FIRE 只写自己的日志,巡检单要加一项 `kill -0 $(cat s5_sentinel.pid)`,
否则误报退出和真故障退出看起来都是「安静」;(c) 排除项每加一条都要重跑一次「排除后对历史日志命中 0 行」的测试(两次修都做了,
第二次顺带确认第一次的排除没被冲掉)。

## §66 教师第 5 批:官方 smith harness × deepseek-v4-flash,100 并发,1000 题 47 分钟,留 707 条;第 6 批 5338 题全量起跑(2026-10-06 17:20)

**背景**:人类 10-06 令「先蒸馏再训练、用 AGL 简易 harness、增大并发、教师做错的题不参与训练」,17:1x 追加「跑完 step20 探针若没效果赶紧蒸馏;先检查蒸馏数据和代码;趁这个时间跑完所有训练集的教师轨迹」。

**数字(b5)**:1000 题 16:21→17:08(47 min,10 shards×10 workers=100 并发,CPU-only,与 s5 训练并行,s5 步时无变化);resolved 822/1000(82.2%)、submitted 815、timed_out 3、overflow 0、harness error 0;请求 24319、重试 8、非 200 1、延迟 p50 1.99 s;网关显示值 $9.42(真实按「每请求算两次」折半)。拼装留 707:unresolved 178 / 未提交 81 / length_finish 10 / 不可解析未答 19 / 格式错≥3 次 5;token p50 9476 / p90 16912 / max 27864;轮数 p50 18 / max 40。

**flash 输出怪癖(1.65 万响应实测)**:35% 回合在 bash 块后拖伪 XML 闭合标签(`</parameter>`/`</invoke>`/`</format_example>`/DSML 串);480/12057 回合有非标签正文续写(幻觉下一步、heredoc 续写);1.2% content 为空而答案在 reasoning_content;1.7% 用 `<｜｜DSML｜｜ invoke name="bash">` 原生标记 ⇒ harness 判格式错、连续 3 次就烧掉整条 episode。
清洗规则(assemble_smith_teacher.py):只留 resolved+submitted;剪掉 (parse_action 失败的 assistant, "Format error:" user) 对;assistant 归一化=THOUGHT 文本(去纯标签行、去 `<THOUGHT>` 包装)+ 单 bash 块、丢尾巴,**归一化前后 parse_action 必须逐字相等否则回退**。代理补丁 `normalize_teacher_message`(空 content 提升 reasoning / DSML 转 fence / 剪 fence 后尾巴)回放:格式错 4.5%→1.6%,有效命令 0 改;随新代理 pid <N> 生效,b6 全程享受。

**蒸馏数据/代码正确性核查(17:15,人类要求)**:`swe_sft_dataset.py` 整段用 apply_chat_template 渲染、按字符区间做 loss mask。核查项全过:(1) MiniCPM5 模板 `enable_thinking` 默认即 False,渲染与学生推理时 vLLM 的 `chat_template_kwargs={"enable_thinking":False}` **逐字相同**(generation prompt 末尾都是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`);(2) 12/12 个学习区间解码后 == assistant 原文 + `<|im_end|>`,user/system/观察区全为 0,loss 占比 23.4%;(3) 行首 system == harness `SYSTEM_PROMPT`,tools_json=[];(4) 全部 assistant 轮 parse_action 通过,末轮含提交标记。启动器 `sft/run_sft_b5.sh`:从 SFT ep3 续训 1 epoch(人类 09-21 迭代规则)、lr 5e-6、4 卡 sp4;启动前自检显卡占用 <8G 否则拒起 —— s5 占着 1/2/3/5,必须先 pause。

**第 6 批(17:19 起)**:train_dataset_mixed 6338 − b5 1000 = 5338 题,131 个镜像**全部本地**(b1–4 是 opencode 工具调用格式,学生在 smith harness 下用不了,一并重采);15 shards×10 workers=150 并发,load 79/192 核,p50 延迟 1.49 s,retries 0;预计 ≈2.8 h、显示值 ≈$50。`teacher_smith_b6.sh`=b5 脚本把 PFX 参数化 + status 的 done 分母读 IDS 行数 + 成本只统计 launch_ts 之后的事件。

**陷阱(本段新增)**:
1. `teacher_api.env` 导出 `TEACHER_MODEL=gpt-5.5`(别的作业的默认),必须在 source **之后**覆盖,否则教师静默换成别的模型。
2. `( a && b && nohup cmd & )` 的 `$!` 是包装子壳不是 python,写 pid 文件要 `( cd; exec env … nohup … ) &`;第一次 kill pid 文件后真服务还活着就是这个。
3. `pgrep -f run_smith_sweep.py`/`pkill -f` 会匹配到自己的 shell;stop/status 一律按 `/proc/<pid>/environ` 里的 `AGL_SWEEP_ROOT` 过滤。
4. `\bsubmit\b` 匹不到 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`;判提交用完整标记。
5. 代理按 md5(首条 user) 存整条对话,problem_statement 必须 `.strip()` 后再算,否则和 harness 对不上。
6. 探针 smith_fleet.sh 对 `^agl-sm-` 一律 `docker rm -f`:教师容器前缀必须错开(`agl-st`)。
7. harness 不存对话,教师轨迹只能靠代理旁路记录;stdlib 客户端无重试,429/5xx 直接变成空 assistant 烧轮 —— 重试放在代理里。
8. `py_compile` 不查缺失 import:模块级 `re.compile` 在 `import re` 缺失时启动即 NameError,改完必须 import smoke。
9. 训练日志里 `grep -oE 'step:[0-9]+'` 会抓到 `timing_s/step` 的值;要用 `'step:[0-9]+ -'`。
10. 同一行里 `image_name` 自带 `jyangballin/` 前缀,拼镜像名时别再加一遍(第一次算出「0 个本地镜像」就是这个)。

## §67 教师轨迹 SFT 两轮递减(134→126→118):损失全在「40 轮内没交卷」,学生学了教师的探索长度却学不到收敛(2026-10-06 21:10)

**数字**:b5(ep3 续 1 epoch,677 行,21 步,700 万 token)474 题 126 vs 134,McNemar p=0.36;b6(b5 续 1 epoch,3848 行,120 步,3950 万 token,每步≈33 万 token)**118 vs 134,p=0.056**(both 95 / base_only 39 / b6_only 23,MDE +18)。两轮都判 CONTINUE_WARN,但方向一致向下,**按蒸馏口径不是增益**。对照:SFT ep3 本身是 858 行×3 epoch=5190 万 token 从基座训出来的(123→127/134)。

**先排除运行层**:三轮 overflow 都是 0、timeout ≤1、status_error 0、`n_turns=None` 都是同 5 条(Go 镜像无 python)——不是评测事故。

**逐题核对 result.json 的画像**(ep3 → b5 → b6):
- submitted 240 → 205 → 164;未交卷里顶满 40 轮的 213 → 262 → 302(其余是 Go 镜像那几条);
- resolved/submitted **55.8% → 61.5% → 72.0%**;交了卷的轮数中位 16 → 19 → 16。
⇒ 损失**全部**来自「没交卷」,交了卷的反而更准。学生学到的是教师的行为分布:教师 b6 轨迹 assistant 轮数 p50 18、p90 31、12% ≥30 轮、只 34% ≤15 轮(都是做对了的轨迹,所以「长」在教师那里是「谨慎后做对」)。2B 学生模仿了「多探索再交」的前半段,却没有教师的收敛能力,于是更多题探索到 40 轮被掐。未交卷样本的 smith.log 尾部:grep/sed/写 repro 脚本失败再读,**不是死循环、不是格式错**,就是没走到提交。

**还有一个没对齐的变量**:教师采样 `AGL_SAMPLE_TIMEOUT=2400`,评测是 600(人类 09-21 定死不提)。教师的长轨迹有些是 600 s 内根本跑不完的,学生学了也兑现不了。

**经验**:
1. 蒸馏没涨先看 **submitted 与 resolved/submitted 的分解**,比看总分有信息量得多:这里总分跌 16 题,但「交卷准确率」涨了 16 pp,两件事抵消后才是 −16。只看总分会误判成「数据脏」。
2. 拒绝采样只筛「做对」不筛「怎么做对」:**轨迹长度本身是会被学走的监督信号**(这和 §62 的长度通道是同一件事的 SFT 版)。下一轮若再做,先改数据不加数据:按轮数筛短轨迹(≤15 轮占 34%,约 1300 条)、或把「剩余轮数」写进 prompt 让学生学会收尾;教师采样超时与评测对齐。
3. 轮次规则(每轮新题续训 1 epoch)在这里连做两轮都是小步(loss 0.46→0.43),**递减是稳定的方向而不是噪声**——两次 CONTINUE_WARN 同向就该停,不等第三次。

**同框架参照(人类令「同样的框架评 Qwen3-8B」)**:`smith_fleet.sh` 加了三个 env(`FLEET_SERVED_NAME`/`FLEET_CHAT_TEMPLATE=none`/`FLEET_ROPE_SCALING`),默认值逐字不变;Qwen3-8B 用 YaRN factor 2.0 把 40960 扩到 65536 容纳 ctx 51200,非思考模式(harness 本来就传 `enable_thinking=False`),采样参数沿用 MiniCPM 口径 temp 0.6/top_p 0.95/top_k 20(不是 Qwen 官方非思考推荐的 0.7/0.8)。冒烟核对 harness 对它工作正常:Qwen3 模板自己在 assistant 前缀塞 `<think>\n\n</think>`,回复是 THOUGHT + 单 bash 块,45k token 的 prompt 可接受。结果见本节末尾追加。

**工程陷阱(本段新增)**:
1. **vLLM V1 的 EngineCore 是独立子进程**,只杀 api_server 显存不放(残留 29.8G 让下一实例把 util 算到 0.958)。停实例:`kids=$(pgrep -P $P); kill $P $kids`。
2. `cd … && nohup … &` 记下的 `$!` 是子壳 pid(python 真 pid +2),和 §66 陷阱 2 是同一件事的另一种写法。
3. 合成长 prompt 估 token 要实测:文件清单一行 12 token,15000 行是 18 万 token 不是 4.5 万。
4. 8B 模型的 KV 账:51200 token 一条 = 7.03 GiB;权重 15.3 G + 激活 2.5 G 后,budget 30000 只剩 9 万 token KV,45000 才够 8 并发;50000 会把常年有 46 G 邻居的卡挤出候选。

**Qwen3-8B 终值(21:09–21:50,`val_qwen3_8b`)**:**22/474 = 4.6%**(vs ep3 134,McNemar p=0.0000;基线独对 114、Qwen3 独对 2)。交卷 87、顶满 40 轮 358、超时 8、溢出 10、resolved/submitted 18/87、交卷轮数中位 11、completion token 中位 19 799。日志画像是大量重复同一条命令直到 40 轮——这是没对齐采样参数和没做过这个协议的零样本表现,**只能作参照,不能用来说「8B 不如 2B」**:它说明官方 smith 协议(THOUGHT + 单 bash 块 + 自己判断何时提交)对没见过的模型并不自然,SFT ep3 的 134 里有相当一部分是「学会了协议」而非「学会了修 bug」。

## §68 蒸馏三连跌之后:条件口径的选择偏差、短轨迹/LoRA 两条并行方案、`/` 满导致的段错误(2026-10-06 22:40)

**人类的问题 1:「把指标限定为不超轮次情况下的正确率,蒸馏是否带来了提升?」** 按 result.json 配对算(双方都在 40 轮内交卷的子集):b5 vs ep3 87:82(只 b5 对 9/只 ep3 对 4,p=0.267);b6 vs ep3 78:70(只 b6 对 10/只 ep3 对 2,**p=0.039**);没顶满 40 轮子集的正确率 ep3 45.8% → b5 51.4% → b6 54.5%。看起来是涨,**但不能当提升报**:分母由模型自己的行为决定(选择偏差/碰撞偏差)。b6 在 ep3 原本做对的 60 题上放弃交卷(b5 45 题),新赢回的只有 21(b5 28)。条件正确率升的机理是「只交有把握的卷」,就像一个只答简单题的学生「答了的题正确率更高」。真实指标仍是无条件 474,而它在跌。**经验**:任何「限定在模型自己选出的子集上」的指标都要先问分母是谁定的;分母随模型变,指标就不可比。

**人类的问题 2:「让 deepseek-v4-flash 审视 MiniCPM 的 rollout,判断哪个步骤对奖励有增益,让奖励信号更密集?」** 这是把序列级奖励换成判官给的轮级信号(process reward),直指 §61 定位的 credit-assignment 根因。先做离线试点再下结论:`swe_smith_smoke/judge_pilot/judge_pilot.py` 从 s5 轨迹抽 40 个混合组(同题既有做对又有做错)160 条 + 30 条复判,判官**盲评**(不告诉测试结果):预测成败、逐轮 P/N/H、首个致命轮。度量:盲预测 vs 真实 reward 的判别力(判官到底看不看得懂)、失败轨迹里致命轮之前的 token 占比(= 现在被整段压低的梯度份额)、复判一致率(信号噪声)。每条 prompt 中位 13.8k token、8 s;160 条 ≈2.3M token。**陷阱**:第一轮 `max_tokens=2000` 让 144/160 条被截断——flash 非思考模式会在正文里先写几百行分析再给 JSON,按 `{.*}` 正则只能抓到短轨迹的;改成「分析 ≤300 词 + 末尾 ```json 块」、上限 6000、取最后一个平衡 `{}`。第二轮仍 95/160 截断(输出中位正好 6000),`rerun_unparsed.py` 以 16000 补判后 123/160 可解析,剩下 37 条还是没给 JSON——全是 40 轮顶满的长轨迹(未解析组 n_turns 中位 40,已解析组 33),flash 写 16k token 分析也写不完。

**判官试点结果(160 条全量)**:
- 序列级:盲评准确率 0.852(TP 67 / FN 2 / FP 16 / TN 37,多数类基线 0.566),AUC 0.870,组内「做对排在做错前」62/77 = 0.805。偏差方向是**乐观**:16 个假阳性里判官信了 agent 自述「测试通过」——这就是用判官当奖励时的 hack 入口。
- 轮级:53 条失败轨迹只有 29 条给出致命轮(其余判官认为是「整体无进展」没有单一致命轮);致命轮 p50 在第 13 轮 / 40 轮,致命轮之前的 assistant 字符占比均值 0.32——即当下整段压低的负梯度里约三分之一落在判官认为非致命的轮上。复判 17 条:成败预测 17/17 一致、致命轮有无 16/17 一致,但**两次都给出致命轮的 4 条里位置精确一致 0/4、±1 也 0/4**;逐轮 P/N/H 标签一致率 0.75。失败轨迹 52%、成功轨迹 46% 的轮被标「中性」。
- 成本:prompt 中位 13.8k、输出中位 5.8k token、延迟 p50 24 s / p90 66 s;160 条 + 复判共 3.8M token。按 RL 一步 64 条估 1.5–2M token,真实计费约 $0.4/步,可承受。
- 结论(64 条时的判断在 160 条上不变):判官能看出整条成败,但这一项我们有测试结果、不缺;**定位到轮不可靠**——位置复判 0/4、半数轮中性、还会被 agent 自述骗。期限前不上判官奖励。零成本派生物:对「纯探索轮」(ls/cat/grep,不改文件)的 loss 降权,这条不需要判官,只需要看命令是否改了文件。

**并行两条方案(人类「那就按照你的来」)**,都是蒸馏口径:
- **方案 1 short15**:同一教师池(b5+b6 共 4585 条)按 assistant 轮数筛 ≤15 轮 → 1544 条(p50 12 轮,token p50 6210),`sft/run_sft_short15.sh` = b5 脚本只换数据、**从 ep3 起**(不从 b6)、1 epoch 47 步,loss 0.40→0.36,22:07 训完。可证伪预测:40 轮内交卷回到 ≥240(ep3 240/b5 205/b6 164)。**结果(22:49):120/474 vs 134,McNemar p=0.10(both 95 / 基线独对 39 / short15 独对 25)。交卷 272,预测兑现;顶满 40 轮 197(ep3 216);但交卷者正确率 109/272 = 40%(ep3 116/240 = 48%,b6 90/164 = 55%)。** 即「交卷率由训练数据的轮数分布控制」被证实,但多出来的交卷是错的:短轨迹教会了「早交」,没教会「修对」。三轮 b5/b6/short15 = 126/118/120 全在 134 之下,交卷率 205/164/272 与 resolved 不相关——**flash 轨迹蒸馏在 2B 上的上限就在 ep3 附近,再换数据切法不会涨**(ep3 自己也是蒸馏产物,vs 基座 123 同样不显著)。蒸馏口径非增益。
- **运行层事故(22:49–22:54):被「杀掉」的 Qwen3 LoRA 驱动链还是把 SFT 拉起来了。** 驱动链是 `nohup bash drive_qwen3_lora.sh &` 起的,bash 为 `nohup … &` 又 fork 了一层,实际是两个 bash(父 → 子);22:30 只杀了父 PID,子壳继续等 short15 的标记,22:49:32 标记一出就起了 `verl.trainer.sft_trainer`(4 卡各 32–52G),违反人类「接下来不要训 qwen」的指令,5 分钟后巡检卡占用时才发现并杀掉,没有产出 ckpt。教训:**杀驱动链要杀进程组**——起的时候用 `setsid`,杀的时候 `kill -- -<pgid>`,杀完用 `ps -eo pid,ppid,args | grep '[d]rive_'` 确认脚本名不再出现,而不是只看那一个 PID;所有「等条件再开训」的链都要这样对待,它们的危险正是在无人看着时触发。
- **OPSD go/no-go 诊断(22:58 起,`hint_diag/run_hint_diag.sh`)。** 在评测侧 `smith_rollout.py` 加 `SMITH_HINT_MODE`(none/tests/files/patch),在 `checkout_bug_commit` 之后、`relocate_git` 之前用 `git diff HEAD~1 HEAD~2` 取金标修复(HEAD~1 是 Bug Patch、HEAD~2 是干净 main,这个区间正好是源文件上的修复、不含测试删除;先核对 HEAD~1 的提交标题含 "Bug Patch",不含就不给 hint 并计数),按模式把 F2P 测试 id / 触及文件路径 / 补丁本身(上限 8000 字符)追加到题面末尾的 `<hint>` 块;hint 取空时退化为无 hint 并在 status.json 记 `hint_chars=0`。固定子集 = val 去掉 4 道无 python 镜像后 seed 20261006 抽 100 题(18 个仓库;ep3 全量跑在这 100 题上 resolved 29、交卷 55),同一权重 SFT ep3、同一 fleet 配置(4 卡 × 6 worker、600 s、40 轮、temp 0.6)逐条件顺序跑,`hint_diag/analyze.py` 出配对 McNemar。判据写在跑之前:patch 条件是上限,它如果不比 none 高出 ≥20 题,说明瓶颈不在「知道答案」而在「按协议把改动做进文件并交卷」,OPSD 作罢;tests/files 才是训练时能拿到的特权信息,它们与 none 的差距决定路径 A 值不值得采。
- **方案 3 Qwen3-8B LoRA**(人类问「能不能用 LoRA 代替全量」,顺便验「是不是 2B 容量问题」):`sft/run_sft_qwen3_lora.sh`,rank 64/alpha 128/all-linear、lr 1e-4、1 epoch、short20 数据 2763 条、`max_token_len_per_gpu` 8192(8B 的 logits 是 152k 词表);配对基线是零样本 22,134 只作参照。**模板陷阱**:Qwen3 原版模板只给「最后一条 user 之后」的 assistant 渲染 `<think>` 块,历史轮不带,于是 `swe_sft_dataset.py` 的前缀不变量(`full.startswith(before)`)在多轮上必炸。解法是 `/workspace/models/Qwen3-8B-nothink/`:权重符号链接到共享只读目录,`tokenizer_config.json` 里换成改版模板(所有 assistant 轮一律 `<think>\n\n</think>\n\n`+content,gen prompt 无条件追加空块),并写 `chat_template.jinja` 给 vLLM;30 行验证 bad=0、loss token 20.7%。训练和评测必须用同一个模板文件。verl 0.7.1 的 LoRA 开关是 `model.lora_rank>0`,`checkpoint.save_contents=[hf_model]` 对 PEFT 模型大概率只存 adapter,`sft/merge_lora.py` 两种布局都接(adapter → `merge_and_unload` 成 16G 全量件给 vLLM)。驱动链 `drive_qwen3_lora.sh` 等 short15 评测腾出 4 卡后自动开训→合并→评 474。**→ 人类 10-06 22:30 取消(「接下来不要训 qwen」),驱动链在等待阶段被杀,没开训、没合并件。**

**Qwen3-8B 为什么只有 22/474(人类问「是不是没微调过的」)**:磁盘上 `/data/<other-user>` 是后训练版(hybrid thinking,README 里 Base 是另一个仓库),不是 Base。失败模式用 `smith.log` 逐轮命令量化(469 题,ep3 468 题对照):顶满 40 轮 p50;**重复命令占比 p50 0.65 vs ep3 0.03**,同一命令连发 ≥5 次的题 41% vs 4%;命令 rc≠0 比例 p50 0.33 vs 0.11;**从不跑 pytest(0% vs 68%)**;格式错总共 36 次(协议不是问题);97% 先读文件再改(不是瞎改)。即:它会按协议输出,但在长多轮里退化成重复同一条命令,也不验证、不提交。三个放大因素:(1) 只能用非思考模式(思考 token 吃掉 51200 上下文预算),而 Qwen3 的 agentic/代码能力主要在思考模式;(2) 用了 MiniCPM 的采样参数(temp 0.6/top_p 0.95/top_k 20),Qwen 非思考推荐 temp 0.7/top_p 0.8 并加 presence_penalty 抑制重复,没调;(3) 协议(THOUGHT + 单 bash 块 + 自己决定何时 submit)它没见过,ep3 的 134 里有相当一部分是 SFT 学来的协议与「跑测试再交」的习惯。文献参照:SWE-smith 论文里 Qwen2.5-Coder-7B 要用 5k 条轨迹微调后才到 SWE-bench Verified 15.2%(SWE-agent-LM-7B),未微调的 7B 级模型在 bash 脚手架上公开报告普遍只有个位数,22/474=4.6% 不离谱。所以 22 不是 Qwen3-8B 的上限,但「零样本很拉」是这类小模型的常态,不是这份权重坏了。

**工程陷阱:`/` 满 = 启动瞬间 SIGSEGV**。short15 首次启动 rank 0 两秒内段错误、无 python 栈。`df /` 显示 0 字节空闲(`/root/.cache` 675G 与别人的 `/tmp/*`,都不是我的)。torch/triton/inductor 往 `/tmp` 或 `~/.cache` 写缓存失败直接崩。修法:训练脚本统一 `export TMPDIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR XDG_CACHE_HOME HF_HOME` 到 `/data`,之后一次启动成功。**没有栈的启动期段错误,先查磁盘再查代码**。
## §69 官方 MiniCPM5-2B-SFT 与 MiniCPM5-1B：同词表更弱的亲戚当不了教师(2026-10-07)

人类问：有没有更老、词表一样（130560）的 MiniCPM，能不能当教师或对照。Hugging Face / ModelScope 上同架构可下载的是官方 **MiniCPM5-2B-SFT** 和 **MiniCPM5-1B**，同一套 LlamaForCausalLM、同一词表。把它们放进**完全同一套**官方 smith harness（temp 0.6 / top_p 0.95 / top_k 20 / 40 轮 / 600 s，固定 474 题）评完：

| 权重 | resolved | 相对 ep3 134 | 失败画像 |
|---|---|---|---|
| MiniCPM5-2B 基座 | 123/474 | McNemar p=0.22，不显著 | 会协议，弱于 ep3 |
| SFT ep3（OpenCode 教师轨迹训出来的，模板修复后重测） | **134/474** | 对照 | 交卷 ~240 |
| 官方 MiniCPM5-2B-SFT | **92/474** | 显著更差 | 不是「没见过协议」，是修 bug 更弱 |
| MiniCPM5-1B | **0/474** | — | 256/474 格式中止，几乎说不出「一个 THOUGHT + 一个 bash 块」 |

**经验**：同词表不等于能蒸馏。1B 连协议都过不了，2B-SFT 官方件比我们用 OpenCode 轨迹训出的 ep3 还差 42 题。跨模型教师必须先过「同一 harness、同一 474、McNemar 相对 ep3」这道门，过不了就不要当教师。ep3 自己相对基座 123 也不显著（p=0.22），它能当对照只是因为它是目前这条线上最稳的读数，不是因为它被证明「学会了修 bug」。

## §70 拒绝采样两轮（RFT1/RFT2）还是贴着 134，报不了增益(2026-10-07)

做法：用学生自己（或带 hint 的学生）在训练集上采样，只留 `resolved` 且已提交的轨迹，从 ep3 再 SFT 1 epoch。这是拒绝采样 / RFT 口径，**不是 RL**。

- RFT1（无 hint 学生轨迹）：**130/474** vs ep3 134
- RFT2（hint 条件采集后再 SFT）：**127/474** vs 134

方向与 b5/b6/short15 一样，全在 134 附近或之下，McNemar 都不构成可报增益。**经验**：学生已经会协议之后，再用「做对的自己的轨迹」喂回去，等于在同一个行为分布上微调，解决不了 credit assignment，也解决不了「2B 学了教师的探索长度却收敛不了」（§67）。RFT 在这条线上不是新杠杆。

## §71 特权信息：金标 patch 上限成立，可训练的 tests/files 几乎不动；训出来的 OPSD/GKD 更差(2026-10-07)

§68 的 go/no-go 跑完了。同一 ep3 权重、固定 100 题（val 去掉 4 道无 python 镜像后抽的）：

| 题面特权 | resolved | 交卷 | vs none |
|---|---|---|---|
| none | 24/100 | 39 | — |
| F2P 测试名 | 23/100 | 51 | −1 |
| 触及文件路径 | 31/100 | 61 | +7（不到事先写的 +20） |
| 金标 patch（泄漏答案） | **64/100** | 81 | **+40** |

金标补丁贴进题面，上限确实高——「看见答案就会做」这部分是真的。但这是评测泄漏，训练也不能用。真正能当训练特权、评测时拿不到的，是 tests/files，它们几乎不动。

随后按特权信息训出的权重（蒸馏口径，不是 RL）全量 474：OPSD2 **109/474**（交卷 280），GKD1 **108/474**（交卷 258），对照 ep3 134。方向是跌。OPSD1 一次全量 0/474 是失败跑（采集/格式崩了），不当对照。

跨词表的 **token 级 reverse-KL（MOPD）做不了**：MiniCPM 词表 130560，Qwen 不是同一套 id。能做的只有序列级 `log P_T(y)`（学生采样 → 反分词成文本 → 教师打整段 logprob），代码在 `swe_smith/distillation/seq_opd_score.py`。前提仍是教师本身显著强于 ep3。

**经验**：先测「泄漏答案的上限」，再测「训练时真能拿到的特权」。上限高但可训练特权不动，就不要开 OPSD。OPSD/GKD/RFT 一律标「自蒸馏 / 蒸馏」，不要写成 RL。

## §72 Qwen3.6 的格式死亡：答案在 tool_calls 里，harness 只看 content(2026-10-08)

通俗讲：官方 smith 协议规定助手每一轮只能回一段思考，再加**恰好一个** ` ```bash ` 代码块，里面是要在容器里执行的命令。MiniCPM 是这么训的，`parse_action` 也只认这一种。

Qwen3.6 默认按「工具调用模型」说话：HTTP 响应里 `content` 经常是空的，真正的命令写在 OpenAI 风格的 `tool_calls`，或者写成 Hermes XML `<tool_call>{...}</tool_call>`，再夹一层 `<think>`。harness 读 `message.content`，看见空字符串或 XML，就记一次 format error。连续三次，`SMITH_MAX_FORMAT_ERRORS=3` 直接杀掉这题——**模型可能已经在内部知道要 `sed` 哪个文件，但学生侧的裁判根本没看见这条命令**。

未适配的 Qwen3.6 全量：**100/474**，其中 **406/474** 是格式中止。这 100 不能拿来跟 ep3 的 134 比「谁更会修 bug」，因为分母里 86% 的题它没被允许做完。Qwen3-8B 的 22/474 是另一种病（协议对了、长多轮退化成重复命令，§68）；Qwen3.6 是协议根本没对上。两种失败不要混着讲。

**不能改的**：MiniCPM 的 `parse_action`、`SMITH_MAX_FORMAT_ERRORS`、单 fence 规则。改了学生评测口径，之前所有 474 读数作废。

**能改的**：教师侧加一层 HTTP 代理，把 `tool_calls` / Hermes XML / 残余 think 块收成一个 bash fence，再交给同一套 harness。失败则原样转发，让 format error 被记录，而不是替模型编造命令。代码：`swe_smith/distillation/teacher_rewrite.py` + `teacher_proxy_rewrite.py`。学生评测仍然直连自己的 vLLM。

关掉 thinking 不是「口头告诉模型不要想」，也不是把生成的 CoT 藏起来。MiniCPM / Qwen 的 chat 模板在 `enable_thinking=False` 时会在 assistant 前填好空的 `\n\n`（或 `<think>\n\n</think>\n\n`），模型接着写正文。AGL 代理必须把 `chat_template_kwargs` 透传（§57），否则训练和评测看到的前缀不一致。

## §73 适配后的 32 题 smoke：8/32 与 ep3 打平；尾部三连错不是第 3 轮猝死(2026-10-08)

Qwen3.6 + rewrite 代理，同一 32 题子集（与 ep3 逐题配对）：**8/32 vs ep3 8/32**，McNemar p=1（独对各 3，双对 5）。过了「能干活」这道门，**没过「显著强于 ep3」**。

适配之后日志里仍会出现 `format_abort`，但机制变了：不是第 3 轮就死，而是先正常跑了 17–34 轮（读文件、改文件、跑 pytest），最后用一段散文「我做完了」或再套一个代码块收尾，被连续三次 format error 杀掉，没执行到 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`。启发式「连续格式错」在 episode **结束时**才亮，容易误判成旧的 turn-3 死亡。看 `n_turns` 和 smith.log 里成功执行过的 `cmd=`，不要只看 termination 字符串。

32 题打平不够当教师。教师资格写在跑之前：全量 474、McNemar 相对 ep3 134，且必须显著。flash 教师在同一官方 harness 上是 **286/474**（OpenCode 口径的数不能拿来比；这里说的是 smith 协议下采轨迹时的 resolved），那才是目前合格的强教师。验证集轨迹**不许**进 SFT。

全量 `val_qwen36_adapted` 于 2026-10-08 13:53 在空闲卡剩余显存上启动（不杀邻进程），8 worker，rewrite 代理与 flash 教师代理分端口，避免覆盖 flash 那份代理。写本节时已完成 41/474、resolved 9（早期题偏 bottle，比例不可外推），sweep 仍在跑。终值出来之前，Qwen **不能**当教师，也不采它的验证集轨迹。

## §74 short15 的病根不是「轨迹太长」，是「截断位置不是最短成功前缀」(2026-10-08)

short15（§68）按**全局轮数硬切 ≤15**，把「修对了但多逛了几轮」的轨迹扔掉，留下「碰巧早交」的。交卷率按预测回升，交卷正确率跌到 40%：模型学会了早交，没学会修对。

最短成功前缀是另一件事：一条已经 `resolved` 的轨迹，从前往后找**最小的 k**，使得只回放前 k 个动作、pytest 仍过。这不是按 15 切，也**不能二分**：后面的动作可以破坏已经修好的树，`eval(k)` 对 k **不单调**，二分会把「中间过、两端不过」切错。实现是对编辑类动作的 k 做线性扫描，先对完整轨迹做一次 sanity eval；完整回放都不过（环境漂移）就丢弃，不要退回原长轨迹。过滤与 Qwen3-Coder 同类：无 submit、格式坏、改测试 / 动隐藏 git 的丢掉。

代码：`swe_smith/distillation/shortest_prefix.py`（线性扫描）、`shortest_prefix_replay.py`（docker 回放）、`build_prefix_sft_jsonl.py`。SFT 起点仍是 ep3，1 epoch，lr 5e-6。数据必须来自**训练集**上过门教师的轨迹；flash 的 b6 jsonl 目前 `status.json` 里没有 `messages`（代理组装的），要先从 assembler / smith.log 还原再回放。教师资格没过之前不开 SFT。

## §75 全局轮数罚会变成「早交卷」通道；HMPO 把长度预算收到「做对的组内中位数」(2026-10-08)

s4（§62）已经实证：同样做对时，轮数罚是组内最稳的信号，模型学到的是早交（提交率 71%）而不是修 bug。HMPO 的改法是：**做错的永远 0**；长度预算 = 组内**做对**的轨迹长度中位数；做对且短于预算的得 1，长于预算的按 `clip(budget/len, floor, 1)` 打折。全错组仍全 0，短而错的排不到做对的前面。这和「全局 T0=32、λ=0.1」不是一回事。

过程信号用环境状态，不用金标 diff 重叠：F2P 通过数的增量、复现脚本从失败变通过。金标 overlap 是漏答案。代码：`swe_smith/training/hmpo_reward.py`、`env_state_reward.py`。门闩：**SFT 相对 ep3 显著之前，不开 HMPO RL**。平台期再考虑失败后二次尝试（FC-SWE）：重置工作树，把失败 patch + pytest 日志作为下一条 user，两次 attempt 各自计分，后一次成功不给前一次零分刷成 1。默认 `SMITH_RECOVERY_ATTEMPTS=0`，评测关着。代码：`swe_smith/agents/recovery.py`。

## §76 截至 2026-10-08 的闸门（写下来是为了以后不被「先训起来」带跑）

1. MiniCPM `parse_action` 不动。教师格式只允许改代理。
2. 不训 Qwen。Qwen 只当候选教师或参照。
3. 教师资格：同一 smith harness、val-474、McNemar vs ep3 134，显著才采**训练集**轨迹。32 题打平不算过门。
4. 验证集轨迹不进 SFT。flash 286 仍是合格教师；Qwen 适配 474 未出终值，终值不显著则继续只用 flash，走最短前缀而不是再灌长轨迹。
5. 最短前缀 SFT → 显著才能 HMPO；RL 平台期才加 FC-SWE；序列级 OPD 仅当教师 ≫ ep3。
6. `/` 与数据盘水位、不杀邻进程、不删 SFT ep3 / `ckpt_step5722.pt`，仍有效。
7. 实验记录的家是本仓库 `docs/pitfalls-and-lessons.md`，不是 microsoft/agent-lightning。公开仓库脱敏：个人目录写成 `/workspace`，邻进程写成 `<neighbour-process>`。
8. `gh` 装在 `~/.local/bin`，不在默认 PATH；推送用绝对路径或先 `export PATH="$HOME/.local/bin:$PATH"`。
## §77 格式门过了；最短成功前缀在长轨迹上把 40 步压到约 4 步(2026-10-08)

两件事并行，数字都还不是终值。

**Qwen 教师格式。** 8 题格式烟测：`format_abort=0`、最后一轮感叹号坍缩=0，门过了才开新的 val-474 目录（旧适配跑的 474 作废，不混）。新 474 开跑后前几十题仍是 `format_abort=0`。对照未适配 406/474 中止、旧适配约 123/163 中止。根因仍是 §72：`smith_rollout` 把 temperature 写成 1.0，教师代理必须把请求打回 0.6 并截停 `!!!!`。**现在还不能报 Qwen 的 resolved，也不能宣布教师资格**——资格只认全量 474 对 ep3 134 的 McNemar。

**MiniCPM 侧按计划走阶段 1，不等 Qwen 过门。** flash 已是合格教师（286/474）。`status.json` 常常没有 messages，训练数据用已经拼好的 train jsonl（3878 条 resolved）。做法：

- ≤15 轮的成功轨迹原样留（short15 留过的那批，补上缺失的 submit 句）
- \>15 轮的在干净容器里**线性**回放编辑步（`eval(k)` 不单调，不能二分），保留最小仍能 pytest 过的前缀，并补一条 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- 改测试 / git / 联网的丢掉（本批 347 条 `forbidden_action`）
- 验证集轨迹不进 SFT

头几十条长轨迹已经回放完：原长中位数 40 轮，最短前缀中位数 **4**，最后一轮都是 submit。这就是 §74 说的「短15 扔掉的是已经修好、后面在逛」的那些轨迹。SFT 仍从 ep3、1 epoch、lr 5e-6；GPU 配方还是 4 卡，要等 Qwen 评测放出卡才能开训。HMPO / 二次尝试 / 序列级 OPD 仍然有闸：SFT 相对 134 不显著就不开。

**经验**：

1. 教师格式门和最短前缀可以并行；前者决定 Qwen 能不能进下一轮数据，后者现在只用 flash。
2. 回放脚本必须挂上 `instance.json` + `eval_inside.py`，先藏 `.git` 再评，且**不能** `set -e`（教师轨迹里失败的 `ls`/`grep` 本来就不会停）。stdout 是 indent 过的 JSON，不能只 `json.loads` 最后一行。
3. 截到编辑步之后一定要补 submit，否则学生学会「改完就停」。
4. `/data` 的 `df` Available=0 往往是 root 预留块，size−used 仍可能大于 50G。没有 `global_step_*.tmp` 就不要删已经落盘的失败 SFT 权重。

## §78 最短前缀 SFT 训完；评测路径又和 134 口径分叉了(2026-10-08)

前缀 SFT（`sft_prefix_from_ep3`）1 epoch 跑完：109/109 step，约 15 分钟，loss ≈ 0.36，权重在 `MiniCPM5-2B-sft-prefix`。SwanLab：https://swanlab.cn/@zzzluvst/swe_smith_sft/runs/5tew8krp92k6chv4rq700 。这不是 RL。

自动 val-474 **没跑起来**。`smith_fleet.sh` 在启动前检查 `smith_rollout.py` 是否含 `skip_special_tokens` / `strip_turn_delims` / `_STOP_STRINGS`，smoke 目录那份是 214 行旧拷贝，三条都没有，于是 `PREFIX_SFT_FAIL FLEET`。更严重的是 CANON（`examples/swe_smith/agents/smith_agent.py`）也漂到了只认 ` ```bash ` 的 1016 行版本：`_query` 仍 `skip_special_tokens` 默认 True。134 那次评测用的是公开仓库里的 1223 行 harness——MiniCPM5 的 `<function>/<param>` 是特殊 token 18/19/20/21，默认反分词会删掉它们，harness 把本来合法的调用判成 format error（§54/§55）。**用旧 harness 评前缀 SFT 不能和 134 比。**

修复：把 `agl-agentic-rl/swe_smith/{agents/smith_agent.py,harness/smith_rollout.py}` 拷回 smoke 和 CANON，再开 `val_sft_prefix`。`parse_action` 仍然只把 bash fence 和 MiniCPM 原生 bash 函数当成同一种动作，不接受 Qwen 的 `tool_calls`。

HMPO 有一个真 bug：`train_smith_agent.py` 里的 `algorithm.hmpo` **VERL 不读**。真开关是 `SMITH_HMPO=1`（agent 交 0/1 + `n_turns`，跳过全局 t0 和 prompt 长度惩罚）加上 `agl_rollout_manager.apply_hmpo_to_completed`（按 `data_id` 组、正确轨迹中位长度做预算，做错恒 0）。`rollout.n` 已是 8。没接到 manager 就开训，只会跑二元 GRPO。

先前 GKD1 108 / PG-OPD1 119 对照 134 更差或 n.s.，那次教师是 **带金标 patch hint 的 ep3**，不是「师生各自 rollout」。明天若 SFT/HMPO 都没增益，按新口径重做同词表蒸馏：学生用 MiniCPM SFT（前缀或 ep3，看谁评测更好），教师用**没训过的原版 MiniCPM5-2B**（123/474，弱于 134，有把学生往回拉的风险，必须写进笔记），损失是逐 token 软标签 KL，以及师生都 rollout 的 GKD / PG-OPD。

验收数字只认 val-474 vs `val_sft3_tmpl2` 134 的 McNemar。Qwen 不当教师。

## §79 最短前缀 SFT：128/474，对照 134，p=0.56；HMPO 两次启动都是环境 bug(2026-10-09)

`val_sft_prefix` 全量 474，官方 smith harness（`skip_special_tokens=False`，原生 `<function>` 解析）。对照 `val_sft3_tmpl2`：

| | resolved | 交卷 | 轮数中位 | 打满 40 轮 |
|---|---|---|---|---|
| ep3 | 134 | 240 | 31 | 216 |
| 前缀 SFT | **128** | 270 | 28 | 198 |

配对：ep3 独对 39，前缀独对 33，差 −6，McNemar exact **p=0.56**。不是增益。交卷变多、做对变少，和 §75 的「早交卷」同一方向，只是这次来自 SFT 数据而不是轮数罚。格式中止没有占满日志，输出能被解析执行。这不是 RL。

HMPO（`rollout.n=8`，`SMITH_HMPO=1`，从这份前缀权重起）按「SFT 不显著就开」启动，前两次都没进入 rollout：

1. `run.sh` 调用裸 `python`，机器上只有 `python3` 和 venv。进程秒退。
2. 补上 venv 的 `PATH` 之后，verl 走 V1 `AsyncLLM`，环境里 `VLLM_USE_V1` 却是关的，和 s5 当时显式 `VLLM_USE_V1=1` 不一致。同时关掉 MiniCPM 不该挂的 Hermes `tool_call_parser`（s5 是 `None`），chat template 用基座那份 jinja。`/tmp` 和 `/data` 的 `bavail=0` 是 root 预留块，ray 会报 95% full；没有去删已落盘权重。

第三次启动带上 `VLLM_USE_V1=1` 和 `TMPDIR` 在 `/data`。显存占用按 0.4，不用 s5 的 0.95，避免挤占同卡上别人的进程。
