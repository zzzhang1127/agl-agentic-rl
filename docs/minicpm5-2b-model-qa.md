# 技术问答:MiniCPM5-2B 选型、架构与显存(2026-09-18)

来源约定:模型事实来自 `/workspace/models/MiniCPM5-2B/` 下的 `config.json`、`generation_config.json`、
`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`README-cn.md`;训练侧事实来自
`examples/swe_smith/train_opencode_agent.py`、`examples/swe_smith/restart_v4_trainer.sh` 和 verl 0.7.x 源码
(`.venv/lib/python3.12/site-packages/verl/`)。README 没写的设计动机,答案里明确标注"官方未说明,以下是我的推断"。

---

## Q0. 为什么选 MiniCPM5-2B,详细讲一下它的架构

先纠正一个名字:用的是 **MiniCPM5-2B**(面壁 MiniCPM 第五代的 2B 版),不是 MiniCPM 2。

**为什么选它**

1. **任务需要 50K 上下文的 agent 轨迹。** 一条 SWE 轨迹 20–30 轮,结束时上下文中位 20k、最大 29k。它原生 131072 上下文(`max_position_embeddings`,`rope_theta` 5e6),我们只用 51200,余量充足。
2. **训练资源是 4 张共享卡,每卡能给我约 30G。** 2.5B 参数 bf16 权重 5G,FSDP 四切之后每卡状态压得住(精确算法见 Q1;真实配置把参数和优化器都卸到了 CPU)。
3. **架构是标准 Llama。** `architectures: LlamaForCausalLM`,`model_type: llama`,verl 的 FSDP 训练侧和 vLLM 推理侧零改动支持,不用 `trust_remote_code`。RL 框架要频繁在训练权重和推理引擎之间同步权重,非标准架构会在这一步反复出问题。
4. **原生 thinking 和工具调用格式。** chat template 内置 `<think>`、`<function name=…><param name=…>`、`<tool_response>` 三套标记和 `enable_thinking` 开关,opencode 的工具调用能直接映射。
5. **后训练路线和我们的目标对口。** README 写明后训练是 SFT(400B token deep-thinking)→ RL(JustRL II)→ OPD,OPD 合并了 16 个 RL 专家,其中 5 个是 agentic 专家;SFT/RL 数据(UltraData-SFT-Agent-2609、UltraData-RL-2609)开源。它出厂就带 agent 能力,我们的 RL 是领域对齐,不是从零教。
6. **小模型的问题也真实。** 验证集基线 26/100,不高,RL 才有可量化的提升空间;它也很容易训塌(v3 第 76 步),动态采样、刹车、探针这套稳定手段都是被它逼出来的。

**架构表**(`config.json`)

| 项 | 值 | 含义 |
|---|---|---|
| 总参数 | 2,516,756,480 | 非嵌入 1,981,982,720(推导见 Q7) |
| 层数 | 42 | 深而窄 |
| hidden_size | 2048 | |
| 注意力头 | 16 × head_dim 128 | 16×128 = 2048 |
| KV 头 | 2 | GQA,8 个 Q 头共享 1 组 KV |
| MLP | SwiGLU(`hidden_act: silu`),intermediate 6144 | 恰好 3× hidden |
| 归一化 | RMSNorm,eps 1e-6 | pre-norm(Llama 结构) |
| 位置编码 | RoPE,theta 5e6,`rope_scaling: null` | |
| 词表 | 130560 | `tie_word_embeddings: false` |
| bos / eos / pad | 0 / [1, 130073] / 1 | 见 Q13 |
| 精度 | bf16 | safetensors 5,033,512,960 B |

**可以展开的点**

- **参数去哪了。** MLP 42 × 37.7M = 1.59B,注意力 42 × 9.4M = 0.40B,两份嵌入 0.53B。GQA 把 K/V 投影压到 2048×256,注意力很小。
- **GQA 对我们的直接价值是 KV cache。** 每 token KV = 42 层 × 2(K,V)× 2 头 × 128 × 2 B ≈ 43 KB,一条 50k 上下文约 2.1 GB。若 16 个 KV 头,同样上下文 17 GB。实测瓶颈不是显存,是 prefill 带宽。
- **深而窄的取舍。** 同参数量下 42 层比 24 层串行延迟高,但小模型上更深通常推理能力更好,MiniCPM 系列从第一代起走这条路线。
- **训练流程。** base training(stable + decay)→ mid-training → SFT → RL → OPD。

如果追问"为什么不用同尺寸 Qwen":同尺寸 Qwen 也能跑,决定性因素是第 1、3、4 条,外加它的 agentic 后训练数据公开,做实验时知道模型见过什么。

---

## Q1. "每卡 5G"怎么算的?激活值和 vLLM 显存各多少?

**先纠正**:"每卡参数+Adam+梯度约 5G"按 verl 真实精度方案算不出来,它只在"全 bf16 优化器"假设下成立。

N = 2,516,756,480,bf16 权重 2N = 5.03 GB。FSDP 全切(`fsdp_size 4`)每卡 1/4:

| 方案 | 每参数字节 | 4 卡总量 | 每卡 |
|---|---|---|---|
| 全 bf16(权重 2 + 梯度 2 + Adam m,v 各 2) | 8 B | 20.1 GB | 5.0 GB |
| 标准混合精度(fp32 主权重 4 + fp32 梯度 4 + Adam fp32 8) | 16 B | 40.3 GB | **10.1 GB** |
| 混合精度 + bf16 计算副本常驻 | 18 B | 45.3 GB | 11.3 GB |

verl 的做法是第二行:actor 以 fp32 加载(`fsdp_workers.py:389`,`torch.float32 if self._is_actor`),FSDP `MixedPrecision(param bf16, reduce fp32, buffer fp32)`(`fsdp_workers.py:565-573`),AdamW 状态 fp32。教科书答案是 **16 B/参数 → 每卡 10 GB**。

而我们真实训练里这 10 GB **不在显存里**:`train_opencode_agent.py:96-97` 开了 `param_offload=True, optimizer_offload=True`,ref 也 `param_offload=True`(:109)。更新时把参数分片搬回 GPU(`fsdp_workers.py:1002`),更新完再搬回 CPU(:1035-1038)。代价是主机内存:step 21 `perf/cpu_memory_used_gb` 449 GB。

**激活值**

配置:micro batch = 1 条序列(`ppo_micro_batch_size_per_gpu 1`),单条最长 51200 token(`ppo_max_token_len_per_gpu`),`use_remove_padding`,flash_attention_2,梯度检查点 + 激活卸载(:116-117)。取最坏 L = 51200,h = 2048,f = 6144。

每层内部激活(bf16,flash attention 不存 L×L):
- attention:norm 输入 h + q(16×128 = h)+ k、v(各 256)+ attn 输出 h + o_proj 输入 h ≈ 4h + 512 = 8704
- MLP:gate、up 输出、SiLU 输出、down 输入 ≈ 4f = 24576
- 合计 ≈ 33,280 个数 × 2 B ≈ **65 KB/token**

| 情形 | 计算 | 显存 |
|---|---|---|
| 不开检查点,42 层全存 | 65 KB × 42 × 51200 | 140 GB,单卡放不下 |
| 检查点(只存每层入口 L×h) | 51200 × 2048 × 2 B × 42 | 8.8 GB + 重算 1 层 3.3 GB |
| 检查点 + 激活卸载(我们) | 检查点搬到 CPU | **≈ 3.3 GB** |

真正的峰值是 **logits**:L × vocab = 51200 × 130560,bf16 13.4 GB,fp32 26.7 GB,log_softmax + 反传再翻倍。所以开了 `use_fused_kernels` 和 `entropy_from_logits_with_chunking`(:98, :113-114),分块算 log_prob/熵,不落全量 logits。

实测 step 21:`perf/max_memory_allocated_gb` 14.9,`reserved` 21.9,和"一层重算 3 GB + 分块 logits + 一层 all-gather 的 bf16 权重 + 梯度分片"对得上。

**vLLM(TP2,每卡一个 rank)**

- 权重:5.03 / 2 = **2.5 GB**
- KV 每 token:2(K,V)× 2 KV 头 × 128 × 2 B × 42 层 = 43 KB;TP2 分掉 KV 头 → 21.5 KB/token/rank
- KV 预算硬写:`num_gpu_blocks_override=25600`(`restart_v4_trainer.sh:80`)× 16 token/块 = 409,600 token = max_num_seqs 8 × max_model_len 51200;409,600 × 21.5 KB = **8.8 GB/rank**(脚本第 7 行注释的 8.2 GiB)
- 运行时中间量:chunked prefill 8192 token × 6144 × 2 B ≈ 0.2 GB;`enforce_eager` 无 CUDA graph;CUDA 上下文 + NCCL ≈ 1 GB
- 合计 **≈ 12–13 GB/卡**

`gpu_memory_utilization 0.95` 只是 vLLM 探测时的上限:0.95 × 97.9 GB 再减去卡上邻居已占显存,小于所需 KV 就报 "No available memory for the cache blocks"(`kv_cache_utils.py:527`,在 override 生效之前检查)。`free_cache_engine=True` 让更新期间 KV 释放,训练峰值和 KV 不叠加。

**每卡合计**:vLLM 12–13 GB(采样期)与训练 15–22 GB(更新期)错峰,观测峰值 25–35 GB。

**"7B 装不下"也要修正**:7B × 16 B / 4 = 28 GB 优化器状态,98 GB 卡加 offload 塞得下;真正卡住的是 KV(Qwen2.5-7B 28 层 × 8 KV 头 = 114 KB/token,8 条 51k 序列每 rank 23 GB)和 CPU 卸载带来的每步时间。准确说法是"7B 在这套 4 卡流水线上 KV 预算和步长时间都不够用"。

---

## Q2. rope_theta 5e6 能推出上下文长度 131072 吗?

**不能。** theta 是必要条件,不是决定量。

RoPE 第 i 对维度的波长 λ_i = 2π · θ^(2i/d),d = head_dim = 128。最高频维度波长 2π ≈ 6 token,最低频维度 λ ≈ 2π · θ^(126/128) ≈ 2.7 × 10^7 token。经验规则是最低频维度在训练长度内不能转满一圈,否则远距离相对位置会混淆,所以 θ 给出的是**下界**:θ = 5e6 支撑 131072 绰绰有余(2π·θ ≫ L),但同一个 θ 也能配 32k 或 256k。

真正决定 131072 的是训练时见过的长度:mid-training 的长上下文阶段用多长序列训,`max_position_embeddings` 就声明多长;theta 只需要配套调大(Llama 2 用 1e4 配 4k,Llama 3 用 5e5 配 8k 再靠 scaling 拉到 128k,Qwen2.5 用 1e6 配 32k 再靠 YaRN 拉到 128k)。MiniCPM5-2B `rope_scaling: null`,说明 128k 是原生训出来的,不是靠插值。

一句话总结:θ 决定"能不能表示这么远",数据决定"会不会用这么远"。

---

## Q3. trust_remote_code 是啥,作用是啥?

HuggingFace `from_pretrained` 的开关。模型仓库可以在 `config.json` 的 `auto_map` 里指向仓库自带的 `modeling_xxx.py` / `configuration_xxx.py` / `tokenization_xxx.py`,`trust_remote_code=True` 允许 transformers 下载并**执行**这些 Python 文件,用来加载 transformers 主干里没有的架构。vLLM/SGLang 的 `--trust-remote-code` 是同一件事。

风险:执行的是仓库作者的任意代码,等于 `pip install` 一个未审计的包;生产环境要 pin revision。

对我们的意义:MiniCPM 前几代(MiniCPM-2B 有 μP 的 `scale_emb`、`dim_model_base` 缩放)需要它;MiniCPM5-2B `model_type: llama`,transformers、verl FSDP、vLLM 都走原生 Llama 路径,不需要。这直接影响 RL 可行性:verl 的权重同步、`model_merger`、vLLM 的 tensor parallel 切分都按 Llama 的参数命名走,自定义架构每一处都要适配。

---

## Q4. 为什么用 opencode,而不是 openclaw / Hermes / dsh / Claude Code / Codex / OpenHands?

按项目里实际发生的事说。

1. **它是 agent-lightning 官方 SWE-smith 示例自带的 harness**(`examples/swe_smith/`,`job-template-opencode.yaml`),k8s 作业模板、LLM 代理、trace 采集链路都是现成的。换 harness 等于重写采集层。
2. **模型无关、可离线。** opencode 通过 `opencode.json` 配 OpenAI-compatible provider(`agents/opencode_agent.py:437-456`:`@ai-sdk/openai-compatible`、`baseURL` 指向本地代理端口),不绑任何厂商账号。Claude Code 只接 Anthropic API;Codex CLI 虽开源,协议按 OpenAI Responses API 设计,接自研 2B 模型要改。
3. **可在容器里无头跑,配置面小。** `OPENCODE_CONFIG` 环境变量 + 一条命令;上下文上限、压缩阈值(`compaction.reserved`、`limit.input`)都是配置项,我们靠这些做 50K 墙(踩坑 §14)。OpenHands 自带 runtime 沙箱和事件流,system prompt 长、依赖重,32 个 pod 并发对 2B 模型的 prompt 预算不友好。
4. **工具集小而对口。** read/edit/bash/grep/glob 这类文件工具,和 SWE-smith"改代码过测试"完全匹配;2B 模型学工具协议的负担小。
5. **对照组存在。** AGL 官方用同一 harness 训过并公布了曲线,我们出问题时能和官方配方比(踩坑 §45.4)。

Hermes(Nous 的通用助手 agent)和 openclaw(个人助手/消息机器人框架)偏通用对话和长期记忆,不是 SWE harness。**dsh 我没有用过,不了解,不评价**。

诚实的一句:选 opencode 首先是因为它来自我们要贡献的上游示例,其次才是它本身合适。

---

## Q5. 400B token deep-thinking SFT,为什么 2B 模型要用这个量级?是 Chinchilla 算的吗?

**不是 Chinchilla。** Chinchilla(Hoffmann 2022)是预训练算力最优的配比,约 20 token/参数,2.5B 对应 50B 预训练 token,而且它优化的是"固定算力下最低 loss",不管推理成本。现在所有小模型都远超这个比例(Llama 3 8B 用 15T,1875 token/参数),因为部署时推理成本才是大头,预训练多花算力换更小模型划算。SFT 阶段更和 Chinchilla 无关。

README 只写"400B tokens deep-thinking SFT 建立深度思考和通用对话能力",没说数量怎么定的。以下是我的推断:

- **deep-thinking 数据每条极长。** 一条带 `<think>` 的推理样本动辄 8k–32k token,400B token 折成样本量只有几千万条,和 UltraData-SFT-2605 这类公开集的规模是一个数量级,可能含多 epoch。
- **小模型靠模仿学推理的效率低。** 2B 容量小,长 CoT 的 SFT 曲线在几百万样本后仍在下降(OpenThoughts、AM-Thinking 一类工作的规模实验都显示这一点),所以"喂到不再涨"的量就会很大。
- **这一阶段实际上是蒸馏。** 数据来自更大的教师,400B 更像把教师的推理分布"压"进 2B,类似 Llama 3.2 1B/3B、Gemma 小模型的做法。

准确的说法是:数量由"SFT loss/下游指标还在涨"的经验曲线决定,不是公式。

---

## Q6. JustRL II 是啥?它的 OPD 是 GKD 还是 PG-OPD?

**JustRL II**(MiniCPM RL 团队,博客 "Scaling Small LLMs to 128k Reasoning with a Critic",代码仓 JustRL-II `docs/method.md`):面向小模型 128k 长 CoT 的 RL 配方。要点:

- **带 critic。** 沿用 GRPO 的分组采样结构,但优势不用组内均值做基线,而用一个独立 critic 做 GAE。critic 是策略同架构的第二份副本,**去掉 LM head 换成标量 value head**("cc-noLM"),只训 value loss。
- **value head 初始化 = 平均奖励。** 权重置零、bias 初始化为期望奖励(发布配置 0.52),加载策略 ckpt 后重新置零,去掉前 25 步的 value 震荡;再加 30 步 critic-only 预热。
- **长度自适应 GAE λ**:λ_i = k^(1/L_i),k = 0.5,让终端奖励传到序列开头的比例与长度无关(优势用 λ_i,value 目标用 λ = 1 的回报,fp32 计算,否则 128k 时 λ 在 bf16 下舍入成 1)。
- **DAPO 软超长惩罚**,但 critic 的回归目标**不含**该惩罚,让 value 只建模"会不会对"。
- 数据侧三阶段清洗、按目标 checkpoint 重标定难度。

**OPD 是 PG-OPD,不是 GKD。** README 原话:"在 response 序列的每个位置分别对学生模型和教师模型 logits 计算全词表的反向 KL 散度作为优势估计值,替代原有的 verification-based advantage"。
- GKD(Agarwal 2023):学生自采样,直接对每个位置最小化学生与教师分布的散度(JSD/KL),是**监督式损失**,梯度经过 logits。
- PG-OPD(Thinking Machines 的 on-policy distillation 那一路):学生自采样,每 token 的 −KL(student‖teacher) 作为**奖励/优势**,走策略梯度更新。
MiniCPM5 用的是后者,和它 RL 阶段的训练器共用一套 advantage 接口,所以 OPD 能直接复用 RL teacher 的 prompt,"无需额外构造语料"。

---

## Q7. 非嵌入参数 1.98B,用公式算嵌入和非嵌入参数

记 V = 130560,h = 2048,L = 42,n_h = 16,n_kv = 2,d = 128,f = 6144。Llama 结构无 bias。

嵌入(不共享,两份):
- embed_tokens:V × h = 130560 × 2048 = 267,386,880
- lm_head:V × h = 267,386,880
- 合计 **534,773,760**(占总量 21%)

每层:
- q_proj:h × (n_h·d) = 2048 × 2048 = 4,194,304
- k_proj:h × (n_kv·d) = 2048 × 256 = 524,288
- v_proj:524,288
- o_proj:(n_h·d) × h = 4,194,304
- 注意力小计 9,437,184
- gate_proj + up_proj + down_proj:3 × h × f = 3 × 2048 × 6144 = 37,748,736
- 两个 RMSNorm:2h = 4,096
- 每层 **47,190,016**

非嵌入 = L × 47,190,016 + 最终 norm h = 1,981,980,672 + 2,048 = **1,981,982,720**(≈ 1.98B,与官方一致)

总参数 = 534,773,760 + 1,981,982,720 = **2,516,756,480**;× 2 B = 5,033,512,960,与 `model.safetensors.index.json` 的 `total_size` 完全相等。

---

## Q8. 手写 GQA

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    """Grouped-Query Attention: n_heads 个 Q 头共享 n_kv_heads 组 K/V。
    MiniCPM5-2B: hidden 2048, n_heads 16, n_kv_heads 2, head_dim 128 -> 每组 KV 供 8 个 Q 头。"""

    def __init__(self, hidden=2048, n_heads=16, n_kv_heads=2, head_dim=128):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.group = n_heads // n_kv_heads                       # 8
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)     # 2048 -> 2048
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)  # 2048 -> 256
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)  # 2048 -> 256
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)     # 2048 -> 2048

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, 16, T, 128]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, 2, T, 128]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # (RoPE 在这里作用于 q、k,略)

        if kv_cache is not None:                     # 解码时只缓存 2 个 KV 头,这就是 GQA 省显存的地方
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # 把每组 KV 复制给它的 8 个 Q 头(repeat_interleave 保证第 i 个 Q 头对应第 i//8 组)
        k = k.repeat_interleave(self.group, dim=1)   # [B, 16, S, 128]
        v = v.repeat_interleave(self.group, dim=1)

        S = k.shape[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)          # [B, 16, T, S]
        causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), new_cache
```

追问要点:
- MHA 是 n_kv = n_heads,MQA 是 n_kv = 1,GQA 介于两者;KV cache 和 K/V 投影参数都按 n_kv/n_heads 缩小(这里 1/8)。
- 实际实现不会真的 `repeat_interleave`,flash-attn / SDPA 的 `enable_gqa=True` 在 kernel 里按索引广播,不复制内存。
- 精度:softmax 用 fp32 再转回 bf16。

---

## Q9. 6144 是 3 倍吗?为什么选 3 而不是 8/3?

6144 / 2048 = **3,精确**。

8/3 的来历:原始 Transformer FFN 是两个矩阵 h→4h→h,参数 8h²。换成 SwiGLU 多了一个 gate 矩阵,变成三个 h→f 的矩阵,参数 3hf。要和原来持平,3hf = 8h² → f = 8h/3。Llama 就是这么定的(再向上取整到 256 的倍数,Llama-7B 4096 → 11008)。所以 8/3 只是"参数量和标准 FFN 对齐"的约定,不是最优解。

官方未说明为什么选 3,以下是推断:
- **小模型倾向更宽的 FFN。** 知识主要存在 FFN 里,参数少时把比例往上抬更划算;Qwen3-1.7B 同样是 2048/6144 = 3,Qwen2.5-1.5B 是 1536/8960 ≈ 5.8,MiniCPM-2B(一代)2304/5760 = 2.5。
- **硬件对齐。** 6144 = 48 × 128,TP2 切完 3072 仍是 128 倍数,矩阵乘和 GQA 的 tile 都整齐;8/3 × 2048 = 5461,要取整到 5504,和 6144 差 12%,不如直接取整数倍。
- 42 层 × 3hf 的 MLP 已占 1.59B(63%),再宽就要减层,他们选了深而窄(见 Q0)。

---

## Q10. SwiGLU 原理,和其他激活函数的对比

**SwiGLU**(Shazeer 2020 "GLU Variants Improve Transformer"):

FFN(x) = W_down · ( Swish(W_gate x) ⊙ (W_up x) ),Swish(z) = z · σ(βz),β = 1 时就是 SiLU(`hidden_act: silu`)。

原理:GLU 把 FFN 拆成"值通路"W_up x 和"门通路"σ-类函数(W_gate x),逐元素相乘。门让每个隐单元根据输入决定通过多少,相当于输入相关的动态稀疏;从表达力上看,乘法交互比单纯的非线性更强(二阶项)。代价是三个矩阵而非两个,所以 f 要缩到 8h/3 或按需取。

对比:

| 激活 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| ReLU | max(0, z) | 计算最便宜,稀疏 | 负半轴梯度为 0,"死神经元";零点不可导 |
| GELU | z·Φ(z) | 平滑,负半轴有小梯度,BERT/GPT-2 的默认 | 需要 erf 或 tanh 近似,略贵 |
| Swish/SiLU | z·σ(z) | 平滑、非单调、有自门控,效果与 GELU 接近 | 同上 |
| GLU 家族(ReGLU/GEGLU/SwiGLU) | act(W1x) ⊙ W2x | 同参数量下 loss 更低(Shazeer 的实验,SwiGLU/GEGLU 最好) | 多一个矩阵乘和一次逐元素乘,访存增加;实现要 fuse |
| Squared ReLU | max(0,z)² | 更稀疏,Primer 提出 | 数值范围大,bf16 要小心 |

为什么 Llama 系列(含 MiniCPM5)都选 SwiGLU:同算力下困惑度稳定优于 GELU/ReLU,且 Swish 在小 z 处近似线性、梯度平滑,配合 pre-norm 训练稳定。Shazeer 本人的解释是"没有理论,归功于神的恩典"(原文如此),可以引用来说明这是经验选择。

---

## Q11. RMSNorm 原理,为什么用它;pre-norm 与位置的影响

**RMSNorm**(Zhang & Sennrich 2019):y = x / RMS(x) · g,RMS(x) = sqrt(mean(x²) + ε),ε = 1e-6(`rms_norm_eps`)。相比 LayerNorm 去掉了减均值和 bias:

- LayerNorm 做两次归约(均值、方差),RMSNorm 一次;在 42 层 × 2 个 norm、序列 51k 时是可观的带宽节省(norm 是访存瓶颈算子)。
- 论文论证 LayerNorm 的收益主要来自**重缩放不变性**,不来自重定心;去掉均值后效果持平。
- 少一个 bias 参数,少一个可能漂移的量;fp32 里做 RMS 再 cast 回 bf16 是所有 Llama 实现的标准。

其他选择及为什么不用:BatchNorm 依赖 batch 统计,自回归/变长序列不适用;LayerNorm 可用但更贵、无增益;DeepNorm 是 post-norm 变体,配合特定初始化,主流预训练不采用。

**pre-norm vs post-norm**

- 原始 Transformer 是 post-norm:x + Sublayer(x) 之后归一化。残差流每层都被重新缩放,深层梯度经过多个 norm 的雅可比,初期梯度大,必须 warmup,层数深了不稳定。
- pre-norm:x + Sublayer(Norm(x))。残差流是干净的恒等通路,梯度直接回传,深层网络不用 warmup 也能训,几乎所有 LLM 都用它。`LlamaForCausalLM` 的 `input_layernorm`、`post_attention_layernorm` 都在子层**之前**,是 pre-norm。
- 代价:残差流范数随层数单调增长,后面几层的子层输出相对残差越来越小(贡献被稀释),同深度下 post-norm 收敛后的效果略好。补救方案:Sandwich norm(子层前后各一,CogView)、Gemma 2/3 的 pre+post 双 norm、QK-norm 稳注意力 logits、Peri-LN。
- 对 RL 训练的实际意义:pre-norm 的稳定让小学习率下更新平滑;我们观察到的 v3 塌缩不是 norm 问题,是策略漂移。

---

## Q12. 为什么 embedding 与 lm_head 不共享?分词方式是什么,为什么选它?

**不共享**(`tie_word_embeddings: false`,safetensors 里 `model.embed_tokens.weight` 和 `lm_head.weight` 各一份)。共享能省 267M 参数(占模型 10.6%),Qwen3-1.7B、Gemma、Llama-3.2-1B/3B 都共享。官方未说明,推断:

- 输入嵌入学的是"token 的语义向量",输出矩阵学的是"隐状态到下一个 token 的判别方向",两者几何角色不同;共享等于强制两组向量同构,大模型上代价小,小模型上会限制容量(有工作显示解绑后小模型困惑度更低)。
- 面壁的目标是"2B 打 4B",宁可多 0.27B 也要保效果;130k 大词表让这份代价更显眼,但他们接受了。
- 工程副作用:vLLM/FSDP 切分时两份矩阵各自处理,`model_merger` 也不用特殊处理 tied 权重。

**分词**:`tokenizer.json` 的 `model.type` 是 **BPE**,byte-level。基础 BPE 词表 130,072(id 0–130071,其中 0–21 是 `<s>`、`</s>`、`<think>` 这类特殊符号),之上 488 个 added token 占 id 130072–130559(`<|im_start|>`、`<|im_end|>`、`/think`、`/no_think` 及大量预留位),合计 130,560 = 1020 × 128,对齐到 128 倍数便于矩阵分块。pre-tokenizer 先把数字按 1–3 位切开(`\p{N}{1,3}`),再用 GPT-4 cl100k 风格的正则切词(缩写、字母串、标点、空白)。

为什么 BPE(byte-level):
- 无 UNK,任何字节串都能编码,代码、日志、混合语言都安全;SWE 任务里大量 diff、路径、hex,这一点最实用。
- 130k 词表对中文压缩率高(单字/常见词一 token),长上下文场景直接省 KV。
- 数字 1–3 位切分让算术更规整(Llama 3、Qwen 同样做法)。
- 对比:SentencePiece unigram(Llama 1/2、T5)需要指定字符覆盖率,罕见字符回退到 byte fallback;WordPiece(BERT)有 UNK。HF fast tokenizer 的 BPE 实现最成熟,和 vLLM 的增量 detokenize 兼容性最好。

---

## Q13. eos 为什么有两个 id?bos 呢?

`eos_token_id: [1, 130073]`(config 与 generation_config 一致):
- **1 = `</s>`**:预训练阶段的文本结束符,tokenizer 层面的 `eos_token`,同时也是 `pad_token`(`pad_token_id: 1`)。
- **130073 = `<|im_end|>`**:ChatML 风格的**轮次结束符**,chat template 每个 message 以 `<|im_end|>\n` 收尾(`chat_template.jinja:21,42,151`)。

两个都列进 eos 是为了让 `generate`/vLLM 遇到任一都停:对话模型正常停在 `<|im_end|>`,少数情况下(续写、模板外用法)会吐 `</s>`,不列的话会继续生成垃圾。这是后训练把 ChatML 标记加进基座后的通用做法,Qwen 也是 `[151645, 151643]` 两个。

**bos 只有一个**:`bos_token_id: 0` = `<s>`。注意 `tokenizer_config.json` 里 `add_bos_token: false`,tokenizer 本身不加,而 chat template 第一行 `{{- bos_token }}` 显式加,所以走 `apply_chat_template` 才有 bos,直接 `tokenizer(text)` 没有。训练侧要保证两条路径一致,否则 log_prob 的第一个 token 会对不上。`<|im_start|>` 是 130072,不是 bos。

---

## Q14. stable + decay 是啥?

README 的 "base training 包含 stable training 与 decay training" 指 **WSD(Warmup-Stable-Decay)学习率调度**,MiniCPM 一代论文(Hu et al. 2024)提出:

1. **Warmup**:短暂线性升到峰值 lr。
2. **Stable**:很长一段**恒定**峰值 lr,这一段可以无限延长、持续加数据,任何时刻的 checkpoint 都能作为后续起点。
3. **Decay**:最后约 10% token 里 lr 快速(指数/余弦)衰减到很小,loss 在这一段急剧下降;这一段同时换成高质量"退火数据"(高质量网页、代码、数学、SFT 风格数据),把能力"收口"。

和 cosine 的区别:cosine 的总步数在开始就得定死,改数据量就要重训;WSD 的 stable 段是"开放的",从任一 stable checkpoint 分支出一个 decay 就得到一个可用模型,所以做 scaling law 实验只需一条 stable 主线 + 多条短 decay,成本降一个量级。MiniCPM 正是靠它在小模型上做了大量配比实验。

在 MiniCPM5-2B 的流程里:stable → decay 构成 base training,之后 mid-training 再强化目标能力(长上下文、代码、推理),再进 SFT/RL/OPD。对我们 RL 的启示是同一思路的缩小版:恒定小 lr(2e-6)+ 探针,而不是预设总步数的余弦。

---

## Q15. 它的"原生工具格式"指的是啥?

指模型在 SFT/RL 时学的、写死在 `chat_template.jinja` 里的工具调用文本协议,**XML 风格**,不是 OpenAI 的 JSON `tool_calls`,也不是 Hermes 的 `<tool_call>{json}</tool_call>`:

- **工具声明**:system 段用 `<tools>...</tools>` 列出函数(模板第 1–21 行),并附使用说明。
- **模型发起调用**(assistant 内容里,模板 :69-73, :97-101, :135-139):
  ```xml
  <function name="bash"><param name="command">pytest -x</param></function>
  ```
  多行或含 `<`、`&` 的值用 `<![CDATA[...]]>` 包裹;可以连续多个 `<function>`。
- **工具返回**:作为 user 轮,内容包在 `<tool_response>...</tool_response>` 里(模板 :30 用它识别工具回合)。
- **思考**:`<think>...</think>` 前缀,`enable_thinking` 关闭时模板填空 think 块(:120-122)。
- 词表里预留了这些标记为 special token:`<think>` 8、`</think>` 9、`<tool_response>` 10/11、`<tools>` 12/13、`<|im_start|>` 130072、`<|im_end|>` 130073,还保留了 Hermes 风格的 `<tool_call>` 2/3 和 `<|tool_call|>` 130077 但模板不用。

"原生"的含义:推理框架要把这段 XML 解析回 OpenAI 格式的 `tool_calls` 给 harness 用。SGLang 内置 `minicpm5` 解析器;vLLM 0.8.5 没有,我们自己写了插件(`restart_v4_trainer.sh:81-82`,`tool_call_parser=minicpm5`,`tool_parser_plugin=minicpm5_parser_plugin_085.py`),并把模型自带的 `chat_template.jinja` 传给 vLLM。早期用 Hermes 模板(`swe_smith_chat_template.jinja`)让模型输出 JSON,等于让它说一门后训练没学过的方言,这是后来切回原生格式的原因。
