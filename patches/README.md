# patches/ — 对 agent-lightning 核心库和 verl 的改动

本目录是本项目对上游框架的全部改动，以 diff 形式给出，可以直接套在上游对应版本上。

| 文件 | 针对 | 内容 |
|---|---|---|
| `agentlightning-core.diff` | microsoft/agent-lightning，基线提交 `218f1f7c`（2026-09 主干） | 5 个文件、+1016/−49 行，见下表 |
| `tests/` | 同上 | 为上述改动写的 pytest 单测（动态采样、mini-batch 适配、rollout 合并、行打乱） |
| `verl_dp_actor_dp_group.patch` + `README-verl-patch.md` | verl 0.7.1 `workers/actor/dp_actor.py` | 修 `compute_log_prob`/`update_policy` 漏传 `dp_group` 导致的跨 rank all-gather 挂死 |

## 应用方法

```bash
git clone https://github.com/microsoft/agent-lightning && cd agent-lightning
git checkout 218f1f7c
git apply --3way /path/to/patches/agentlightning-core.diff
cp -r /path/to/patches/tests/* tests/verl/
pytest tests/verl -q

# verl 补丁（装好 verl 0.7.1 之后）
V=$(python -c 'import verl,os;print(os.path.dirname(verl.__file__))')
patch -p0 -d $V/workers/actor < /path/to/patches/verl_dp_actor_dp_group.patch
```

## agentlightning-core.diff 改了什么

按文件说明，每一条都对应 `docs/pitfalls-and-lessons.md` 里一次真实事故。

### `agentlightning/server/proxy.py`（+330）

代理是 trainer 和 agent 之间的 OpenAI 兼容网关，所有训练样本都从这里采集。改动都是为了让「采到的 token 序列」与「模型真正看到/生成的序列」逐 token 一致：

- **工具调用轮不再被丢**：原版对带 `tool_calls` 的响应渲染 chat template 会抛错并被静默吞掉，agent 的所有工具调用轮都没进训练 batch，采集率只有 8.1%（pitfalls §41/§44）。现在把 `tool_calls` 内联成文本后再渲染（`_inline_tool_calls`、`_render_tool_calls_xml`、`_coerce_tool_call_arguments`）。
- **透传 `chat_template_kwargs`**（如 `enable_thinking`）：原版丢掉它，导致代理渲染的 prompt 与 vLLM 实际渲染的不一致，多轮 episode 合并 89% 失败（§57）。
- **缺失的 token id 从 logprobs 回填并与 `usage` 交叉校验**（`_response_ids_from_logprobs`、`_check_against_usage`、`_fill_missing_token_ids`）。
- **采集率统计与硬闸门**：`CaptureTally` 按分支计数，`AGL_MIN_CAPTURE_RATE`（默认 0.9）低于阈值直接报错，而不是悄悄用 8% 的数据训练。
- 模型名路由：trainer 的模型路径与 `default_proxy.model_name` 不一致时，路由到唯一注册的模型。

### `agentlightning/verl/rollout_adapter.py`（+313）

把一条多轮 episode 的若干次 LLM 调用合并成一个训练样本：

- **文本级续接**（`text_level_continuation`）：当新一轮 prompt 的前缀与上一轮「prompt+response」在 token 上不相等（chat template 的 `<think>` 不对称、`</think>` 重切，§56/§58）时退回文本级比对，并产出逐 token 分歧报告（`_diverge_report`）。
- **合并失败的样本落盘**（`_save_trace_merge_mismatches_locally`）用于取证；修复前每条 episode 被打碎成「一轮一行」，前向算力浪费 18.3×（§56）。

### `agentlightning/verl/agl_rollout_manager.py`（+80）

- 一个 batch 零 LLM 调用被采集时直接失败（而不是训练出一个空 batch）。
- 每步把 rollout 轨迹、合并统计、随机样本写到本地并记入 SwanLab（`_save_rollout_trajectories_locally`、`_log_rollout_step_metrics_to_swanlab`、`_log_random_rollout_sample_to_swanlab`）。

### `agentlightning/verl/trainer.py`（+340）

- **DAPO 风格动态采样**（`agentlightning.dynamic_sampling.{enabled,max_gen_batches,min_valid_groups}`）：GRPO 下同奖励组优势为 0、梯度为 0，整组丢弃并按需补采（`_split_zero_adv_groups`、`_collect_train_batch`）。
- **mini-batch 适配**（`_mini_batch_fit`）：合并修好后一步只有几十行，按 `ppo_mini_batch_size` 和 `max_ppo_update_times` 推出实际更新次数，不够时用中性复制补齐（`_pad_with_neutral_duplicates`）。
- **行打乱**（`agentlightning.shuffle_train_rows`，默认开）：mini-batch 切分前打乱行，避免同一组样本总落在同一个 mini-batch。
- **原子化保存与旧步清理**（`_save_checkpoint`、`_atomic_replace_dir`、`_prune_old_step_dirs`）：磁盘吃紧的共享机上半写 ckpt 曾经毁掉 resume。

### `agentlightning/server/routes/proxy.py`（2 行）

路由层把模型名传给 `prepare_body`。
