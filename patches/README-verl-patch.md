# 本地打在 venv 里的补丁(不随 pip 安装走,重建 venv 后要重打)

| 补丁 | 目标 | 应用 |
|---|---|---|
| `verl_dp_actor_dp_group.patch` | `verl 0.7.1` `verl/workers/actor/dp_actor.py` | 见下 |

## verl_dp_actor_dp_group.patch(2026-10-03)

`dp_actor.py` 的 `compute_log_prob` / `update_policy` 调 `prepare_dynamic_batch()` 时漏传
`dp_group`,导致 `verl/utils/seqlen_balancing.py:402` 的跨 rank `all_reduce(MAX)` 被静默跳过,
各 DP rank 按自己分片各算 `num_micro_batches`,少的那个提前退出循环、不再参与 FSDP 逐层
`_ALLGATHER_BASE`,其余 rank 自旋到 `nccl_timeout`(3600s)才被 watchdog 打掉。
实测 3 步命中 2 次。根因、判别法和"为什么判定数值中性"见
[`../docs/pitfalls-and-lessons.md`](../docs/pitfalls-and-lessons.md) §52.1 / §52.2。

```bash
V=$(python -c "import verl,os;print(os.path.dirname(verl.__file__))")
cp $V/workers/actor/dp_actor.py $V/workers/actor/dp_actor.py.orig
patch -p0 -d $V/workers/actor < examples/swe_smith/patches/verl_dp_actor_dp_group.patch
```

验证(**不能用 `inspect.getsource`**,那两个方法被 `verl/utils/profiler/performance.py:104`
的装饰器包了且没用 `functools.wraps`,会给假阴性):

```python
import verl.workers.actor.dp_actor as m
def unwrap(f, depth=6):
    for _ in range(depth):
        cl = getattr(f, "__closure__", None)
        if not cl: break
        cands = [c.cell_contents for c in cl
                 if callable(c.cell_contents) and hasattr(c.cell_contents, "__code__")]
        if not cands: break
        f = cands[0]
    return f
for n in ("compute_log_prob", "update_policy"):
    assert "_dp_group" in unwrap(getattr(m.DataParallelPPOActor, n)).__code__.co_names, n
```

**上游**:值得往 verl / AGL 提,和 AGL #589 同类(失败被静默跳过,症状出现在一小时后的另一个地方)。
提之前要人类确认(对外动作)。
