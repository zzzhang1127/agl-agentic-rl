# OpenCode任务终止判定修复（2026-10-16）

旧runner在任意 `step_finish(reason=stop)` 时停止进程。OpenCode压缩摘要也发出
这个事件，导致任务尚未完成就被评分。审计和旧训练的解释边界见
[step200记录中的纠正](critic-final-20261016.md)。step200保留作诊断，actor继续禁用。

## 修复行为

`opencode_terminal.py`按事件的session ID和message ID读取固定版OpenCode的
会话导出，核对assistant身份、agent、summary与finish字段。只有非summary的
build agent真正stop才可终止；compaction继续主任务，未知或导出失败不批准奖励。
循环内终止和最终structured评分均检查，覆盖CLI自然退出时缓冲的最后事件。

明确关闭子进程stdin，避免CLI等待父进程管道EOF。长会话导出使用匿名临时文件，
不通过stdout管道：所用1.18.28版本在真实任务中出现了16 KiB管道截断，而完整
文件导出为107127字节、30条message，能按ID识别compaction。临时文件位于runner
挂载的日志目录，关闭后释放，不保留另一份模型或会话副本。

## 已验证与待验证

- 固定版CLI加本地假provider真实运行：`build → compaction → build`；两次stop
  分别被识别为compaction和task_stop。假provider模拟token usage触发压缩，
  不调用GPU、不改变真实生成预算。
- 真实任务容器的长会话导出已完整解析并按ID识别compaction。
- 任务终止元数据、长导出、runner评分合约与proxy SSE的针对性回归64项通过。
- 发布包48个overlay哈希及核心补丁逆向检查通过；包内27个回归模块在本地为
  373 passed、9 skipped。另一个 `test_agl_rollout_manager` 模块因本地缺少
  `httpx_retries` 未运行，没有为此安装或升级依赖。
- 隔离v9快照的真实烟测已走通主任务终止、结构化评分、token/logprob对齐与
  初始化/step200的有限value推理；runner超时轨迹被拒绝，不补零。
  两轮各只有一个批准样本，不能合并当作独立质量评估。
- 固定留出集的四卡双分片评估正在运行；最终质量指标尚未产生，actor继续禁用。
  详见[独立评估进展与边界](critic-value-eval-20261016.md)。

框架核心补丁保持不变；overlay新增helper和测试，并更新agent及manifest。
应用方式仍见[critic包说明](../swe_smith/critic_warmup/README.md)。完成真实任务验收
后重新采集固定留出名单，不使用此前摘要终止轨迹补足负样本，不自动重开训练。
