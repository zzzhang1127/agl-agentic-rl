# Critic-only 续训验收 · 2026-10-16

数据盘余量恢复后，从唯一完整的 step40 checkpoint 继续训练。未保存的
step41–59没有作为恢复来源，也没有重新初始化critic。

## 已观察结果

- 两个rank原生恢复model、optimizer、RNG与scheduler，恢复日志明确
  `CRITIC_ONLY_RESUME step=40`、`actor_loaded=0`。
- 42层critic attention保持冻结，首个新的有效更新为**step42**。
  前一个不足批次因终止协议/奖励批准条件被拒绝，不伪记为有效更新。
- step42：batch reward=0.5，actor update=0，vf loss=0.34131，
  grad norm=152.24，explained variance=-0.24219；无训练截断或prefix skip。
- 检查时5条归档中4条显式批准、1条无效终止拒绝；19次模型请求token IDs
  与logprobs长度对齐。不能把拒绝轨迹的零值当作获准训练的零奖励。
- SwanLab新运行已建立，critic-only/attention冻结/save20与配置脱敏均已读回。
  最新完整保存仍为step40，下一次常规保存为step60。

## 存储策略修正

此前额外设置的根分区12GiB硬阈值，使数据盘余量恢复后仍无法启动。
当前所有训练权重、轨迹、日志、Ray临时文件和编译缓存均配置在数据盘，
因此移除根分区空间作为启动/运行拦截条件；根盘余量仅用于观测。
保留实际写入分区的准入64GiB、运行保护32GiB，以及数值/actor/奖励供给保护。

新增CUDA cache与FlashInfer workspace显式路径。运行中检查本任务53个进程，
当时未发现它们持有`/root`或`/tmp`下普通文件的可写句柄。这是运行时快照，
不把它表述为对所有未来库写入行为的证明。

portable launcher与storage helper已同步修正，overlay manifest重新计算。
存储回归2 passed、1 skipped；实际启动/运行准入表达式5项检查通过。
训练核心补丁与code快照未因这次存储策略修正而改变。

## 质量边界

单个小batch的EV接近零不构成质量验收。继续保持固定actor和冻结attention，
按200步有界预算观察奖励和value趋势，之后用独立正负留出轨迹验收。
不因工程恢复成功自动打开actor，也不宣称VAL474提升。
