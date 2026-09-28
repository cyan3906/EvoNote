# EvoRAG 实体治理后续阶段

阶段 1 已完成：实体别名映射层

- MySQL `evorag_entity_aliases`
- Redis alias cache
- 合并前优先查 alias，命中则跳过 ES/Milvus

阶段 2 已完成：入库任务持久化

- MySQL `evorag_ingest_jobs`
- MySQL `evorag_incoming_entities`
- `/ingest` 同步完成抽取后写入 job 和 incoming entities

阶段状态：

## 阶段 3：Redis Stream 队列和后台 Worker

- 使用 Redis Stream 投递 incoming entity 任务。
- Worker 从 Stream 消费任务。
- Worker 抢锁、重试、失败转 dead letter。
- 服务恢复后从 MySQL pending / expired processing 任务重建队列。

状态：已完成。

## 阶段 4：自动合并策略

- 先查 alias cache / alias 表。
- 未命中再查 ES / Milvus。
- `score >= 0.9` 自动合并。
- `0.7 <= score < 0.9` 进入人工审核。
- `score < 0.7` 新建实体或进入低优先级审核。

## 阶段 5：人工审核和负样本

- 建 review task 表。
- 建 rejection / negative pair 表。
- 人工确认合并后反哺 alias 表和 Redis cache。
- 人工拒绝后记录负样本，后续检索降权或过滤。

## 阶段 6：观测和前端进度展示

- job status API。
- 前端展示 job 处理进度。
- 展示 queued / processing / auto merged / review / failed。
- 展示 worker 队列长度、平均耗时、失败重试数。
