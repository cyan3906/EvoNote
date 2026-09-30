# memory_guarded_hybrid 实体消歧策略

`memory_guarded_hybrid` 是实体消歧评测中的经验记忆增强策略。它的核心目标不是替代 `hybrid_rrf_baseline`，而是在 hybrid 前增加几层守门：先复用 Redis 快路径，再用 LLM 判断 incoming 本身是不是实体，之后才查 MySQL 关系经验；如果这些都不能给出可靠结论，才回退到 ES + Milvus hybrid RRF。

## 运行命令

```powershell
cd E:\huadian\evonote\backend\python
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid
```

只跑某个类别，例如 `nil_new_entity`：

```powershell
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid --category nil_new_entity
```

只跑某个类别的前 10 条：

```powershell
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid --category nil_new_entity --limit 10
```

如需重建评测 seed：

```powershell
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid --rebuild-seeds
```

也可以组合重建 seed 和分类过滤：

```powershell
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid --category nil_new_entity --limit 10 --rebuild-seeds
```

## 报告输出路径

不传 `--out` 时，评估报告会按参数自动分目录保存，避免不同实验互相覆盖：

```text
tests/EvoRAG/results/
  memory_guarded_hybrid__category-nil_new_entity__limit-10__top-k-5/
    entity_resolution_eval_report__20260930_143012_123456.json
```

路径签名由 `strategy / category / limit / top-k / rebuild-seeds` 拼接得到。同一组参数重复运行时，文件名会追加时间戳，因此也不会覆盖上一次结果。

如果需要固定输出文件，可以显式传 `--out`：

```powershell
python -m app.EvoRAG.entity_resolution_eval --strategy memory_guarded_hybrid --category nil_new_entity --limit 10 --out tests\EvoRAG\results\my_report.json
```

显式 `--out` 会完全使用你指定的路径；如果路径相同，会覆盖旧文件。

## 总体流程

```text
incoming entity
  -> Redis direct relation
      -> allow：直接 matched，结束
      -> 未命中或非 allow：继续
  -> DeepSeek admission guard
      -> entity：继续
      -> attribute / section_title / process_step / metric_or_property / not_entity：输出 new，结束
  -> MySQL relation memory top30
      -> 无关系：回退 hybrid
      -> 有关系：进入 DeepSeek relation guard
          -> matched + matched_entity_id 命中关系记录：直接 matched，结束
          -> attribute_of_entity / section_title_of_entity / process_step_of_entity / metric_or_property：输出 new，写 reject 经验，结束
          -> not_matched / ambiguous：回退 hybrid
  -> hybrid_rrf_baseline
      -> incoming embedding
      -> Milvus 向量检索
      -> Elasticsearch 关键词检索
      -> RRF 融合
      -> DeepSeek final judge
      -> 按规则沉淀经验
      -> 输出最终结果
```

## 1. Redis direct relation

第一层先查 Redis，key 由 incoming 的 `scope + normalized_name + entity_type` 组成。

Redis 只承担快路径职责：

- 如果直接命中白名单关系，例如 `allow / matched / same / equal`，直接输出 `matched`。
- 此时不会调用 admission guard、MySQL relation memory、hybrid，也不会生成 incoming embedding。
- 输出的 `strategy_trace.stage` 是 `redis_direct_allow`。

如果 Redis 没命中，或命中的是非 allow 关系，则不会直接合并，继续进入 admission guard。

## 2. DeepSeek admission guard

第二层判断 incoming 本身是否值得作为实体消歧对象。这一层不依赖候选实体，也不查 MySQL top30。

输入：

- `case_id`
- incoming snapshot：`name / normalized_name / entity_type / aliases / identity_description / description_for_match / scope`

允许输出：

```text
entity
attribute
section_title
process_step
metric_or_property
not_entity
```

处理规则：

- `entity`：认为 incoming 可以继续进入实体消歧流程。
- `attribute / section_title / process_step / metric_or_property / not_entity`：直接输出 `new`，不再查 MySQL，也不再走 hybrid。

这一层没有候选实体，所以不会写关系表经验。它的作用是拦住冷启动场景下的非实体输入，例如“优点”“缺点”“应用场景”“第一步”“锁粒度”等。

被拦截时：

```text
judgment.decision = new
judgment.matched_seed_id = null
strategy_trace.stage = llm_admission_guard_reject
strategy_trace.admission_decision = <LLM decision>
```

## 3. MySQL relation memory top30

只有 admission guard 输出 `entity` 后，才查询 MySQL 关系记忆表 `evorag_entity_resolution_relation_memory`。

查询规则：

- 必须在同一个 `scope` 下。
- incoming 可以匹配关系表左侧实体，也可以匹配右侧实体。
- 按 `hit_count DESC, confidence DESC, updated_at DESC` 排序。
- 最多取前 30 条。

关系会分成两类提供给后续 LLM：

- 白名单：`allow / white / whitelist / same / same_entity / matched / equal`
- 黑名单：`reject / black / blacklist / different / not_same / not_equal / new`

如果 MySQL 没有查到任何关系，流程直接回退到 hybrid，`strategy_trace.stage = no_relation_memory_fallback_hybrid`。

## 4. DeepSeek relation guard

如果 MySQL top30 有关系，系统会把 incoming、白名单关系、黑名单关系一起交给 DeepSeek。注意这一层和 admission guard 不同：它判断的是 incoming 与候选实体之间的关系。

允许输出：

```text
matched
not_matched
ambiguous
attribute_of_entity
section_title_of_entity
process_step_of_entity
metric_or_property
```

处理规则：

- `matched`：如果 `matched_entity_id` 能对应到 top30 关系里的候选实体，直接输出 `matched`。
- `attribute_of_entity / section_title_of_entity / process_step_of_entity / metric_or_property`：说明 incoming 不是该候选的同一实体，而是候选的属性、章节、步骤或指标；直接输出 `new`，并写入 reject 经验。
- `not_matched / ambiguous`：不直接下结论，继续回退 hybrid。

matched 快路径输出：

```text
strategy_trace.stage = mysql_relation_memory_judge
judgment.decision = matched
```

非合并关系输出：

```text
strategy_trace.stage = llm_relation_guard_reject
strategy_trace.relation_guard_decision = attribute_of_entity | section_title_of_entity | process_step_of_entity | metric_or_property
judgment.decision = new
```

同时写入关系经验：

```text
decision = reject
relation_type = <relation_guard_decision>
source = llm_guard
```

## 5. 回退 hybrid RRF

如果 Redis、admission guard、MySQL relation guard 都不能直接结束流程，则回退到原来的 `hybrid_rrf_baseline`。

步骤：

1. 为 incoming 生成 embedding。
2. 从 Milvus 做向量检索。
3. 从 Elasticsearch 做关键词检索。
4. 使用 RRF 融合两路候选。
5. 把融合候选交给 DeepSeek final judge。
6. 根据结果和分数写入经验。

hybrid 回退阶段可能出现两个 trace：

```text
strategy_trace.stage = memory_judge_fallback_hybrid
```

表示 MySQL top30 有关系，但 relation guard 输出 `not_matched / ambiguous`，所以继续 hybrid。

```text
strategy_trace.stage = no_relation_memory_fallback_hybrid
```

表示 admission 通过后，MySQL top30 没有可用关系，直接进入 hybrid。

## 6. 经验写入规则

经验写入发生在两处：relation guard 非合并关系，以及 hybrid final judge 后。

### 6.1 relation guard 非合并关系

当 relation guard 输出：

```text
attribute_of_entity
section_title_of_entity
process_step_of_entity
metric_or_property
```

写入：

```text
decision = reject
relation_type = <relation_guard_decision>
source = llm_guard
confidence = LLM confidence
```

### 6.2 hybrid 低分直接失败

当 hybrid top1 分数 `< 0.5` 时，认为该候选不是同一实体，写入：

```text
decision = reject
relation_type = low_score_direct_reject
source = auto
confidence = 1.0 - top1_score
```

### 6.3 hybrid 人工评估触发

以下两种情况会根据 final judge 结果写入人工评估经验：

- hybrid top1 分数在 `0.7-0.85` 之间。
- hybrid top1 的 ES 分数和 Milvus 分数差值 `> 0.4`。

如果 final judge 输出 `new`，且 hybrid top1 分数 `>= 0.85`，说明候选非常像但最终确认不是同一实体，会优先写入强 reject 经验：

```text
decision = reject
relation_type = high_score_llm_reject
source = llm_judge
confidence = 1.0
```

如果 final judge 输出 `matched`：

```text
decision = allow
relation_type = manual_match
source = manual
confidence = 1.0
```

如果 final judge 不是 `matched`：

```text
decision = reject
relation_type = manual_reject
source = manual
confidence = 1.0
```

## 7. 访问次数和排序

关系记忆表使用唯一键约束同一个 scope 下的实体对。

如果重复写入同一关系，不会创建重复记录，而是更新关系信息，并执行：

```sql
hit_count = hit_count + 1
```

后续查询 MySQL top30 时，高频且高置信的经验会排在前面：

```text
ORDER BY hit_count DESC, confidence DESC, updated_at DESC
```

## 8. 报告字段

每个 case 会输出：

- `retrieval`：最终用于报告的候选列表。
- `judgment`：最终裁判结果。
- `usage.judge`：本 case 的 LLM judge 用量汇总，包括 admission guard、relation guard、final judge。
- `strategy_trace`：该 case 实际走过的关键分支。
- `flow`：从 `strategy_trace` 和 `experience` 推导出的流程可解释字段。
- `experience`：如果本 case 写入了关系经验，会记录写入摘要。

常见 `strategy_trace.stage`：

```text
redis_direct_allow
llm_admission_guard_reject
mysql_relation_memory_judge
llm_relation_guard_reject
memory_judge_fallback_hybrid
no_relation_memory_fallback_hybrid
hybrid_rrf
```

每个 case 的 `flow` 会把流程拆成更直观的布尔字段：

```json
{
  "stage": "memory_judge_fallback_hybrid",
  "used_redis": true,
  "used_admission_guard": true,
  "used_mysql_memory": true,
  "used_relation_guard": true,
  "used_hybrid": true,
  "wrote_experience": true,
  "manual_review_triggered": true,
  "experience_source": "manual",
  "experience_relation_type": "manual_match"
}
```

整体报告会额外输出 `flow_metrics`：

- `by_stage`：每条流程路径命中了多少 case。
- `by_category_stage`：每个数据集类别分别走了哪些流程路径。
- `accuracy_by_stage`：每条流程路径下的正确数、总数和准确率。
- `accuracy_by_category_stage`：每个类别内、每条流程路径下的准确率。
- `component_usage_counts`：Redis、admission guard、MySQL memory、relation guard、hybrid、经验写入、人工评估触发分别用了多少次。
- `hybrid_fallback_rate`：最终进入 hybrid 的比例。
- `experience_write_count`：写入经验的 case 数。
- `manual_review_trigger_count`：触发人工评估经验沉淀的 case 数。

比率类指标如果当前数据集没有适用分母，会输出 `null`，而不是 `0.0`。例如只评估 `nil_new_entity` 时没有 expected matched 样本，因此 `recall@1/3/5`、`mrr`、`matched_entity_accuracy`、`false_new_rate` 会是 `null`。计数字段仍保留 `0`，方便看样本分布。

## 相关文件

- 策略入口：`backend/python/app/EvoRAG/entity_resolution_eval.py`
- 关系记忆读取：`backend/python/app/EvoRAG/entity_store/relation_memory.py`
- 关系记忆入库：`backend/python/app/EvoRAG/entity_store/repository.py`
- 生产 ingest/人工审核经验写入：`backend/python/app/EvoRAG/entity_store/ingestor.py`
- 数据表：`backend/python/app/EvoRAG/sql/entity_schema.sql`

