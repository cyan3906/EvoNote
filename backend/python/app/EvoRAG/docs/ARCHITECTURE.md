# EvoRAG 项目架构文档

更新时间：2026-09-26

EvoRAG 是 Evonote 后端中的实体记忆 RAG 子系统。它的核心目标是把用户输入的长文本切分成语义 Block，从 Block 中抽取“可作为知识入口”的实体和属性，写入实体库，并通过 MySQL、Elasticsearch、Milvus 组合完成实体检索、融合和回答生成。

## 1. 总体架构

```text
用户长文本
  |
  v
BlockSplitter
  |
  v
EntityExtractor
  |
  v
EvoRAGPreprocessResult
  |
  +-- /extract：只返回 Block 和实体抽取结果，不入库
  |
  v
EntityIngestor
  |
  +-- 实体去重与属性合并
  +-- 生成实体 embedding
  +-- EntityResolver 判断新建 / 合并 / 模糊
  +-- MySQL 写入实体、属性、证据、审计
  +-- Elasticsearch 写入关键词索引
  +-- Milvus 写入向量索引
```

检索和问答链路：

```text
用户输入实体
  |
  v
EvoRAGRetriever
  |
  +-- MySQL 精确名称匹配
  +-- Elasticsearch 关键词检索
  +-- Milvus 向量检索
  +-- RRF 融合排序
  +-- MySQL 批量回填完整属性和证据
  |
  v
EvoRAGQueryService
  |
  +-- 构建实体依赖图
  +-- 调用 LLM 生成回答
```

## 2. 目录结构

```text
app/EvoRAG
├─ config.py                  配置：LLM、Embedding、MySQL、ES、Milvus
├─ constants.py               属性类型等常量
├─ prompts.py                 Block 切分和实体抽取提示词
├─ cli.py                     命令行测试入口
├─ README.md                  项目说明
├─ docs/
│  └─ ARCHITECTURE.md         当前架构文档
├─ frontend/
│  ├─ index.html              EvoRAG 调试页面
│  ├─ app.js                  前端交互逻辑
│  └─ styles.css              前端样式
├─ llm/
│  ├─ client.py               LLM JSON 调用封装
│  ├─ retry.py                重试逻辑
│  └─ circuit_breaker.py      熔断逻辑
├─ models/
│  ├─ __init__.py
│  └─ schemas.py              Pydantic 数据结构
├─ services/
│  ├─ processor.py            预处理总流程
│  ├─ block_splitter.py       长文本切分 Block
│  ├─ entity_extractor.py     从 Block 抽取实体和属性
│  ├─ retriever.py            实体检索、ES/Milvus 融合、MySQL 回填
│  ├─ query.py                查询服务入口
│  ├─ dependency_graph.py     构建实体依赖图
│  └─ answer_generator.py     生成最终回答
├─ entity_store/
│  ├─ models.py               实体库 dataclass 模型
│  ├─ normalizer.py           实体规范化、去重、属性合并
│  ├─ repository.py           MySQL 读写
│  ├─ resolver.py             实体消歧和合并判断
│  └─ ingestor.py             入库编排
├─ indexes/
│  ├─ embedding.py            Embedding 客户端
│  └─ entity_hybrid.py        ES/Milvus 建索引、写入、检索、融合
├─ sql/
│  └─ entity_schema.sql       MySQL 表结构
└─ scripts/                   Demo 和辅助脚本
```

## 3. API 入口

API 路由位于：

```text
app/api/routes/evorag.py
```

路由统一挂载在：

```text
/api/evorag
```

当前接口：

| 接口 | 作用 |
| --- | --- |
| `POST /api/evorag/extract` | 只执行 Block 切分和实体抽取，不写入 MySQL/ES/Milvus |
| `POST /api/evorag/ingest` | 执行切分、抽取、实体合并、MySQL 入库、ES/Milvus 索引写入 |
| `POST /api/evorag/query` | 根据实体检索、构建依赖图、生成回答 |
| `POST /api/evorag/index-search` | 调试 ES、Milvus、融合结果，并返回各阶段耗时 |

所有接口都依赖主项目鉴权：

```text
app.core.security.require_auth
```

## 4. 前端调试页

前端目录：

```text
app/EvoRAG/frontend
```

FastAPI 挂载地址：

```text
/evorag
```

页面能力：

- 登录后调用 EvoRAG API。
- 输入长文本后执行“抽取预览”，只测试 Block 切分和实体抽取，不入库。
- 输入长文本后执行“提交并入库”，完整执行入库链路。
- 展示每个 Block 的文本内容。
- 展示每个 Block 中抽取出的实体、属性、证据、置信度。
- 展示实体准入评分和准入理由。
- 展示预处理阶段耗时：启动到首次切分前、实体锚点预切分、物理二切、实体抽取、预处理总耗时。
- 展示 ES、Milvus、融合检索结果。
- 展示健康检查、Embedding、ES 检索、Milvus 检索、融合、MySQL 回填等阶段耗时。

## 5. 预处理流程

预处理入口：

```text
app/EvoRAG/services/processor.py
```

核心流程：

```text
EvoRAGProcessor.preprocess(text)
  |
  +-- 校验空文本
  +-- BlockSplitter.split(text)
  +-- EntityExtractor.extract_many(blocks)
  +-- 返回 EvoRAGPreprocessResult
```

### 5.1 Block 切分

文件：

```text
app/EvoRAG/services/block_splitter.py
```

职责：

- 调用 LLM 将用户长文本先切成多个“实体锚点 Block”。
- 每个 Block 尽量围绕一个 `anchor_entity`，并附带候选实体、切分原因和锚点置信度。
- 后端再按 `max_block_chars` 做确定性的物理二次切分，超长 Block 会拆成多个 chunk，不再截断丢内容。
- 传给 LLM 的关键参数包括：
  - `text`
  - `max_blocks`
  - `max_block_chars`
- 对 LLM 返回结果做规范化：
  - 丢弃空 Block。
  - 超长 Block 会按自然边界尽量切成多个 chunk。
  - chunk 会继承原 Block 的 `anchor_entity`、`candidate_entities`、`anchor_confidence`。
  - chunk 的 `split_reason` 会标记为 `max_chars_chunk`。
  - 重新按顺序编号 `block_index`。
  - heading 为空时补成 `Block 1`、`Block 2`。
  - 返回结构不合法时抛错。

Block 结构核心字段：

```text
block_index
heading
anchor_entity
candidate_entities
l1_text
split_reason
anchor_confidence
parent_block_index
chunk_index
chunk_count
char_start
char_end
```

### 5.2 实体抽取

文件：

```text
app/EvoRAG/services/entity_extractor.py
app/EvoRAG/prompts.py
```

职责：

- 对每个 Block 调用 LLM 抽取实体、属性和证据。
- 支持并发抽取多个 Block。
- 使用 `asyncio.gather(..., return_exceptions=True)` 隔离单个 Block 的失败。
- 单个 Block 抽取失败时，该 Block 返回空实体和 warning，不影响其他 Block。

实体抽取现在有 5 维准入评分：

| 维度 | 含义 |
| --- | --- |
| `definable` | 是否能被定义，能回答“它是什么” |
| `query_entry` | 是否适合作为用户查询入口 |
| `independent_scope` | 是否有独立机制、目的、约束、组成或边界 |
| `stable_relations` | 是否和多个其他实体有稳定关联 |
| `key_sentence` | 是否出现在标题、主题句、定义句或关键句中 |

后端会重新计算：

```text
aggregate_score = 五个维度平均值
```

只有：

```text
aggregate_score >= 0.7
```

才会保留为实体。低于阈值的候选会被丢弃，并写入当前 Block 的 warning。

例如：

```text
CI（持续集成）        -> 实体
拉取最新代码          -> CI.mechanism 属性
安装依赖              -> CI.mechanism 属性
编译 / 打包           -> CI.mechanism 属性
生成报告              -> CI.mechanism 属性
```

## 6. 入库流程

入口：

```text
app/EvoRAG/entity_store/ingestor.py
```

核心流程：

```text
EntityIngestor.ingest(preprocess_result)
  |
  +-- dedupe_extracted_entities()
  +-- 为每个 incoming entity 生成 embedding
  +-- repository.list_entities(scope)
  +-- EntityResolver.resolve_many()
  +-- apply_decisions()
       |
       +-- ambiguous：只写审计，不入库
       +-- matched：合并到已有实体
       +-- new：创建新实体
       |
       +-- repository.upsert_entity()
       +-- hybrid_index.upsert_entity()
       +-- repository.record_resolution_audit()
```

### 6.1 实体去重与属性合并

文件：

```text
app/EvoRAG/entity_store/normalizer.py
```

职责：

- 按实体名和实体类型做基础去重。
- 合并 aliases。
- 合并同类属性。
- 属性去重目前主要是文本 fingerprint 去重。
- 构造 `description_for_match`，供 ES 检索和匹配使用。

### 6.2 实体消歧

文件：

```text
app/EvoRAG/entity_store/resolver.py
```

职责：

- 判断新抽取实体和已有实体是否是同一个。
- 决策类型：
  - `matched`
  - `new`
  - `ambiguous`
- ambiguous 的实体不会直接入库，会写入审计表，避免误合并。

### 6.3 MySQL 入库

文件：

```text
app/EvoRAG/entity_store/repository.py
app/EvoRAG/sql/entity_schema.sql
```

MySQL 存储完整数据：

- 实体基础信息。
- aliases。
- identity_description。
- summary。
- description_for_match。
- embedding JSON。
- 属性。
- evidence。
- 实体消歧审计记录。

## 7. ES / Milvus 索引

核心文件：

```text
app/EvoRAG/indexes/entity_hybrid.py
```

### 7.1 Elasticsearch

ES 索引名默认：

```text
evorag_entities
```

写入字段：

```text
entity_id
workspace_id
project_id
collection_id
domain
canonical_name
normalized_name
entity_type
aliases
identity_description
summary
description_for_match
```

ES 查询逻辑：

```text
term normalized_name boost 8
term entity_type boost 2
multi_match:
  canonical_name^6
  aliases^5
  identity_description^3
  summary^2
  description_for_match
```

并使用 scope 过滤：

```text
workspace_id
project_id
collection_id
domain
```

ES 更适合：

- 实体名精确命中。
- 别名命中。
- 关键词匹配。
- 描述文本匹配。

### 7.2 Milvus

Milvus collection 默认：

```text
evorag_entities
```

主要字段：

```text
id
vector
entity_id
workspace_id
project_id
collection_id
domain
canonical_name
entity_type
identity_description
```

向量索引：

```text
index_type = AUTOINDEX
metric_type = COSINE
```

实体入库时的 embedding 文本优先级：

```text
identity_description
or description_for_match
or entity.name
```

查询时会对用户输入实体生成 embedding，然后用 Milvus 做向量相似检索。

Milvus 更适合：

- 语义相似召回。
- 用户没有使用实体精确名称时的召回。
- 概念描述类查询。

### 7.3 RRF 融合

融合函数：

```text
reciprocal_rank_fusion_entities()
```

融合公式：

```text
score += 1 / (rrf_k + rank)
```

默认：

```text
rrf_k = 60
```

如果一个实体同时被 ES 和 Milvus 召回，它会获得两边排名贡献，通常更容易排到前面。

## 8. 检索与问答

检索入口：

```text
app/EvoRAG/services/retriever.py
```

正式查询入口：

```text
app/EvoRAG/services/query.py
```

### 8.1 `/index-search` 调试链路

```text
EvoRAGRetriever.search_indexes(query)
  |
  +-- backend_status()
  +-- embed_text(query)
  +-- ensure_elasticsearch_index()
  +-- search_elasticsearch()
  +-- ensure_milvus_collection()
  +-- search_milvus()
  +-- reciprocal_rank_fusion_entities()
  +-- hydrate_entities_for_candidates()
  +-- 返回 ES / Milvus / fused 三组结果
```

返回耗时：

```text
backend_status_ms
embedding_ms
elasticsearch_ensure_ms
elasticsearch_search_ms
milvus_ensure_ms
milvus_search_ms
fusion_ms
mysql_hydration_ms
total_ms
```

### 8.2 `/query` 正式问答链路

```text
EvoRAGQueryService.query(entity)
  |
  +-- EvoRAGRetriever.retrieve()
  +-- 找 root_entity
  +-- build_dependency_graph()
  +-- AnswerGenerator.generate()
  +-- 返回 EvoRAGQueryResult
```

返回内容：

- root entity。
- retrieved entities。
- dependency graph。
- answer。
- warnings。

## 9. 启动集成

主项目入口：

```text
app/main.py
```

EvoRAG API 挂载：

```text
app.include_router(evorag_router, prefix="/api")
```

EvoRAG 前端挂载：

```text
app.mount("/evorag", StaticFiles(directory=EVORAG_FRONTEND_ROOT, html=True), name="evorag")
```

启动时初始化：

```text
initialize_evorag_database()
initialize_retrieval_backends()
```

EvoRAG MySQL 连接封装在：

```text
app/core/evorag_database.py
```

ES / Milvus 会尽量复用主项目已有连接状态和客户端。

## 10. 测试结构

测试目录：

```text
tests/EvoRAG
```

当前重点测试：

- BlockSplitter 是否正确调用 LLM。
- BlockSplitter 是否校验和规范化 LLM 返回。
- EntityExtractor 是否能解析固定 JSON。
- EntityExtractor 是否兼容 legacy `source_text`。
- EntityExtractor 是否过滤低准入分候选实体。
- EntityExtractor.extract_many 是否隔离单个 Block 失败。
- EvoRAGProcessor.preprocess 是否串起切分和抽取。
- index-search 是否返回 ES、Milvus、融合结果并批量回填。
- extract route 是否只预处理、不入库。

常用测试命令：

```bash
pytest tests/EvoRAG -q
```

前端 JS 检查：

```bash
node --check app/EvoRAG/frontend/app.js
```

## 11. 当前架构特点

优点：

- 抽取、入库、检索、问答分层比较清楚。
- `/extract` 可以不入库调试抽取效果。
- `/index-search` 可以分别观察 ES、Milvus 和融合结果。
- MySQL 回填已改为批量查询，避免每个候选实体单独查库。
- 实体抽取已加入准入评分，能减少“步骤/属性被误当实体”的问题。
- ES/Milvus 检索都有 scope 隔离，便于多项目、多集合扩展。

仍可优化：

- Embedding 查询可以加缓存，减少重复 query 的耗时。
- 属性去重目前主要是文本级去重，后续可加语义去重。
- 实体类型体系还可以进一步收敛，例如固定枚举和中文显示映射。
- Milvus 复用主项目连接时，需要保证 host、port、token 配置完全一致。
- ES/Milvus 检索权重和 RRF 参数后续可以基于真实测试集调参。
- ambiguous 实体可以做人工审核页面，而不只是记录审计。

