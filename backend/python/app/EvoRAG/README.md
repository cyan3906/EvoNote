# EvoRAG

EvoRAG is a preprocessing module for entity-memory RAG.

Current scope:

- Split user text into blocks with an LLM.
- Extract entities from each block concurrently.
- Record seven controlled attribute groups for each entity:
  `definition`, `purpose`, `core_idea`, `mechanism`, `components`,
  `constraints`, and `related`.
- Keep explicit `evidence` text at the attribute level.
- Provide retry, semaphore-based concurrency control, and a circuit breaker.
- Normalize, deduplicate, resolve, and merge extracted entities into MySQL.
- Maintain scoped ES and Milvus entity indexes for hybrid entity retrieval.
- Retrieve an entity memory set, build semantic/conditional dependency graphs,
  detect cycles, topologically order related entities, and generate a structured
  long-form answer.

Structure:

```text
EvoRAG/
  config.py              independent EvoRAG settings
  constants.py           controlled attribute names
  prompts.py             LLM prompts
  cli.py                 command-line entry
  models/                pydantic data models
  llm/                   OpenAI-compatible client, retry, circuit breaker
  services/              block splitting, extraction, retrieval, graphing, generation
  entity_store/          entity dedupe, resolution, MySQL persistence
  sql/                   MySQL table schema
  processor.py           compatibility re-export
  schemas.py             compatibility re-export
```

MySQL tables:

- `evorag_entities`: global entity records and matching summaries.
- `evorag_entity_attributes`: one row per physical attribute value.
- `evorag_entity_attribute_evidence`: evidence rows attached to attributes.
- `evorag_entity_resolution_audit`: audit trail for matched/new/ambiguous decisions.

Model settings:

- `EVORAG_INFERENCE_MODEL`: chat/inference model for block splitting, entity extraction, and entity resolution.
- `EVORAG_EMBEDDING_MODEL`: embedding model reserved for entity vector generation.

Entity resolution:

- ES retrieves entity name, aliases, type, and identity text.
- Milvus retrieves the vector of `identity_description`.
- Results are fused with RRF.
- High confidence candidates are merged automatically, middle confidence candidates are judged by the LLM, and low confidence candidates are left ambiguous for manual review.

Local indexes:

- Milvus uses the local service at `EVORAG_MILVUS_HOST:EVORAG_MILVUS_PORT`. In the Evonote Docker Compose setup, its physical data is mounted at `backend/python/data/milvus`.
- Elasticsearch uses the local service at `EVORAG_ES_URL`. In the Evonote Docker Compose setup, its physical data is mounted at `backend/python/data/elasticsearch`.

Run from `evonote/backend/python`:

```powershell
python -m app.EvoRAG.cli --text "MVCC 是 InnoDB 实现读不加锁、读写不阻塞的并发机制。"
```

Or:

```powershell
python -m app.EvoRAG.cli --file .\sample.txt
```

Query an ingested entity and generate structured content:

```powershell
python -m app.EvoRAG.cli --query "Transformer" --top-k 5
```
