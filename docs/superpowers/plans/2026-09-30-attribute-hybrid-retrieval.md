# Attribute Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested attribute candidate-retrieval subsystem with exact-match short-circuiting, adaptive MySQL full reads, Elasticsearch multi-search, concurrent Milvus batch search, and per-input RRF.

**Architecture:** MySQL remains authoritative and supplies exact matches, group counts, small-group results, index backfill records, and final candidate hydration. A focused `AttributeHybridIndex` owns Elasticsearch and Milvus details; an async `AttributeCandidateRetriever` orchestrates threshold selection, embedding, index repair, backend degradation, and fusion without making semantic merge decisions.

**Tech Stack:** Python 3, asyncio, PyMySQL, Elasticsearch 8 client, pymilvus 2.4+, pytest

**Spec:** `docs/superpowers/specs/2026-09-30-attribute-hybrid-retrieval-design.md`

## Global Constraints

- Evaluate the threshold per `(entity_id, attr_type, status=active)`.
- Use MySQL for groups containing 0 through 19 active attributes and hybrid retrieval from 20 onward.
- Exact `value_fingerprint` matches bypass embedding and both external backends.
- Use one Elasticsearch `_msearch` request for all large-group incoming attributes.
- Batch Milvus vectors by shared entity, type, and scope filter; run groups concurrently.
- Fuse rankings independently for every incoming-list position.
- Index and query text is exactly `attribute type: {attr_type}\nattribute value: {value_text}`.
- Do not implement or invoke LLM `MERGE / ADD / UPDATE / CONFLICT` decisions.
- Do not change current `_merge_attributes` behavior in this phase.

## Review Focus

- Empty incoming attribute batches return immediately without database, embedding, or backend calls; Task 3 pins this.
- Duplicate incoming values keep separate positional results and do not overwrite each other; Task 3 pins this.
- The exact 19/20 threshold boundary chooses different modes deterministically; Task 3 pins this.
- Stale, cross-entity, inactive, and wrong-type search hits never reach returned candidates; Task 4 pins this.
- A single backend failure degrades with warnings, while two backend failures raise a typed error; Task 4 pins this.

---

### Task 1: Attribute Records, Results, and Repository Queries

**Files:**
- Modify: `backend/python/app/EvoRAG/entity_store/models.py`
- Modify: `backend/python/app/EvoRAG/entity_store/repository.py`
- Modify: `backend/python/app/EvoRAG/config.py`
- Test: `backend/python/tests/EvoRAG/test_attribute_repository.py`

**Interfaces:**
- Produces: `StoredEntityAttribute`, `AttributeCandidate`, `AttributeRetrievalResult`, and `AttributeRetrievalError`.
- Produces: `MySQLEntityRepository.count_active_attributes_by_type(entity_id: int, attr_types: list[str]) -> dict[str, int]`.
- Produces: `MySQLEntityRepository.find_exact_active_attributes(entity_id: int, fingerprints_by_type: dict[str, set[str]]) -> dict[tuple[str, str], StoredEntityAttribute]`.
- Produces: `MySQLEntityRepository.list_active_attributes_for_types(entity_id: int, attr_types: list[str]) -> list[StoredEntityAttribute]`.
- Produces: `MySQLEntityRepository.get_active_attributes_by_ids(attribute_ids: list[int]) -> list[StoredEntityAttribute]`.

- [ ] **Step 1: Write failing model and configuration tests**

Add tests asserting defaults: threshold `20`, hybrid Top-K `10`, RRF constant `60`, Milvus concurrency `4`, backfill batch size `100`, ES index `evorag_entity_attributes`, and Milvus collection `evorag_entity_attributes`. Assert result models preserve `input_index`, raw backend scores, fused score, warnings, and mode.

- [ ] **Step 2: Run the model tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_repository.py -k "config or model"`

Expected: FAIL because the settings and models do not exist.

- [ ] **Step 3: Add the models and settings**

Implement the dataclasses and exact settings named in the Interfaces and Step 1. `StoredEntityAttribute` includes ID, entity ID, scope, type, value, fingerprint, confidence, and status. Retrieval modes are `exact`, `full_scan`, and `hybrid`.

- [ ] **Step 4: Run model tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_repository.py -k "config or model"`

Expected: PASS.

- [ ] **Step 5: Write failing repository query tests**

Use a recording fake cursor to assert each method issues one bounded query, always filters `status = 'active'`, maps rows to `StoredEntityAttribute`, returns missing type counts as zero, deduplicates requested IDs, and returns deterministic ID ordering.

- [ ] **Step 6: Run repository tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_repository.py -k repository`

Expected: FAIL because repository methods are missing.

- [ ] **Step 7: Implement the repository methods**

Add the four exact signatures from Interfaces and a private row mapper. Use parameterized SQL only. Query candidate pairs in batches by entity ID plus requested types/fingerprints, then enforce exact pair membership in Python.

- [ ] **Step 8: Run Task 1 tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_repository.py`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add backend/python/app/EvoRAG/config.py backend/python/app/EvoRAG/entity_store/models.py backend/python/app/EvoRAG/entity_store/repository.py backend/python/tests/EvoRAG/test_attribute_repository.py
git commit -m "feat: add attribute retrieval repository primitives"
```

### Task 2: Attribute Elasticsearch and Milvus Index

**Files:**
- Create: `backend/python/app/EvoRAG/indexes/attribute_hybrid.py`
- Modify: `backend/python/app/EvoRAG/indexes/__init__.py`
- Test: `backend/python/tests/EvoRAG/test_attribute_hybrid_index.py`

**Interfaces:**
- Consumes: `StoredEntityAttribute`, `AttributeCandidate`, and new attribute index settings from Task 1.
- Produces: `attribute_text_for_embedding(attr_type: str, value_text: str) -> str`.
- Produces: `AttributeHybridIndex.ensure_indexes() -> None`.
- Produces: batched `upsert_elasticsearch_attributes(records)` and `upsert_milvus_attributes(records, vectors)`.
- Produces: `existing_elasticsearch_attribute_ids(attribute_ids) -> set[int]` and `existing_milvus_attribute_ids(attribute_ids) -> set[int]`.
- Produces: `search_elasticsearch_many(requests, top_k) -> dict[int, list[AttributeCandidate]]`.
- Produces: `search_milvus_group(vectors, *, entity_id, attr_type, scope, top_k) -> list[list[AttributeCandidate]]`.

- [ ] **Step 1: Write failing index-schema and vector-text tests**

Assert the ES mapping has keyword/numeric filters and text `value_text`. Assert the Milvus schema uses INT64 `attribute_id` as stable identity, a FLOAT_VECTOR at configured dimensions, COSINE/AUTOINDEX, scope/type/status metadata, and that the embedding template matches Global Constraints exactly.

- [ ] **Step 2: Run schema tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_hybrid_index.py -k "schema or embedding_text"`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement index creation and document builders**

Create `AttributeHybridIndex` following the existing entity index client-selection pattern. Add pure ES document and Milvus record builders so schema/content behavior is testable without services.

- [ ] **Step 4: Run schema tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_hybrid_index.py -k "schema or embedding_text"`

Expected: PASS.

- [ ] **Step 5: Write failing batch-operation tests**

Assert ES upsert uses one bulk request, Milvus upsert receives one document list aligned with vectors, ID existence methods return sets, `_msearch` sends one header/body pair per request, response order maps to `input_index`, and one Milvus call accepts multiple vectors sharing a filter.

- [ ] **Step 6: Run batch tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_hybrid_index.py -k "upsert or existing or search"`

Expected: FAIL because batch operations are missing.

- [ ] **Step 7: Implement batch upsert, existence checks, and searches**

Use the Elasticsearch 8 client `bulk(operations=[...])` and `msearch(searches=[...])` APIs. Use Milvus `upsert(data=[...])`, `query`, and one `search(data=vectors, filter=...)`. Return lightweight candidates keyed by input index; do not trust indexed text as final hydrated data.

- [ ] **Step 8: Run Task 2 tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_hybrid_index.py`

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add backend/python/app/EvoRAG/indexes/attribute_hybrid.py backend/python/app/EvoRAG/indexes/__init__.py backend/python/tests/EvoRAG/test_attribute_hybrid_index.py
git commit -m "feat: add attribute ES and Milvus index"
```

### Task 3: Exact and Adaptive Full-Scan Retrieval

**Files:**
- Create: `backend/python/app/EvoRAG/entity_store/attribute_retriever.py`
- Test: `backend/python/tests/EvoRAG/test_attribute_retriever.py`

**Interfaces:**
- Consumes: repository methods from Task 1, `EvoRAGEmbeddingClient`, and `AttributeHybridIndex`.
- Produces: `AttributeCandidateRetriever.retrieve(entity: StoredEntity, incoming_attributes: list[EntityAttributeInput]) -> list[AttributeRetrievalResult]`.
- Produces: private positional request records retaining `input_index`, type, text, fingerprint, and group size.

- [ ] **Step 1: Write failing no-op and exact-match tests**

Assert an empty input returns `[]` with no dependency calls. Assert exact lookup occurs once for the batch, every exact result uses mode `exact`, and neither counts, embeddings, ES, nor Milvus are called. Include two identical inputs and assert results retain indices `0` and `1`.

- [ ] **Step 2: Run exact tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "empty or exact"`

Expected: FAIL because the retriever does not exist.

- [ ] **Step 3: Implement request preparation and exact short-circuiting**

Build fingerprints with the existing `text_fingerprint`; make one repository exact lookup; construct one result per original list position in original order.

- [ ] **Step 4: Run exact tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "empty or exact"`

Expected: PASS.

- [ ] **Step 5: Write failing threshold/full-scan tests**

Assert count `19` returns all same-type active rows as mode `full_scan`; count `20` does not call the full-scan branch. Assert multiple small attribute types are loaded in one repository call and candidates cannot leak between types.

- [ ] **Step 6: Run threshold tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "threshold or full_scan"`

Expected: FAIL because adaptive routing is incomplete.

- [ ] **Step 7: Implement count routing and grouped full scans**

Count only unmatched types, load all small types in one call, partition rows by type, and map them to candidates in deterministic ID order. Leave large requests to a dedicated hybrid method added in Task 4.

- [ ] **Step 8: Run Task 3 tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "empty or exact or threshold or full_scan"`

Expected: PASS.

- [ ] **Step 9: Commit Task 3**

```bash
git add backend/python/app/EvoRAG/entity_store/attribute_retriever.py backend/python/tests/EvoRAG/test_attribute_retriever.py
git commit -m "feat: add adaptive attribute full-scan retrieval"
```

### Task 4: Hybrid Orchestration, Index Repair, and RRF

**Files:**
- Modify: `backend/python/app/EvoRAG/entity_store/attribute_retriever.py`
- Modify: `backend/python/app/EvoRAG/entity_store/__init__.py`
- Test: `backend/python/tests/EvoRAG/test_attribute_retriever.py`

**Interfaces:**
- Consumes: all Task 1-3 interfaces.
- Produces: `reciprocal_rank_fusion_attributes(rankings, *, rrf_k: int, top_k: int) -> list[AttributeCandidate]`.
- Completes: `AttributeCandidateRetriever.retrieve(...)` hybrid mode and warning/error behavior.

- [ ] **Step 1: Write failing embedding, ES, and Milvus batching tests**

Assert all large-group query texts use one embedding call. Assert ES receives all large requests once. Assert Milvus receives one multi-vector call per shared entity/type/scope group, and use synchronization events in the fake client to prove two groups overlap under configured concurrency.

- [ ] **Step 2: Run hybrid batching tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "embedding_batch or es_msearch or milvus_group"`

Expected: FAIL because hybrid orchestration is incomplete.

- [ ] **Step 3: Implement batched hybrid searches**

Embed query texts once, call ES once through `asyncio.to_thread`, group Milvus requests by filter, and execute groups with `asyncio.gather` under an `asyncio.Semaphore`.

- [ ] **Step 4: Run hybrid batching tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "embedding_batch or es_msearch or milvus_group"`

Expected: PASS.

- [ ] **Step 5: Write failing index-repair tests**

Provide active DB records where each backend lacks a different subset. Assert one union embedding batch is made, then each backend receives only its missing records. Assert fully indexed groups perform no backfill embedding or upsert.

- [ ] **Step 6: Run repair tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k index_repair`

Expected: FAIL because index repair is missing.

- [ ] **Step 7: Implement pre-search index verification and repair**

Load active records for large types, query existing IDs in both stores, embed the union of missing records once in configured chunks, and upsert each backend's missing subset with aligned vectors.

- [ ] **Step 8: Run repair tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k index_repair`

Expected: PASS.

- [ ] **Step 9: Write failing RRF, hydration, and failure tests**

Assert RRF scores and ordering are isolated per input index. Return fake stale, inactive, wrong-entity, and wrong-type IDs and assert hydration removes them. Assert one backend failure returns the other backend with a warning; assert two search-backend failures raise `AttributeRetrievalError`.

- [ ] **Step 10: Run fusion/failure tests and verify failure**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py -k "rrf or hydration or backend_failure"`

Expected: FAIL because final fusion and degradation are incomplete.

- [ ] **Step 11: Implement per-input fusion, authoritative hydration, and failure semantics**

Fuse IDs first, hydrate the union in one MySQL call, validate entity/type/status against each request, reconstruct candidates from authoritative rows, preserve raw scores and warnings, and raise only when neither backend produced a usable search response.

- [ ] **Step 12: Run Task 4 tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_retriever.py`

Expected: PASS.

- [ ] **Step 13: Commit Task 4**

```bash
git add backend/python/app/EvoRAG/entity_store/attribute_retriever.py backend/python/app/EvoRAG/entity_store/__init__.py backend/python/tests/EvoRAG/test_attribute_retriever.py
git commit -m "feat: add batched hybrid attribute retrieval"
```

### Task 5: Regression Verification

**Files:**
- Verify: `backend/python/app/EvoRAG/**`
- Verify: `backend/python/tests/EvoRAG/**`

**Interfaces:**
- Consumes: completed attribute retrieval subsystem.
- Produces: no new API; verifies no regression in current entity resolution and ingestion.

- [ ] **Step 1: Run focused attribute tests**

Run: `pytest -q backend/python/tests/EvoRAG/test_attribute_repository.py backend/python/tests/EvoRAG/test_attribute_hybrid_index.py backend/python/tests/EvoRAG/test_attribute_retriever.py`

Expected: PASS.

- [ ] **Step 2: Run all EvoRAG tests**

Run: `pytest -q backend/python/tests/EvoRAG`

Expected: PASS, with live tests skipped when their credentials are absent.

- [ ] **Step 3: Run syntax compilation**

Run: `python -m compileall -q backend/python/app/EvoRAG`

Expected: exit code 0.

- [ ] **Step 4: Inspect final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only plan/spec and scoped implementation/test files are changed.

- [ ] **Step 5: Commit verification adjustments if needed**

```bash
git add backend/python/app/EvoRAG backend/python/tests/EvoRAG docs/superpowers
git commit -m "test: verify attribute hybrid retrieval"
```
