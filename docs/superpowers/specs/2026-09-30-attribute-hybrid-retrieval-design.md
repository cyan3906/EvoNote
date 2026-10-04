# Attribute Hybrid Retrieval Design

## Goal

Add the attribute candidate-retrieval stage that runs after an incoming entity has matched an existing entity. The stage must efficiently retrieve relevant existing attributes for each incoming attribute while guaranteeing that exact duplicates are never lost to a Top-K search.

This phase does not implement the later LLM decisions (`MERGE`, `ADD`, `UPDATE`, or `CONFLICT`) and does not change the current attribute write semantics.

## Scope

The implementation provides deterministic fingerprint lookup, adaptive retrieval per `(entity_id, attr_type)`, direct MySQL loading for groups smaller than 20, Elasticsearch `_msearch`, concurrent grouped Milvus multi-vector search, per-attribute reciprocal-rank fusion, independent attribute indexes, and index backfill/self-healing primitives.

It does not call an LLM, change `_merge_attributes`, introduce claim versioning or conflicts, or add UI.

## Retrieval Unit

The threshold is evaluated independently for each matched entity, attribute type, and active status. A group with 19 active attributes uses MySQL; a group with exactly 20 uses hybrid retrieval. Results use the incoming list position as a stable request key so identical input text cannot collide.

## Data Model

`AttributeCandidate` carries `attribute_id`, `entity_id`, `attr_type`, `value_text`, confidence, rank, source, and separate Elasticsearch, vector, and fused scores.

`AttributeRetrievalResult` carries the request key, retrieval mode, group size, optional exact match, candidates, and backend warnings.

## Deterministic Exact Match

The repository first performs one batched lookup by:

```text
entity_id + attr_type + value_fingerprint + status=active
```

An exact match is terminal for retrieval. It bypasses embeddings, Elasticsearch, and Milvus. Fingerprints only detect deterministic textual duplicates; paraphrases continue to adaptive retrieval.

## Small Groups

For each unmatched group with fewer than 20 active attributes, one grouped MySQL query loads all active attributes of that type in deterministic ID order. No external index is used.

## Hybrid Retrieval

All unmatched inputs in large groups are embedded in one `embed_texts` call. Index and query use the same text:

```text
attribute type: {attr_type}
attribute value: {value_text}
```

Entity name, summary, and evidence are excluded because identity and type are filters and evidence adds source noise.

Elasticsearch receives one `_msearch` request with one filtered query per input. Each query searches `value_text` and filters by entity ID, type, scope, and active status.

Milvus requests are grouped by entity ID, type, and scope because every vector in a Milvus search shares one filter. Each group sends all vectors in one `data=[...]` call. Groups run concurrently through bounded `asyncio.gather` and `asyncio.to_thread`.

Fusion runs independently for every input and joins rankings by attribute ID:

```text
rrf_score = sum(1 / (rrf_k + rank))
```

Final order uses the RRF score while raw backend scores remain available.

## Attribute Indexes

The Elasticsearch document stores one active attribute with attribute ID, entity ID, scope, type, status, fingerprint, value text, and confidence. Value text is searchable; identity and lifecycle fields are filters.

The Milvus record uses `attribute_id` as both stable primary key and metadata, stores the same filters and value text, and stores a cosine vector built from the declared type/value template.

MySQL remains the source of truth. Both indexes support batched idempotent upserts. Existing rows are backfilled in batches. Before searching a large group, the retriever verifies that its active attribute IDs exist in both indexes and backfills missing records. Concurrent backfills are safe because the attribute ID is stable.

## Failure Behavior

Exact matches and small groups work without external services. If one hybrid backend fails, the other supplies candidates and the result records a warning. If both fail, retrieval raises a typed error so a future decision layer can retry or review rather than assume an addition. MySQL hydration discards stale, cross-entity, or wrong-type IDs.

## Integration Boundary

This phase exposes an async batch interface:

```python
retrieve(
    entity: StoredEntity,
    incoming_attributes: list[EntityAttributeInput],
) -> list[AttributeRetrievalResult]
```

It is ready to call after entity matching but does not feed results into `_merge_attributes`. The next phase will add LLM decisions and transactional semantic writes.

## Configuration

Configuration covers attribute index names, threshold `20`, hybrid Top-K, RRF constant, Milvus concurrency, and backfill batch size.

## Tests

Tests cover exact short-circuiting, the 19/20 boundary, complete small-group loads, ES request batching, Milvus vector grouping and concurrency, request/result mapping, per-input RRF isolation, candidate hydration guards, backend degradation, total backend failure, and index/vector document construction.
