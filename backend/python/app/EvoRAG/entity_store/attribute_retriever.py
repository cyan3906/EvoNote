import asyncio
from dataclasses import dataclass
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import (
    AttributeCandidate,
    AttributeRetrievalError,
    AttributeRetrievalResult,
    EntityAttributeInput,
    EntityScope,
    StoredEntity,
    StoredEntityAttribute,
)
from app.EvoRAG.entity_store.normalizer import text_fingerprint
from app.EvoRAG.indexes.attribute_hybrid import AttributeHybridIndex, attribute_text_for_embedding
from app.EvoRAG.indexes.embedding import EvoRAGEmbeddingClient


@dataclass(slots=True)
class _AttributeRequest:
    input_index: int
    attr_type: str
    value_text: str
    fingerprint: str
    group_size: int = 0


class AttributeCandidateRetriever:
    def __init__(
        self,
        repository: Any,
        embedding_client: EvoRAGEmbeddingClient | Any | None = None,
        index: AttributeHybridIndex | Any | None = None,
        config: EvoRAGSettings = settings,
    ) -> None:
        self.repository = repository
        self.embedding_client = embedding_client or EvoRAGEmbeddingClient(config)
        self.index = index or AttributeHybridIndex(config)
        self.config = config

    async def retrieve(
        self,
        entity: StoredEntity,
        incoming_attributes: list[EntityAttributeInput],
    ) -> list[AttributeRetrievalResult]:
        if not incoming_attributes:
            return []

        requests = [
            _AttributeRequest(
                input_index=index,
                attr_type=str(attribute.attr_type or ""),
                value_text=str(attribute.value_text or ""),
                fingerprint=text_fingerprint(attribute.value_text),
            )
            for index, attribute in enumerate(incoming_attributes)
        ]
        results: dict[int, AttributeRetrievalResult] = {}

        fingerprints_by_type: dict[str, set[str]] = {}
        for request in requests:
            fingerprints_by_type.setdefault(request.attr_type, set()).add(request.fingerprint)
        exact_matches = self.repository.find_exact_active_attributes(entity.id, fingerprints_by_type)

        unmatched: list[_AttributeRequest] = []
        for request in requests:
            exact = exact_matches.get((request.attr_type, request.fingerprint))
            if exact is not None:
                results[request.input_index] = AttributeRetrievalResult(
                    input_index=request.input_index,
                    attr_type=request.attr_type,
                    value_text=request.value_text,
                    mode="exact",
                    group_size=1,
                    exact_match=exact,
                    candidates=[candidate_from_record(exact, rank=1, source="exact")],
                )
            else:
                unmatched.append(request)

        if unmatched:
            counts = self.repository.count_active_attributes_by_type(entity.id, [request.attr_type for request in unmatched])
            small_types = unique_ordered([request.attr_type for request in unmatched if counts.get(request.attr_type, 0) < self.config.attribute_full_scan_threshold])
            large_requests = [request for request in unmatched if counts.get(request.attr_type, 0) >= self.config.attribute_full_scan_threshold]

            for request in unmatched:
                request.group_size = int(counts.get(request.attr_type, 0))

            if small_types:
                full_scan_records = self.repository.list_active_attributes_for_types(entity.id, small_types)
                records_by_type: dict[str, list[StoredEntityAttribute]] = {}
                for record in full_scan_records:
                    records_by_type.setdefault(record.attr_type, []).append(record)
                for request in unmatched:
                    if request.attr_type not in small_types:
                        continue
                    candidates = [
                        candidate_from_record(record, rank=rank, source="mysql")
                        for rank, record in enumerate(records_by_type.get(request.attr_type, []), start=1)
                    ]
                    history_summaries = self._history_summaries(candidates)
                    results[request.input_index] = AttributeRetrievalResult(
                        input_index=request.input_index,
                        attr_type=request.attr_type,
                        value_text=request.value_text,
                        mode="full_scan",
                        group_size=request.group_size,
                        candidates=candidates,
                        history_summaries=history_summaries,
                    )

            if large_requests:
                hybrid_results = await self._retrieve_hybrid(entity, large_requests)
                results.update({result.input_index: result for result in hybrid_results})

        return [results[index] for index in range(len(incoming_attributes))]

    async def _retrieve_hybrid(self, entity: StoredEntity, requests: list[_AttributeRequest]) -> list[AttributeRetrievalResult]:
        large_types = sorted({request.attr_type for request in requests})
        active_records = self.repository.list_active_attributes_for_types(entity.id, large_types)
        repair_warnings = await self._repair_indexes(active_records, entity.scope)

        query_texts = [attribute_text_for_embedding(request.attr_type, request.value_text) for request in requests]
        vectors = await self.embedding_client.embed_texts(query_texts)

        es_results: dict[int, list[AttributeCandidate]] = {}
        milvus_results: dict[int, list[AttributeCandidate]] = {}
        warnings: list[str] = list(repair_warnings)
        es_failed = False
        milvus_failed = False

        es_requests = [
            {
                "input_index": request.input_index,
                "entity_id": entity.id,
                "attr_type": request.attr_type,
                "scope": entity.scope,
                "query": request.value_text,
            }
            for request in requests
        ]
        try:
            es_results = await asyncio.to_thread(
                self.index.search_elasticsearch_many,
                es_requests,
                self.config.attribute_resolution_top_k,
            )
        except Exception as exc:
            es_failed = True
            warnings.append(f"elasticsearch failed: {exc}")

        try:
            milvus_results = await self._search_milvus_groups(entity, requests, vectors)
        except Exception as exc:
            milvus_failed = True
            warnings.append(f"milvus failed: {exc}")

        if es_failed and milvus_failed:
            raise AttributeRetrievalError("attribute hybrid retrieval failed in both Elasticsearch and Milvus")

        fused_by_input: dict[int, list[AttributeCandidate]] = {}
        all_candidate_ids: list[int] = []
        for request in requests:
            fused = reciprocal_rank_fusion_attributes(
                [es_results.get(request.input_index, []), milvus_results.get(request.input_index, [])],
                rrf_k=self.config.attribute_resolution_rrf_k,
                top_k=self.config.attribute_resolution_top_k,
            )
            fused_by_input[request.input_index] = fused
            all_candidate_ids.extend(candidate.attribute_id for candidate in fused)

        hydrated = {record.id: record for record in self.repository.get_active_attributes_by_ids(all_candidate_ids)}
        results: list[AttributeRetrievalResult] = []
        for request in requests:
            candidates: list[AttributeCandidate] = []
            for fused in fused_by_input.get(request.input_index, []):
                record = hydrated.get(fused.attribute_id)
                if record is None:
                    continue
                if record.entity_id != entity.id or record.attr_type != request.attr_type or record.status != "active":
                    continue
                candidates.append(candidate_from_record(record, rank=len(candidates) + 1, source=fused.source, fused=fused))
            results.append(
                AttributeRetrievalResult(
                    input_index=request.input_index,
                    attr_type=request.attr_type,
                    value_text=request.value_text,
                    mode="hybrid",
                    group_size=request.group_size,
                    candidates=candidates,
                    warnings=list(warnings),
                    history_summaries=self._history_summaries(candidates),
                )
            )
        return results

    def _history_summaries(self, candidates: list[AttributeCandidate]) -> dict[int, dict[str, Any]]:
        reader = getattr(self.repository, "attribute_event_summaries", None)
        if reader is None or not candidates:
            return {}
        try:
            return reader([candidate.attribute_id for candidate in candidates[:5]])
        except Exception:
            return {}

    async def _repair_indexes(self, records: list[StoredEntityAttribute], scope: EntityScope) -> list[str]:
        warnings: list[str] = []
        if not records:
            return warnings

        attribute_ids = [record.id for record in records]
        missing_es: list[StoredEntityAttribute] = []
        missing_milvus: list[StoredEntityAttribute] = []

        try:
            ensure_es = getattr(self.index, "ensure_elasticsearch_index", None)
            if ensure_es is not None:
                await asyncio.to_thread(ensure_es)
            existing_es = await asyncio.to_thread(self.index.existing_elasticsearch_attribute_ids, attribute_ids)
            missing_es = [record for record in records if record.id not in existing_es]
        except Exception as exc:
            warnings.append(f"elasticsearch repair failed: {exc}")

        try:
            ensure_milvus = getattr(self.index, "ensure_milvus_collection", None)
            if ensure_milvus is not None:
                await asyncio.to_thread(ensure_milvus)
            existing_milvus = await asyncio.to_thread(self.index.existing_milvus_attribute_ids, attribute_ids, scope)
            missing_milvus = [record for record in records if record.id not in existing_milvus]
        except Exception as exc:
            warnings.append(f"milvus repair failed: {exc}")

        missing_by_id = {record.id: record for record in [*missing_es, *missing_milvus]}
        if not missing_by_id:
            return warnings

        missing_es_ids = {record.id for record in missing_es}
        missing_milvus_ids = {record.id for record in missing_milvus}
        batch_size = max(1, int(self.config.attribute_index_backfill_batch_size))
        for batch in chunked(list(missing_by_id.values()), batch_size):
            vectors = await self.embedding_client.embed_texts(
                [attribute_text_for_embedding(record.attr_type, record.value_text) for record in batch]
            )
            vectors_by_id = {record.id: vector for record, vector in zip(batch, vectors)}
            es_batch = [record for record in batch if record.id in missing_es_ids]
            milvus_batch = [record for record in batch if record.id in missing_milvus_ids]
            if es_batch:
                try:
                    await asyncio.to_thread(self.index.upsert_elasticsearch_attributes, es_batch)
                except Exception as exc:
                    warnings.append(f"elasticsearch repair failed: {exc}")
            if milvus_batch:
                try:
                    await asyncio.to_thread(
                        self.index.upsert_milvus_attributes,
                        milvus_batch,
                        [vectors_by_id[record.id] for record in milvus_batch],
                    )
                except Exception as exc:
                    warnings.append(f"milvus repair failed: {exc}")
        return warnings
    async def _search_milvus_groups(
        self,
        entity: StoredEntity,
        requests: list[_AttributeRequest],
        vectors: list[list[float]],
    ) -> dict[int, list[AttributeCandidate]]:
        semaphore = asyncio.Semaphore(max(1, int(self.config.attribute_milvus_max_concurrency)))
        grouped: dict[tuple[int, str, EntityScope], list[tuple[_AttributeRequest, list[float]]]] = {}
        for request, vector in zip(requests, vectors):
            grouped.setdefault((entity.id, request.attr_type, entity.scope), []).append((request, vector))

        async def run_group(key: tuple[int, str, EntityScope], items: list[tuple[_AttributeRequest, list[float]]]):
            async with semaphore:
                entity_id, attr_type, scope = key
                group_vectors = [vector for _, vector in items]
                group_hits = await asyncio.to_thread(
                    self.index.search_milvus_group,
                    group_vectors,
                    entity_id=entity_id,
                    attr_type=attr_type,
                    scope=scope,
                    top_k=self.config.attribute_resolution_top_k,
                )
                return [(request.input_index, hits) for (request, _), hits in zip(items, group_hits)]

        results: dict[int, list[AttributeCandidate]] = {}
        groups = await asyncio.gather(*(run_group(key, items) for key, items in grouped.items()))
        for group in groups:
            for input_index, hits in group:
                results[input_index] = hits
        return results


def candidate_from_record(
    record: StoredEntityAttribute,
    *,
    rank: int,
    source: str,
    fused: AttributeCandidate | None = None,
) -> AttributeCandidate:
    return AttributeCandidate(
        attribute_id=record.id,
        entity_id=record.entity_id,
        attr_type=record.attr_type,
        value_text=record.value_text,
        confidence=record.confidence,
        rank=rank,
        source=source,
        es_score=fused.es_score if fused else 0.0,
        vector_score=fused.vector_score if fused else 0.0,
        fused_score=fused.fused_score if fused else 0.0,
    )


def reciprocal_rank_fusion_attributes(
    rankings: list[list[AttributeCandidate]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[AttributeCandidate]:
    scores: dict[int, float] = {}
    best: dict[int, AttributeCandidate] = {}
    sources: dict[int, list[str]] = {}
    es_scores: dict[int, float] = {}
    vector_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            attribute_id = int(hit.attribute_id)
            scores[attribute_id] = scores.get(attribute_id, 0.0) + (1.0 / (rrf_k + rank))
            sources.setdefault(attribute_id, []).append(hit.source)
            es_scores[attribute_id] = max(es_scores.get(attribute_id, 0.0), hit.es_score)
            vector_scores[attribute_id] = max(vector_scores.get(attribute_id, 0.0), hit.vector_score)
            if attribute_id not in best:
                best[attribute_id] = hit

    fused: list[AttributeCandidate] = []
    for rank, attribute_id in enumerate(sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k], start=1):
        hit = best[attribute_id]
        fused.append(
            AttributeCandidate(
                attribute_id=hit.attribute_id,
                entity_id=hit.entity_id,
                attr_type=hit.attr_type,
                value_text=hit.value_text,
                confidence=hit.confidence,
                rank=rank,
                source="+".join(unique_ordered(sources.get(attribute_id, []))),
                es_score=es_scores.get(attribute_id, 0.0),
                vector_score=vector_scores.get(attribute_id, 0.0),
                fused_score=scores[attribute_id],
            )
        )
    return fused


def unique_ordered(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

def chunked(records: list[StoredEntityAttribute], size: int) -> list[list[StoredEntityAttribute]]:
    return [records[index:index + size] for index in range(0, len(records), size)]
