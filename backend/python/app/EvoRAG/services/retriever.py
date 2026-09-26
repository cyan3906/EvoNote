from time import perf_counter

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.indexes.entity_hybrid import reciprocal_rank_fusion_entities
from app.EvoRAG.models import EvoRAGIndexSearchResult, RetrievedEntity, StoredAttribute


class EvoRAGRetriever:
    def __init__(
        self,
        repository: MySQLEntityRepository | None = None,
        hybrid_index: EntityHybridIndex | None = None,
        embedding_client: EvoRAGEmbeddingClient | None = None,
        config: EvoRAGSettings = settings,
        scope: EntityScope | None = None,
    ) -> None:
        self.config = config
        self.scope = scope or EntityScope(
            workspace_id=config.default_workspace_id,
            project_id=config.default_project_id,
            collection_id=config.default_collection_id,
            domain=config.default_domain,
        )
        self.repository = repository or MySQLEntityRepository(config)
        self.hybrid_index = hybrid_index or EntityHybridIndex(config)
        self.embedding_client = embedding_client or EvoRAGEmbeddingClient(config)

    async def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[list[RetrievedEntity], list[str]]:
        normalized_query = normalize_name(query)
        if not normalized_query:
            return [], ["query is empty"]

        limit = top_k or self.config.entity_resolution_top_k
        warnings: list[str] = []
        incoming = IncomingEntity(
            name=query.strip(),
            normalized_name=normalized_query,
            entity_type="",
            scope=self.scope,
            identity_description=query.strip(),
            description_for_match=query.strip(),
        )

        try:
            incoming.embedding = await self.embedding_client.embed_text(query)
        except Exception as exc:
            warnings.append(f"embedding skipped: {exc}")

        candidates: list[CandidateEntity] = []
        exact = self.repository.find_by_normalized_name(normalized_query, scope=self.scope)
        if exact is not None:
            candidates.append(CandidateEntity(entity=exact, score=1.0, rank=1, source="exact_name"))

        try:
            candidates.extend(self.hybrid_index.search(incoming, top_k=limit))
        except Exception as exc:
            warnings.append(f"hybrid retrieval skipped: {exc}")
            candidates.extend(self.fallback_candidates(normalized_query, limit=limit))

        candidates = dedupe_candidates(candidates, limit=limit)
        entities = self.hydrate_candidates(candidates)
        if not entities:
            warnings.append("no matching entity found")
        return entities, warnings

    async def search_indexes(self, query: str, *, top_k: int | None = None) -> EvoRAGIndexSearchResult:
        total_started_at = perf_counter()
        timings: dict[str, float] = {}
        normalized_query = normalize_name(query)
        if not normalized_query:
            timings["total_ms"] = elapsed_ms(total_started_at)
            return EvoRAGIndexSearchResult(query=query, timings=timings, warnings=["query is empty"])

        limit = top_k or self.config.entity_resolution_top_k
        warnings: list[str] = []
        stage_started_at = perf_counter()
        backend_status = self.backend_status(warnings)
        timings["backend_status_ms"] = elapsed_ms(stage_started_at)
        incoming = IncomingEntity(
            name=query.strip(),
            normalized_name=normalized_query,
            entity_type="",
            scope=self.scope,
            identity_description=query.strip(),
            description_for_match=query.strip(),
        )

        try:
            stage_started_at = perf_counter()
            incoming.embedding = await self.embedding_client.embed_text(query)
            timings["embedding_ms"] = elapsed_ms(stage_started_at)
        except Exception as exc:
            timings["embedding_ms"] = elapsed_ms(stage_started_at)
            warnings.append(f"embedding skipped: {exc}")

        es_candidates: list[CandidateEntity] = []
        try:
            stage_started_at = perf_counter()
            self.hybrid_index.ensure_elasticsearch_index()
            timings["elasticsearch_ensure_ms"] = elapsed_ms(stage_started_at)
            stage_started_at = perf_counter()
            es_candidates = self.hybrid_index.search_elasticsearch(incoming, top_k=limit)
            timings["elasticsearch_search_ms"] = elapsed_ms(stage_started_at)
        except Exception as exc:
            timings.setdefault("elasticsearch_ensure_ms", elapsed_ms(stage_started_at))
            warnings.append(f"elasticsearch search skipped: {exc}")

        milvus_candidates: list[CandidateEntity] = []
        try:
            stage_started_at = perf_counter()
            self.hybrid_index.ensure_milvus_collection()
            timings["milvus_ensure_ms"] = elapsed_ms(stage_started_at)
            stage_started_at = perf_counter()
            milvus_candidates = self.hybrid_index.search_milvus(incoming.embedding, top_k=limit, scope=self.scope)
            timings["milvus_search_ms"] = elapsed_ms(stage_started_at)
        except Exception as exc:
            timings.setdefault("milvus_ensure_ms", elapsed_ms(stage_started_at))
            warnings.append(f"milvus search skipped: {exc}")

        stage_started_at = perf_counter()
        fused_candidates = reciprocal_rank_fusion_entities(
            [es_candidates, milvus_candidates],
            rrf_k=self.config.entity_resolution_rrf_k,
            top_k=limit,
        )
        timings["fusion_ms"] = elapsed_ms(stage_started_at)

        stage_started_at = perf_counter()
        entity_by_id, attributes_by_entity = self.safe_hydrate_candidate_data(
            [*es_candidates, *milvus_candidates, *fused_candidates],
            warnings=warnings,
        )
        elasticsearch_results = hydrate_candidates_from_maps(es_candidates, entity_by_id, attributes_by_entity)
        milvus_results = hydrate_candidates_from_maps(milvus_candidates, entity_by_id, attributes_by_entity)
        fused_results = hydrate_candidates_from_maps(fused_candidates, entity_by_id, attributes_by_entity)
        timings["mysql_hydration_ms"] = elapsed_ms(stage_started_at)
        timings["total_ms"] = elapsed_ms(total_started_at)

        return EvoRAGIndexSearchResult(
            query=query,
            backend_status=backend_status,
            timings=timings,
            elasticsearch_results=elasticsearch_results,
            milvus_results=milvus_results,
            fused_results=fused_results,
            warnings=warnings,
        )

    def backend_status(self, warnings: list[str] | None = None) -> dict[str, object]:
        try:
            return self.hybrid_index.backend_status()
        except Exception as exc:
            if warnings is not None:
                warnings.append(f"backend health check skipped: {exc}")
            return {}

    def safe_hydrate_candidate_data(
        self,
        candidates: list[CandidateEntity],
        *,
        warnings: list[str],
    ) -> tuple[dict[int, StoredEntity], dict[int, list[StoredAttribute]]]:
        entity_ids = [candidate.entity.id for candidate in candidates if candidate.entity.id]
        if not entity_ids:
            return {}, {}
        try:
            return self.repository.hydrate_entities_for_candidates(entity_ids)
        except Exception as exc:
            warnings.append(f"mysql hydration skipped: {exc}")
            return {}, {}

    def safe_hydrate_candidates(
        self,
        candidates: list[CandidateEntity],
        *,
        warnings: list[str],
        label: str,
    ) -> list[RetrievedEntity]:
        try:
            return self.hydrate_candidates(candidates)
        except Exception as exc:
            warnings.append(f"{label} mysql hydration skipped: {exc}")
            return [
                stored_entity_to_retrieved(
                    candidate.entity,
                    attributes=[],
                    score=candidate.score,
                    rank=candidate.rank,
                    source=candidate.source,
                )
                for candidate in candidates
            ]

    def fallback_candidates(self, normalized_query: str, *, limit: int) -> list[CandidateEntity]:
        candidates: list[CandidateEntity] = []
        for entity in self.repository.list_entities(self.scope):
            score = lexical_score(normalized_query, entity)
            if score <= 0:
                continue
            candidates.append(CandidateEntity(entity=entity, score=score, source="mysql_fallback"))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        for rank, candidate in enumerate(candidates[:limit], start=1):
            candidate.rank = rank
        return candidates[:limit]

    def hydrate_candidates(self, candidates: list[CandidateEntity]) -> list[RetrievedEntity]:
        stored_entities: list[StoredEntity] = []
        for candidate in candidates:
            entity = self.repository.get_entity(candidate.entity.id) or candidate.entity
            stored_entities.append(entity)

        attributes_by_entity = self.repository.list_attributes_for_entities([entity.id for entity in stored_entities])
        retrieved: list[RetrievedEntity] = []
        score_by_id = {candidate.entity.id: candidate for candidate in candidates}
        for rank, entity in enumerate(stored_entities, start=1):
            candidate = score_by_id.get(entity.id)
            retrieved.append(
                stored_entity_to_retrieved(
                    entity,
                    attributes=attributes_by_entity.get(entity.id, []),
                    score=candidate.score if candidate else 0.0,
                    rank=rank,
                    source=candidate.source if candidate else "",
                )
            )
        return retrieved


def dedupe_candidates(candidates: list[CandidateEntity], *, limit: int) -> list[CandidateEntity]:
    best_by_id: dict[int, CandidateEntity] = {}
    for candidate in candidates:
        entity_id = candidate.entity.id
        existing = best_by_id.get(entity_id)
        if existing is None or candidate.score > existing.score:
            best_by_id[entity_id] = candidate
        elif candidate.source and candidate.source not in existing.source:
            existing.source = "+".join(part for part in [existing.source, candidate.source] if part)
    deduped = sorted(best_by_id.values(), key=lambda candidate: candidate.score, reverse=True)[:limit]
    for rank, candidate in enumerate(deduped, start=1):
        candidate.rank = rank
    return deduped


def lexical_score(normalized_query: str, entity: StoredEntity) -> float:
    names = [entity.normalized_name, *(normalize_name(alias) for alias in entity.aliases)]
    if normalized_query in names:
        return 1.0
    searchable = normalize_name(
        " ".join(
            part
            for part in [
                entity.canonical_name,
                " ".join(entity.aliases),
                entity.entity_type,
                entity.identity_description,
                entity.summary,
                entity.description_for_match,
            ]
            if part
        )
    )
    if normalized_query and normalized_query in searchable:
        return 0.7
    query_terms = set(normalized_query.split())
    searchable_terms = set(searchable.split())
    if not query_terms or not searchable_terms:
        return 0.0
    return len(query_terms & searchable_terms) / len(query_terms)


def stored_entity_to_retrieved(
    entity: StoredEntity,
    *,
    attributes: list[StoredAttribute],
    score: float,
    rank: int,
    source: str,
) -> RetrievedEntity:
    return RetrievedEntity(
        id=entity.id,
        canonical_name=entity.canonical_name,
        normalized_name=entity.normalized_name,
        entity_type=entity.entity_type,
        aliases=entity.aliases,
        identity_description=entity.identity_description,
        summary=entity.summary,
        description_for_match=entity.description_for_match,
        score=score,
        rank=rank,
        source=source,
        attributes=attributes,
    )


def hydrate_candidates_from_maps(
    candidates: list[CandidateEntity],
    entity_by_id: dict[int, StoredEntity],
    attributes_by_entity: dict[int, list[StoredAttribute]],
) -> list[RetrievedEntity]:
    retrieved: list[RetrievedEntity] = []
    for rank, candidate in enumerate(candidates, start=1):
        entity = entity_by_id.get(candidate.entity.id, candidate.entity)
        retrieved.append(
            stored_entity_to_retrieved(
                entity,
                attributes=attributes_by_entity.get(entity.id, []),
                score=candidate.score,
                rank=rank,
                source=candidate.source,
            )
        )
    return retrieved


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
