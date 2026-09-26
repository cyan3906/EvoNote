from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.models import RetrievedEntity, StoredAttribute


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
