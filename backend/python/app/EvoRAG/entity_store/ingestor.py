import asyncio

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import EntityResolutionDecision, EntityScope, EntityUpsertResult
from app.EvoRAG.entity_store.normalizer import dedupe_extracted_entities, merge_unique
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.entity_store.resolver import EntityResolver
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.models import EvoRAGPreprocessResult


class EntityIngestor:
    def __init__(
        self,
        repository: MySQLEntityRepository | None = None,
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
        self.embedding_client = EvoRAGEmbeddingClient(config)
        self.hybrid_index = EntityHybridIndex(config)

    async def ingest(self, result: EvoRAGPreprocessResult) -> list[EntityUpsertResult]:
        incoming_entities = dedupe_extracted_entities(result.blocks, scope=self.scope)
        vectors = await self.embedding_client.embed_texts(
            [entity.identity_description or entity.description_for_match or entity.name for entity in incoming_entities]
        )
        for entity, vector in zip(incoming_entities, vectors, strict=False):
            entity.embedding = vector

        existing_entities = await asyncio.to_thread(self.repository.list_entities, self.scope)
        resolver = EntityResolver(existing_entities, config=self.config, hybrid_index=self.hybrid_index)
        decisions = await resolver.resolve_many(incoming_entities)
        return await self.apply_decisions(decisions)

    async def apply_decisions(self, decisions: list[EntityResolutionDecision]) -> list[EntityUpsertResult]:
        semaphore = asyncio.Semaphore(max(1, self.config.entity_merge_max_concurrency))
        tasks = [self._apply_one(decision, semaphore) for decision in decisions if decision.decision != "ambiguous"]
        return await asyncio.gather(*tasks)

    async def _apply_one(self, decision: EntityResolutionDecision, semaphore: asyncio.Semaphore) -> EntityUpsertResult:
        async with semaphore:
            matched_id = decision.matched_entity.id if decision.matched_entity else None
            if decision.matched_entity:
                decision.incoming.aliases = merge_unique(
                    [
                        *decision.matched_entity.aliases,
                        decision.matched_entity.canonical_name,
                        decision.incoming.name,
                        *decision.incoming.aliases,
                    ]
                )
            result = await asyncio.to_thread(self.repository.upsert_entity, decision.incoming, matched_entity_id=matched_id)
            stored = await asyncio.to_thread(self.repository.get_entity, result.entity_id)
            if stored is not None:
                await asyncio.to_thread(self.hybrid_index.upsert_entity, stored)
            await asyncio.to_thread(
                self.repository.record_resolution_audit,
                decision.incoming,
                decision.decision,
                matched_id,
                decision.score,
                decision.reason,
            )
            return result
