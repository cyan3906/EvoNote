import asyncio
from dataclasses import replace
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.attribute_decider import AttributeDecisionMaker
from app.EvoRAG.entity_store.attribute_retriever import AttributeCandidateRetriever
from app.EvoRAG.entity_store.models import EntityIngestQueueResult, EntityResolutionDecision, EntityScope, EntityUpsertResult, IncomingEntity, IncomingEntityTask
from app.EvoRAG.entity_store.normalizer import dedupe_extracted_entities, merge_unique, normalize_name
from app.EvoRAG.entity_store.relation_memory import EntityRelationMemory
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.entity_store.resolver import EntityResolver
from app.EvoRAG.indexes import AttributeHybridIndex, EntityHybridIndex, EvoRAGEmbeddingClient, attribute_text_for_embedding
from app.EvoRAG.models import EvoRAGPreprocessResult


LOW_SCORE_DIRECT_REJECT_THRESHOLD = 0.5


class EntityIngestor:
    def __init__(
        self,
        repository: MySQLEntityRepository | None = None,
        config: EvoRAGSettings = settings,
        scope: EntityScope | None = None,
        embedding_client: EvoRAGEmbeddingClient | Any | None = None,
        hybrid_index: EntityHybridIndex | Any | None = None,
        attribute_retriever: AttributeCandidateRetriever | Any | None = None,
        attribute_decider: AttributeDecisionMaker | Any | None = None,
        attribute_index: AttributeHybridIndex | Any | None = None,
        queue_client: Any | None = None,
    ) -> None:
        self.config = config
        self.scope = scope or EntityScope(
            workspace_id=config.default_workspace_id,
            project_id=config.default_project_id,
            collection_id=config.default_collection_id,
            domain=config.default_domain,
        )
        self.repository = repository or MySQLEntityRepository(config)
        self.embedding_client = embedding_client or EvoRAGEmbeddingClient(config)
        self.hybrid_index = hybrid_index or EntityHybridIndex(config)
        self.attribute_index = attribute_index or AttributeHybridIndex(config)
        self.attribute_retriever = attribute_retriever or AttributeCandidateRetriever(
            self.repository,
            self.embedding_client,
            self.attribute_index,
            config,
        )
        self.attribute_decider = attribute_decider or AttributeDecisionMaker()
        self.queue_client = queue_client

    async def queue(
        self,
        result: EvoRAGPreprocessResult,
        *,
        source_note_id: str = "",
        task_name: str = "",
    ) -> EntityIngestQueueResult:
        incoming_entities = dedupe_extracted_entities(result.blocks, scope=self.scope, source_note_id=source_note_id)
        source_blocks_by_key = source_blocks_for_incoming(result, incoming_entities)
        queue_result = await asyncio.to_thread(
            self.repository.create_ingest_job,
            input_text=result.input_text,
            preprocess=result,
            scope=self.scope,
            incoming_entities=incoming_entities,
            source_blocks_by_key=source_blocks_by_key,
            source_note_id=source_note_id,
            task_name=task_name,
        )
        return queue_result

    async def ingest(self, result: EvoRAGPreprocessResult) -> list[EntityUpsertResult]:
        incoming_entities = dedupe_extracted_entities(result.blocks, scope=self.scope)
        vectors = await self.embedding_client.embed_texts(
            [entity.identity_description or entity.description_for_match or entity.name for entity in incoming_entities]
        )
        for entity, vector in zip(incoming_entities, vectors, strict=False):
            entity.embedding = vector

        existing_entities = await asyncio.to_thread(self.repository.list_entities, self.scope)
        resolver = EntityResolver(
            existing_entities,
            config=self.config,
            hybrid_index=self.hybrid_index,
            alias_lookup=self.repository.find_alias_for_incoming,
            rejection_lookup=self.repository.rejected_candidate_ids,
            relation_memory=EntityRelationMemory(self.repository),
        )
        decisions = await resolver.resolve_many(incoming_entities)
        return await self.apply_decisions(decisions)

    async def process_incoming_entity_id(self, incoming_entity_id: int, *, worker_id: str) -> str:
        task = await asyncio.to_thread(
            self.repository.claim_incoming_entity,
            incoming_entity_id,
            worker_id=worker_id,
            lock_seconds=self.config.entity_worker_lock_seconds,
        )
        if task is None:
            return "skipped"

        try:
            status, decision = await self.process_incoming_task(task)
        except Exception as exc:
            return await asyncio.to_thread(
                self.repository.mark_incoming_failure,
                incoming_entity_id,
                error=str(exc),
                max_attempts=self.config.entity_worker_max_attempts,
            )

        matched_id = decision.matched_entity.id if decision.matched_entity else None
        await asyncio.to_thread(
            self.repository.finish_incoming_entity,
            incoming_entity_id,
            status=status,
            decision=decision.decision,
            matched_entity_id=matched_id,
            score=decision.score,
            reason=decision.reason,
        )
        return status

    async def process_incoming_task(self, task: IncomingEntityTask) -> tuple[str, EntityResolutionDecision]:
        incoming = task.incoming
        vectors = await self.embedding_client.embed_texts(
            [incoming.identity_description or incoming.description_for_match or incoming.name]
        )
        incoming.embedding = vectors[0] if vectors else []

        existing_entities = await asyncio.to_thread(self.repository.list_entities, incoming.scope)
        resolver = EntityResolver(
            existing_entities,
            config=self.config,
            hybrid_index=self.hybrid_index,
            alias_lookup=self.repository.find_alias_for_incoming,
            rejection_lookup=self.repository.rejected_candidate_ids,
            relation_memory=EntityRelationMemory(self.repository),
        )
        decision = await resolver.resolve_one(incoming)
        results = await self.apply_decisions([decision])
        if decision.decision == "ambiguous":
            await asyncio.to_thread(
                self.repository.create_review_task,
                incoming_entity_id=task.id,
                task=task,
                candidates=decision.candidates,
                reason=decision.reason,
            )
            return "needs_review", decision
        if decision.decision == "new":
            return "new_created", decision
        return "auto_merged" if results else "completed", decision

    async def manual_merge_review_task(
        self,
        review_task_id: int,
        *,
        entity_id: int,
        reason: str = "",
        decided_by: str = "manual",
    ) -> EntityUpsertResult:
        review, task = await self._load_review_task(review_task_id)
        matched_entity = await asyncio.to_thread(self.repository.get_entity, entity_id)
        if matched_entity is None:
            raise ValueError(f"entity {entity_id} not found")

        decision = EntityResolutionDecision(
            incoming=task.incoming,
            decision="matched",
            matched_entity=matched_entity,
            score=1.0,
            reason=reason or "manual review merged",
        )
        results = await self.apply_decisions([decision])
        result = results[0] if results else EntityUpsertResult(
            entity_id=matched_entity.id,
            canonical_name=matched_entity.canonical_name,
            created=False,
            attribute_count=0,
            evidence_count=0,
        )
        await asyncio.to_thread(
            self.repository.finish_incoming_entity,
            task.id,
            status="manual_merged",
            decision="matched",
            matched_entity_id=matched_entity.id,
            score=1.0,
            reason=decision.reason,
        )
        await asyncio.to_thread(
            self.repository.complete_review_task,
            review["id"],
            status="merged",
            decision="matched",
            decided_entity_id=matched_entity.id,
            decided_by=decided_by,
            reason=decision.reason,
        )
        await self._record_relation_memory(
            incoming=task.incoming,
            candidate=matched_entity,
            decision="allow",
            relation_type="manual_match",
            confidence=1.0,
            source=decided_by,
            reason=decision.reason,
        )
        return result

    async def manual_create_new_review_task(
        self,
        review_task_id: int,
        *,
        reason: str = "",
        decided_by: str = "manual",
    ) -> EntityUpsertResult:
        review, task = await self._load_review_task(review_task_id)
        decision = EntityResolutionDecision(
            incoming=task.incoming,
            decision="new",
            score=1.0,
            reason=reason or "manual review created new entity",
        )
        results = await self.apply_decisions([decision])
        if not results:
            raise ValueError("manual new entity did not produce an upsert result")
        result = results[0]
        await asyncio.to_thread(
            self.repository.finish_incoming_entity,
            task.id,
            status="new_created",
            decision="new",
            matched_entity_id=result.entity_id,
            score=1.0,
            reason=decision.reason,
        )
        await asyncio.to_thread(
            self.repository.complete_review_task,
            review["id"],
            status="new_created",
            decision="new",
            decided_entity_id=result.entity_id,
            decided_by=decided_by,
            reason=decision.reason,
        )
        await self._record_manual_reject_experience_for_review_candidates(
            review,
            incoming=task.incoming,
            reason=decision.reason,
            decided_by=decided_by,
        )
        return result

    async def reject_review_candidates(
        self,
        review_task_id: int,
        *,
        candidate_entity_ids: list[int],
        reason: str = "",
        decided_by: str = "manual",
    ) -> dict[str, object]:
        review, task = await self._load_review_task(review_task_id)
        rejected_ids = sorted({int(entity_id) for entity_id in candidate_entity_ids if int(entity_id) > 0})
        if not rejected_ids:
            raise ValueError("candidate_entity_ids is required")

        await asyncio.to_thread(
            self.repository.record_rejections,
            incoming=task.incoming,
            candidate_entity_ids=rejected_ids,
            reason=reason or "manual review rejected candidates",
            decided_by=decided_by,
        )
        await self._record_manual_reject_experience_for_candidate_ids(
            incoming=task.incoming,
            candidate_entity_ids=rejected_ids,
            reason=reason or "manual review rejected candidates",
            decided_by=decided_by,
        )
        await asyncio.to_thread(
            self.repository.complete_review_task,
            review["id"],
            status="rejected",
            decision="rejected",
            decided_by=decided_by,
            reason=reason or "manual review rejected candidates",
        )
        await asyncio.to_thread(self.repository.reset_incoming_for_retry, task.id)
        return {"review_task_id": review["id"], "incoming_entity_id": task.id, "rejected_entity_ids": rejected_ids, "status": "requeued"}

    async def _load_review_task(self, review_task_id: int) -> tuple[dict[str, Any], IncomingEntityTask]:
        review = await asyncio.to_thread(self.repository.get_review_task, review_task_id)
        if review is None:
            raise ValueError(f"review task {review_task_id} not found")
        if str(review.get("status") or "") != "pending":
            raise ValueError(f"review task {review_task_id} is already {review.get('status')}")
        task = await asyncio.to_thread(self.repository.get_incoming_entity_task, int(review["incoming_entity_id"]))
        if task is None:
            raise ValueError(f"incoming entity {review['incoming_entity_id']} not found")
        return review, task

    async def apply_decisions(self, decisions: list[EntityResolutionDecision]) -> list[EntityUpsertResult]:
        semaphore = asyncio.Semaphore(max(1, self.config.entity_merge_max_concurrency))
        tasks = [self._apply_one(decision, semaphore) for decision in decisions]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    async def _apply_one(self, decision: EntityResolutionDecision, semaphore: asyncio.Semaphore) -> EntityUpsertResult | None:
        async with semaphore:
            matched_id = decision.matched_entity.id if decision.matched_entity else None
            if decision.decision == "ambiguous":
                await asyncio.to_thread(
                    self.repository.record_resolution_audit,
                    decision.incoming,
                    decision.decision,
                    matched_id,
                    decision.score,
                    decision.reason,
                )
                return None

            await self._record_resolution_experience(decision)

            if decision.matched_entity:
                decision.incoming.aliases = merge_unique(
                    [
                        *decision.matched_entity.aliases,
                        decision.matched_entity.canonical_name,
                        decision.incoming.name,
                        *decision.incoming.aliases,
                    ]
                )
            incoming_for_upsert = decision.incoming
            attribute_decisions = []
            if decision.matched_entity and decision.incoming.attributes:
                retrieval_results = await self.attribute_retriever.retrieve(decision.matched_entity, decision.incoming.attributes)
                attribute_decisions = self.attribute_decider.decide(decision.incoming.attributes, retrieval_results)
                incoming_for_upsert = replace(decision.incoming, attributes=[])

            result = await asyncio.to_thread(self.repository.upsert_entity, incoming_for_upsert, matched_entity_id=matched_id)
            if attribute_decisions:
                apply_result = await asyncio.to_thread(self.repository.apply_attribute_decisions, result.entity_id, attribute_decisions)
                result.attribute_count += apply_result.attribute_count
                result.evidence_count += apply_result.evidence_count
                result.conflict_count += apply_result.conflict_count
                result.changed_attribute_ids = merge_ints(result.changed_attribute_ids, apply_result.changed_attribute_ids)

            stored = await asyncio.to_thread(self.repository.get_entity, result.entity_id)
            if stored is not None:
                await asyncio.to_thread(self.repository.upsert_aliases_for_entity, stored, source=decision.decision, confidence=decision.score or 1.0)
                await asyncio.to_thread(self.hybrid_index.upsert_entity, stored)
            await self._upsert_changed_attribute_indexes(result.changed_attribute_ids)
            await asyncio.to_thread(
                self.repository.record_resolution_audit,
                decision.incoming,
                decision.decision,
                matched_id,
                decision.score,
                decision.reason,
            )
            return result

    async def _upsert_changed_attribute_indexes(self, attribute_ids: list[int]) -> None:
        unique_ids = merge_ints(attribute_ids, [])
        if not unique_ids:
            return
        records = await asyncio.to_thread(self.repository.get_active_attributes_by_ids, unique_ids)
        if not records:
            return
        vectors = await self.embedding_client.embed_texts(
            [attribute_text_for_embedding(record.attr_type, record.value_text) for record in records]
        )
        await asyncio.to_thread(self.attribute_index.upsert_elasticsearch_attributes, records)
        await asyncio.to_thread(self.attribute_index.upsert_milvus_attributes, records, vectors)

    async def _record_resolution_experience(self, decision: EntityResolutionDecision) -> None:
        if not decision.candidates:
            return
        best = decision.candidates[0]
        if decision.record_experience:
            await self._record_relation_memory(
                incoming=decision.incoming,
                candidate=best.entity,
                decision=decision.experience_decision or "reject",
                relation_type=decision.experience_relation_type or "llm_guard_reject",
                confidence=decision.experience_confidence or decision.score or best.score,
                source=decision.experience_source or "llm_guard",
                reason=decision.reason,
            )
            return
        if decision.decision != "new":
            return
        if best.score >= LOW_SCORE_DIRECT_REJECT_THRESHOLD:
            return
        await self._record_relation_memory(
            incoming=decision.incoming,
            candidate=best.entity,
            decision="reject",
            relation_type="low_score_direct_reject",
            confidence=round(1.0 - float(best.score), 6),
            source="auto",
            reason=decision.reason,
        )

    async def _record_manual_reject_experience_for_review_candidates(
        self,
        review: dict[str, Any],
        *,
        incoming: IncomingEntity,
        reason: str,
        decided_by: str,
    ) -> None:
        candidate_ids = [
            entity_id
            for entity_id in (candidate_entity_id_from_snapshot(candidate) for candidate in review.get("candidates") or [])
            if entity_id > 0
        ]
        await self._record_manual_reject_experience_for_candidate_ids(
            incoming=incoming,
            candidate_entity_ids=candidate_ids,
            reason=reason,
            decided_by=decided_by,
        )

    async def _record_manual_reject_experience_for_candidate_ids(
        self,
        *,
        incoming: IncomingEntity,
        candidate_entity_ids: list[int],
        reason: str,
        decided_by: str,
    ) -> None:
        for candidate_id in sorted({int(entity_id) for entity_id in candidate_entity_ids if int(entity_id) > 0}):
            candidate = await asyncio.to_thread(self.repository.get_entity, candidate_id)
            if candidate is None:
                continue
            await self._record_relation_memory(
                incoming=incoming,
                candidate=candidate,
                decision="reject",
                relation_type="manual_reject",
                confidence=1.0,
                source=decided_by,
                reason=reason,
            )

    async def _record_relation_memory(
        self,
        *,
        incoming: IncomingEntity,
        candidate,
        decision: str,
        relation_type: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> None:
        writer = getattr(self.repository, "upsert_entity_resolution_relation_memory", None)
        if writer is None:
            return
        await asyncio.to_thread(
            writer,
            incoming=incoming,
            candidate=candidate,
            decision=decision,
            relation_type=relation_type,
            confidence=confidence,
            source=source,
            reason=reason,
        )


def source_blocks_for_incoming(
    result: EvoRAGPreprocessResult,
    incoming_entities: list[IncomingEntity],
) -> dict[str, list[dict[str, Any]]]:
    keys = {incoming_entity_key(entity) for entity in incoming_entities}
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    seen: set[tuple[str, int]] = set()

    for block_result in result.blocks:
        block = block_result.block
        for entity in block_result.entities:
            key = f"{normalize_name(entity.name)}::{normalize_name(entity.entity_type)}"
            if key not in grouped:
                continue
            seen_key = (key, block.block_index)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            grouped[key].append(
                {
                    "block_index": block.block_index,
                    "heading": block.heading,
                    "anchor_entity": block.anchor_entity,
                    "chunk_index": block.chunk_index,
                    "chunk_count": block.chunk_count,
                    "l1_text": block.l1_text,
                }
            )
    return grouped


def candidate_entity_id_from_snapshot(candidate: dict[str, Any]) -> int:
    for key in ("id", "entity_id", "candidate_entity_id"):
        try:
            entity_id = int(candidate.get(key) or 0)
        except (TypeError, ValueError):
            entity_id = 0
        if entity_id > 0:
            return entity_id
    return 0


def incoming_entity_key(incoming: IncomingEntity) -> str:
    return f"{normalize_name(incoming.name)}::{normalize_name(incoming.entity_type)}"


def merge_ints(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in [*left, *right]:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
