import asyncio
from typing import Any

from app.EvoRAG.entity_store.ingestor import EntityIngestor, MemoryGuardedHybridPreview
from app.EvoRAG.entity_store.models import (
    CandidateEntity,
    EntityIngestQueueResult,
    EntityResolutionDecision,
    EntityScope,
    IncomingEntity,
    StoredEntity,
)
from app.EvoRAG.models import EvoRAGPreprocessResult
from app.EvoRAG.services.processor import EvoRAGProcessor


async def run_debug_pipeline(
    text: str,
    *,
    scope: EntityScope,
    source_note_id: str = "",
    task_name: str = "pipeline-debug",
    processor: Any | None = None,
    ingestor: Any | None = None,
) -> dict[str, object]:
    processor = processor or EvoRAGProcessor()
    ingestor = ingestor or EntityIngestor(scope=scope)

    preprocess = await processor.preprocess(text)
    preview = await ingestor.preview_memory_guarded_hybrid(preprocess, source_note_id=source_note_id)
    incoming_entities, decisions = unpack_preview(preview)
    queue_result = await ingestor.queue(preprocess, source_note_id=source_note_id, task_name=task_name)
    stage3 = await merge_persistence_stage_payload(ingestor, queue_result)

    return {
        "input_text": preprocess.input_text,
        "scope": scope.as_dict(),
        "stages": {
            "stage1_extraction": extraction_stage_payload(preprocess),
            "stage2_memory_guarded_hybrid": memory_guarded_hybrid_stage_payload(incoming_entities, decisions),
            "stage3_merge_persistence": stage3,
        },
    }


def unpack_preview(preview: Any) -> tuple[list[IncomingEntity], list[EntityResolutionDecision]]:
    if isinstance(preview, MemoryGuardedHybridPreview):
        return preview.incoming_entities, preview.decisions
    if isinstance(preview, tuple) and len(preview) == 2:
        return list(preview[0]), list(preview[1])
    incoming_entities = getattr(preview, "incoming_entities", [])
    decisions = getattr(preview, "decisions", [])
    return list(incoming_entities), list(decisions)


def extraction_stage_payload(preprocess: EvoRAGPreprocessResult) -> dict[str, object]:
    block_split_blocks: list[dict[str, object]] = []
    entity_extraction_blocks: list[dict[str, object]] = []

    for block_result in preprocess.blocks:
        block = block_result.block.model_dump()
        block_split_blocks.append(block)
        entity_extraction_blocks.append(
            {
                "block_index": block_result.block.block_index,
                "heading": block_result.block.heading,
                "anchor_entity": block_result.block.anchor_entity,
                "entities": [entity.model_dump() for entity in block_result.entities],
                "warnings": list(block_result.warnings),
            }
        )

    return {
        "input_text": preprocess.input_text,
        "block_count": len(block_split_blocks),
        "entity_count": preprocess.entity_count,
        "block_split": {"blocks": block_split_blocks},
        "entity_extraction": {"blocks": entity_extraction_blocks},
        "timings": preprocess.timings,
    }


def memory_guarded_hybrid_stage_payload(
    incoming_entities: list[IncomingEntity],
    decisions: list[EntityResolutionDecision],
) -> dict[str, object]:
    return {
        "incoming_count": len(incoming_entities),
        "decision_count": len(decisions),
        "incoming_entities": [incoming_entity_payload(entity) for entity in incoming_entities],
        "decisions": [resolution_decision_payload(decision) for decision in decisions],
        "traces": [resolution_trace_payload(decision) for decision in decisions],
    }


async def merge_persistence_stage_payload(
    ingestor: Any,
    queue_result: EntityIngestQueueResult,
) -> dict[str, object]:
    processed: list[dict[str, object]] = []
    for incoming_entity_id in queue_result.incoming_entity_ids:
        status = await ingestor.process_incoming_entity_id(incoming_entity_id, worker_id="debug-pipeline")
        processed.append({"incoming_entity_id": incoming_entity_id, "status": status})

    job_status = None
    repository = getattr(ingestor, "repository", None)
    get_job_status = getattr(repository, "get_ingest_job_status", None)
    if get_job_status is not None:
        job_status = await asyncio.to_thread(get_job_status, queue_result.job_id)

    return {
        "job_id": queue_result.job_id,
        "status": queue_result.status,
        "queued_count": queue_result.queued_count,
        "incoming_entity_ids": list(queue_result.incoming_entity_ids),
        "processed": processed,
        "job_status": job_status,
    }


def incoming_entity_payload(entity: IncomingEntity) -> dict[str, object]:
    return {
        "name": entity.name,
        "normalized_name": entity.normalized_name,
        "entity_type": entity.entity_type,
        "scope": entity.scope.as_dict(),
        "aliases": list(entity.aliases),
        "identity_description": entity.identity_description,
        "description_for_match": entity.description_for_match,
        "source_count": entity.source_count,
        "attribute_count": len(entity.attributes),
        "attributes": [
            {
                "attr_type": attribute.attr_type,
                "value_text": attribute.value_text,
                "evidence": attribute.evidence,
                "confidence": attribute.confidence,
                "note_id": attribute.note_id,
                "block_id": attribute.block_id,
                "block_index": attribute.block_index,
            }
            for attribute in entity.attributes
        ],
    }


def stored_entity_payload(entity: StoredEntity | None) -> dict[str, object] | None:
    if entity is None:
        return None
    return {
        "id": entity.id,
        "canonical_name": entity.canonical_name,
        "normalized_name": entity.normalized_name,
        "entity_type": entity.entity_type,
        "scope": entity.scope.as_dict(),
        "aliases": list(entity.aliases),
        "identity_description": entity.identity_description,
        "summary": entity.summary,
        "description_for_match": entity.description_for_match,
    }


def candidate_payload(candidate: CandidateEntity) -> dict[str, object]:
    return {
        "entity": stored_entity_payload(candidate.entity),
        "entity_id": candidate.entity.id,
        "canonical_name": candidate.entity.canonical_name,
        "score": candidate.score,
        "rank": candidate.rank,
        "source": candidate.source,
        "vector_score": candidate.vector_score,
        "es_score": candidate.es_score,
    }


def resolution_decision_payload(decision: EntityResolutionDecision) -> dict[str, object]:
    return {
        "incoming": incoming_entity_payload(decision.incoming),
        "decision": decision.decision,
        "matched_entity": stored_entity_payload(decision.matched_entity),
        "score": decision.score,
        "reason": decision.reason,
        "candidates": [candidate_payload(candidate) for candidate in decision.candidates],
        "experience": {
            "record_experience": decision.record_experience,
            "decision": decision.experience_decision,
            "relation_type": decision.experience_relation_type,
            "source": decision.experience_source,
            "confidence": decision.experience_confidence,
        },
    }


def resolution_trace_payload(decision: EntityResolutionDecision) -> dict[str, object]:
    return {
        "incoming_entity": incoming_entity_payload(decision.incoming),
        "final_decision": resolution_decision_payload(decision),
        "exit_stage": decision.resolution_exit_stage,
        "steps": list(decision.resolution_trace),
    }
