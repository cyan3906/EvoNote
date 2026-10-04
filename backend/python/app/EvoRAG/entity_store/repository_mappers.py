import json
from dataclasses import asdict
from typing import Any

from app.EvoRAG.entity_store.models import (
    CandidateEntity,
    EntityAttributeInput,
    EntityScope,
    IncomingEntity,
    IncomingEntityTask,
    StoredEntity,
    StoredEntityAttribute,
)
from app.EvoRAG.models import StoredAttribute, StoredAttributeEvidence


def row_to_entity(row: dict[str, Any]) -> StoredEntity:
    return StoredEntity(
        id=int(row["id"]),
        canonical_name=str(row["canonical_name"]),
        normalized_name=str(row["normalized_name"]),
        entity_type=str(row["entity_type"]),
        scope=EntityScope(
            workspace_id=str(row.get("workspace_id") or "local"),
            project_id=str(row.get("project_id") or "evorag"),
            collection_id=str(row.get("collection_id") or "default"),
            domain=str(row.get("domain") or "general"),
        ),
        aliases=json.loads(row.get("aliases_json") or "[]"),
        identity_description=str(row.get("identity_description") or ""),
        summary=str(row.get("summary") or ""),
        description_for_match=str(row.get("description_for_match") or ""),
        embedding=json.loads(row.get("embedding_json") or "[]"),
    )


def row_to_entity_attribute(row: dict[str, Any]) -> StoredEntityAttribute:
    return StoredEntityAttribute(
        id=int(row["attribute_id"]),
        entity_id=int(row["entity_id"]),
        scope=EntityScope(
            workspace_id=str(row.get("workspace_id") or "local"),
            project_id=str(row.get("project_id") or "evorag"),
            collection_id=str(row.get("collection_id") or "default"),
            domain=str(row.get("domain") or "general"),
        ),
        attr_type=str(row["attr_type"]),
        value_text=str(row.get("value_text") or ""),
        value_fingerprint=str(row.get("value_fingerprint") or ""),
        confidence=float(row.get("confidence") or 0.7),
        status=str(row.get("status") or "active"),
    )


def row_to_incoming_task(row: dict[str, Any]) -> IncomingEntityTask:
    attributes = [
        EntityAttributeInput(
            attr_type=str(item.get("attr_type") or ""),
            value_text=str(item.get("value_text") or ""),
            evidence=str(item.get("evidence") or ""),
            confidence=float(item.get("confidence") or 0.7),
            note_id=str(item.get("note_id") or ""),
            block_id=str(item.get("block_id") or ""),
            block_index=int(item.get("block_index") if item.get("block_index") is not None else -1),
        )
        for item in json.loads(row.get("attributes_json") or "[]")
        if isinstance(item, dict)
    ]
    incoming = IncomingEntity(
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        entity_type=str(row["entity_type"] or "concept"),
        scope=EntityScope(
            workspace_id=str(row.get("workspace_id") or "local"),
            project_id=str(row.get("project_id") or "evorag"),
            collection_id=str(row.get("collection_id") or "default"),
            domain=str(row.get("domain") or "general"),
        ),
        aliases=json.loads(row.get("aliases_json") or "[]"),
        identity_description=str(row.get("identity_description") or ""),
        attributes=attributes,
        description_for_match=str(row.get("description_for_match") or ""),
        source_count=int(row.get("source_count") or 1),
    )
    return IncomingEntityTask(
        id=int(row["id"]),
        job_id=int(row["job_id"]),
        incoming=incoming,
        status=str(row.get("status") or ""),
        attempt_count=int(row.get("attempt_count") or 0),
    )


def incoming_snapshot(incoming: IncomingEntity) -> dict[str, Any]:
    return {
        "name": incoming.name,
        "normalized_name": incoming.normalized_name,
        "entity_type": incoming.entity_type,
        "aliases": incoming.aliases,
        "identity_description": incoming.identity_description,
        "description_for_match": incoming.description_for_match,
        "source_count": incoming.source_count,
        "attributes": [asdict(attribute) for attribute in incoming.attributes],
        "scope": incoming.scope.as_dict(),
    }


def candidate_snapshot(candidate: CandidateEntity) -> dict[str, Any]:
    entity = candidate.entity
    return {
        "id": entity.id,
        "canonical_name": entity.canonical_name,
        "normalized_name": entity.normalized_name,
        "entity_type": entity.entity_type,
        "aliases": entity.aliases,
        "identity_description": entity.identity_description,
        "summary": entity.summary,
        "description_for_match": entity.description_for_match,
        "score": candidate.score,
        "rank": candidate.rank,
        "source": candidate.source,
        "vector_score": candidate.vector_score,
        "es_score": candidate.es_score,
    }


def review_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["incoming_snapshot"] = json.loads(item.pop("incoming_snapshot_json") or "{}")
    item["candidates"] = json.loads(item.pop("candidates_json") or "[]")
    for key in ("created_at", "updated_at"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    return item


def job_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("created_at", "updated_at"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    return item


def incoming_status_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("created_at", "updated_at", "locked_until"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    if item.get("review_task_id") is not None:
        item["review_task_id"] = int(item["review_task_id"])
    return item


def build_job_progress(status_counts: dict[str, int], total: int) -> dict[str, Any]:
    finished = sum(
        status_counts.get(status, 0)
        for status in ("auto_merged", "manual_merged", "new_created", "completed")
    )
    needs_review = status_counts.get("needs_review", 0)
    failed = status_counts.get("failed", 0) + status_counts.get("dead_letter", 0)
    active = status_counts.get("pending", 0) + status_counts.get("processing", 0)
    done = finished + needs_review + failed
    return {
        "total": total,
        "active": active,
        "finished": finished,
        "needs_review": needs_review,
        "failed": failed,
        "done": done,
        "percent": round((done / total) * 100, 1) if total else 100.0,
    }


def group_attribute_rows(rows: list[dict[str, Any]]) -> dict[int, list[StoredAttribute]]:
    grouped: dict[int, dict[int, StoredAttribute]] = {}
    for row in rows:
        entity_id = int(row["entity_id"])
        attribute_id = int(row["attribute_id"])
        entity_attributes = grouped.setdefault(entity_id, {})
        attribute = entity_attributes.get(attribute_id)
        if attribute is None:
            attribute = StoredAttribute(
                id=attribute_id,
                attr_type=str(row["attr_type"]),
                value_text=str(row["value_text"] or ""),
                confidence=float(row.get("confidence") or 0.7),
            )
            entity_attributes[attribute_id] = attribute
        if row.get("evidence_text"):
            attribute.evidence.append(
                StoredAttributeEvidence(
                    evidence_text=str(row.get("evidence_text") or ""),
                    note_id=str(row.get("note_id") or ""),
                    block_id=str(row.get("block_id") or ""),
                    block_index=int(row.get("block_index") if row.get("block_index") is not None else -1),
                )
            )
    return {entity_id: list(attributes.values()) for entity_id, attributes in grouped.items()}


def build_summary(incoming: IncomingEntity) -> str:
    definitions = [item.value_text for item in incoming.attributes if item.attr_type == "definition"]
    if definitions:
        return definitions[0]
    if incoming.identity_description:
        return incoming.identity_description[:500]
    return incoming.description_for_match[:500]


def attribute_snapshot(attribute_id: int | None, attr_type: str, value_text: str, confidence: float) -> dict[str, Any] | None:
    if not attribute_id:
        return None
    return {
        "attribute_id": int(attribute_id),
        "attr_type": attr_type,
        "value_text": value_text or "",
        "confidence": float(confidence or 0.0),
    }


def attribute_event_diff(
    *,
    action: str,
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
    evidence_added: int,
    target_attribute_id: int | None,
) -> dict[str, Any]:
    before_value = str((before_snapshot or {}).get("value_text") or "")
    after_value = str((after_snapshot or {}).get("value_text") or "")
    return {
        "action": action,
        "target_attribute_id": int(target_attribute_id) if target_attribute_id else None,
        "value_text": {
            "before": before_value,
            "after": after_value,
            "changed": before_value != after_value,
        },
        "evidence_added": int(evidence_added),
    }


def review_stats(events: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for event in events:
        status = str(event.get("review_status") or "pending")
        stats[status] = stats.get(status, 0) + 1
    return stats


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": str(event.get("action") or ""),
        "review_status": str(event.get("review_status") or ""),
        "reason": str(event.get("decision_reason") or ""),
        "confidence": float(event.get("confidence") or 0.0),
        "source": str(event.get("source") or ""),
        "created_at": str(event.get("created_at") or ""),
    }
