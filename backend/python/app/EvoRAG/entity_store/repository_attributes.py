import json
from typing import Any

from app.EvoRAG.entity_store.models import (
    AttributeDecision,
    AttributeDecisionApplyResult,
    EntityAttributeInput,
)
from app.EvoRAG.entity_store.normalizer import text_fingerprint
from app.EvoRAG.entity_store.repository_mappers import (
    attribute_event_diff,
    attribute_snapshot,
    event_summary,
    review_stats,
)


class AttributeDecisionRepositoryMixin:
    def apply_attribute_decisions(self, entity_id: int, decisions: list[AttributeDecision]) -> AttributeDecisionApplyResult:
        outcome = AttributeDecisionApplyResult()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for decision in decisions:
                    attribute = decision.incoming_attribute
                    changed_id: int | None = None
                    event_attribute_id: int | None = None
                    before_snapshot: dict[str, Any] | None = None
                    after_snapshot: dict[str, Any] | None = None
                    evidence_added = 0
                    if decision.action == "add":
                        changed_id, evidence_added = self._insert_attribute_from_input(cursor, entity_id, attribute)
                        event_attribute_id = changed_id
                        after_snapshot = attribute_snapshot(changed_id, attribute.attr_type, attribute.value_text, attribute.confidence)
                        outcome.attribute_count += 1
                        outcome.evidence_count += evidence_added
                    elif decision.action == "merge" and decision.target_attribute_id:
                        changed_id = int(decision.target_attribute_id)
                        event_attribute_id = changed_id
                        before_snapshot = attribute_snapshot(changed_id, attribute.attr_type, decision.target_value_text, attribute.confidence)
                        cursor.execute(
                            """
                            UPDATE evorag_entity_attributes
                            SET confidence = GREATEST(confidence, %s),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s AND entity_id = %s AND status = 'active'
                            """,
                            (attribute.confidence, changed_id, int(entity_id)),
                        )
                        evidence_added = self._insert_attribute_evidence(cursor, changed_id, attribute)
                        after_snapshot = attribute_snapshot(changed_id, attribute.attr_type, decision.target_value_text, attribute.confidence)
                        outcome.evidence_count += evidence_added
                    elif decision.action == "enrich" and decision.target_attribute_id:
                        changed_id = int(decision.target_attribute_id)
                        event_attribute_id = changed_id
                        new_value_text = decision.new_value_text or attribute.value_text
                        before_snapshot = attribute_snapshot(changed_id, attribute.attr_type, decision.target_value_text, attribute.confidence)
                        cursor.execute(
                            """
                            UPDATE evorag_entity_attributes
                            SET value_text = %s,
                                value_fingerprint = %s,
                                confidence = GREATEST(confidence, %s),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s AND entity_id = %s AND status = 'active'
                            """,
                            (new_value_text, text_fingerprint(new_value_text), attribute.confidence, changed_id, int(entity_id)),
                        )
                        outcome.attribute_count += 1
                        evidence_added = self._insert_attribute_evidence(cursor, changed_id, attribute)
                        after_snapshot = attribute_snapshot(changed_id, attribute.attr_type, new_value_text, attribute.confidence)
                        outcome.evidence_count += evidence_added
                    elif decision.action == "update" and decision.target_attribute_id:
                        changed_id = int(decision.target_attribute_id)
                        event_attribute_id = changed_id
                        new_value_text = decision.new_value_text or attribute.value_text
                        before_snapshot = attribute_snapshot(changed_id, attribute.attr_type, decision.target_value_text, attribute.confidence)
                        cursor.execute(
                            """
                            UPDATE evorag_entity_attributes
                            SET value_text = %s,
                                value_fingerprint = %s,
                                confidence = GREATEST(confidence, %s),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s AND entity_id = %s AND status = 'active'
                            """,
                            (new_value_text, text_fingerprint(new_value_text), attribute.confidence, changed_id, int(entity_id)),
                        )
                        outcome.attribute_count += 1
                        evidence_added = self._insert_attribute_evidence(cursor, changed_id, attribute)
                        after_snapshot = attribute_snapshot(changed_id, attribute.attr_type, new_value_text, attribute.confidence)
                        outcome.evidence_count += evidence_added
                    elif decision.action == "conflict" and decision.target_attribute_id:
                        event_attribute_id = int(decision.target_attribute_id)
                        before_snapshot = attribute_snapshot(event_attribute_id, attribute.attr_type, decision.target_value_text, attribute.confidence)
                        after_snapshot = before_snapshot
                        self._insert_attribute_conflict(cursor, entity_id, decision)
                        outcome.conflict_count += 1
                    self._insert_attribute_decision_audit(cursor, entity_id, decision, changed_id)
                    if event_attribute_id:
                        self._insert_attribute_event(
                            cursor,
                            entity_id=entity_id,
                            attribute_id=event_attribute_id,
                            decision=decision,
                            before_snapshot=before_snapshot,
                            after_snapshot=after_snapshot,
                            evidence_added=evidence_added,
                        )
                    if changed_id is not None and changed_id not in outcome.changed_attribute_ids:
                        outcome.changed_attribute_ids.append(changed_id)
            connection.commit()
        return outcome

    def attribute_event_summaries(self, attribute_ids: list[int], *, recent_limit: int = 3) -> dict[int, dict[str, Any]]:
        unique_ids = sorted({int(attribute_id) for attribute_id in attribute_ids if int(attribute_id) > 0})
        if not unique_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(unique_ids))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT attribute_id, action, review_status, decision_reason,
                           confidence, source, corrected_by_event_id, created_at
                    FROM evorag_entity_attribute_events
                    WHERE attribute_id IN ({placeholders})
                    ORDER BY attribute_id, id DESC
                    """,
                    tuple(unique_ids),
                )
                rows = cursor.fetchall()

        grouped: dict[int, list[dict[str, Any]]] = {attribute_id: [] for attribute_id in unique_ids}
        for row in rows:
            attribute_id = int(row["attribute_id"])
            grouped.setdefault(attribute_id, []).append(row)

        summaries: dict[int, dict[str, Any]] = {}
        for attribute_id, events in grouped.items():
            recent = events[: max(1, int(recent_limit))]
            corrected = [event for event in events if event.get("corrected_by_event_id")]
            summaries[attribute_id] = {
                "event_count": len(events),
                "last_actions": [str(event.get("action") or "") for event in recent],
                "review_stats": review_stats(events),
                "recent_corrections": [
                    {
                        "original_action": str(event.get("action") or ""),
                        "review_status": str(event.get("review_status") or ""),
                        "reason": str(event.get("decision_reason") or ""),
                    }
                    for event in corrected[: max(1, int(recent_limit))]
                ],
                "last_event": event_summary(recent[0]) if recent else None,
            }
        return summaries

    def _insert_attribute_from_input(self, cursor: Any, entity_id: int, attribute: EntityAttributeInput) -> tuple[int, int]:
        value_fingerprint = text_fingerprint(attribute.value_text)
        cursor.execute(
            """
            INSERT INTO evorag_entity_attributes (
                entity_id, attr_type, value_text, value_fingerprint, confidence
            )
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                confidence = GREATEST(confidence, VALUES(confidence)),
                updated_at = CURRENT_TIMESTAMP
            """,
            (entity_id, attribute.attr_type, attribute.value_text, value_fingerprint, attribute.confidence),
        )
        cursor.execute(
            """
            SELECT id FROM evorag_entity_attributes
            WHERE entity_id = %s AND attr_type = %s AND value_fingerprint = %s
            LIMIT 1
            """,
            (entity_id, attribute.attr_type, value_fingerprint),
        )
        row = cursor.fetchone()
        if not row:
            return 0, 0
        attribute_id = int(row["id"])
        return attribute_id, self._insert_attribute_evidence(cursor, attribute_id, attribute)

    def _insert_attribute_evidence(self, cursor: Any, attribute_id: int, attribute: EntityAttributeInput) -> int:
        evidence_text = attribute.evidence or attribute.value_text
        if not evidence_text:
            return 0
        cursor.execute(
            """
            INSERT INTO evorag_entity_attribute_evidence (
                attribute_id, note_id, block_id, block_index, evidence_text, evidence_fingerprint
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                evidence_text = VALUES(evidence_text)
            """,
            (
                int(attribute_id),
                attribute.note_id,
                attribute.block_id,
                attribute.block_index,
                evidence_text,
                text_fingerprint(evidence_text),
            ),
        )
        return 1

    def _insert_attribute_conflict(self, cursor: Any, entity_id: int, decision: AttributeDecision) -> None:
        attribute = decision.incoming_attribute
        cursor.execute(
            """
            INSERT INTO evorag_entity_attribute_conflicts (
                entity_id, attribute_id, attr_type, incoming_value_text,
                incoming_evidence, reason, confidence, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
            """,
            (
                int(entity_id),
                int(decision.target_attribute_id or 0),
                attribute.attr_type,
                attribute.value_text,
                attribute.evidence,
                decision.reason,
                decision.confidence,
            ),
        )

    def _insert_attribute_decision_audit(
        self,
        cursor: Any,
        entity_id: int,
        decision: AttributeDecision,
        changed_attribute_id: int | None,
    ) -> None:
        attribute = decision.incoming_attribute
        cursor.execute(
            """
            INSERT INTO evorag_entity_attribute_decision_audit (
                entity_id, input_index, action, attr_type, incoming_value_text,
                target_attribute_id, changed_attribute_id, new_value_text,
                confidence, reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                int(entity_id),
                int(decision.input_index),
                decision.action,
                attribute.attr_type,
                attribute.value_text,
                decision.target_attribute_id,
                changed_attribute_id,
                decision.new_value_text,
                decision.confidence,
                decision.reason,
            ),
        )

    def _insert_attribute_event(
        self,
        cursor: Any,
        *,
        entity_id: int,
        attribute_id: int,
        decision: AttributeDecision,
        before_snapshot: dict[str, Any] | None,
        after_snapshot: dict[str, Any] | None,
        evidence_added: int,
    ) -> None:
        attribute = decision.incoming_attribute
        incoming_snapshot = {
            "attr_type": attribute.attr_type,
            "value_text": attribute.value_text,
            "evidence": attribute.evidence,
            "confidence": attribute.confidence,
            "note_id": attribute.note_id,
            "block_id": attribute.block_id,
            "block_index": attribute.block_index,
        }
        candidate_snapshot = None
        if decision.target_attribute_id:
            candidate_snapshot = {
                "attribute_id": int(decision.target_attribute_id),
                "attr_type": attribute.attr_type,
                "value_text": decision.target_value_text,
            }
        diff = attribute_event_diff(
            action=decision.action,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            evidence_added=evidence_added,
            target_attribute_id=decision.target_attribute_id,
        )
        cursor.execute(
            """
            INSERT INTO evorag_entity_attribute_events (
                entity_id, attribute_id, attr_type, action, review_status,
                before_json, after_json, incoming_json, candidate_json, diff_json,
                decision_reason, confidence, source,
                note_id, block_id, block_index
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'auto', %s, %s, %s)
            """,
            (
                int(entity_id),
                int(attribute_id),
                attribute.attr_type,
                decision.action,
                "pending",
                json.dumps(before_snapshot, ensure_ascii=False),
                json.dumps(after_snapshot, ensure_ascii=False),
                json.dumps(incoming_snapshot, ensure_ascii=False),
                json.dumps(candidate_snapshot, ensure_ascii=False),
                json.dumps(diff, ensure_ascii=False),
                decision.reason,
                decision.confidence,
                attribute.note_id,
                attribute.block_id,
                attribute.block_index,
            ),
        )

    def _merge_attributes(self, cursor: Any, entity_id: int, attributes: list[EntityAttributeInput]) -> tuple[int, int]:
        attribute_count = 0
        evidence_count = 0
        for attribute in attributes:
            value_fingerprint = text_fingerprint(attribute.value_text)
            cursor.execute(
                """
                INSERT INTO evorag_entity_attributes (
                    entity_id, attr_type, value_text, value_fingerprint, confidence
                )
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    confidence = GREATEST(confidence, VALUES(confidence)),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (entity_id, attribute.attr_type, attribute.value_text, value_fingerprint, attribute.confidence),
            )
            attribute_count += 1
            cursor.execute(
                """
                SELECT id FROM evorag_entity_attributes
                WHERE entity_id = %s AND attr_type = %s AND value_fingerprint = %s
                LIMIT 1
                """,
                (entity_id, attribute.attr_type, value_fingerprint),
            )
            row = cursor.fetchone()
            if not row:
                continue
            attribute_id = int(row["id"])
            if attribute.evidence:
                cursor.execute(
                    """
                    INSERT INTO evorag_entity_attribute_evidence (
                        attribute_id, note_id, block_id, block_index, evidence_text, evidence_fingerprint
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        evidence_text = VALUES(evidence_text)
                    """,
                    (
                        attribute_id,
                        attribute.note_id,
                        attribute.block_id,
                        attribute.block_index,
                        attribute.evidence,
                        text_fingerprint(attribute.evidence),
                    ),
                )
                evidence_count += 1
            self._insert_attribute_event(
                cursor,
                entity_id=entity_id,
                attribute_id=attribute_id,
                decision=AttributeDecision(
                    input_index=-1,
                    action="add",
                    incoming_attribute=attribute,
                    target_attribute_id=None,
                    confidence=attribute.confidence,
                    reason="initial entity attribute add",
                ),
                before_snapshot=None,
                after_snapshot=attribute_snapshot(attribute_id, attribute.attr_type, attribute.value_text, attribute.confidence),
                evidence_added=1 if attribute.evidence else 0,
            )
        return attribute_count, evidence_count
