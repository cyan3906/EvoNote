from __future__ import annotations

import json
from typing import Any

from app.EvoRAG.entity_store.alias_cache import scope_hash
from app.EvoRAG.entity_store.models import EntityRelationMemoryRecord, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.core.config import settings as core_settings


ALLOW_DECISIONS = {"allow", "white", "whitelist", "same", "same_entity", "matched", "equal"}
REJECT_DECISIONS = {"reject", "black", "blacklist", "different", "not_same", "not_equal", "new"}


class EntityRelationMemory:
    def __init__(
        self,
        repository: MySQLEntityRepository,
        *,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        password: str | None = None,
    ) -> None:
        self.repository = repository
        self.redis_url = redis_url if redis_url is not None else core_settings.redis_url
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else core_settings.redis_cache_ttl_seconds
        self.password = password if password is not None else core_settings.redis_password
        self._client = None
        self._available = True

    def init_schema(self) -> None:
        self.repository.init_schema()

    def get_direct_relation(self, incoming: IncomingEntity) -> EntityRelationMemoryRecord | None:
        client = self.client()
        if client is None:
            return None
        key = relation_direct_cache_key(incoming.scope, incoming.normalized_name or normalize_name(incoming.name), incoming.entity_type or "")
        try:
            raw = client.get(key)
        except Exception:
            self._available = False
            return None
        return self._record_from_cache_payload(raw, incoming)

    def list_top_relations(self, incoming: IncomingEntity, *, limit: int = 30) -> list[EntityRelationMemoryRecord]:
        return self.repository.list_entity_resolution_relation_memory(incoming, limit=limit)

    def record_relation(
        self,
        *,
        incoming: IncomingEntity,
        candidate: StoredEntity,
        decision: str,
        relation_type: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> None:
        self.repository.upsert_entity_resolution_relation_memory(
            incoming=incoming,
            candidate=candidate,
            decision=decision,
            relation_type=relation_type,
            confidence=confidence,
            source=source,
            reason=reason,
        )

    def _record_from_cache_payload(self, raw: Any, incoming: IncomingEntity) -> EntityRelationMemoryRecord | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return None

        candidate_payload = payload.get("candidate") or {}
        candidate_id = int(payload.get("candidate_entity_id") or candidate_payload.get("id") or 0)
        if candidate_id <= 0:
            return None
        entity = self.repository.get_entity(candidate_id)
        if entity is None:
            entity = StoredEntity(
                id=candidate_id,
                canonical_name=str(candidate_payload.get("canonical_name") or payload.get("candidate_name") or ""),
                normalized_name=str(candidate_payload.get("normalized_name") or normalize_name(str(candidate_payload.get("canonical_name") or payload.get("candidate_name") or ""))),
                entity_type=str(candidate_payload.get("entity_type") or payload.get("candidate_type") or incoming.entity_type or ""),
                scope=incoming.scope,
                aliases=[str(alias) for alias in candidate_payload.get("aliases") or []],
                identity_description=str(candidate_payload.get("identity_description") or ""),
                summary=str(candidate_payload.get("summary") or ""),
                description_for_match=str(candidate_payload.get("description_for_match") or ""),
            )
        return EntityRelationMemoryRecord(
            id=int(payload.get("id") or 0),
            decision=str(payload.get("decision") or ""),
            relation_type=str(payload.get("relation_type") or ""),
            candidate=entity,
            confidence=float(payload.get("confidence") or 0.0),
            hit_count=int(payload.get("hit_count") or 0),
            source=str(payload.get("source") or "redis"),
            reason=str(payload.get("reason") or ""),
        )

    def client(self):
        if not self._available:
            return None
        if self._client is not None:
            return self._client
        try:
            from redis import Redis

            self._client = Redis.from_url(self.redis_url, password=self.password, decode_responses=False)
            return self._client
        except Exception:
            self._available = False
            return None


def relation_direct_cache_key(scope: EntityScope, normalized_name: str, entity_type: str) -> str:
    return f"evorag:entity_resolution:direct:{scope_hash(scope)}:{normalized_name}:{entity_type}"


def relation_pair_cache_key(scope: EntityScope, left_normalized_name: str, left_type: str, right_entity_id: int) -> str:
    return f"evorag:entity_resolution:pair:{scope_hash(scope)}:{left_normalized_name}:{left_type}:{int(right_entity_id)}"


def is_allow_relation(decision: str) -> bool:
    return str(decision or "").strip().lower() in ALLOW_DECISIONS


def is_reject_relation(decision: str) -> bool:
    return str(decision or "").strip().lower() in REJECT_DECISIONS
