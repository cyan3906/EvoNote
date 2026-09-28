import asyncio
from types import SimpleNamespace

from app.EvoRAG.entity_store.models import EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.resolver import EntityResolver


def run(coro):
    return asyncio.run(coro)


def config() -> SimpleNamespace:
    return SimpleNamespace(
        entity_resolution_top_k=3,
        entity_resolution_auto_match_threshold=0.9,
        entity_resolution_llm_threshold=0.7,
        entity_resolution_manual_threshold=0.6,
    )


def stored_entity() -> StoredEntity:
    return StoredEntity(
        id=12,
        canonical_name="HTTP/1.1",
        normalized_name="http 1 1",
        entity_type="protocol",
        scope=EntityScope(),
        aliases=["HTTP1"],
        identity_description="HTTP protocol version.",
    )


class FailingHybridIndex:
    def search(self, incoming: IncomingEntity, *, top_k: int):
        raise AssertionError("hybrid search should not run when alias map matches")


class FakeLLM:
    pass


def test_resolver_uses_alias_lookup_before_hybrid_search() -> None:
    entity = stored_entity()
    incoming = IncomingEntity(
        name="http1",
        normalized_name="http1",
        entity_type="protocol",
        scope=EntityScope(),
    )

    resolver = EntityResolver(
        existing_entities=[],
        config=config(),
        llm_client=FakeLLM(),
        hybrid_index=FailingHybridIndex(),
        alias_lookup=lambda item: entity if item.normalized_name == "http1" else None,
    )

    decision = run(resolver.resolve_one(incoming))

    assert decision.decision == "matched"
    assert decision.matched_entity == entity
    assert decision.score == 1.0
    assert decision.reason == "alias map matched"
