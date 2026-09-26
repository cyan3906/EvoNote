import argparse
import asyncio
import json
from collections import defaultdict
from typing import Any

from app.EvoRAG.entity_store.models import EntityAttributeInput, EntityScope, IncomingEntity
from app.EvoRAG.entity_store.normalizer import dedupe_extracted_entities, normalize_name
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.services import EvoRAGProcessor


DEMO_TEXT = """
在 EvoRAG 里，Block 是从用户输入文本切出来的语义片段，后续系统会从每个 Block 抽取实体、属性和 evidence。
这里的 Block 不是编辑器页面里的内容块，也不是区块链里的区块；它更像知识抽取前的最小处理单元。
""".strip()


EVORAG_SCOPE = EntityScope(
    workspace_id="local",
    project_id="evorag",
    collection_id="default",
    domain="general",
)

MAIN_PROJECT_SCOPE = EntityScope(
    workspace_id="local",
    project_id="main_project",
    collection_id="editor_notes",
    domain="product",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show scoped ES/Milvus retrieval for one ambiguous text.")
    parser.add_argument("--text", default=DEMO_TEXT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-seed", action="store_true", help="Do not seed the ambiguous Block fixtures.")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    repository = MySQLEntityRepository()
    repository.init_schema()

    hybrid_index = EntityHybridIndex()
    hybrid_index.ensure_indexes()
    embedding_client = EvoRAGEmbeddingClient()

    if not args.skip_seed:
        await seed_ambiguous_block_entities(repository, hybrid_index, embedding_client)

    print("=== input text ===")
    print(args.text)
    print()

    print_scope_entities("=== current entity library: evorag scope ===", repository, EVORAG_SCOPE)
    print_scope_entities("=== current entity library: main_project scope ===", repository, MAIN_PROJECT_SCOPE)

    result = await EvoRAGProcessor().preprocess(args.text)
    incoming_entities = dedupe_extracted_entities(result.blocks, scope=EVORAG_SCOPE)
    vectors = await embedding_client.embed_texts(
        [entity.identity_description or entity.description_for_match or entity.name for entity in incoming_entities]
    )
    for entity, vector in zip(incoming_entities, vectors, strict=False):
        entity.embedding = vector

    print("=== extracted entities from text ===")
    for entity in incoming_entities:
        print(json.dumps(incoming_entity_payload(entity), ensure_ascii=False, indent=2))
    print()

    print("=== filtered retrieval: evorag scope ===")
    print_candidates(hybrid_index, incoming_entities, EVORAG_SCOPE, args.top_k)

    print("=== filtered retrieval: main_project scope using same extracted entities ===")
    comparison_entities = [copy_with_scope(entity, MAIN_PROJECT_SCOPE) for entity in incoming_entities]
    print_candidates(hybrid_index, comparison_entities, MAIN_PROJECT_SCOPE, args.top_k)
    return 0


async def seed_ambiguous_block_entities(
    repository: MySQLEntityRepository,
    hybrid_index: EntityHybridIndex,
    embedding_client: EvoRAGEmbeddingClient,
) -> None:
    fixtures = [
        IncomingEntity(
            name="Block",
            normalized_name=normalize_name("Block"),
            entity_type="概念",
            scope=EVORAG_SCOPE,
            aliases=["语义块", "文本块"],
            identity_description="EvoRAG 中从用户输入文本切分出来的语义片段，用于后续实体、属性和证据抽取。",
            attributes=[
                EntityAttributeInput(
                    attr_type="definition",
                    value_text="从输入文本切分出的语义处理单元。",
                    evidence="Block 是从用户输入文本切出来的语义片段。",
                    confidence=0.9,
                ),
                EntityAttributeInput(
                    attr_type="purpose",
                    value_text="承载实体抽取、属性抽取和 evidence 对齐的上下文。",
                    evidence="系统会从每个 Block 抽取实体、属性和 evidence。",
                    confidence=0.9,
                ),
            ],
        ),
        IncomingEntity(
            name="Block",
            normalized_name=normalize_name("Block"),
            entity_type="概念",
            scope=MAIN_PROJECT_SCOPE,
            aliases=["编辑器块", "内容块"],
            identity_description="主项目编辑器中的页面内容块，表示用户在笔记页面里编辑和拖拽的结构化 UI 单元。",
            attributes=[
                EntityAttributeInput(
                    attr_type="definition",
                    value_text="笔记编辑器里的结构化内容单元。",
                    evidence="编辑器页面里的内容块。",
                    confidence=0.9,
                ),
                EntityAttributeInput(
                    attr_type="components",
                    value_text="通常包含文本、样式、排序位置和父子层级等 UI 状态。",
                    evidence="页面内容块表示可编辑和拖拽的结构化 UI 单元。",
                    confidence=0.8,
                ),
            ],
        ),
    ]
    vectors = await embedding_client.embed_texts([fixture.identity_description for fixture in fixtures])
    for fixture, vector in zip(fixtures, vectors, strict=False):
        fixture.embedding = vector
        fixture.description_for_match = fixture.identity_description
        existing = repository.find_by_normalized_name(fixture.normalized_name, fixture.entity_type, fixture.scope)
        result = repository.upsert_entity(fixture, matched_entity_id=existing.id if existing else None)
        stored = repository.get_entity(result.entity_id)
        if stored:
            hybrid_index.upsert_entity(stored)


def print_scope_entities(title: str, repository: MySQLEntityRepository, scope: EntityScope) -> None:
    entities = repository.list_entities(scope)
    attributes_by_entity = load_attributes(repository, [entity.id for entity in entities])
    print(title)
    print(f"scope={scope.as_dict()} count={len(entities)}")
    for entity in entities:
        print(f"- id={entity.id} name={entity.canonical_name} type={entity.entity_type}")
        print(f"  identity={entity.identity_description}")
        for attribute in attributes_by_entity.get(entity.id, [])[:8]:
            print(f"  {attribute['attr_type']}: {attribute['value_text']}")
            if attribute.get("evidence_text"):
                print(f"    evidence: {attribute['evidence_text']}")
    print()


def load_attributes(repository: MySQLEntityRepository, entity_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not entity_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(entity_ids))
    sql = f"""
        SELECT a.entity_id, a.attr_type, a.value_text, a.confidence, e.evidence_text
        FROM evorag_entity_attributes a
        LEFT JOIN evorag_entity_attribute_evidence e ON e.attribute_id = a.id
        WHERE a.entity_id IN ({placeholders}) AND a.status = 'active'
        ORDER BY a.entity_id, a.attr_type, a.id, e.id
    """
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(entity_ids))
            rows = cursor.fetchall()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["entity_id"])].append(row)
    return grouped


def print_candidates(hybrid_index: EntityHybridIndex, entities: list[IncomingEntity], scope: EntityScope, top_k: int) -> None:
    for entity in entities:
        entity.scope = scope
        candidates = hybrid_index.search(entity, top_k=top_k)
        print(f"entity={entity.name} scope={scope.project_id}/{scope.collection_id}/{scope.domain}")
        if not candidates:
            print("  <none>")
        for candidate in candidates:
            print(
                "  "
                f"id={candidate.entity.id} name={candidate.entity.canonical_name} "
                f"scope={candidate.entity.scope.project_id}/{candidate.entity.scope.collection_id}/{candidate.entity.scope.domain} "
                f"score={candidate.score:.4f} source={candidate.source} "
                f"vector={candidate.vector_score:.4f} es={candidate.es_score:.4f}"
            )
    print()


def incoming_entity_payload(entity: IncomingEntity) -> dict[str, Any]:
    attributes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attribute in entity.attributes:
        attributes[attribute.attr_type].append(
            {
                "value": attribute.value_text,
                "evidence": attribute.evidence,
                "confidence": attribute.confidence,
            }
        )
    return {
        "name": entity.name,
        "type": entity.entity_type,
        "identity_description": entity.identity_description,
        "aliases": entity.aliases,
        "attributes": attributes,
    }


def copy_with_scope(entity: IncomingEntity, scope: EntityScope) -> IncomingEntity:
    return IncomingEntity(
        name=entity.name,
        normalized_name=entity.normalized_name,
        entity_type=entity.entity_type,
        scope=scope,
        aliases=list(entity.aliases),
        identity_description=entity.identity_description,
        attributes=list(entity.attributes),
        description_for_match=entity.description_for_match,
        source_count=entity.source_count,
        embedding=list(entity.embedding),
    )


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
