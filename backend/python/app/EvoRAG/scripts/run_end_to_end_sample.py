import argparse
import asyncio
import json
from pathlib import Path

from app.EvoRAG.config import PROJECT_DATA_DIR
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.normalizer import dedupe_extracted_entities
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.entity_store.resolver import EntityResolver
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.services import EvoRAGProcessor


DEFAULT_DATASET = PROJECT_DATA_DIR / "evorag" / "computer_interview_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one EvoRAG dataset item end to end.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset JSON path.")
    parser.add_argument("--index", type=int, default=0, help="Zero-based item index to run.")
    parser.add_argument("--no-ingest", action="store_true", help="Only preprocess and retrieve candidates; skip persistence.")
    parser.add_argument("--workspace-id", default="local")
    parser.add_argument("--project-id", default="evorag")
    parser.add_argument("--collection-id", default="default")
    parser.add_argument("--domain", default="general")
    return parser.parse_args()


def load_item(dataset_path: Path, index: int) -> dict[str, str]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise RuntimeError(f"dataset has no items: {dataset_path}")
    if index < 0 or index >= len(items):
        raise RuntimeError(f"index {index} out of range, dataset size={len(items)}")
    return items[index]


async def main_async() -> int:
    args = parse_args()
    scope = EntityScope(
        workspace_id=args.workspace_id,
        project_id=args.project_id,
        collection_id=args.collection_id,
        domain=args.domain,
    )
    item = load_item(Path(args.dataset), args.index)
    repository = MySQLEntityRepository()
    repository.init_schema()
    hybrid_index = EntityHybridIndex()
    hybrid_index.ensure_indexes()

    print(f"sample: {item['id']} - {item['title']}")
    result = await EvoRAGProcessor().preprocess(item["text"])
    print(f"blocks: {len(result.blocks)}")
    for block_result in result.blocks:
        block = block_result.block
        print(f"  block[{block.block_index}] heading={block.heading!r} chars={len(block.l1_text)} entities={len(block_result.entities)}")
        for entity in block_result.entities:
            attr_count = sum(len(values) for _, values in entity.attributes)
            print(f"    entity: {entity.name} type={entity.entity_type} attrs={attr_count}")
            print(f"      identity: {entity.identity_description[:160]}")

    incoming_entities = dedupe_extracted_entities(result.blocks, scope=scope)
    embedding_client = EvoRAGEmbeddingClient()
    vectors = await embedding_client.embed_texts(
        [entity.identity_description or entity.description_for_match or entity.name for entity in incoming_entities]
    )
    for entity, vector in zip(incoming_entities, vectors, strict=False):
        entity.embedding = vector

    existing = await asyncio.to_thread(repository.list_entities, scope)
    resolver = EntityResolver(existing, hybrid_index=hybrid_index)
    print(f"incoming entities after local dedupe: {len(incoming_entities)}")

    for entity in incoming_entities:
        candidates = hybrid_index.search(entity)
        print(f"  candidates for {entity.name}:")
        if not candidates:
            print("    <none>")
        for candidate in candidates:
            print(
                "    "
                f"id={candidate.entity.id} name={candidate.entity.canonical_name} "
                f"score={candidate.score:.4f} source={candidate.source} "
                f"vector={candidate.vector_score:.4f} es={candidate.es_score:.4f}"
            )
        decision = await resolver.resolve_one(entity)
        matched_name = decision.matched_entity.canonical_name if decision.matched_entity else ""
        print(
            f"    decision={decision.decision} score={decision.score:.4f} "
            f"matched={matched_name or '<none>'} reason={decision.reason}"
        )

    if args.no_ingest:
        print("ingest skipped")
        return 0

    ingest_result = await EntityIngestor(repository=repository, scope=scope).ingest(result)
    print("ingest results:")
    for output in ingest_result:
        print(
            f"  entity_id={output.entity_id} name={output.canonical_name} "
            f"created={output.created} attributes={output.attribute_count} evidence={output.evidence_count}"
        )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
