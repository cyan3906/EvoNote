import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.services import EvoRAGProcessor, EvoRAGQueryService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoRAG preprocessing, ingestion, or entity-memory query.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to preprocess. Add --ingest to persist extracted entities.")
    source.add_argument("--file", help="UTF-8 text file to preprocess. Add --ingest to persist extracted entities.")
    source.add_argument("--query", help="Entity name to retrieve, graph, and generate.")
    parser.add_argument("--ingest", action="store_true", help="Persist --text/--file extraction into MySQL, Elasticsearch, and Milvus.")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum entities to retrieve for --query.")
    parser.add_argument("--out", help="Optional JSON output file.")
    parser.add_argument("--workspace-id", default=None, help="Scope workspace id for --ingest and --query.")
    parser.add_argument("--project-id", default=None, help="Scope project id for --ingest and --query.")
    parser.add_argument("--collection-id", default=None, help="Scope collection id for --ingest and --query.")
    parser.add_argument("--domain", default=None, help="Scope domain for --ingest and --query.")
    args = parser.parse_args()
    if args.query is not None and args.ingest:
        parser.error("--ingest can only be used with --text or --file")
    return args


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.file).read_text(encoding="utf-8")


def build_scope(args: argparse.Namespace) -> EntityScope | None:
    values = [args.workspace_id, args.project_id, args.collection_id, args.domain]
    if not any(value is not None for value in values):
        return None
    default = EntityScope()
    return EntityScope(
        workspace_id=args.workspace_id or default.workspace_id,
        project_id=args.project_id or default.project_id,
        collection_id=args.collection_id or default.collection_id,
        domain=args.domain or default.domain,
    )


async def main_async() -> int:
    args = parse_args()
    scope = build_scope(args)
    if args.query is not None:
        result = await EvoRAGQueryService(scope=scope).query(args.query, top_k=args.top_k)
        output_payload = result.model_dump()
    else:
        text = load_text(args)
        processor = EvoRAGProcessor()
        preprocess = await processor.preprocess(text)
        if args.ingest:
            ingest_results = await EntityIngestor(scope=scope).ingest(preprocess)
            output_payload = {
                "preprocess": preprocess.model_dump(),
                "ingest_results": [asdict(result) for result in ingest_results],
                "warnings": [],
            }
        else:
            output_payload = preprocess.model_dump()

    output = json.dumps(output_payload, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
