import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.EvoRAG.services import EvoRAGProcessor, EvoRAGQueryService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoRAG preprocessing or entity-memory query.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to preprocess.")
    source.add_argument("--file", help="UTF-8 text file to preprocess.")
    source.add_argument("--query", help="Entity name to retrieve, graph, and generate.")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum entities to retrieve for --query.")
    parser.add_argument("--out", help="Optional JSON output file.")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.file).read_text(encoding="utf-8")


async def main_async() -> int:
    args = parse_args()
    if args.query is not None:
        result = await EvoRAGQueryService().query(args.query, top_k=args.top_k)
    else:
        text = load_text(args)
        processor = EvoRAGProcessor()
        result = await processor.preprocess(text)
    output = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
