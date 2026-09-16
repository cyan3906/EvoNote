from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.embeddings import embed_text, embed_texts  # noqa: E402


@dataclass
class BenchmarkMetrics:
    name: str
    total_texts: int
    api_requests: int
    succeeded: int
    failed: int
    elapsed_ms: float
    mean_item_ms: float
    p95_request_ms: float
    dimensions: int
    batch_size: int = 1
    max_concurrency: int = 1
    fallback: int = 0
    failed_batches: int = 0
    circuit_open: bool = False


def load_fixture_claim_texts(limit: int | None = None) -> list[str]:
    fixture_path = REPO_ROOT / "backend" / "testdata" / "408_notes_claims_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    texts: list[str] = []

    for note in data.get("notes", []):
        expected = note.get("expected", {})
        for claim in expected.get("claims", []):
            claim_text = str(claim.get("claim", "")).strip()
            if claim_text:
                texts.append(claim_text)

    return texts[:limit] if limit else texts


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * 0.95)
    return ordered[index]


def run_serial(texts: list[str]) -> BenchmarkMetrics:
    request_durations: list[float] = []
    dimensions: list[int] = []
    failed = 0
    started = time.perf_counter()

    for text in texts:
        request_started = time.perf_counter()
        try:
            vector = embed_text(text)
            if not vector:
                failed += 1
                continue
            dimensions.append(len(vector))
        except Exception:
            failed += 1
            continue
        finally:
            request_durations.append((time.perf_counter() - request_started) * 1000)

    elapsed_ms = (time.perf_counter() - started) * 1000
    succeeded = len(texts) - failed

    return BenchmarkMetrics(
        name="serial_single_request",
        total_texts=len(texts),
        api_requests=len(texts),
        succeeded=succeeded,
        failed=failed,
        elapsed_ms=round(elapsed_ms, 2),
        mean_item_ms=round(elapsed_ms / len(texts), 2) if texts else 0.0,
        p95_request_ms=round(percentile_95(request_durations), 2),
        dimensions=dimensions[0] if dimensions else 0,
    )


def run_parallel(texts: list[str], *, batch_size: int, max_concurrency: int) -> BenchmarkMetrics:
    started = time.perf_counter()
    vectors, stats = embed_texts(
        texts,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        fallback_vector=None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    dimensions = [len(vector) for vector in vectors if vector]
    failed = sum(1 for vector in vectors if not vector)

    return BenchmarkMetrics(
        name="parallel_batch_request",
        total_texts=len(texts),
        api_requests=stats.batches,
        succeeded=len(texts) - failed,
        failed=failed,
        elapsed_ms=round(elapsed_ms, 2),
        mean_item_ms=round(elapsed_ms / len(texts), 2) if texts else 0.0,
        p95_request_ms=0.0,
        dimensions=dimensions[0] if dimensions else 0,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        fallback=stats.fallback,
        failed_batches=stats.failed_batches,
        circuit_open=stats.circuit_open,
    )


def print_metrics(serial: BenchmarkMetrics, parallel: BenchmarkMetrics) -> None:
    saved_ms = serial.elapsed_ms - parallel.elapsed_ms
    saved_percent = (saved_ms / serial.elapsed_ms * 100) if serial.elapsed_ms else 0.0
    speedup = (serial.elapsed_ms / parallel.elapsed_ms) if parallel.elapsed_ms else 0.0

    print(f"Embedding model: {settings.embedding_model}")
    print(f"Texts: {serial.total_texts}")
    print("")
    print("name,total_texts,api_requests,succeeded,failed,elapsed_ms,mean_item_ms,p95_request_ms,batch_size,max_concurrency,dim,fallback,failed_batches,circuit_open")

    for metric in (serial, parallel):
        print(
            ",".join(
                str(value)
                for value in (
                    metric.name,
                    metric.total_texts,
                    metric.api_requests,
                    metric.succeeded,
                    metric.failed,
                    metric.elapsed_ms,
                    metric.mean_item_ms,
                    metric.p95_request_ms,
                    metric.batch_size,
                    metric.max_concurrency,
                    metric.dimensions,
                    metric.fallback,
                    metric.failed_batches,
                    metric.circuit_open,
                )
            )
        )

    print("")
    print("comparison,saved_ms,saved_percent,speedup")
    print(f"parallel_vs_serial,{round(saved_ms, 2)},{round(saved_percent, 2)},{round(speedup, 2)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AB benchmark serial embedding vs batched concurrent embedding.")
    parser.add_argument("--limit", type=int, default=0, help="Limit fixture claim texts. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    texts = load_fixture_claim_texts(limit=args.limit or None)
    if not texts:
        raise RuntimeError("No claim texts found in fixture.")

    serial = run_serial(texts)
    parallel = run_parallel(texts, batch_size=args.batch_size, max_concurrency=args.max_concurrency)

    saved_ms = serial.elapsed_ms - parallel.elapsed_ms
    payload = {
        "serial": asdict(serial),
        "parallel": asdict(parallel),
        "comparison": {
            "saved_ms": round(saved_ms, 2),
            "saved_percent": round((saved_ms / serial.elapsed_ms * 100) if serial.elapsed_ms else 0.0, 2),
            "speedup": round((serial.elapsed_ms / parallel.elapsed_ms) if parallel.elapsed_ms else 0.0, 2),
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_metrics(serial, parallel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
