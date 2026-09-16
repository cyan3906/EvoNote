from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402


DEFAULT_LEVELS = (4, 8, 16, 32, 64)


@dataclass
class ProbeResult:
    name: str
    level: int
    total: int
    succeeded: int
    failed: int
    rate_limited: int
    server_errors: int
    timeout_errors: int
    other_errors: int
    elapsed_ms: float
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    dimensions: int
    sample_error: str = ""


def build_client() -> OpenAI:
    api_key = settings.embedding_api_key.strip() or settings.agent_api_key.strip()
    base_url = settings.embedding_base_url.strip() or settings.agent_api_base_url.strip()

    if not api_key:
        raise RuntimeError("No embedding_api_key or agent_api_key configured.")

    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=settings.agent_timeout_seconds,
    )


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()

    if "429" in text or "rate" in text:
        return "rate_limited"
    if any(code in text for code in ("500", "502", "503", "504")):
        return "server_errors"
    if "timeout" in text or "timed out" in text:
        return "timeout_errors"
    return "other_errors"


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = int((len(ordered) - 1) * 0.95)
    return ordered[index]


def summarize(
    *,
    name: str,
    level: int,
    total: int,
    durations_ms: list[float],
    dimensions: list[int],
    errors: list[tuple[str, str]],
    elapsed_ms: float,
) -> ProbeResult:
    counts = {
        "rate_limited": 0,
        "server_errors": 0,
        "timeout_errors": 0,
        "other_errors": 0,
    }

    for category, _message in errors:
        counts[category] += 1

    return ProbeResult(
        name=name,
        level=level,
        total=total,
        succeeded=len(durations_ms),
        failed=len(errors),
        rate_limited=counts["rate_limited"],
        server_errors=counts["server_errors"],
        timeout_errors=counts["timeout_errors"],
        other_errors=counts["other_errors"],
        elapsed_ms=round(elapsed_ms, 2),
        mean_ms=round(statistics.fmean(durations_ms), 2) if durations_ms else 0.0,
        p95_ms=round(percentile_95(durations_ms), 2),
        min_ms=round(min(durations_ms), 2) if durations_ms else 0.0,
        max_ms=round(max(durations_ms), 2) if durations_ms else 0.0,
        dimensions=dimensions[0] if dimensions else 0,
        sample_error=errors[0][1][:300] if errors else "",
    )


def embedding_request(client: OpenAI, text: str) -> tuple[float, int]:
    started = time.perf_counter()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, len(response.data[0].embedding)


def batch_embedding_request(client: OpenAI, texts: list[str]) -> tuple[float, int, int]:
    started = time.perf_counter()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    dimension = len(response.data[0].embedding) if response.data else 0
    return elapsed_ms, len(response.data), dimension


def probe_batch_support(client: OpenAI, levels: tuple[int, ...]) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    for level in levels:
        texts = [f"evonote batch probe text {index}" for index in range(level)]
        started = time.perf_counter()

        try:
            elapsed_ms, returned_count, dimension = batch_embedding_request(client, texts)
            errors: list[tuple[str, str]] = []
            durations = [elapsed_ms]
            dimensions = [dimension] if returned_count == level else []
            if returned_count != level:
                errors.append(("other_errors", f"Expected {level} embeddings, got {returned_count}."))
                durations = []
        except Exception as exc:  # noqa: BLE001
            errors = [(classify_error(exc), f"{type(exc).__name__}: {exc}")]
            durations = []
            dimensions = []

        total_elapsed_ms = (time.perf_counter() - started) * 1000
        results.append(
            summarize(
                name="batch_size",
                level=level,
                total=1,
                durations_ms=durations,
                dimensions=dimensions,
                errors=errors,
                elapsed_ms=total_elapsed_ms,
            )
        )

    return results


def probe_single_request_concurrency(client: OpenAI, levels: tuple[int, ...]) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    for level in levels:
        durations: list[float] = []
        dimensions: list[int] = []
        errors: list[tuple[str, str]] = []
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=level, thread_name_prefix="embedding-probe") as executor:
            futures = [
                executor.submit(
                    embedding_request,
                    client,
                    f"evonote concurrency probe level {level}, request {index}",
                )
                for index in range(level)
            ]

            for future in as_completed(futures):
                try:
                    elapsed_ms, dimension = future.result()
                    durations.append(elapsed_ms)
                    dimensions.append(dimension)
                except Exception as exc:  # noqa: BLE001
                    errors.append((classify_error(exc), f"{type(exc).__name__}: {exc}"))

        elapsed_ms = (time.perf_counter() - started) * 1000
        results.append(
            summarize(
                name="single_request_concurrency",
                level=level,
                total=level,
                durations_ms=durations,
                dimensions=dimensions,
                errors=errors,
                elapsed_ms=elapsed_ms,
            )
        )

    return results


def print_csv_table(title: str, results: list[ProbeResult]) -> None:
    print(f"\n{title}")
    print("level,total,ok,fail,429,5xx,timeout,other,elapsed_ms,mean_ms,p95_ms,min_ms,max_ms,dim")

    for result in results:
        print(
            ",".join(
                str(value)
                for value in (
                    result.level,
                    result.total,
                    result.succeeded,
                    result.failed,
                    result.rate_limited,
                    result.server_errors,
                    result.timeout_errors,
                    result.other_errors,
                    result.elapsed_ms,
                    result.mean_ms,
                    result.p95_ms,
                    result.min_ms,
                    result.max_ms,
                    result.dimensions,
                )
            )
        )

    failed = [result for result in results if result.sample_error]
    if failed:
        print("\nSample errors:")
        for result in failed:
            print(f"- {result.name} {result.level}: {result.sample_error}")


def print_markdown_table(title: str, results: list[ProbeResult], *, level_label: str) -> None:
    print(f"\n## {title}")
    print("")
    print(
        "| "
        + " | ".join(
            [
                level_label,
                "请求数",
                "成功",
                "失败",
                "429",
                "5xx",
                "超时",
                "其他错误",
                "总耗时",
                "平均耗时",
                "P95",
                "维度",
                "结论",
            ]
        )
        + " |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for result in results:
        status = "通过" if result.failed == 0 else "失败"
        print(
            "| "
            + " | ".join(
                [
                    str(result.level),
                    str(result.total),
                    str(result.succeeded),
                    str(result.failed),
                    str(result.rate_limited),
                    str(result.server_errors),
                    str(result.timeout_errors),
                    str(result.other_errors),
                    format_ms(result.elapsed_ms),
                    format_ms(result.mean_ms),
                    format_ms(result.p95_ms),
                    str(result.dimensions),
                    status,
                ]
            )
            + " |"
        )

    failed = [result for result in results if result.sample_error]
    if failed:
        print("\n### 错误样例")
        for result in failed:
            print(f"- {level_label} {result.level}: {result.sample_error}")


def print_summary(batch_results: list[ProbeResult], concurrency_results: list[ProbeResult]) -> None:
    successful_batches = [result for result in batch_results if result.failed == 0]
    successful_concurrency = [result for result in concurrency_results if result.failed == 0]
    best_batch = min(successful_batches, key=lambda item: item.elapsed_ms, default=None)
    largest_batch = max(successful_batches, key=lambda item: item.level, default=None)
    largest_concurrency = max(successful_concurrency, key=lambda item: item.level, default=None)
    fastest_concurrency = min(successful_concurrency, key=lambda item: item.elapsed_ms, default=None)

    print("\n## 结论")
    print("")

    if largest_batch:
        print(f"- Batch embedding：至少支持到 batch_size={largest_batch.level}，本轮没有失败。")
    else:
        print("- Batch embedding：本轮没有成功档位。")

    if largest_concurrency:
        print(f"- 单条请求并发：至少支持到 concurrency={largest_concurrency.level}，本轮没有失败。")
    else:
        print("- 单条请求并发：本轮没有成功档位。")

    if best_batch:
        print(f"- 本轮最快 batch 档位：batch_size={best_batch.level}，耗时 {format_ms(best_batch.elapsed_ms)}。")

    if fastest_concurrency:
        print(
            f"- 本轮最快单条并发档位：concurrency={fastest_concurrency.level}，"
            f"总耗时 {format_ms(fastest_concurrency.elapsed_ms)}。"
        )

    print("- 生产建议：优先使用 batch embedding；当前代码默认 batch_size=32、max_concurrency=4。")


def format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels:
        raise argparse.ArgumentTypeError("At least one level is required.")
    return levels


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe embedding batch support and concurrency limits.")
    parser.add_argument("--levels", type=parse_levels, default=DEFAULT_LEVELS)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--csv", action="store_true", help="Print compact CSV tables.")
    args = parser.parse_args()

    client = build_client()
    batch_results = probe_batch_support(client, args.levels)
    concurrency_results = probe_single_request_concurrency(client, args.levels)

    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "levels": args.levels,
        "batch_results": [asdict(result) for result in batch_results],
        "concurrency_results": [asdict(result) for result in concurrency_results],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.csv:
        print(f"Embedding model: {settings.embedding_model}")
        print_csv_table("Batch embedding support", batch_results)
        print_csv_table("Single-request concurrency", concurrency_results)
    else:
        effective_base_url = settings.embedding_base_url.strip() or settings.agent_api_base_url.strip() or "<default-openai>"
        print("# Embedding 并发探测报告")
        print("")
        print(f"- 模型：`{settings.embedding_model}`")
        print(f"- Base URL：`{effective_base_url}`")
        print(f"- 测试档位：`{', '.join(str(level) for level in args.levels)}`")
        print_markdown_table("Batch Embedding 支持", batch_results, level_label="batch_size")
        print_markdown_table("单条请求并发", concurrency_results, level_label="concurrency")
        print_summary(batch_results, concurrency_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
