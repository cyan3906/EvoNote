import asyncio

from app.EvoRAG.entity_store.models import EntityUpsertResult
from app.api.routes import evorag


def run(coro):
    return asyncio.run(coro)


class FakeRepository:
    def __init__(self) -> None:
        self.list_calls: list[tuple[str, int]] = []

    def list_review_tasks(self, *, status: str = "pending", limit: int = 50) -> list[dict[str, object]]:
        self.list_calls.append((status, limit))
        return [{"id": 1, "status": status}]


class FakeIngestor:
    repository = FakeRepository()
    merge_calls: list[tuple[int, int, str, str]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def manual_merge_review_task(
        self,
        review_task_id: int,
        *,
        entity_id: int,
        reason: str = "",
        decided_by: str = "manual",
    ) -> EntityUpsertResult:
        self.merge_calls.append((review_task_id, entity_id, reason, decided_by))
        return EntityUpsertResult(
            entity_id=entity_id,
            canonical_name="HTTP/1.1",
            created=False,
            attribute_count=2,
            evidence_count=2,
        )


def test_review_task_list_route_reads_pending_tasks(monkeypatch) -> None:
    FakeIngestor.repository = FakeRepository()
    monkeypatch.setattr(evorag, "EntityIngestor", FakeIngestor)

    result = run(evorag.list_review_tasks(status_filter="pending", limit=7))

    assert result.tasks == [{"id": 1, "status": "pending"}]
    assert FakeIngestor.repository.list_calls == [("pending", 7)]


def test_review_merge_route_delegates_manual_decision(monkeypatch) -> None:
    FakeIngestor.merge_calls = []
    monkeypatch.setattr(evorag, "EntityIngestor", FakeIngestor)

    result = run(
        evorag.merge_review_task(
            5,
            evorag.EvoRAGReviewMergeRequest(entity_id=12, reason="same entity", decided_by="tester"),
        )
    )

    assert result["status"] == "merged"
    assert result["result"]["entity_id"] == 12
    assert FakeIngestor.merge_calls == [(5, 12, "same entity", "tester")]


class FakeJobRepository:
    def list_ingest_jobs(self, *, limit: int = 20) -> list[dict[str, object]]:
        return [{"id": 3, "status": "processing", "limit": limit}]

    def ingest_observability_summary(self) -> dict[str, object]:
        return {"retry_count": 2, "failed_count": 1, "average_completed_ms": 123.0}


class FakeQueue:
    def stats(self) -> dict[str, object]:
        return {"stream_length": 4, "pending_count": 1, "consumer_count": 2}


def test_ingest_job_list_route_reads_recent_jobs(monkeypatch) -> None:
    monkeypatch.setattr(evorag, "MySQLEntityRepository", lambda: FakeJobRepository())

    result = run(evorag.list_ingest_jobs(limit=6))

    assert result.jobs == [{"id": 3, "status": "processing", "limit": 6}]


def test_worker_status_route_combines_runtime_queue_and_mysql(monkeypatch) -> None:
    monkeypatch.setattr(evorag, "MySQLEntityRepository", lambda: FakeJobRepository())
    monkeypatch.setattr(evorag, "EntityMergeQueue", lambda: FakeQueue())
    monkeypatch.setattr(evorag, "entity_worker_runtime_status", lambda: {"started": True, "worker_count": 2})

    result = run(evorag.get_worker_status())

    assert result["worker"] == {"started": True, "worker_count": 2}
    assert result["queue"]["available"] is True
    assert result["queue"]["stream_length"] == 4
    assert result["mysql"]["retry_count"] == 2
    assert result["warnings"] == []
