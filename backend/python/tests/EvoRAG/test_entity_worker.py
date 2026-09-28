from app.EvoRAG.entity_store.worker import _handle_message


class FakeQueue:
    def __init__(self) -> None:
        self.acked: list[str] = []
        self.enqueued: list[tuple[int, int]] = []
        self.dead_letters: list[tuple[str, dict[str, str], str]] = []

    def enqueue(self, incoming_entity_id: int, *, job_id: int = 0) -> str:
        self.enqueued.append((incoming_entity_id, job_id))
        return "requeued-1"

    def ack(self, message_id: str) -> None:
        self.acked.append(message_id)

    def dead_letter(self, message_id: str, payload: dict[str, str], *, error: str) -> None:
        self.dead_letters.append((message_id, payload, error))


class FakeIngestor:
    def __init__(self, status: str) -> None:
        self.status = status
        self.calls: list[tuple[int, str]] = []

    async def process_incoming_entity_id(self, incoming_entity_id: int, *, worker_id: str) -> str:
        self.calls.append((incoming_entity_id, worker_id))
        return self.status


def test_worker_requeues_retryable_pending_status() -> None:
    queue = FakeQueue()
    ingestor = FakeIngestor("pending")

    _handle_message(queue, ingestor, "1-0", {"incoming_entity_id": "7", "job_id": "3"}, "worker-a")

    assert ingestor.calls == [(7, "worker-a")]
    assert queue.enqueued == [(7, 3)]
    assert queue.acked == ["1-0"]
    assert queue.dead_letters == []


def test_worker_sends_dead_letter_status_to_dead_letter_stream() -> None:
    queue = FakeQueue()
    ingestor = FakeIngestor("dead_letter")

    _handle_message(queue, ingestor, "1-1", {"incoming_entity_id": "8", "job_id": "4"}, "worker-a")

    assert ingestor.calls == [(8, "worker-a")]
    assert queue.enqueued == []
    assert queue.acked == ["1-1"]
    assert queue.dead_letters == [("1-1", {"incoming_entity_id": "8", "job_id": "4"}, "max attempts exceeded")]


def test_worker_acks_invalid_message_after_dead_letter() -> None:
    queue = FakeQueue()
    ingestor = FakeIngestor("auto_merged")

    _handle_message(queue, ingestor, "1-2", {"incoming_entity_id": "bad"}, "worker-a")

    assert ingestor.calls == []
    assert queue.acked == ["1-2"]
    assert queue.dead_letters == [("1-2", {"incoming_entity_id": "bad"}, "invalid incoming_entity_id")]
