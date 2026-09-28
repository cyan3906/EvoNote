import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.queue import EntityMergeQueue
from app.EvoRAG.entity_store.repository import MySQLEntityRepository


_LOCK = Lock()
_STOP_EVENT = Event()
_EXECUTOR: ThreadPoolExecutor | None = None
_STARTED = False


def start_entity_merge_workers(config: EvoRAGSettings = settings) -> None:
    global _EXECUTOR, _STARTED
    if not config.entity_worker_enabled:
        return

    with _LOCK:
        if _STARTED:
            return
        _STOP_EVENT.clear()
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=max(1, config.entity_worker_count),
            thread_name_prefix="evorag-entity-worker",
        )
        recover_pending_entity_tasks(config)
        for index in range(max(1, config.entity_worker_count)):
            _EXECUTOR.submit(_worker_loop, f"{socket.gethostname()}-{index}", config)
        _STARTED = True


def shutdown_entity_merge_workers() -> None:
    global _EXECUTOR, _STARTED
    with _LOCK:
        _STOP_EVENT.set()
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=False, cancel_futures=False)
        _EXECUTOR = None
        _STARTED = False


def entity_worker_runtime_status(config: EvoRAGSettings = settings) -> dict[str, object]:
    with _LOCK:
        started = _STARTED
        executor_alive = _EXECUTOR is not None
    return {
        "enabled": config.entity_worker_enabled,
        "started": started,
        "executor_alive": executor_alive,
        "worker_count": max(1, config.entity_worker_count),
        "batch_size": config.entity_worker_batch_size,
        "block_ms": config.entity_worker_block_ms,
        "lock_seconds": config.entity_worker_lock_seconds,
        "max_attempts": config.entity_worker_max_attempts,
    }


def recover_pending_entity_tasks(config: EvoRAGSettings = settings) -> int:
    repository = MySQLEntityRepository(config)
    queue = EntityMergeQueue(config)
    incoming_ids = repository.list_resumable_incoming_entity_ids(limit=config.entity_worker_recover_limit)
    for incoming_id in incoming_ids:
        try:
            queue.enqueue(incoming_id)
        except Exception:
            break
    return len(incoming_ids)


def _worker_loop(worker_id: str, config: EvoRAGSettings) -> None:
    queue = EntityMergeQueue(config)
    ingestor = EntityIngestor(config=config)
    while not _STOP_EVENT.is_set():
        try:
            messages = queue.read(
                consumer_name=worker_id,
                count=config.entity_worker_batch_size,
                block_ms=config.entity_worker_block_ms,
            )
        except Exception:
            _STOP_EVENT.wait(2)
            continue

        for message_id, payload in messages:
            if _STOP_EVENT.is_set():
                break
            _handle_message(queue, ingestor, message_id, payload, worker_id)


def _handle_message(
    queue: EntityMergeQueue,
    ingestor: EntityIngestor,
    message_id: str,
    payload: dict[str, str],
    worker_id: str,
) -> None:
    try:
        incoming_entity_id = int(payload.get("incoming_entity_id") or 0)
    except ValueError:
        queue.dead_letter(message_id, payload, error="invalid incoming_entity_id")
        queue.ack(message_id)
        return

    if incoming_entity_id <= 0:
        queue.dead_letter(message_id, payload, error="missing incoming_entity_id")
        queue.ack(message_id)
        return

    try:
        status = asyncio.run(ingestor.process_incoming_entity_id(incoming_entity_id, worker_id=worker_id))
    except Exception as exc:
        queue.dead_letter(message_id, payload, error=str(exc))
        queue.ack(message_id)
        return

    if status == "pending":
        try:
            queue.enqueue(incoming_entity_id, job_id=int(payload.get("job_id") or 0))
        except Exception:
            pass
    elif status == "dead_letter":
        queue.dead_letter(message_id, payload, error="max attempts exceeded")
    queue.ack(message_id)
