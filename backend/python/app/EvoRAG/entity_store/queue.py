from typing import Any

from redis.exceptions import ResponseError

from app.EvoRAG.config import EvoRAGSettings, settings
from app.core.config import settings as core_settings


class EntityMergeQueue:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config
        self._client = None

    def ensure_group(self) -> None:
        try:
            self.client().xgroup_create(
                name=self.config.entity_worker_stream,
                groupname=self.config.entity_worker_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, incoming_entity_id: int, *, job_id: int = 0) -> str:
        self.ensure_group()
        message_id = self.client().xadd(
            self.config.entity_worker_stream,
            {
                "incoming_entity_id": str(int(incoming_entity_id)),
                "job_id": str(int(job_id)),
            },
        )
        return decode_value(message_id)

    def enqueue_many(self, incoming_entity_ids: list[int], *, job_id: int = 0) -> list[str]:
        return [self.enqueue(incoming_id, job_id=job_id) for incoming_id in incoming_entity_ids]

    def stats(self) -> dict[str, Any]:
        self.ensure_group()
        client = self.client()
        stream_length = int(client.xlen(self.config.entity_worker_stream) or 0)
        pending_count = 0
        consumers: list[dict[str, Any]] = []
        try:
            pending = client.xpending(self.config.entity_worker_stream, self.config.entity_worker_group)
            if isinstance(pending, dict):
                pending_count = int(pending.get("pending") or 0)
            elif isinstance(pending, (list, tuple)) and pending:
                pending_count = int(pending[0] or 0)
        except Exception:
            pending_count = 0

        try:
            for consumer in client.xinfo_consumers(self.config.entity_worker_stream, self.config.entity_worker_group):
                consumers.append(decode_info_dict(consumer))
        except Exception:
            consumers = []

        return {
            "stream": self.config.entity_worker_stream,
            "group": self.config.entity_worker_group,
            "stream_length": stream_length,
            "pending_count": pending_count,
            "consumer_count": len(consumers),
            "consumers": consumers,
        }

    def read(self, *, consumer_name: str, count: int, block_ms: int) -> list[tuple[str, dict[str, str]]]:
        self.ensure_group()
        response = self.client().xreadgroup(
            groupname=self.config.entity_worker_group,
            consumername=consumer_name,
            streams={self.config.entity_worker_stream: ">"},
            count=max(1, int(count)),
            block=max(1, int(block_ms)),
        )
        messages: list[tuple[str, dict[str, str]]] = []
        for _, stream_messages in response:
            for message_id, payload in stream_messages:
                messages.append((decode_value(message_id), decode_payload(payload)))
        return messages

    def ack(self, message_id: str) -> None:
        self.client().xack(self.config.entity_worker_stream, self.config.entity_worker_group, message_id)

    def dead_letter(self, message_id: str, payload: dict[str, Any], *, error: str) -> None:
        self.client().xadd(
            self.config.entity_worker_dead_letter_stream,
            {
                "source_message_id": message_id,
                "incoming_entity_id": str(payload.get("incoming_entity_id") or ""),
                "job_id": str(payload.get("job_id") or ""),
                "error": error,
            },
        )

    def client(self):
        if self._client is not None:
            return self._client

        from redis import Redis

        self._client = Redis.from_url(core_settings.redis_url, password=core_settings.redis_password, decode_responses=False)
        return self._client


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_payload(payload: dict[Any, Any]) -> dict[str, str]:
    return {decode_value(key): decode_value(value) for key, value in payload.items()}


def decode_info_dict(payload: dict[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        decoded_key = decode_value(key)
        if isinstance(value, bytes):
            result[decoded_key] = decode_value(value)
        else:
            result[decoded_key] = value
    return result
