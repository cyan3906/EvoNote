import hashlib

from app.EvoRAG.entity_store.models import EntityScope
from app.core.config import settings as core_settings


class EntityAliasCache:
    def __init__(self, *, redis_url: str | None = None, ttl_seconds: int | None = None, password: str | None = None) -> None:
        self.redis_url = redis_url if redis_url is not None else core_settings.redis_url
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else core_settings.redis_cache_ttl_seconds
        self.password = password if password is not None else core_settings.redis_password
        self._client = None
        self._available = True

    def get_entity_id(self, scope: EntityScope, normalized_alias: str) -> int | None:
        client = self.client()
        if client is None:
            return None
        try:
            value = client.get(alias_cache_key(scope, normalized_alias))
        except Exception:
            self._available = False
            return None
        if value is None:
            return None
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return int(value)
        except (TypeError, ValueError):
            return None

    def set_entity_id(self, scope: EntityScope, normalized_alias: str, entity_id: int) -> None:
        client = self.client()
        if client is None or not normalized_alias or not entity_id:
            return
        try:
            client.set(alias_cache_key(scope, normalized_alias), str(int(entity_id)), ex=max(1, int(self.ttl_seconds)))
        except Exception:
            self._available = False

    def client(self):
        if not self._available:
            return None
        if self._client is not None:
            return self._client
        try:
            from redis import Redis

            self._client = Redis.from_url(self.redis_url, password=self.password, decode_responses=False)
            return self._client
        except Exception:
            self._available = False
            return None


def alias_cache_key(scope: EntityScope, normalized_alias: str) -> str:
    return f"evorag:alias:{scope_hash(scope)}:{normalized_alias}"


def scope_hash(scope: EntityScope) -> str:
    raw = "|".join(
        [
            scope.workspace_id,
            scope.project_id,
            scope.collection_id,
            scope.domain,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
