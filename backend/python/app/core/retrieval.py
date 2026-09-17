from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.embeddings import embed_text, embed_texts
from app.core import metrics
from app.core.config import settings
from app.core.retry import retry_call


class RetrievalBackendError(RuntimeError):
    pass


_client_lock = Lock()
_milvus_client: Any | None = None
_elasticsearch_client: Any | None = None
_backends_initialized = False


@dataclass(frozen=True)
class RetrievalHit:
    claim_id: str
    note_id: str
    score: float
    rank: int
    source: str
    title: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NoteRetrievalHit:
    note_id: str
    score: float
    rank: int
    source: str
    title: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def safe_embed_text(text: str) -> list[float]:
    try:
        return embed_text(text)
    except Exception:
        return []


def sync_note_indexes(
    note: dict[str, str],
    representation: dict[str, object],
    claims: list[dict[str, object]],
) -> dict[str, object]:
    documents = build_claim_documents(note, representation, claims)
    result: dict[str, object] = {
        "milvus_claims": 0,
        "es_claims": 0,
        "errors": [],
    }

    try:
        result["es_claims"] = sync_elasticsearch_claims(note["id"], documents)
    except Exception as exc:
        result["errors"].append(f"Elasticsearch indexing skipped: {exc}")

    try:
        result["milvus_claims"] = sync_milvus_claims(note["id"], documents)
    except Exception as exc:
        result["errors"].append(f"Milvus indexing skipped: {exc}")

    return result


def build_claim_documents(
    note: dict[str, str],
    representation: dict[str, object],
    claims: list[dict[str, object]],
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    note_keywords = normalize_keywords(representation.get("keywords", []))

    for claim in claims:
        claim_keywords = normalize_keywords(claim.get("keywords", []))
        keywords = unique_values([*note_keywords, *claim_keywords])
        l3_text = str(representation.get("l3_text", ""))
        l2_summary = str(representation.get("l2_summary", ""))
        claim_text = str(claim.get("claim_text", ""))
        vector = normalize_vector(claim.get("vector", []))

        if len(vector) != settings.embedding_dimensions:
            vector = safe_embed_text(" ".join([*claim_keywords, claim_text]))

        documents.append(
            {
                "id": str(claim["id"]),
                "claim_id": str(claim["id"]),
                "note_id": str(claim["note_id"]),
                "block_id": str(claim.get("block_id", "")),
                "title": note.get("title", ""),
                "tags": note.get("tags", ""),
                "l2_summary": l2_summary,
                "l3_text": l3_text,
                "claim_text": claim_text,
                "subject": str(claim.get("subject", "")),
                "predicate": str(claim.get("predicate", "")),
                "object_text": str(claim.get("object_text", "")),
                "source_text": str(claim.get("source_text", "")),
                "keywords": keywords,
                "keywords_text": " ".join(keywords),
                "vector": vector,
            }
        )

    return documents


def delete_note_indexes(note_id: str) -> dict[str, object]:
    result: dict[str, object] = {
        "milvus_deleted": False,
        "es_deleted": False,
        "errors": [],
    }

    try:
        delete_elasticsearch_claims(note_id)
        result["es_deleted"] = True
    except Exception as exc:
        result["errors"].append(f"Elasticsearch cleanup skipped: {exc}")

    try:
        delete_milvus_claims(note_id)
        result["milvus_deleted"] = True
    except Exception as exc:
        result["errors"].append(f"Milvus cleanup skipped: {exc}")

    return result


def sync_milvus_claims(note_id: str, documents: list[dict[str, object]]) -> int:
    vector_documents = [
        document
        for document in documents
        if len(normalize_vector(document.get("vector", []))) == settings.embedding_dimensions
    ]

    with metrics.api_call(operation_name="milvus_index_sync", provider="milvus") as call:
        delete_milvus_claims(note_id)

        if not vector_documents:
            metrics.set_call_response_size(call, 0)
            return 0

        client = milvus_client()
        retry_call(
            lambda: client.insert(
                collection_name=settings.milvus_collection,
                data=[milvus_document(document) for document in vector_documents],
            ),
            operation_name="Milvus insert claims",
        )
        retry_call(
            lambda: client.flush(collection_name=settings.milvus_collection),
            operation_name="Milvus flush claims",
        )
        metrics.set_call_response_size(call, len(vector_documents))

    return len(vector_documents)


def delete_milvus_claims(note_id: str) -> None:
    ensure_retrieval_backends_initialized()
    client = milvus_client()
    retry_call(
        lambda: client.delete(
            collection_name=settings.milvus_collection,
            filter=f'note_id == "{escape_milvus_value(note_id)}"',
        ),
        operation_name="Milvus delete claims",
    )
    retry_call(
        lambda: client.flush(collection_name=settings.milvus_collection),
        operation_name="Milvus flush deleted claims",
    )


def query_milvus_by_vector(
    query_vector: list[float],
    *,
    top_k: int | None = None,
    exclude_note_id: str | None = None,
) -> list[RetrievalHit]:
    if len(query_vector) != settings.embedding_dimensions:
        return []

    ensure_retrieval_backends_initialized()
    client = milvus_client()
    filter_expr = ""

    if exclude_note_id:
        filter_expr = f'note_id != "{escape_milvus_value(exclude_note_id)}"'

    search_kwargs = {
        "collection_name": settings.milvus_collection,
        "data": [query_vector],
        "limit": top_k or settings.retrieval_top_k,
        "output_fields": ["claim_id", "note_id", "title", "claim_text"],
    }

    if filter_expr:
        search_kwargs["filter"] = filter_expr

    results = retry_call(
        lambda: client.search(**search_kwargs),
        operation_name="Milvus vector search",
    )
    hits: list[RetrievalHit] = []

    for rank, hit in enumerate(results[0] if results else [], start=1):
        entity = hit.get("entity", {})
        hits.append(
            RetrievalHit(
                claim_id=str(entity.get("claim_id") or hit.get("id")),
                note_id=str(entity.get("note_id", "")),
                score=float(hit.get("distance", 0.0)),
                rank=rank,
                source="milvus",
                title=str(entity.get("title", "")),
                reason=str(entity.get("claim_text", "")),
            )
        )

    return hits


def sync_elasticsearch_claims(note_id: str, documents: list[dict[str, object]]) -> int:
    with metrics.api_call(operation_name="elasticsearch_index_sync", provider="elasticsearch") as call:
        client = delete_elasticsearch_claims(note_id)

        for document in documents:
            retry_call(
                lambda document=document: client.index(
                    index=settings.es_index,
                    id=str(document["claim_id"]),
                    document=elasticsearch_document(document),
                    refresh=False,
                ),
                operation_name="Elasticsearch index claim",
            )

        if documents:
            retry_call(
                lambda: client.indices.refresh(index=settings.es_index),
                operation_name="Elasticsearch refresh index",
            )

        metrics.set_call_response_size(call, len(documents))

    return len(documents)


def delete_elasticsearch_claims(note_id: str):
    ensure_retrieval_backends_initialized()
    client = elasticsearch_client()
    retry_call(
        lambda: client.delete_by_query(
            index=settings.es_index,
            query={"term": {"note_id": note_id}},
            ignore_unavailable=True,
            conflicts="proceed",
            refresh=True,
        ),
        operation_name="Elasticsearch delete old claims",
    )
    return client


def query_elasticsearch_by_keywords(
    query_text: str,
    keywords: list[str],
    *,
    top_k: int | None = None,
    exclude_note_id: str | None = None,
) -> list[RetrievalHit]:
    ensure_retrieval_backends_initialized()
    client = elasticsearch_client()
    search_text = " ".join(unique_values([query_text, *keywords])).strip()

    if not search_text:
        return []

    must_not = []

    if exclude_note_id:
        must_not.append({"term": {"note_id": exclude_note_id}})

    response = retry_call(
        lambda: client.search(
            index=settings.es_index,
            size=top_k or settings.retrieval_top_k,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": search_text,
                                "fields": [
                                    "claim_text^4",
                                    "keywords_text^3",
                                    "l3_text^2",
                                    "l2_summary",
                                    "title",
                                    "tags",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "must_not": must_not,
                }
            },
        ),
        operation_name="Elasticsearch keyword search",
    )
    hits: list[RetrievalHit] = []

    for rank, item in enumerate(response.get("hits", {}).get("hits", []), start=1):
        source = item.get("_source", {})
        hits.append(
            RetrievalHit(
                claim_id=str(source.get("claim_id") or item.get("_id")),
                note_id=str(source.get("note_id", "")),
                score=float(item.get("_score", 0.0) or 0.0),
                rank=rank,
                source="elasticsearch",
                title=str(source.get("title", "")),
                reason=str(source.get("claim_text", "")),
            )
        )

    return hits


def query_milvus_notes_by_l2_vector(
    query_vector: list[float],
    *,
    top_k: int | None = None,
    exclude_note_id: str | None = None,
) -> list[NoteRetrievalHit]:
    with metrics.api_call(operation_name="milvus_note_search", provider="milvus") as call:
        claim_hits = query_milvus_by_vector(
            query_vector,
            top_k=max((top_k or settings.retrieval_top_k) * 10, top_k or settings.retrieval_top_k),
            exclude_note_id=exclude_note_id,
        )
        metrics.set_call_response_size(call, len(claim_hits))

    hits: list[NoteRetrievalHit] = []
    seen: set[str] = set()

    for claim_hit in claim_hits:
        if not claim_hit.note_id or claim_hit.note_id in seen:
            continue

        seen.add(claim_hit.note_id)
        hits.append(
            NoteRetrievalHit(
                note_id=claim_hit.note_id,
                score=claim_hit.score,
                rank=len(hits) + 1,
                source="milvus",
                title=claim_hit.title,
                reason=claim_hit.reason,
            )
        )

        if len(hits) >= (top_k or settings.retrieval_top_k):
            break

    return hits


def query_elasticsearch_notes_by_l3(
    query_text: str,
    keywords: list[str],
    *,
    top_k: int | None = None,
    exclude_note_id: str | None = None,
) -> list[NoteRetrievalHit]:
    ensure_retrieval_backends_initialized()
    client = elasticsearch_client()
    search_text = " ".join(unique_values([query_text, *keywords])).strip()

    if not search_text:
        return []

    must_not = []

    if exclude_note_id:
        must_not.append({"term": {"note_id": exclude_note_id}})

    with metrics.api_call(
        operation_name="elasticsearch_note_search",
        provider="elasticsearch",
        request_size_chars=len(search_text),
    ) as call:
        response = retry_call(
            lambda: client.search(
                index=settings.es_index,
                size=top_k or settings.retrieval_top_k,
                collapse={"field": "note_id"},
                query={
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": search_text,
                                    "fields": [
                                        "l3_text^4",
                                        "keywords_text^2",
                                        "l2_summary",
                                        "title",
                                        "tags",
                                    ],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "must_not": must_not,
                    }
                },
            ),
            operation_name="Elasticsearch note keyword search",
        )
        metrics.set_call_response_size(call, response)

    hits: list[NoteRetrievalHit] = []

    for rank, item in enumerate(response.get("hits", {}).get("hits", []), start=1):
        source = item.get("_source", {})
        note_id = str(source.get("note_id", ""))

        if not note_id:
            continue

        hits.append(
            NoteRetrievalHit(
                note_id=note_id,
                score=float(item.get("_score", 0.0) or 0.0),
                rank=rank,
                source="elasticsearch",
                title=str(source.get("title", "")),
                reason=str(source.get("l3_text", "")),
            )
        )

    return hits


def hybrid_retrieve_notes(
    representation: dict[str, object],
    *,
    exclude_note_id: str | None = None,
    top_k: int | None = None,
) -> list[NoteRetrievalHit]:
    vector_hits: list[NoteRetrievalHit] = []
    keyword_hits: list[NoteRetrievalHit] = []
    query_vector = normalize_vector(representation.get("vector", []))

    try:
        vector_hits = query_milvus_notes_by_l2_vector(
            query_vector,
            top_k=top_k,
            exclude_note_id=exclude_note_id,
        )
    except Exception:
        vector_hits = []

    try:
        keyword_hits = query_elasticsearch_notes_by_l3(
            str(representation.get("l3_text", "")),
            normalize_keywords(representation.get("keywords", [])),
            top_k=top_k,
            exclude_note_id=exclude_note_id,
        )
    except Exception:
        keyword_hits = []

    return reciprocal_rank_fusion_notes(
        [vector_hits, keyword_hits],
        rrf_k=settings.rrf_k,
        top_k=top_k or settings.retrieval_top_k,
    )


def hybrid_retrieve_claims(
    representation: dict[str, object],
    *,
    exclude_note_id: str | None = None,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    vector_hits: list[RetrievalHit] = []
    keyword_hits: list[RetrievalHit] = []
    query_vector = normalize_vector(representation.get("vector", []))

    try:
        vector_hits = query_milvus_by_vector(
            query_vector,
            top_k=top_k,
            exclude_note_id=exclude_note_id,
        )
    except Exception:
        vector_hits = []

    try:
        keyword_hits = query_elasticsearch_by_keywords(
            str(representation.get("l3_text", "")),
            normalize_keywords(representation.get("keywords", [])),
            top_k=top_k,
            exclude_note_id=exclude_note_id,
        )
    except Exception:
        keyword_hits = []

    return reciprocal_rank_fusion(
        [vector_hits, keyword_hits],
        rrf_k=settings.rrf_k,
        top_k=top_k or settings.retrieval_top_k,
    )


def reciprocal_rank_fusion_notes(
    rankings: list[list[NoteRetrievalHit]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[NoteRetrievalHit]:
    scores: dict[str, float] = {}
    best_hit: dict[str, NoteRetrievalHit] = {}
    sources: dict[str, list[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            if not hit.note_id:
                continue

            scores[hit.note_id] = scores.get(hit.note_id, 0.0) + (1.0 / (rrf_k + rank))
            sources.setdefault(hit.note_id, []).append(hit.source)

            if hit.note_id not in best_hit or hit.score > best_hit[hit.note_id].score:
                best_hit[hit.note_id] = hit

    fused: list[NoteRetrievalHit] = []

    for rank, note_id in enumerate(
        sorted(scores, key=lambda item: scores[item], reverse=True)[:top_k],
        start=1,
    ):
        hit = best_hit[note_id]
        fused.append(
            NoteRetrievalHit(
                note_id=note_id,
                score=round(scores[note_id], 6),
                rank=rank,
                source="+".join(unique_values(sources.get(note_id, []))),
                title=hit.title,
                reason=hit.reason,
            )
        )

    return fused


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalHit]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_hit: dict[str, RetrievalHit] = {}
    sources: dict[str, list[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            if not hit.claim_id:
                continue

            scores[hit.claim_id] = scores.get(hit.claim_id, 0.0) + (1.0 / (rrf_k + rank))
            sources.setdefault(hit.claim_id, []).append(hit.source)

            if hit.claim_id not in best_hit or hit.score > best_hit[hit.claim_id].score:
                best_hit[hit.claim_id] = hit

    fused: list[RetrievalHit] = []

    for rank, claim_id in enumerate(
        sorted(scores, key=lambda item: scores[item], reverse=True)[:top_k],
        start=1,
    ):
        hit = best_hit[claim_id]
        fused.append(
            RetrievalHit(
                claim_id=claim_id,
                note_id=hit.note_id,
                score=round(scores[claim_id], 6),
                rank=rank,
                source="+".join(unique_values(sources.get(claim_id, []))),
                title=hit.title,
                reason=hit.reason,
            )
        )

    return fused


def rebuild_hybrid_indexes() -> dict[str, object]:
    from app.core import database

    indexed_notes = 0
    indexed_claims = 0
    errors: list[str] = []

    for note in database.list_notes():
        representation = database.get_note_representation(note["id"])
        claims = database.list_claims(note_id=note["id"])

        if not representation or not claims:
            continue

        result = sync_note_indexes(note, representation, claims)
        indexed_notes += 1
        indexed_claims += int(result.get("es_claims", 0) or result.get("milvus_claims", 0) or 0)
        errors.extend(str(error) for error in result.get("errors", []))

    return {
        "indexed_notes": indexed_notes,
        "indexed_claims": indexed_claims,
        "errors": errors,
    }


def initialize_retrieval_backends() -> None:
    global _milvus_client, _elasticsearch_client, _backends_initialized

    with _client_lock:
        if _backends_initialized:
            return

        _milvus_client = _create_milvus_client()
        _elasticsearch_client = _create_elasticsearch_client()
        ensure_milvus_collection(_milvus_client)
        ensure_elasticsearch_index(_elasticsearch_client)
        _backends_initialized = True


def ensure_retrieval_backends_initialized() -> None:
    if not _backends_initialized:
        initialize_retrieval_backends()


def close_retrieval_backends() -> None:
    global _milvus_client, _elasticsearch_client, _backends_initialized

    with _client_lock:
        for client in (_milvus_client, _elasticsearch_client):
            close = getattr(client, "close", None)

            if callable(close):
                close()

        _milvus_client = None
        _elasticsearch_client = None
        _backends_initialized = False


def milvus_client():
    global _milvus_client

    if _milvus_client is not None:
        return _milvus_client

    with _client_lock:
        if _milvus_client is None:
            _milvus_client = _create_milvus_client()

        return _milvus_client


def _create_milvus_client():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RetrievalBackendError("未安装 pymilvus，请先安装 requirements.txt") from exc

    token = settings.milvus_token.strip() or None
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    return MilvusClient(uri=uri, token=token)


def ensure_milvus_collection(client: Any) -> None:
    try:
        if retry_call(
            lambda: client.has_collection(settings.milvus_collection),
            operation_name="Milvus has collection",
        ):
            return

        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dimensions)
        schema.add_field("claim_id", DataType.VARCHAR, max_length=128)
        schema.add_field("note_id", DataType.VARCHAR, max_length=128)
        schema.add_field("block_id", DataType.VARCHAR, max_length=128)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("keywords_text", DataType.VARCHAR, max_length=2048)
        schema.add_field("l3_text", DataType.VARCHAR, max_length=4096)
        schema.add_field("claim_text", DataType.VARCHAR, max_length=4096)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        retry_call(
            lambda: client.create_collection(
                collection_name=settings.milvus_collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            ),
            operation_name="Milvus create collection",
        )
    except Exception as exc:
        raise RetrievalBackendError(f"Milvus collection 初始化失败：{exc}") from exc


def milvus_document(document: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(document["id"])[:128],
        "vector": normalize_vector(document["vector"]),
        "claim_id": str(document["claim_id"])[:128],
        "note_id": str(document["note_id"])[:128],
        "block_id": str(document.get("block_id", ""))[:128],
        "title": str(document.get("title", ""))[:512],
        "keywords_text": str(document.get("keywords_text", ""))[:2048],
        "l3_text": str(document.get("l3_text", ""))[:4096],
        "claim_text": str(document.get("claim_text", ""))[:4096],
    }


def elasticsearch_client():
    global _elasticsearch_client

    if _elasticsearch_client is not None:
        return _elasticsearch_client

    with _client_lock:
        if _elasticsearch_client is None:
            _elasticsearch_client = _create_elasticsearch_client()

        return _elasticsearch_client


def _create_elasticsearch_client():
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise RetrievalBackendError("未安装 elasticsearch，请先安装 requirements.txt") from exc

    return Elasticsearch(settings.es_url)


def ensure_elasticsearch_index(client: Any) -> None:
    if retry_call(
        lambda: client.indices.exists(index=settings.es_index),
        operation_name="Elasticsearch index exists",
    ):
        return

    try:
        retry_call(
            lambda: client.indices.create(
                index=settings.es_index,
                mappings={
                    "properties": {
                        "note_id": {"type": "keyword"},
                        "claim_id": {"type": "keyword"},
                        "block_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "tags": {"type": "text"},
                        "l2_summary": {"type": "text"},
                        "l3_text": {"type": "text"},
                        "claim_text": {"type": "text"},
                        "subject": {"type": "text"},
                        "predicate": {"type": "keyword"},
                        "object_text": {"type": "text"},
                        "source_text": {"type": "text"},
                        "keywords": {"type": "keyword"},
                        "keywords_text": {"type": "text"},
                    }
                },
            ),
            operation_name="Elasticsearch create index",
        )
    except Exception as exc:
        if "resource_already_exists_exception" in str(exc):
            return

        raise


def elasticsearch_document(document: dict[str, object]) -> dict[str, object]:
    return {
        "note_id": document["note_id"],
        "claim_id": document["claim_id"],
        "block_id": document.get("block_id", ""),
        "title": document.get("title", ""),
        "tags": document.get("tags", ""),
        "l2_summary": document.get("l2_summary", ""),
        "l3_text": document.get("l3_text", ""),
        "claim_text": document.get("claim_text", ""),
        "subject": document.get("subject", ""),
        "predicate": document.get("predicate", ""),
        "object_text": document.get("object_text", ""),
        "source_text": document.get("source_text", ""),
        "keywords": document.get("keywords", []),
        "keywords_text": document.get("keywords_text", ""),
    }


def normalize_keywords(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def normalize_vector(value: object) -> list[float]:
    if not isinstance(value, list):
        return []

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def unique_values(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = str(value).strip()

        if not item or item in seen:
            continue

        seen.add(item)
        unique.append(item)

    return unique


def escape_milvus_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    initialize_retrieval_backends()
    print("retrieval backends initialized")






