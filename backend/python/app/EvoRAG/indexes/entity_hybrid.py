from typing import Any
from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name


class EntityHybridIndex:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config

    def ensure_indexes(self) -> None:
        self.ensure_elasticsearch_index()
        self.ensure_milvus_collection()

    def upsert_entity(self, entity: StoredEntity) -> None:
        self.ensure_indexes()
        self.upsert_elasticsearch_entity(entity)
        self.upsert_milvus_entity(entity)

    def search(self, incoming: IncomingEntity, *, top_k: int | None = None) -> list[CandidateEntity]:
        self.ensure_indexes()
        limit = top_k or self.config.entity_resolution_top_k
        es_hits = self.search_elasticsearch(incoming, top_k=limit)
        milvus_hits = self.search_milvus(incoming.embedding, top_k=limit, scope=incoming.scope)
        return reciprocal_rank_fusion_entities(
            [es_hits, milvus_hits],
            rrf_k=self.config.entity_resolution_rrf_k,
            top_k=limit,
        )

    def ensure_elasticsearch_index(self) -> None:
        client = self.elasticsearch_client()
        if client.indices.exists(index=self.config.es_entity_index):
            client.indices.put_mapping(
                index=self.config.es_entity_index,
                properties=elasticsearch_entity_properties(),
            )
            return
        client.indices.create(
            index=self.config.es_entity_index,
            mappings={"properties": elasticsearch_entity_properties()},
        )

    def upsert_elasticsearch_entity(self, entity: StoredEntity) -> None:
        self.elasticsearch_client().index(
            index=self.config.es_entity_index,
            id=str(entity.id),
            document={
                "entity_id": entity.id,
                "workspace_id": entity.scope.workspace_id,
                "project_id": entity.scope.project_id,
                "collection_id": entity.scope.collection_id,
                "domain": entity.scope.domain,
                "canonical_name": entity.canonical_name,
                "normalized_name": entity.normalized_name,
                "entity_type": entity.entity_type,
                "aliases": entity.aliases,
                "identity_description": entity.identity_description,
                "summary": entity.summary,
                "description_for_match": entity.description_for_match,
            },
            refresh=False,
        )

    def search_elasticsearch(self, incoming: IncomingEntity, *, top_k: int) -> list[CandidateEntity]:
        query_text = " ".join(
            part
            for part in [
                incoming.name,
                " ".join(incoming.aliases),
                incoming.entity_type,
                incoming.identity_description,
            ]
            if part
        ).strip()
        if not query_text:
            return []

        response = self.elasticsearch_client().search(
            index=self.config.es_entity_index,
            size=top_k,
            query={
                "bool": {
                    "should": [
                        {"term": {"normalized_name": {"value": incoming.normalized_name, "boost": 8}}},
                        {"term": {"entity_type": {"value": incoming.entity_type, "boost": 2}}},
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "canonical_name^6",
                                    "aliases^5",
                                    "identity_description^3",
                                    "summary^2",
                                    "description_for_match",
                                ],
                            }
                        },
                    ],
                    "filter": elasticsearch_scope_filters(incoming.scope),
                    "minimum_should_match": 1,
                }
            },
        )

        hits: list[CandidateEntity] = []
        raw_hits = response.get("hits", {}).get("hits", [])
        max_score = max((float(item.get("_score") or 0.0) for item in raw_hits), default=0.0)
        for rank, item in enumerate(raw_hits, start=1):
            source = item.get("_source", {})
            raw_score = float(item.get("_score") or 0.0)
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            hits.append(
                CandidateEntity(
                    entity=source_to_entity(source),
                    score=normalized_score,
                    rank=rank,
                    source="elasticsearch",
                    es_score=normalized_score,
                )
            )
        return hits

    def ensure_milvus_collection(self) -> None:
        client = self.milvus_client()
        if client.has_collection(self.config.milvus_entity_collection):
            self.ensure_milvus_scope_fields(client)
            return

        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.embedding_dimensions)
        schema.add_field("entity_id", DataType.INT64)
        schema.add_field("workspace_id", DataType.VARCHAR, max_length=128)
        schema.add_field("project_id", DataType.VARCHAR, max_length=128)
        schema.add_field("collection_id", DataType.VARCHAR, max_length=128)
        schema.add_field("domain", DataType.VARCHAR, max_length=128)
        schema.add_field("canonical_name", DataType.VARCHAR, max_length=255)
        schema.add_field("entity_type", DataType.VARCHAR, max_length=64)
        schema.add_field("identity_description", DataType.VARCHAR, max_length=2048)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_collection(
            collection_name=self.config.milvus_entity_collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def upsert_milvus_entity(self, entity: StoredEntity) -> None:
        if not entity.embedding:
            return
        client = self.milvus_client()
        document = {
            "id": int(entity.id),
            "vector": entity.embedding,
            "entity_id": int(entity.id),
            "workspace_id": entity.scope.workspace_id[:128],
            "project_id": entity.scope.project_id[:128],
            "collection_id": entity.scope.collection_id[:128],
            "domain": entity.scope.domain[:128],
            "canonical_name": entity.canonical_name[:255],
            "entity_type": entity.entity_type[:64],
            "identity_description": entity.identity_description[:2048],
        }
        if hasattr(client, "upsert"):
            client.upsert(collection_name=self.config.milvus_entity_collection, data=[document])
        else:
            client.delete(collection_name=self.config.milvus_entity_collection, filter=f"entity_id == {int(entity.id)}")
            client.insert(collection_name=self.config.milvus_entity_collection, data=[document])
        client.flush(collection_name=self.config.milvus_entity_collection)

    def search_milvus(self, vector: list[float], *, top_k: int, scope: EntityScope | None = None) -> list[CandidateEntity]:
        if not vector:
            return []
        results = self.milvus_client().search(
            collection_name=self.config.milvus_entity_collection,
            data=[vector],
            limit=top_k,
            filter=milvus_scope_filter(scope) if scope else "",
            output_fields=[
                "entity_id",
                "workspace_id",
                "project_id",
                "collection_id",
                "domain",
                "canonical_name",
                "entity_type",
                "identity_description",
            ],
        )
        hits: list[CandidateEntity] = []
        for rank, hit in enumerate(results[0] if results else [], start=1):
            entity = hit.get("entity", {})
            score = float(hit.get("distance") or 0.0)
            entity_id = int(entity.get("entity_id") or hit.get("id"))
            hits.append(
                CandidateEntity(
                    entity=StoredEntity(
                        id=entity_id,
                        canonical_name=str(entity.get("canonical_name") or ""),
                        normalized_name=normalize_name(str(entity.get("canonical_name") or "")),
                        entity_type=str(entity.get("entity_type") or ""),
                        scope=EntityScope(
                            workspace_id=str(entity.get("workspace_id") or "local"),
                            project_id=str(entity.get("project_id") or "evorag"),
                            collection_id=str(entity.get("collection_id") or "default"),
                            domain=str(entity.get("domain") or "general"),
                        ),
                        identity_description=str(entity.get("identity_description") or ""),
                    ),
                    score=score,
                    rank=rank,
                    source="milvus",
                    vector_score=score,
                )
            )
        return hits

    def elasticsearch_client(self):
        from elasticsearch import Elasticsearch

        return Elasticsearch(self.config.es_url)

    def milvus_client(self):
        from pymilvus import MilvusClient

        token = self.config.milvus_token.strip() or None
        return MilvusClient(uri=f"http://{self.config.milvus_host}:{self.config.milvus_port}", token=token)

    def ensure_milvus_scope_fields(self, client: Any) -> None:
        existing = extract_milvus_field_names(client.describe_collection(collection_name=self.config.milvus_entity_collection))
        missing = [field for field in ("workspace_id", "project_id", "collection_id", "domain") if field not in existing]
        if not missing:
            return

        from pymilvus import DataType

        for field in missing:
            client.add_collection_field(
                collection_name=self.config.milvus_entity_collection,
                field_name=field,
                data_type=DataType.VARCHAR,
                max_length=128,
                nullable=True,
            )


def reciprocal_rank_fusion_entities(
    rankings: list[list[CandidateEntity]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[CandidateEntity]:
    scores: dict[int, float] = {}
    best: dict[int, CandidateEntity] = {}
    sources: dict[int, list[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            entity_id = hit.entity.id
            scores[entity_id] = scores.get(entity_id, 0.0) + (1.0 / (rrf_k + rank))
            sources.setdefault(entity_id, []).append(hit.source)
            previous = best.get(entity_id)
            if previous is None or hit.score > previous.score:
                best[entity_id] = hit
            else:
                previous.vector_score = max(previous.vector_score, hit.vector_score)
                previous.es_score = max(previous.es_score, hit.es_score)

    fused: list[CandidateEntity] = []
    for rank, entity_id in enumerate(sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k], start=1):
        hit = best[entity_id]
        fused.append(
            CandidateEntity(
                entity=hit.entity,
                score=max(hit.score, hit.vector_score, hit.es_score),
                rank=rank,
                source="+".join(unique_ordered(sources.get(entity_id, []))),
                vector_score=hit.vector_score,
                es_score=hit.es_score,
            )
        )
    return fused


def source_to_entity(source: dict[str, Any]) -> StoredEntity:
    return StoredEntity(
        id=int(source.get("entity_id") or 0),
        canonical_name=str(source.get("canonical_name") or ""),
        normalized_name=str(source.get("normalized_name") or normalize_name(str(source.get("canonical_name") or ""))),
        entity_type=str(source.get("entity_type") or ""),
        scope=EntityScope(
            workspace_id=str(source.get("workspace_id") or "local"),
            project_id=str(source.get("project_id") or "evorag"),
            collection_id=str(source.get("collection_id") or "default"),
            domain=str(source.get("domain") or "general"),
        ),
        aliases=[str(item) for item in source.get("aliases") or []],
        identity_description=str(source.get("identity_description") or ""),
        summary=str(source.get("summary") or ""),
        description_for_match=str(source.get("description_for_match") or ""),
    )


def unique_ordered(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def elasticsearch_entity_properties() -> dict[str, Any]:
    return {
        "entity_id": {"type": "long"},
        "workspace_id": {"type": "keyword"},
        "project_id": {"type": "keyword"},
        "collection_id": {"type": "keyword"},
        "domain": {"type": "keyword"},
        "canonical_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "normalized_name": {"type": "keyword"},
        "entity_type": {"type": "keyword"},
        "aliases": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "identity_description": {"type": "text"},
        "summary": {"type": "text"},
        "description_for_match": {"type": "text"},
    }


def elasticsearch_scope_filters(scope: EntityScope) -> list[dict[str, Any]]:
    return [{"term": {field: value}} for field, value in scope.as_dict().items() if value]


def milvus_scope_filter(scope: EntityScope | None) -> str:
    if scope is None:
        return ""
    parts = [f'{field} == "{escape_milvus_string(value)}"' for field, value in scope.as_dict().items() if value]
    return " and ".join(parts)


def escape_milvus_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def extract_milvus_field_names(description: Any) -> set[str]:
    if isinstance(description, dict):
        fields = description.get("fields") or description.get("schema", {}).get("fields") or []
        result: set[str] = set()
        for field in fields:
            if isinstance(field, dict) and field.get("name"):
                result.add(str(field["name"]))
            elif hasattr(field, "name"):
                result.add(str(field.name))
        return result
    return set()
