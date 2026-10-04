from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import AttributeCandidate, EntityScope, StoredEntityAttribute
from app.EvoRAG.indexes.entity_hybrid import escape_milvus_string, normalize_url
from app.core import retrieval as core_retrieval
from app.core.config import settings as core_settings


def attribute_text_for_embedding(attr_type: str, value_text: str) -> str:
    return f"attribute type: {str(attr_type or '').strip()}\nattribute value: {str(value_text or '').strip()}"


class AttributeHybridIndex:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config

    def ensure_indexes(self) -> None:
        self.ensure_elasticsearch_index()
        self.ensure_milvus_collection()

    def ensure_elasticsearch_index(self) -> None:
        client = self.elasticsearch_client()
        if client.indices.exists(index=self.config.es_attribute_index):
            if hasattr(client.indices, "put_mapping"):
                client.indices.put_mapping(
                    index=self.config.es_attribute_index,
                    properties=elasticsearch_attribute_properties(),
                )
            return
        client.indices.create(
            index=self.config.es_attribute_index,
            mappings={"properties": elasticsearch_attribute_properties()},
        )

    def ensure_milvus_collection(self) -> None:
        client = self.milvus_client()
        if client.has_collection(self.config.milvus_attribute_collection):
            return

        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("attribute_id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.embedding_dimensions)
        schema.add_field("entity_id", DataType.INT64)
        schema.add_field("workspace_id", DataType.VARCHAR, max_length=128)
        schema.add_field("project_id", DataType.VARCHAR, max_length=128)
        schema.add_field("collection_id", DataType.VARCHAR, max_length=128)
        schema.add_field("domain", DataType.VARCHAR, max_length=128)
        schema.add_field("attr_type", DataType.VARCHAR, max_length=64)
        schema.add_field("status", DataType.VARCHAR, max_length=32)
        schema.add_field("value_text", DataType.VARCHAR, max_length=4096)

        index_params = client.prepare_index_params()
        params = self.milvus_index_params()
        index_params.add_index(**params)
        client.create_collection(
            collection_name=self.config.milvus_attribute_collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def milvus_schema_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "attribute_id": {"type": "INT64", "is_primary": True},
            "vector": {"type": "FLOAT_VECTOR", "dim": self.config.embedding_dimensions},
            "entity_id": {"type": "INT64"},
            "workspace_id": {"type": "VARCHAR", "max_length": 128},
            "project_id": {"type": "VARCHAR", "max_length": 128},
            "collection_id": {"type": "VARCHAR", "max_length": 128},
            "domain": {"type": "VARCHAR", "max_length": 128},
            "attr_type": {"type": "VARCHAR", "max_length": 64},
            "status": {"type": "VARCHAR", "max_length": 32},
            "value_text": {"type": "VARCHAR", "max_length": 4096},
        }

    def milvus_index_params(self) -> dict[str, str]:
        return {"field_name": "vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"}

    def upsert_elasticsearch_attributes(self, records: list[StoredEntityAttribute]) -> None:
        if not records:
            return
        operations: list[dict[str, Any]] = []
        for record in records:
            operations.append({"index": {"_index": self.config.es_attribute_index, "_id": str(record.id)}})
            operations.append(elasticsearch_attribute_document(record))
        response = self.elasticsearch_client().bulk(operations=operations)
        if isinstance(response, dict) and response.get("errors"):
            raise RuntimeError("Elasticsearch attribute bulk upsert failed")

    def upsert_milvus_attributes(self, records: list[StoredEntityAttribute], vectors: list[list[float]]) -> None:
        if not records:
            return
        data = [milvus_attribute_document(record, vector) for record, vector in zip(records, vectors)]
        client = self.milvus_client()
        if hasattr(client, "upsert"):
            client.upsert(collection_name=self.config.milvus_attribute_collection, data=data)
        else:
            ids = ", ".join(str(record.id) for record in records)
            client.delete(collection_name=self.config.milvus_attribute_collection, filter=f"attribute_id in [{ids}]")
            client.insert(collection_name=self.config.milvus_attribute_collection, data=data)
        if hasattr(client, "flush"):
            client.flush(collection_name=self.config.milvus_attribute_collection)

    def existing_elasticsearch_attribute_ids(self, attribute_ids: list[int]) -> set[int]:
        unique_ids = sorted({int(attribute_id) for attribute_id in attribute_ids})
        if not unique_ids:
            return set()
        response = self.elasticsearch_client().search(
            index=self.config.es_attribute_index,
            size=len(unique_ids),
            source_includes=["attribute_id"],
            query={"bool": {"filter": [{"terms": {"attribute_id": unique_ids}}, {"term": {"status": "active"}}]}},
        )
        return {
            int(hit.get("_source", {}).get("attribute_id"))
            for hit in response.get("hits", {}).get("hits", [])
            if hit.get("_source", {}).get("attribute_id") is not None
        }

    def existing_milvus_attribute_ids(self, attribute_ids: list[int], scope: EntityScope | None = None) -> set[int]:
        unique_ids = sorted({int(attribute_id) for attribute_id in attribute_ids})
        if not unique_ids:
            return set()
        id_filter = f"attribute_id in [{', '.join(str(attribute_id) for attribute_id in unique_ids)}]"
        scope_filter = milvus_scope_filter(scope)
        filter_expression = " and ".join(part for part in (id_filter, scope_filter, 'status == "active"') if part)
        results = self.milvus_client().query(
            collection_name=self.config.milvus_attribute_collection,
            filter=filter_expression,
            output_fields=["attribute_id"],
            limit=len(unique_ids),
        )
        return {int(item["attribute_id"]) for item in results if item.get("attribute_id") is not None}

    def search_elasticsearch_many(self, requests: list[dict[str, Any]], top_k: int) -> dict[int, list[AttributeCandidate]]:
        if not requests:
            return {}
        searches: list[dict[str, Any]] = []
        for request in requests:
            searches.append({"index": self.config.es_attribute_index})
            searches.append(
                {
                    "size": top_k,
                    "query": {
                        "bool": {
                            "must": [{"match": {"value_text": request["query"]}}],
                            "filter": elasticsearch_attribute_filters(
                                entity_id=int(request["entity_id"]),
                                attr_type=str(request["attr_type"]),
                                scope=request.get("scope"),
                            ),
                        }
                    },
                }
            )
        response = self.elasticsearch_client().msearch(searches=searches)
        results: dict[int, list[AttributeCandidate]] = {}
        for request, item in zip(requests, response.get("responses", [])):
            if item.get("error") or int(item.get("status") or 200) >= 400:
                raise RuntimeError(f"Elasticsearch attribute msearch failed: {item.get('error') or item.get('status')}")
            hits: list[AttributeCandidate] = []
            for rank, hit in enumerate(item.get("hits", {}).get("hits", []), start=1):
                source = hit.get("_source", {})
                hits.append(
                    AttributeCandidate(
                        attribute_id=int(source.get("attribute_id") or 0),
                        entity_id=int(source.get("entity_id") or request["entity_id"]),
                        attr_type=str(source.get("attr_type") or request["attr_type"]),
                        value_text=str(source.get("value_text") or ""),
                        confidence=float(source.get("confidence") or 0.0),
                        rank=rank,
                        source="elasticsearch",
                        es_score=float(hit.get("_score") or 0.0),
                    )
                )
            results[int(request["input_index"])] = hits
        return results

    def search_milvus_group(
        self,
        vectors: list[list[float]],
        *,
        entity_id: int,
        attr_type: str,
        scope: EntityScope,
        top_k: int,
    ) -> list[list[AttributeCandidate]]:
        if not vectors:
            return []
        results = self.milvus_client().search(
            collection_name=self.config.milvus_attribute_collection,
            data=vectors,
            limit=top_k,
            filter=milvus_attribute_filter(entity_id, attr_type, scope),
            output_fields=["attribute_id", "entity_id", "attr_type", "value_text", "confidence"],
        )
        grouped: list[list[AttributeCandidate]] = []
        for vector_hits in results:
            candidates: list[AttributeCandidate] = []
            for rank, hit in enumerate(vector_hits, start=1):
                entity = hit.get("entity", {})
                candidates.append(
                    AttributeCandidate(
                        attribute_id=int(entity.get("attribute_id") or hit.get("id") or 0),
                        entity_id=int(entity.get("entity_id") or entity_id),
                        attr_type=str(entity.get("attr_type") or attr_type),
                        value_text=str(entity.get("value_text") or ""),
                        confidence=float(entity.get("confidence") or 0.0),
                        rank=rank,
                        source="milvus",
                        vector_score=float(hit.get("distance") or 0.0),
                    )
                )
            grouped.append(candidates)
        return grouped

    def elasticsearch_client(self):
        if self.can_use_core_elasticsearch_client():
            return core_retrieval.elasticsearch_client()
        from elasticsearch import Elasticsearch
        return Elasticsearch(self.config.es_url)

    def milvus_client(self):
        if self.can_use_core_milvus_client():
            return core_retrieval.milvus_client()
        from pymilvus import MilvusClient
        token = self.config.milvus_token.strip() or None
        return MilvusClient(uri=f"http://{self.config.milvus_host}:{self.config.milvus_port}", token=token)

    def can_use_core_elasticsearch_client(self) -> bool:
        return normalize_url(self.config.es_url) == normalize_url(core_settings.es_url)

    def can_use_core_milvus_client(self) -> bool:
        return (
            str(self.config.milvus_host).strip().lower() == str(core_settings.milvus_host).strip().lower()
            and int(self.config.milvus_port) == int(core_settings.milvus_port)
            and str(self.config.milvus_token or "").strip() == str(core_settings.milvus_token or "").strip()
        )


def elasticsearch_attribute_properties() -> dict[str, Any]:
    return {
        "attribute_id": {"type": "long"},
        "entity_id": {"type": "long"},
        "workspace_id": {"type": "keyword"},
        "project_id": {"type": "keyword"},
        "collection_id": {"type": "keyword"},
        "domain": {"type": "keyword"},
        "attr_type": {"type": "keyword"},
        "status": {"type": "keyword"},
        "value_fingerprint": {"type": "keyword"},
        "value_text": {"type": "text"},
        "confidence": {"type": "double"},
    }


def elasticsearch_attribute_document(record: StoredEntityAttribute) -> dict[str, Any]:
    return {
        "attribute_id": int(record.id),
        "entity_id": int(record.entity_id),
        "workspace_id": record.scope.workspace_id,
        "project_id": record.scope.project_id,
        "collection_id": record.scope.collection_id,
        "domain": record.scope.domain,
        "attr_type": record.attr_type,
        "status": record.status,
        "value_fingerprint": record.value_fingerprint,
        "value_text": record.value_text,
        "confidence": record.confidence,
    }


def milvus_attribute_document(record: StoredEntityAttribute, vector: list[float]) -> dict[str, Any]:
    return {
        "attribute_id": int(record.id),
        "vector": vector,
        "entity_id": int(record.entity_id),
        "workspace_id": record.scope.workspace_id[:128],
        "project_id": record.scope.project_id[:128],
        "collection_id": record.scope.collection_id[:128],
        "domain": record.scope.domain[:128],
        "attr_type": record.attr_type[:64],
        "status": record.status[:32],
        "value_text": record.value_text[:4096],
    }


def elasticsearch_attribute_filters(*, entity_id: int, attr_type: str, scope: EntityScope | None) -> list[dict[str, Any]]:
    filters = [{"term": {"entity_id": entity_id}}, {"term": {"attr_type": attr_type}}, {"term": {"status": "active"}}]
    if scope is not None:
        filters.extend({"term": {field: value}} for field, value in scope.as_dict().items() if value)
    return filters


def milvus_scope_filter(scope: EntityScope | None) -> str:
    if scope is None:
        return ""
    return " and ".join(
        f'{field} == "{escape_milvus_string(value)}"'
        for field, value in scope.as_dict().items()
        if value
    )


def milvus_attribute_filter(entity_id: int, attr_type: str, scope: EntityScope | None) -> str:
    parts = [
        f"entity_id == {int(entity_id)}",
        f'attr_type == "{escape_milvus_string(attr_type)}"',
        'status == "active"',
        milvus_scope_filter(scope),
    ]
    return " and ".join(part for part in parts if part)