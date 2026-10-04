import pytest

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.models import AttributeCandidate, EntityScope, StoredEntityAttribute
from app.EvoRAG.indexes.attribute_hybrid import (
    AttributeHybridIndex,
    attribute_text_for_embedding,
    elasticsearch_attribute_properties,
    milvus_attribute_filter,
)


def record(attribute_id: int = 7, *, attr_type: str = "definition") -> StoredEntityAttribute:
    return StoredEntityAttribute(
        id=attribute_id,
        entity_id=3,
        scope=EntityScope(),
        attr_type=attr_type,
        value_text="A definition",
        value_fingerprint="fp-a",
        confidence=0.8,
        status="active",
    )


def test_attribute_index_schema_and_embedding_text_target_attribute_value() -> None:
    config = EvoRAGSettings(_env_file=None, embedding_dimensions=5)
    index = AttributeHybridIndex(config)

    assert attribute_text_for_embedding("definition", "A definition") == (
        "attribute type: definition\nattribute value: A definition"
    )
    properties = elasticsearch_attribute_properties()
    assert properties["attribute_id"]["type"] == "long"
    assert properties["entity_id"]["type"] == "long"
    assert properties["attr_type"]["type"] == "keyword"
    assert properties["status"]["type"] == "keyword"
    assert properties["value_text"]["type"] == "text"

    assert index.milvus_schema_fields()["attribute_id"]["is_primary"] is True
    assert index.milvus_schema_fields()["vector"]["dim"] == 5
    assert index.milvus_index_params() == {"field_name": "vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"}


class FakeESIndices:
    def __init__(self) -> None:
        self.created = []

    def exists(self, *, index: str) -> bool:
        return False

    def create(self, **kwargs):
        self.created.append(kwargs)


class FakeES:
    def __init__(self) -> None:
        self.indices = FakeESIndices()
        self.bulk_operations = None
        self.search_query = None
        self.msearches = None

    def bulk(self, *, operations):
        self.bulk_operations = operations
        return {"errors": False}

    def search(self, **kwargs):
        self.search_query = kwargs
        return {"hits": {"hits": [{"_source": {"attribute_id": 7}}]}}

    def msearch(self, *, searches):
        self.msearches = searches
        return {
            "responses": [
                {"hits": {"hits": [{"_score": 4.0, "_source": {"attribute_id": 7}}]}},
                {"hits": {"hits": [{"_score": 2.0, "_source": {"attribute_id": 8}}]}},
            ]
        }


class FakeMilvus:
    def __init__(self) -> None:
        self.upsert_payload = None
        self.query_filter = None
        self.search_payload = None

    def has_collection(self, collection_name: str) -> bool:
        return True

    def upsert(self, **kwargs):
        self.upsert_payload = kwargs

    def flush(self, **kwargs):
        pass

    def query(self, **kwargs):
        self.query_filter = kwargs
        return [{"attribute_id": 7}]

    def search(self, **kwargs):
        self.search_payload = kwargs
        return [
            [{"distance": 0.91, "entity": {"attribute_id": 7}}],
            [{"distance": 0.73, "entity": {"attribute_id": 8}}],
        ]


class FakeIndex(AttributeHybridIndex):
    def __init__(self, config, es, milvus) -> None:
        super().__init__(config)
        self.es = es
        self.milvus = milvus

    def elasticsearch_client(self):
        return self.es

    def milvus_client(self):
        return self.milvus


def test_attribute_index_batches_upserts_existence_and_searches() -> None:
    es = FakeES()
    milvus = FakeMilvus()
    index = FakeIndex(EvoRAGSettings(_env_file=None, embedding_dimensions=3), es, milvus)
    records = [record(7), record(8, attr_type="constraints")]

    index.upsert_elasticsearch_attributes(records)
    assert len(es.bulk_operations) == 4
    assert es.bulk_operations[0]["index"]["_id"] == "7"
    assert es.bulk_operations[1]["value_text"] == "A definition"

    index.upsert_milvus_attributes(records, [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    assert [item["attribute_id"] for item in milvus.upsert_payload["data"]] == [7, 8]
    assert milvus.upsert_payload["data"][0]["vector"] == [0.1, 0.2, 0.3]

    assert index.existing_elasticsearch_attribute_ids([8, 7]) == {7}
    assert index.existing_milvus_attribute_ids([8, 7], EntityScope()) == {7}

    es_hits = index.search_elasticsearch_many(
        [
            {"input_index": 0, "entity_id": 3, "attr_type": "definition", "scope": EntityScope(), "query": "A"},
            {"input_index": 1, "entity_id": 3, "attr_type": "definition", "scope": EntityScope(), "query": "B"},
        ],
        top_k=5,
    )
    assert [hit.attribute_id for hit in es_hits[0]] == [7]
    assert [hit.attribute_id for hit in es_hits[1]] == [8]
    assert len(es.msearches) == 4

    milvus_hits = index.search_milvus_group(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        entity_id=3,
        attr_type="definition",
        scope=EntityScope(),
        top_k=5,
    )
    assert [hit.attribute_id for hit in milvus_hits[0]] == [7]
    assert [hit.attribute_id for hit in milvus_hits[1]] == [8]
    assert milvus.search_payload["data"] == [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    assert 'entity_id == 3' in milvus_attribute_filter(3, "definition", EntityScope())

def test_elasticsearch_msearch_item_error_raises_backend_failure() -> None:
    class ErrorES(FakeES):
        def msearch(self, *, searches):
            return {"responses": [{"status": 503, "error": {"type": "unavailable"}}]}

    index = FakeIndex(EvoRAGSettings(_env_file=None), ErrorES(), FakeMilvus())

    with pytest.raises(RuntimeError, match="Elasticsearch attribute msearch failed"):
        index.search_elasticsearch_many(
            [{"input_index": 0, "entity_id": 3, "attr_type": "definition", "scope": EntityScope(), "query": "A"}],
            top_k=5,
        )