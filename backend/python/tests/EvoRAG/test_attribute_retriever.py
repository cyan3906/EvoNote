import asyncio
import time

import pytest

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.attribute_retriever import AttributeCandidateRetriever
from app.EvoRAG.entity_store.models import AttributeCandidate, EntityAttributeInput, EntityScope, StoredEntity, StoredEntityAttribute
from app.EvoRAG.entity_store.normalizer import text_fingerprint
from app.EvoRAG.indexes.attribute_hybrid import attribute_text_for_embedding


def entity() -> StoredEntity:
    return StoredEntity(
        id=3,
        canonical_name="Redis",
        normalized_name="redis",
        entity_type="system",
        scope=EntityScope(),
    )


def incoming(attr_type: str, value: str) -> EntityAttributeInput:
    return EntityAttributeInput(attr_type=attr_type, value_text=value, evidence="")


def stored_attr(attribute_id: int, attr_type: str, value: str, *, entity_id: int = 3, status: str = "active") -> StoredEntityAttribute:
    return StoredEntityAttribute(
        id=attribute_id,
        entity_id=entity_id,
        scope=EntityScope(),
        attr_type=attr_type,
        value_text=value,
        value_fingerprint=text_fingerprint(value),
        confidence=0.8,
        status=status,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.exact_matches = {}
        self.counts = {}
        self.rows = []
        self.hydrated = []
        self.calls = []

    def find_exact_active_attributes(self, entity_id, fingerprints_by_type):
        self.calls.append(("exact", entity_id, fingerprints_by_type))
        return self.exact_matches

    def count_active_attributes_by_type(self, entity_id, attr_types):
        self.calls.append(("count", entity_id, list(attr_types)))
        return {attr_type: self.counts.get(attr_type, 0) for attr_type in set(attr_types)}

    def list_active_attributes_for_types(self, entity_id, attr_types):
        self.calls.append(("list", entity_id, list(attr_types)))
        return [row for row in self.rows if row.entity_id == entity_id and row.attr_type in set(attr_types)]

    def get_active_attributes_by_ids(self, attribute_ids):
        self.calls.append(("hydrate", list(attribute_ids)))
        allowed = set(attribute_ids)
        return [row for row in self.hydrated if row.id in allowed]


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls = []

    async def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


class FakeIndex:
    def __init__(self) -> None:
        self.es_calls = []
        self.milvus_calls = []
        self.es_existing = set()
        self.milvus_existing = set()
        self.es_results = {}
        self.milvus_results = []
        self.milvus_results_by_type = {}
        self.overlap = False
        self._running = 0
        self.es_upserts = []
        self.milvus_upserts = []
        self.ensure_es_calls = 0
        self.ensure_milvus_calls = 0
        self.fail_existing_es = False
        self.fail_existing_milvus = False
        self.fail_es_search = False
        self.fail_milvus_search = False

    def ensure_elasticsearch_index(self):
        self.ensure_es_calls += 1

    def ensure_milvus_collection(self):
        self.ensure_milvus_calls += 1

    def existing_elasticsearch_attribute_ids(self, ids):
        if self.fail_existing_es:
            raise RuntimeError("es repair down")
        return set(self.es_existing)

    def existing_milvus_attribute_ids(self, ids, scope):
        if self.fail_existing_milvus:
            raise RuntimeError("milvus repair down")
        return set(self.milvus_existing)

    def upsert_elasticsearch_attributes(self, records):
        self.es_upserts.append([record.id for record in records])

    def upsert_milvus_attributes(self, records, vectors):
        self.milvus_upserts.append(([record.id for record in records], vectors))

    def search_elasticsearch_many(self, requests, top_k):
        if self.fail_es_search:
            raise RuntimeError("es search down")
        self.es_calls.append((requests, top_k))
        return self.es_results

    def search_milvus_group(self, vectors, *, entity_id, attr_type, scope, top_k):
        if self.fail_milvus_search:
            raise RuntimeError("milvus search down")
        self._running += 1
        if self._running > 1:
            self.overlap = True
        time.sleep(0.05)
        self._running -= 1
        self.milvus_calls.append((vectors, entity_id, attr_type, scope, top_k))
        if attr_type in self.milvus_results_by_type:
            return self.milvus_results_by_type[attr_type]
        return self.milvus_results.pop(0)

def test_retriever_empty_and_exact_matches_short_circuit_dependencies() -> None:
    repo = FakeRepository()
    embedding = FakeEmbedding()
    index = FakeIndex()
    exact = stored_attr(7, "definition", "A definition")
    repo.exact_matches = {("definition", text_fingerprint("A definition")): exact}
    retriever = AttributeCandidateRetriever(repo, embedding, index, EvoRAGSettings(_env_file=None))

    assert asyncio.run(retriever.retrieve(entity(), [])) == []
    results = asyncio.run(retriever.retrieve(entity(), [incoming("definition", "A definition"), incoming("definition", "A definition")]))

    assert [result.input_index for result in results] == [0, 1]
    assert all(result.mode == "exact" for result in results)
    assert all(result.exact_match == exact for result in results)
    assert [call[0] for call in repo.calls] == ["exact"]
    assert embedding.calls == []
    assert index.es_calls == []
    assert index.milvus_calls == []


def test_retriever_full_scans_attribute_groups_below_threshold_without_type_leakage() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 19, "constraints": 1}
    repo.rows = [stored_attr(7, "definition", "A definition"), stored_attr(8, "constraints", "A constraint")]
    retriever = AttributeCandidateRetriever(repo, FakeEmbedding(), FakeIndex(), EvoRAGSettings(_env_file=None))

    results = asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming"), incoming("constraints", "Incoming")]))

    assert [result.mode for result in results] == ["full_scan", "full_scan"]
    assert [candidate.attribute_id for candidate in results[0].candidates] == [7]
    assert [candidate.attribute_id for candidate in results[1].candidates] == [8]
    assert ("list", 3, ["definition", "constraints"]) in repo.calls


def test_retriever_hybrid_batches_es_groups_milvus_concurrently_and_hydrates_fused_results() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20, "constraints": 20}
    repo.rows = [stored_attr(7, "definition", "Indexed definition"), stored_attr(8, "constraints", "Indexed constraint")]
    repo.hydrated = list(repo.rows)
    embedding = FakeEmbedding()
    index = FakeIndex()
    index.es_existing = {7, 8}
    index.milvus_existing = {7, 8}
    index.es_results = {
        0: [AttributeCandidate(7, 3, "definition", "", 0.0, rank=1, source="elasticsearch", es_score=4.0)],
        1: [AttributeCandidate(8, 3, "constraints", "", 0.0, rank=1, source="elasticsearch", es_score=3.0)],
    }
    index.milvus_results_by_type = {
        "definition": [[AttributeCandidate(7, 3, "definition", "", 0.0, rank=1, source="milvus", vector_score=0.9)]],
        "constraints": [[AttributeCandidate(8, 3, "constraints", "", 0.0, rank=1, source="milvus", vector_score=0.8)]],
    }
    config = EvoRAGSettings(_env_file=None, attribute_milvus_max_concurrency=2)
    retriever = AttributeCandidateRetriever(repo, embedding, index, config)

    results = asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming def"), incoming("constraints", "Incoming constraint")]))

    assert [result.mode for result in results] == ["hybrid", "hybrid"]
    assert [result.candidates[0].attribute_id for result in results] == [7, 8]
    assert results[0].candidates[0].source == "elasticsearch+milvus"
    assert len(index.es_calls) == 1
    assert [request["input_index"] for request in index.es_calls[0][0]] == [0, 1]
    assert len(index.milvus_calls) == 2
    assert index.overlap is True
    assert embedding.calls[0] == [
        attribute_text_for_embedding("definition", "Incoming def"),
        attribute_text_for_embedding("constraints", "Incoming constraint"),
    ]


def test_retriever_repairs_missing_attribute_indexes_before_hybrid_search() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20}
    repo.rows = [stored_attr(7, "definition", "Indexed definition"), stored_attr(8, "definition", "Missing definition")]
    repo.hydrated = list(repo.rows)
    embedding = FakeEmbedding()
    index = FakeIndex()
    index.es_existing = {7}
    index.milvus_existing = {8}
    index.es_results = {0: []}
    index.milvus_results = [[[]]]
    retriever = AttributeCandidateRetriever(repo, embedding, index, EvoRAGSettings(_env_file=None))

    asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming")]))

    assert index.es_upserts == [[8]]
    assert index.milvus_upserts[0][0] == [7]
    assert attribute_text_for_embedding("definition", "Missing definition") in embedding.calls[0]
    assert attribute_text_for_embedding("definition", "Indexed definition") in embedding.calls[0]

def test_retriever_repair_failure_in_one_backend_still_uses_other_backend() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20}
    repo.rows = [stored_attr(7, "definition", "Indexed definition")]
    repo.hydrated = list(repo.rows)
    embedding = FakeEmbedding()
    index = FakeIndex()
    index.fail_existing_es = True
    index.milvus_existing = {7}
    index.es_results = {0: []}
    index.milvus_results = [[[AttributeCandidate(7, 3, "definition", "", 0.0, rank=1, source="milvus", vector_score=0.9)]]]
    retriever = AttributeCandidateRetriever(repo, embedding, index, EvoRAGSettings(_env_file=None))

    results = asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming")]))

    assert [candidate.attribute_id for candidate in results[0].candidates] == [7]
    assert any("elasticsearch repair failed" in warning for warning in results[0].warnings)
    assert index.ensure_es_calls == 1
    assert index.ensure_milvus_calls == 1


def test_retriever_filters_hydrated_wrong_entity_type_and_inactive_hits() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20}
    repo.rows = [stored_attr(7, "definition", "Good")]
    repo.hydrated = [
        stored_attr(7, "definition", "Good"),
        stored_attr(8, "definition", "Wrong entity", entity_id=99),
        stored_attr(9, "constraints", "Wrong type"),
        stored_attr(10, "definition", "Inactive", status="inactive"),
    ]
    index = FakeIndex()
    index.es_existing = {7, 8, 9, 10}
    index.milvus_existing = {7, 8, 9, 10}
    index.es_results = {
        0: [
            AttributeCandidate(8, 99, "definition", "", 0.0, rank=1, source="elasticsearch", es_score=4.0),
            AttributeCandidate(9, 3, "constraints", "", 0.0, rank=2, source="elasticsearch", es_score=3.0),
            AttributeCandidate(10, 3, "definition", "", 0.0, rank=3, source="elasticsearch", es_score=2.0),
            AttributeCandidate(7, 3, "definition", "", 0.0, rank=4, source="elasticsearch", es_score=1.0),
        ]
    }
    index.milvus_results = [[[]]]
    retriever = AttributeCandidateRetriever(repo, FakeEmbedding(), index, EvoRAGSettings(_env_file=None))

    results = asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming")]))

    assert [candidate.attribute_id for candidate in results[0].candidates] == [7]


def test_retriever_raises_typed_error_when_both_search_backends_fail() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20}
    repo.rows = [stored_attr(7, "definition", "Indexed definition")]
    repo.hydrated = list(repo.rows)
    index = FakeIndex()
    index.es_existing = {7}
    index.milvus_existing = {7}
    index.fail_es_search = True
    index.fail_milvus_search = True
    retriever = AttributeCandidateRetriever(repo, FakeEmbedding(), index, EvoRAGSettings(_env_file=None))

    with pytest.raises(Exception, match="attribute hybrid retrieval failed"):
        asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming")]))


def test_retriever_batches_index_repair_embeddings_by_configured_size() -> None:
    repo = FakeRepository()
    repo.counts = {"definition": 20}
    repo.rows = [stored_attr(7, "definition", "A"), stored_attr(8, "definition", "B"), stored_attr(9, "definition", "C")]
    repo.hydrated = list(repo.rows)
    embedding = FakeEmbedding()
    index = FakeIndex()
    index.es_existing = set()
    index.milvus_existing = set()
    index.es_results = {0: []}
    index.milvus_results = [[[]]]
    config = EvoRAGSettings(_env_file=None, attribute_index_backfill_batch_size=2)
    retriever = AttributeCandidateRetriever(repo, embedding, index, config)

    asyncio.run(retriever.retrieve(entity(), [incoming("definition", "Incoming")]))

    repair_calls = embedding.calls[:-1]
    assert [len(call) for call in repair_calls] == [2, 1]
    assert index.es_upserts == [[7, 8], [9]]
    assert [item[0] for item in index.milvus_upserts] == [[7, 8], [9]]