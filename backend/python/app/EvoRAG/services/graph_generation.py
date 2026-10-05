import asyncio
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import EntityScope, StoredEntity
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.models import DependencyGraph, RetrievedEntity, StoredAttribute
from app.EvoRAG.services.answer_generator import StructuredAnswerGenerator
from app.EvoRAG.services.dependency_graph import build_dependency_graph
from app.EvoRAG.services.retriever import EvoRAGRetriever, stored_entity_to_retrieved


class EvoRAGGraphGenerationService:
    def __init__(
        self,
        *,
        retriever: EvoRAGRetriever | Any | None = None,
        repository: MySQLEntityRepository | Any | None = None,
        answer_generator: StructuredAnswerGenerator | None = None,
        config: EvoRAGSettings = settings,
        scope: EntityScope | None = None,
    ) -> None:
        self.config = config
        self.scope = scope or EntityScope(
            workspace_id=config.default_workspace_id,
            project_id=config.default_project_id,
            collection_id=config.default_collection_id,
            domain=config.default_domain,
        )
        self.repository = repository or MySQLEntityRepository(config)
        self.retriever = retriever or EvoRAGRetriever(repository=self.repository, config=config, scope=self.scope)
        self.answer_generator = answer_generator or StructuredAnswerGenerator()

    async def candidates(self, entity_name: str, *, top_k: int | None = None) -> dict[str, object]:
        entities, warnings = await self.retriever.retrieve(entity_name, top_k=top_k)
        if not entities:
            return {
                "query": entity_name,
                "root_entity": None,
                "candidates": [],
                "default_selected_entity_ids": [],
                "graph": graph_payload(DependencyGraph(root_entity_id=0)),
                "warnings": warnings,
            }

        root = entities[0]
        graph = build_dependency_graph(entities, root_entity_id=root.id)
        return {
            "query": entity_name,
            "root_entity": retrieved_entity_payload(root),
            "candidates": [retrieved_entity_payload(entity) for entity in entities],
            "default_selected_entity_ids": [entity.id for entity in entities],
            "graph": graph_payload(graph),
            "warnings": warnings,
        }

    async def generate(
        self,
        *,
        root_entity_id: int,
        selected_entity_ids: list[int],
    ) -> dict[str, object]:
        entity_ids = ordered_unique_ints([root_entity_id, *selected_entity_ids])
        entities = await asyncio.to_thread(self._load_entities, entity_ids)
        if not entities:
            graph = DependencyGraph(root_entity_id=root_entity_id)
            return {
                "root_entity": None,
                "entities": [],
                "graph": graph_payload(graph),
                "outline": [],
                "article": "",
                "warnings": ["no selected entities found"],
            }

        root = next((entity for entity in entities if entity.id == root_entity_id), entities[0])
        graph = build_dependency_graph(entities, root_entity_id=root.id)
        article = self.answer_generator.generate(graph)
        return {
            "root_entity": retrieved_entity_payload(root),
            "entities": [retrieved_entity_payload(entity) for entity in entities],
            "graph": graph_payload(graph),
            "outline": outline_payload(graph),
            "article": article,
            "warnings": [],
        }

    def _load_entities(self, entity_ids: list[int]) -> list[RetrievedEntity]:
        stored_entities: list[StoredEntity] = []
        for entity_id in entity_ids:
            entity = self.repository.get_entity(entity_id)
            if entity is not None:
                stored_entities.append(entity)
        attributes_by_entity = self.repository.list_attributes_for_entities([entity.id for entity in stored_entities])
        return [
            stored_entity_to_retrieved(
                entity,
                attributes=attributes_by_entity.get(entity.id, []),
                score=1.0,
                rank=rank,
                source="selected",
            )
            for rank, entity in enumerate(stored_entities, start=1)
        ]


def ordered_unique_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def retrieved_entity_payload(entity: RetrievedEntity) -> dict[str, object]:
    return entity.model_dump()


def graph_payload(graph: DependencyGraph) -> dict[str, object]:
    return graph.model_dump()


def outline_payload(graph: DependencyGraph) -> list[dict[str, object]]:
    entity_by_id = {entity.id: entity for entity in graph.entities}
    ordered_ids = graph.ordered_entity_ids or [entity.id for entity in graph.entities]
    outline: list[dict[str, object]] = []
    for index, entity_id in enumerate(ordered_ids, start=1):
        entity = entity_by_id.get(entity_id)
        if entity is None:
            continue
        outline.append(
            {
                "entity_id": entity.id,
                "title": entity.canonical_name,
                "order": index,
                "sections": [attribute.attr_type for attribute in entity.attributes[:8]],
            }
        )
    return outline
