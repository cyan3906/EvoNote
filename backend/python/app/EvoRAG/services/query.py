from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.models import EvoRAGQueryResult
from app.EvoRAG.services.answer_generator import StructuredAnswerGenerator
from app.EvoRAG.services.dependency_graph import build_dependency_graph
from app.EvoRAG.services.retriever import EvoRAGRetriever


class EvoRAGQueryService:
    def __init__(
        self,
        retriever: EvoRAGRetriever | None = None,
        answer_generator: StructuredAnswerGenerator | None = None,
        config: EvoRAGSettings = settings,
        scope: EntityScope | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever or EvoRAGRetriever(config=config, scope=scope)
        self.answer_generator = answer_generator or StructuredAnswerGenerator()

    async def query(self, entity_name: str, *, top_k: int | None = None) -> EvoRAGQueryResult:
        entities, warnings = await self.retriever.retrieve(entity_name, top_k=top_k)
        if not entities:
            return EvoRAGQueryResult(query=entity_name, warnings=warnings)

        root = entities[0]
        graph = build_dependency_graph(entities, root_entity_id=root.id)
        answer = self.answer_generator.generate(graph)
        return EvoRAGQueryResult(
            query=entity_name,
            root_entity=root,
            retrieved_entities=entities,
            graph=graph,
            answer=answer,
            warnings=warnings,
        )
