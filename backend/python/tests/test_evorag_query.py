from app.EvoRAG.models import RetrievedEntity, StoredAttribute
from app.EvoRAG.services.answer_generator import StructuredAnswerGenerator
from app.EvoRAG.services.dependency_graph import build_dependency_graph


def entity(entity_id: int, name: str, attributes: list[StoredAttribute]) -> RetrievedEntity:
    return RetrievedEntity(
        id=entity_id,
        canonical_name=name,
        normalized_name=name.lower(),
        entity_type="concept",
        identity_description=f"{name} identity",
        summary=f"{name} summary",
        attributes=attributes,
    )


def attribute(attr_type: str, value: str) -> StoredAttribute:
    return StoredAttribute(attr_type=attr_type, value_text=value, confidence=0.9)


def test_dependency_graph_extracts_semantic_and_conditional_edges() -> None:
    transformer = entity(
        1,
        "Transformer",
        [
            attribute("mechanism", "Transformer uses Attention to model token relations."),
            attribute("constraints", "Transformer depends on positional encoding when sequence order matters."),
        ],
    )
    attention = entity(2, "Attention", [attribute("related", "Attention is part of Transformer.")])
    positional_encoding = entity(3, "positional encoding", [attribute("definition", "Position signal for a sequence.")])

    graph = build_dependency_graph([transformer, attention, positional_encoding], root_entity_id=1)

    assert any(edge.source_id == 1 and edge.target_id == 2 for edge in graph.semantic_edges)
    assert any(edge.source_id == 1 and edge.target_id == 3 for edge in graph.conditional_edges)
    assert graph.cycles == [[1, 2]]
    assert graph.ordered_entity_ids[0] == 1


def test_structured_answer_generator_outputs_ordered_sections_and_cycles() -> None:
    transformer = entity(
        1,
        "Transformer",
        [
            attribute("definition", "Transformer is a neural network architecture."),
            attribute("related", "Transformer uses Attention."),
        ],
    )
    attention = entity(2, "Attention", [attribute("definition", "Attention scores token relevance for Transformer.")])
    graph = build_dependency_graph([transformer, attention], root_entity_id=1)

    answer = StructuredAnswerGenerator().generate(graph)

    assert "# Transformer" in answer
    assert "## Transformer" in answer
    assert "### 定义" in answer
    assert "Transformer -> Attention" in answer
    assert "环处理" in answer
