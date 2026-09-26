from typing import Literal

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.EvoRAG.constants import ATTRIBUTE_TYPES


AttributeType = Literal[
    "definition",
    "purpose",
    "core_idea",
    "mechanism",
    "components",
    "constraints",
    "related",
]


class AttributeValue(BaseModel):
    value: str = Field(default="", description="Attribute value extracted from the block.")
    evidence: str = Field(default="", description="Evidence copied from the block text.")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_evidence(cls, data: Any) -> Any:
        if isinstance(data, dict) and "evidence" not in data and "source_text" in data:
            data = {**data, "evidence": data.get("source_text", "")}
        return data

    @field_validator("value", "evidence")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class AttributeBucket(BaseModel):
    definition: list[AttributeValue] = Field(default_factory=list)
    purpose: list[AttributeValue] = Field(default_factory=list)
    core_idea: list[AttributeValue] = Field(default_factory=list)
    mechanism: list[AttributeValue] = Field(default_factory=list)
    components: list[AttributeValue] = Field(default_factory=list)
    constraints: list[AttributeValue] = Field(default_factory=list)
    related: list[AttributeValue] = Field(default_factory=list)

    def iter_values(self) -> list[tuple[AttributeType, AttributeValue]]:
        items: list[tuple[AttributeType, AttributeValue]] = []
        for attr_type in ATTRIBUTE_TYPES:
            values = getattr(self, attr_type)
            items.extend((attr_type, value) for value in values)
        return items


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = Field(default="concept")
    aliases: list[str] = Field(default_factory=list)
    identity_description: str = Field(
        default="",
        description="Short model-generated description used only for entity identity resolution.",
    )
    attributes: AttributeBucket = Field(default_factory=AttributeBucket)

    @field_validator("name", "entity_type", "identity_description")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in values:
            alias = " ".join(str(item or "").split())
            if alias and alias not in seen:
                seen.add(alias)
                cleaned.append(alias)
        return cleaned


class TextBlock(BaseModel):
    block_index: int = Field(ge=0)
    heading: str = ""
    l1_text: str

    @field_validator("heading", "l1_text")
    @classmethod
    def clean_block_text(cls, value: str) -> str:
        return str(value or "").strip()


class BlockSplitResult(BaseModel):
    blocks: list[TextBlock] = Field(default_factory=list)


class BlockEntityExtraction(BaseModel):
    block: TextBlock
    entities: list[ExtractedEntity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BlockExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvoRAGPreprocessResult(BaseModel):
    input_text: str
    blocks: list[BlockEntityExtraction] = Field(default_factory=list)

    @property
    def entity_count(self) -> int:
        return sum(len(block.entities) for block in self.blocks)


class StoredAttributeEvidence(BaseModel):
    evidence_text: str = ""
    note_id: str = ""
    block_id: str = ""
    block_index: int = -1


class StoredAttribute(BaseModel):
    id: int = 0
    attr_type: AttributeType
    value_text: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: list[StoredAttributeEvidence] = Field(default_factory=list)


class RetrievedEntity(BaseModel):
    id: int
    canonical_name: str
    normalized_name: str = ""
    entity_type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    identity_description: str = ""
    summary: str = ""
    description_for_match: str = ""
    score: float = 0.0
    rank: int = 0
    source: str = ""
    attributes: list[StoredAttribute] = Field(default_factory=list)

    def attributes_by_type(self, attr_type: AttributeType) -> list[StoredAttribute]:
        return [attribute for attribute in self.attributes if attribute.attr_type == attr_type]


class DependencyEdge(BaseModel):
    source_id: int
    target_id: int
    relation: str
    graph_type: Literal["semantic", "conditional"]
    evidence: str = ""
    weight: float = Field(default=1.0, ge=0.0)


class DependencyGraph(BaseModel):
    root_entity_id: int
    entities: list[RetrievedEntity] = Field(default_factory=list)
    semantic_edges: list[DependencyEdge] = Field(default_factory=list)
    conditional_edges: list[DependencyEdge] = Field(default_factory=list)
    ordered_entity_ids: list[int] = Field(default_factory=list)
    cycles: list[list[int]] = Field(default_factory=list)


class EvoRAGQueryResult(BaseModel):
    query: str
    root_entity: RetrievedEntity | None = None
    retrieved_entities: list[RetrievedEntity] = Field(default_factory=list)
    graph: DependencyGraph | None = None
    answer: str = ""
    warnings: list[str] = Field(default_factory=list)
