from dataclasses import dataclass, field
from typing import Literal


ResolutionDecision = Literal["matched", "new", "ambiguous"]


@dataclass(slots=True, frozen=True)
class EntityScope:
    workspace_id: str = "local"
    project_id: str = "evorag"
    collection_id: str = "default"
    domain: str = "general"

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "collection_id": self.collection_id,
            "domain": self.domain,
        }


@dataclass(slots=True)
class EntityAttributeInput:
    attr_type: str
    value_text: str
    evidence: str
    confidence: float = 0.7
    note_id: str = ""
    block_id: str = ""
    block_index: int = -1


@dataclass(slots=True)
class IncomingEntity:
    name: str
    normalized_name: str
    entity_type: str
    scope: EntityScope = field(default_factory=EntityScope)
    aliases: list[str] = field(default_factory=list)
    identity_description: str = ""
    attributes: list[EntityAttributeInput] = field(default_factory=list)
    description_for_match: str = ""
    source_count: int = 1
    embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class StoredEntity:
    id: int
    canonical_name: str
    normalized_name: str
    entity_type: str
    scope: EntityScope = field(default_factory=EntityScope)
    aliases: list[str] = field(default_factory=list)
    identity_description: str = ""
    summary: str = ""
    description_for_match: str = ""
    embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CandidateEntity:
    entity: StoredEntity
    score: float
    rank: int = 0
    source: str = ""
    vector_score: float = 0.0
    es_score: float = 0.0


@dataclass(slots=True)
class EntityResolutionDecision:
    incoming: IncomingEntity
    decision: ResolutionDecision
    matched_entity: StoredEntity | None = None
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class EntityUpsertResult:
    entity_id: int
    canonical_name: str
    created: bool
    attribute_count: int
    evidence_count: int
