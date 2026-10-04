from dataclasses import dataclass, field
from typing import Literal


ResolutionDecision = Literal["matched", "new", "ambiguous"]
AttributeRetrievalMode = Literal["exact", "full_scan", "hybrid"]
AttributeDecisionAction = Literal["merge", "enrich", "update", "conflict", "add", "review"]


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
class StoredEntityAttribute:
    id: int
    entity_id: int
    scope: EntityScope
    attr_type: str
    value_text: str
    value_fingerprint: str
    confidence: float = 0.7
    status: str = "active"


@dataclass(slots=True)
class AttributeCandidate:
    attribute_id: int
    entity_id: int
    attr_type: str
    value_text: str
    confidence: float
    rank: int = 0
    source: str = ""
    es_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0


@dataclass(slots=True)
class AttributeRetrievalResult:
    input_index: int
    attr_type: str
    value_text: str
    mode: AttributeRetrievalMode
    group_size: int
    exact_match: StoredEntityAttribute | None = None
    candidates: list[AttributeCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    history_summaries: dict[int, dict] = field(default_factory=dict)


class AttributeRetrievalError(RuntimeError):
    pass
@dataclass(slots=True)
class AttributeDecision:
    input_index: int
    action: AttributeDecisionAction
    incoming_attribute: EntityAttributeInput
    target_attribute_id: int | None = None
    target_value_text: str = ""
    new_value_text: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class AttributeDecisionApplyResult:
    attribute_count: int = 0
    evidence_count: int = 0
    conflict_count: int = 0
    changed_attribute_ids: list[int] = field(default_factory=list)

@dataclass(slots=True)
class EntityResolutionDecision:
    incoming: IncomingEntity
    decision: ResolutionDecision
    matched_entity: StoredEntity | None = None
    score: float = 0.0
    reason: str = ""
    candidates: list[CandidateEntity] = field(default_factory=list)
    record_experience: bool = False
    experience_decision: str = ""
    experience_relation_type: str = ""
    experience_source: str = ""
    experience_confidence: float = 0.0


@dataclass(slots=True)
class EntityRelationMemoryRecord:
    id: int
    decision: str
    relation_type: str
    candidate: StoredEntity
    confidence: float = 1.0
    hit_count: int = 0
    source: str = ""
    reason: str = ""


@dataclass(slots=True)
class EntityUpsertResult:
    entity_id: int
    canonical_name: str
    created: bool
    attribute_count: int
    evidence_count: int
    conflict_count: int = 0
    changed_attribute_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class EntityIngestQueueResult:
    job_id: int
    status: str
    queued_count: int
    incoming_entity_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class IncomingEntityTask:
    id: int
    job_id: int
    incoming: IncomingEntity
    status: str
    attempt_count: int
