from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents.knowledge_agent import (
    DEFAULT_KNOWLEDGE_ITEMS,
    DEFAULT_SOURCE_KNOWLEDGE,
    KnowledgeAssociationAgent,
    KnowledgeItem,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeItemPayload(BaseModel):
    id: str | None = None
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    category: str = "计算机基础"
    keywords: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class AssociateKnowledgeRequest(BaseModel):
    source: KnowledgeItemPayload
    candidates: tuple[KnowledgeItemPayload, ...] | None = None


class KnowledgeItemResponse(BaseModel):
    id: str
    title: str
    content: str
    category: str
    keywords: tuple[str, ...]
    tags: tuple[str, ...]


class KnowledgeRelationResponse(BaseModel):
    source_id: str
    target_id: str
    relation: str
    reason: str
    evidence: tuple[str, ...]
    confidence: float


class KnowledgeAssociationResponse(BaseModel):
    source: KnowledgeItemResponse
    candidates: tuple[KnowledgeItemResponse, ...]
    relations: tuple[KnowledgeRelationResponse, ...]
    merged_content: str
    prompt: str


def _to_knowledge_item(payload: KnowledgeItemPayload) -> KnowledgeItem:
    return KnowledgeItem(
        id=payload.id or payload.title,
        title=payload.title,
        content=payload.content,
        category=payload.category,
        keywords=payload.keywords,
        tags=payload.tags,
    )


def _to_response(result) -> KnowledgeAssociationResponse:
    return KnowledgeAssociationResponse(
        source=KnowledgeItemResponse(**vars(result.source)),
        candidates=tuple(KnowledgeItemResponse(**vars(candidate)) for candidate in result.candidates),
        relations=tuple(KnowledgeRelationResponse(**vars(relation)) for relation in result.relations),
        merged_content=result.merged_content,
        prompt=result.prompt,
    )


@router.get("/defaults", response_model=tuple[KnowledgeItemResponse, ...])
def list_default_knowledge() -> tuple[KnowledgeItemResponse, ...]:
    return tuple(KnowledgeItemResponse(**vars(item)) for item in DEFAULT_KNOWLEDGE_ITEMS)


@router.get("/graph", response_model=KnowledgeAssociationResponse)
def get_demo_knowledge_graph() -> KnowledgeAssociationResponse:
    result = KnowledgeAssociationAgent().associate(
        source=DEFAULT_SOURCE_KNOWLEDGE,
        candidates=DEFAULT_KNOWLEDGE_ITEMS,
    )
    return _to_response(result)


@router.post("/associate", response_model=KnowledgeAssociationResponse)
@router.get("/associate", response_model=KnowledgeAssociationResponse)
def associate_knowledge(request: AssociateKnowledgeRequest) -> KnowledgeAssociationResponse:
    source = _to_knowledge_item(request.source)
    candidates = None

    if request.candidates is not None:
        candidates = tuple(_to_knowledge_item(candidate) for candidate in request.candidates)

    result = KnowledgeAssociationAgent().associate(source=source, candidates=candidates)
    return _to_response(result)
