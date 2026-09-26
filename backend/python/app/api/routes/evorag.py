from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.models import EvoRAGIndexSearchResult, EvoRAGPreprocessResult, EvoRAGQueryResult
from app.EvoRAG.services import EvoRAGProcessor, EvoRAGQueryService, EvoRAGRetriever
from app.core.security import require_auth


router = APIRouter(prefix="/evorag", tags=["evorag"], dependencies=[Depends(require_auth)])


class EvoRAGScopePayload(BaseModel):
    workspace_id: str = "local"
    project_id: str = "evorag"
    collection_id: str = "default"
    domain: str = "general"


class EvoRAGIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    scope: EvoRAGScopePayload = Field(default_factory=EvoRAGScopePayload)


class EvoRAGQueryRequest(BaseModel):
    entity: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    scope: EvoRAGScopePayload = Field(default_factory=EvoRAGScopePayload)


class EvoRAGIngestResponse(BaseModel):
    preprocess: EvoRAGPreprocessResult
    ingest_results: list[dict[str, object]]
    warnings: list[str] = Field(default_factory=list)


@router.post("/extract", response_model=EvoRAGPreprocessResult)
async def extract_text(request: EvoRAGIngestRequest) -> EvoRAGPreprocessResult:
    try:
        return await EvoRAGProcessor().preprocess(request.text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/ingest", response_model=EvoRAGIngestResponse)
async def ingest_text(request: EvoRAGIngestRequest) -> EvoRAGIngestResponse:
    scope = to_scope(request.scope)
    try:
        preprocess = await EvoRAGProcessor().preprocess(request.text)
        ingest_results = await EntityIngestor(scope=scope).ingest(preprocess)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return EvoRAGIngestResponse(
        preprocess=preprocess,
        ingest_results=[asdict(result) for result in ingest_results],
    )


@router.post("/query", response_model=EvoRAGQueryResult)
async def query_entity(request: EvoRAGQueryRequest) -> EvoRAGQueryResult:
    try:
        return await EvoRAGQueryService(scope=to_scope(request.scope)).query(request.entity, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/index-search", response_model=EvoRAGIndexSearchResult)
async def search_entity_indexes(request: EvoRAGQueryRequest) -> EvoRAGIndexSearchResult:
    try:
        return await EvoRAGRetriever(scope=to_scope(request.scope)).search_indexes(request.entity, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


def to_scope(payload: EvoRAGScopePayload) -> EntityScope:
    return EntityScope(
        workspace_id=payload.workspace_id,
        project_id=payload.project_id,
        collection_id=payload.collection_id,
        domain=payload.domain,
    )
