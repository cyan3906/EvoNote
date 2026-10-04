import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.entity_store.worker import entity_worker_runtime_status
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
    note_id: str = ""
    task_name: str = ""
    scope: EvoRAGScopePayload = Field(default_factory=EvoRAGScopePayload)


class EvoRAGQueryRequest(BaseModel):
    entity: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    scope: EvoRAGScopePayload = Field(default_factory=EvoRAGScopePayload)


class EvoRAGIngestResponse(BaseModel):
    preprocess: EvoRAGPreprocessResult
    job_id: int = 0
    status: str = "queued"
    queued_count: int = 0
    incoming_entity_ids: list[int] = Field(default_factory=list)
    ingest_results: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvoRAGReviewTaskListResponse(BaseModel):
    tasks: list[dict[str, object]] = Field(default_factory=list)


class EvoRAGIngestJobListResponse(BaseModel):
    jobs: list[dict[str, object]] = Field(default_factory=list)


class EvoRAGReviewMergeRequest(BaseModel):
    entity_id: int = Field(..., ge=1)
    reason: str = ""
    decided_by: str = "manual"


class EvoRAGReviewActionRequest(BaseModel):
    reason: str = ""
    decided_by: str = "manual"


class EvoRAGReviewRejectRequest(EvoRAGReviewActionRequest):
    candidate_entity_ids: list[int] = Field(default_factory=list)


@router.post("/extract", response_model=EvoRAGPreprocessResult)
async def extract_text(request: EvoRAGIngestRequest) -> EvoRAGPreprocessResult:
    try:
        return await EvoRAGProcessor().preprocess(request.text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/debug/extract")
async def debug_extract_text(request: EvoRAGIngestRequest) -> dict[str, object]:
    try:
        preprocess = await EvoRAGProcessor().preprocess(request.text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return debug_extract_payload(preprocess)


def debug_extract_payload(preprocess: EvoRAGPreprocessResult) -> dict[str, object]:
    block_split_blocks: list[dict[str, object]] = []
    entity_extraction_blocks: list[dict[str, object]] = []

    for block_result in preprocess.blocks:
        block = block_result.block.model_dump()
        block_split_blocks.append(block)
        entity_extraction_blocks.append(
            {
                "block_index": block_result.block.block_index,
                "heading": block_result.block.heading,
                "anchor_entity": block_result.block.anchor_entity,
                "entities": [entity.model_dump() for entity in block_result.entities],
                "warnings": list(block_result.warnings),
            }
        )

    return {
        "input_text": preprocess.input_text,
        "block_split": {"blocks": block_split_blocks},
        "entity_extraction": {"blocks": entity_extraction_blocks},
        "timings": preprocess.timings,
    }


@router.post("/ingest", response_model=EvoRAGIngestResponse)
async def ingest_text(request: EvoRAGIngestRequest) -> EvoRAGIngestResponse:
    scope = to_scope(request.scope)
    try:
        preprocess = await EvoRAGProcessor().preprocess(request.text)
        queue_result = await EntityIngestor(scope=scope).queue(
            preprocess,
            source_note_id=request.note_id,
            task_name=request.task_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return EvoRAGIngestResponse(
        preprocess=preprocess,
        job_id=queue_result.job_id,
        status=queue_result.status,
        queued_count=queue_result.queued_count,
        incoming_entity_ids=queue_result.incoming_entity_ids,
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


@router.get("/ingest-jobs", response_model=EvoRAGIngestJobListResponse)
async def list_ingest_jobs(limit: int = 20) -> EvoRAGIngestJobListResponse:
    repository = MySQLEntityRepository()
    try:
        jobs = await asyncio.to_thread(repository.list_ingest_jobs, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return EvoRAGIngestJobListResponse(jobs=jobs)


@router.get("/ingest-jobs/{job_id}")
async def get_ingest_job(job_id: int) -> dict[str, object]:
    repository = MySQLEntityRepository()
    try:
        job = await asyncio.to_thread(repository.get_ingest_job_status, job_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ingest job not found")
    return job


@router.get("/worker-status")
async def get_worker_status() -> dict[str, object]:
    repository = MySQLEntityRepository()
    response: dict[str, object] = {
        "worker": entity_worker_runtime_status(),
        "queue": {"available": False, "backend": "mysql"},
        "mysql": {},
        "warnings": [],
    }
    try:
        response["mysql"] = await asyncio.to_thread(repository.ingest_observability_summary)
    except Exception as exc:
        response["warnings"].append(f"mysql summary unavailable: {exc}")  # type: ignore[index]
    return response


@router.get("/review-tasks", response_model=EvoRAGReviewTaskListResponse)
async def list_review_tasks(status_filter: str = Query("pending", alias="status"), limit: int = 50) -> EvoRAGReviewTaskListResponse:
    ingestor = EntityIngestor()
    try:
        tasks = await asyncio.to_thread(ingestor.repository.list_review_tasks, status=status_filter, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return EvoRAGReviewTaskListResponse(tasks=tasks)


@router.get("/review-tasks/{review_task_id}")
async def get_review_task(review_task_id: int) -> dict[str, object]:
    ingestor = EntityIngestor()
    try:
        task = await asyncio.to_thread(ingestor.repository.get_review_task, review_task_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review task not found")
    return task


@router.post("/review-tasks/{review_task_id}/merge")
async def merge_review_task(review_task_id: int, request: EvoRAGReviewMergeRequest) -> dict[str, object]:
    try:
        result = await EntityIngestor().manual_merge_review_task(
            review_task_id,
            entity_id=request.entity_id,
            reason=request.reason,
            decided_by=request.decided_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return {"status": "merged", "result": asdict(result)}


@router.post("/review-tasks/{review_task_id}/new")
async def create_new_from_review_task(review_task_id: int, request: EvoRAGReviewActionRequest) -> dict[str, object]:
    try:
        result = await EntityIngestor().manual_create_new_review_task(
            review_task_id,
            reason=request.reason,
            decided_by=request.decided_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return {"status": "new_created", "result": asdict(result)}


@router.post("/review-tasks/{review_task_id}/reject")
async def reject_review_task_candidates(review_task_id: int, request: EvoRAGReviewRejectRequest) -> dict[str, object]:
    try:
        result = await EntityIngestor().reject_review_candidates(
            review_task_id,
            candidate_entity_ids=request.candidate_entity_ids,
            reason=request.reason,
            decided_by=request.decided_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return result


def to_scope(payload: EvoRAGScopePayload) -> EntityScope:
    return EntityScope(
        workspace_id=payload.workspace_id,
        project_id=payload.project_id,
        collection_id=payload.collection_id,
        domain=payload.domain,
    )
