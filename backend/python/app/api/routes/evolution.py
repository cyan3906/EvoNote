from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core import database
from app.core.evolution import apply_suggestion, get_note_evolution_snapshot, reject_suggestion
from app.core.evolution_jobs import get_scan_job, queue_note_scan
from app.core.retrieval import hybrid_retrieve_claims, rebuild_hybrid_indexes
from app.core.security import require_auth

router = APIRouter(prefix="/evolution", tags=["evolution"], dependencies=[Depends(require_auth)])


class RepresentationResponse(BaseModel):
    note_id: str
    l2_summary: str
    l3_text: str
    keywords: list[str]
    vector: list[float]
    updated_at: str | None = None


class BlockResponse(BaseModel):
    id: str
    note_id: str
    block_index: int
    heading: str
    l1_text: str
    l2_summary: str
    l3_text: str
    keywords: list[str]
    created_at: str | None = None


class ClaimResponse(BaseModel):
    id: str
    note_id: str
    block_id: str
    claim_index: int
    claim_text: str
    subject: str
    predicate: str
    object_text: str
    source_text: str
    keywords: list[str]
    vector: list[float]
    confidence: float
    created_at: str | None = None


class SuggestionResponse(BaseModel):
    id: str
    source_note_id: str
    target_note_id: str
    source_claim_id: str | None = None
    target_claim_id: str | None = None
    relation: str
    confidence: float
    risk_level: str
    reason: str
    patch: dict[str, Any]
    status: str
    created_at: str
    updated_at: str


class EvolutionStateResponse(BaseModel):
    note: dict[str, str]
    representation: RepresentationResponse | None = None
    blocks: list[BlockResponse]
    claims: list[ClaimResponse]
    suggestions: list[SuggestionResponse]
    stale: bool = False
    analysis_status: str = "ready"
    error: str = ""


class RetrievalHitResponse(BaseModel):
    claim_id: str
    note_id: str
    score: float
    rank: int
    source: str
    title: str = ""
    reason: str = ""


class IndexRebuildResponse(BaseModel):
    indexed_notes: int
    indexed_claims: int
    errors: list[str]


class ScanJobResponse(BaseModel):
    id: str
    note_id: str
    status: str
    error: str = ""
    created_at: str
    updated_at: str


def _state_or_404(note_id: str) -> dict[str, object]:
    state = get_note_evolution_snapshot(note_id)

    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    job = get_scan_job(note_id)
    state["analysis_status"] = job.status if job else ("stale" if state.get("stale") else "ready")
    state["error"] = job.error if job and job.status == "failed" else ""
    return state


@router.get("/notes/{note_id}", response_model=EvolutionStateResponse)
def read_note_evolution(note_id: str) -> dict[str, object]:
    return _state_or_404(note_id)


@router.post("/notes/{note_id}/scan", response_model=ScanJobResponse, status_code=status.HTTP_202_ACCEPTED)
def scan_note_evolution(note_id: str) -> dict[str, object]:
    if not database.get_note(note_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    return queue_note_scan(note_id).to_dict()


@router.get("/notes/{note_id}/scan/status", response_model=ScanJobResponse)
def read_scan_status(note_id: str) -> dict[str, object]:
    if not database.get_note(note_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    job = get_scan_job(note_id)

    if not job:
        return {
            "id": "",
            "note_id": note_id,
            "status": "idle",
            "error": "",
            "created_at": "",
            "updated_at": "",
        }

    return job.to_dict()


@router.get("/notes/{note_id}/retrieve", response_model=list[RetrievalHitResponse])
def retrieve_related_claims(note_id: str) -> list[dict[str, object]]:
    state = _state_or_404(note_id)

    if not state.get("representation"):
        return []

    hits = hybrid_retrieve_claims(
        state["representation"],
        exclude_note_id=note_id,
    )
    return [hit.to_dict() for hit in hits]


@router.post("/index/rebuild", response_model=IndexRebuildResponse)
def rebuild_indexes() -> dict[str, object]:
    return rebuild_hybrid_indexes()


@router.get("/suggestions", response_model=list[SuggestionResponse])
def list_suggestions(status_filter: str = "pending") -> list[dict[str, object]]:
    return database.list_merge_suggestions(status=status_filter)


@router.post("/suggestions/{suggestion_id}/apply", response_model=SuggestionResponse)
def apply_merge_suggestion(suggestion_id: str) -> dict[str, object]:
    suggestion = apply_suggestion(suggestion_id)

    if not suggestion:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="合并建议不存在")

    return suggestion


@router.post("/suggestions/{suggestion_id}/reject", response_model=SuggestionResponse)
def reject_merge_suggestion(suggestion_id: str) -> dict[str, object]:
    suggestion = reject_suggestion(suggestion_id)

    if not suggestion:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="合并建议不存在")

    return suggestion

