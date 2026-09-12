from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core import database
from app.core.evolution import (
    EvolutionModelError,
    apply_suggestion,
    get_note_evolution_state,
    reject_suggestion,
    scan_note,
)
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
    representation: RepresentationResponse
    blocks: list[BlockResponse]
    claims: list[ClaimResponse]
    suggestions: list[SuggestionResponse]


def _state_or_404(note_id: str) -> dict[str, object]:
    try:
        state = get_note_evolution_state(note_id)
    except EvolutionModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    return state


@router.get("/notes/{note_id}", response_model=EvolutionStateResponse)
def read_note_evolution(note_id: str) -> dict[str, object]:
    return _state_or_404(note_id)


@router.post("/notes/{note_id}/scan", response_model=EvolutionStateResponse)
def scan_note_evolution(note_id: str) -> dict[str, object]:
    try:
        state = scan_note(note_id)
    except EvolutionModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    return state


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
