from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel

from app.core import database
from app.core.evolution import scan_note
from app.core.exports import note_filename, note_to_docx, note_to_pdf, note_to_text
from app.core.security import require_auth

router = APIRouter(prefix="/notes", tags=["notes"], dependencies=[Depends(require_auth)])


class NotePayload(BaseModel):
    title: str = ""
    tags: str = ""
    body: str = ""


class NoteResponse(NotePayload):
    id: str
    created_at: str
    updated_at: str


@router.get("", response_model=list[NoteResponse])
def list_notes(query: str | None = None) -> list[dict[str, str]]:
    return database.list_notes(query=query)


@router.post("", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
def create_note(payload: NotePayload) -> dict[str, str]:
    note = database.create_note(
        title=payload.title,
        tags=payload.tags,
        body=payload.body,
    )
    scan_note(note["id"])
    return note


@router.get("/{note_id}", response_model=NoteResponse)
def get_note(note_id: str) -> dict[str, str]:
    note = database.get_note(note_id)

    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    scan_note(note["id"])
    return note


@router.put("/{note_id}", response_model=NoteResponse)
def update_note(note_id: str, payload: NotePayload) -> dict[str, str]:
    note = database.update_note(
        note_id=note_id,
        title=payload.title,
        tags=payload.tags,
        body=payload.body,
    )

    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    return note


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: str) -> Response:
    deleted = database.delete_note(note_id)

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{note_id}/export")
def export_note(note_id: str, format: str = Query("txt", pattern="^(txt|pdf|docx)$")) -> Response:
    note = database.get_note(note_id)

    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    if format == "txt":
        content = note_to_text(note).encode("utf-8")
        media_type = "text/plain; charset=utf-8"
    elif format == "pdf":
        try:
            content = note_to_pdf(note)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        media_type = "application/pdf"
    else:
        try:
            content = note_to_docx(note)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    filename = note_filename(note, format)

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
