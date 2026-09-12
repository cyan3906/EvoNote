# Evonote FastAPI

FastAPI backend scaffold for Evonote.

## Quick start

```powershell
cd E:\huadian\evonote\backend\python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ for the Evonote login screen.
Open http://127.0.0.1:8000/evolution for the standalone self-evolving note interface after login.
Open http://127.0.0.1:8000/docs for the interactive API docs.

## Auth API

- `POST /api/auth/login`: submit a password and receive a bearer token for the current development login flow.

The default development password is `evonote2026`. Override it in `.env` before sharing or deploying the app:

```text
AUTH_PASSWORD=your-password
AUTH_TOKEN=replace-with-a-long-random-token
```

## Notes API

Evonote stores notes in SQLite by default:

```text
data/evonote.sqlite3
```

Override the path in `.env` when needed:

```text
SQLITE_DATABASE_PATH=data/evonote.sqlite3
```

- `GET /api/notes`: list notes.
- `POST /api/notes`: create a note.
- `GET /api/notes/{note_id}`: read one note.
- `PUT /api/notes/{note_id}`: update one note.
- `DELETE /api/notes/{note_id}`: delete one note.
- `GET /api/notes/{note_id}/export?format=txt`: export TXT.
- `GET /api/notes/{note_id}/export?format=pdf`: export PDF.
- `GET /api/notes/{note_id}/export?format=docx`: export Word.

PDF and Word export are generated with Python's standard library.

## Evolution API

Evonote maintains extra knowledge layers for every note:

- `L1`: the original Markdown note body.
- `L2`: a short summary for reading and model context.
- `L3`: keywords and a lightweight local vector for retrieval.
- `Claims`: small comparable knowledge points extracted from L1.
- `Patches`: merge suggestions that can append, mark duplicates, or record conflicts.

Related notes are found with L3 keywords/vectors and then compared at claim level. Suggestions are stored first, so risky changes can be reviewed before they alter a note.

- `GET /api/evolution/notes/{note_id}`: read L2/L3, blocks, claims, and pending suggestions.
- `POST /api/evolution/notes/{note_id}/scan`: rebuild analysis and generate merge suggestions.
- `GET /api/evolution/suggestions`: list pending suggestions.
- `POST /api/evolution/suggestions/{suggestion_id}/apply`: apply one patch suggestion.
- `POST /api/evolution/suggestions/{suggestion_id}/reject`: reject one patch suggestion.

Evolution data is stored in these SQLite tables:

- `note_representations`
- `note_blocks`
- `knowledge_claims`
- `merge_suggestions`
- `note_versions`

## Knowledge association API

The knowledge agent accepts one source knowledge item and compares it with candidate knowledge items. If candidates are omitted, the API uses the built-in five computer interview knowledge items as mock retrieval results.

- `GET /api/knowledge/defaults`: list the built-in five knowledge items.
- `GET /api/knowledge/graph`: demo association using the built-in source item.
- `POST /api/knowledge/associate`: submit one source knowledge item and optionally provide candidates.

Example request:

```json
{
  "source": {
    "id": "REQ-1",
    "title": "HTTP 与 TCP 的关系",
    "content": "HTTP 是应用层协议，通常基于 TCP 连接传输请求和响应。",
    "category": "计算机网络",
    "keywords": ["HTTP", "TCP", "连接"],
    "tags": ["网络"]
  }
}
```

Agent model configuration is read from `.env` through `app.core.config.settings`:

```text
AGENT_API_BASE_URL=...
AGENT_API_KEY=...
AGENT_MODEL=...
AGENT_TIMEOUT_SECONDS=30
```
