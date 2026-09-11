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

Open http://127.0.0.1:8000/docs for the interactive API docs.

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