# Evonote Frontend

Frontend for the Evonote note app. It includes the login screen, Markdown note workspace, export controls, and a separate self-evolving note interface.

## Structure

```text
frontend/
  index.html
  src/
    app.mjs
    markdown.mjs
  styles/
    main.css
  tests/
    markdown.test.mjs
```

## Run

Start the FastAPI backend first:

```powershell
cd E:\huadian\evonote\backend\python
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/ in a browser. The backend serves the login screen as the default note-space entry.

The default development password is `evonote2026`.

Notes are stored by the backend SQLite database, not browser cache. Typing still autosaves
the raw Markdown note. Clicking `保存` submits the current note for L2/L3/Claims analysis,
hybrid retrieval indexing, and merge-suggestion generation.

The main workspace remains focused on Markdown writing. Click `智能整理` or open `/evolution` to view L2 summaries, L3 retrieval keywords, extracted claims, and merge suggestions for the active note.

## Test

```powershell
cd E:\huadian\evonote\frontend
node .\tests\markdown.test.mjs
```
