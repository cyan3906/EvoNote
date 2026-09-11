# Evonote Frontend

Frontend for the Evonote note app. It includes the login screen, note workspace, and Markdown editor preview.

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

## Test

```powershell
cd E:\huadian\evonote\frontend
node .\tests\markdown.test.mjs
```
