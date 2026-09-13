import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT.parent.parent
FRONTEND_ROOT = APP_ROOT / "frontend"
# print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import router as auth_router
from app.api.routes.evolution import router as evolution_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.notes import router as notes_router
from app.core.database import init_db
from app.core.config import settings
from app.core.retrieval import close_retrieval_backends, initialize_retrieval_backends

app = FastAPI(title=settings.app_name, debug=settings.app_debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")
app.include_router(notes_router, prefix="/api")
app.include_router(evolution_router, prefix="/api")


@app.on_event("startup")
def startup() -> None:
    init_db()
    initialize_retrieval_backends()


@app.on_event("shutdown")
def shutdown() -> None:
    close_retrieval_backends()

app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/notes")
def notes() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/evolution")
def evolution() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/info/markdown")
def markdown_info() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "info" / "markdown.html")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


