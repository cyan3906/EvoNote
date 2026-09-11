import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.core.config import settings

app = FastAPI(title=settings.app_name, debug=settings.app_debug)

app.include_router(health_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.app_name}"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
