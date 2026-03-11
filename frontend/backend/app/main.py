import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .services.job_manager import job_manager
from .routers import datasets, generation, evaluation, models, embeddings, profiler, sage, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.set_loop(asyncio.get_event_loop())
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)
app.include_router(generation.router)
app.include_router(evaluation.router)
app.include_router(models.router)
app.include_router(embeddings.router)
app.include_router(profiler.router)
app.include_router(sage.router)
app.include_router(jobs.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str):
    await websocket.accept()
    await job_manager.register_ws(job_id, websocket)

    # Send current state immediately
    job = job_manager.get_job(job_id)
    if job:
        await websocket.send_json(job.to_dict())

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await job_manager.unregister_ws(job_id, websocket)
