"""QBioCode UI — FastAPI backend application."""

# ── Environment fixes (MUST be first) ────────────────────────────────────
# These prevent grpcio mutex deadlocks on macOS and suppress noisy logs.
# They must be set before *any* library that touches grpc is imported.
import os

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("MPLBACKEND", "Agg")

# ── Pre-register qbiocode stubs so __init__.py never runs ────────────────
from .qbiocode_loader import install_lazy_stubs  # noqa: E402

install_lazy_stubs()

# ── Normal imports ───────────────────────────────────────────────────────
import asyncio  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .config import settings  # noqa: E402
from .services.job_manager import job_manager  # noqa: E402
from .routers import datasets, generation, evaluation, models, embeddings, profiler, sage, jobs  # noqa: E402


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
