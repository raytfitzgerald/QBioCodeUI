from __future__ import annotations

import asyncio
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import WebSocket


@dataclass
class Job:
    id: str
    type: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.ws_connections: dict[str, set[WebSocket]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def create_job(self, job_type: str, params: dict | None = None) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            type=job_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            params=params or {},
        )
        self.jobs[job_id] = job
        return job

    async def run_job(self, job: Job, fn: Callable, *args, **kwargs):
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        await self._broadcast(job.id, job.to_dict())

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fn(*args, progress_callback=self._make_sync_callback(job.id), **kwargs)
            )
            job.status = "completed"
            job.progress = 1.0
            job.result = result if isinstance(result, dict) else {"data": str(result)}
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.message = "Completed successfully"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.message = f"Failed: {e}"
            traceback.print_exc()

        await self._broadcast(job.id, job.to_dict())

    def _make_sync_callback(self, job_id: str):
        def callback(progress: float, message: str = ""):
            job = self.jobs.get(job_id)
            if job:
                job.progress = progress
                job.message = message
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast(job_id, job.to_dict()), self._loop
                    )
        return callback

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_jobs(self, status: str | None = None) -> list[Job]:
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status == "running":
            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    async def register_ws(self, job_id: str, ws: WebSocket):
        if job_id not in self.ws_connections:
            self.ws_connections[job_id] = set()
        self.ws_connections[job_id].add(ws)

    async def unregister_ws(self, job_id: str, ws: WebSocket):
        if job_id in self.ws_connections:
            self.ws_connections[job_id].discard(ws)

    async def _broadcast(self, job_id: str, data: dict):
        if job_id not in self.ws_connections:
            return
        dead = set()
        for ws in self.ws_connections[job_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.ws_connections[job_id] -= dead


job_manager = JobManager()
