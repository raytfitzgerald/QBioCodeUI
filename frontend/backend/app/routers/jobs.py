from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(status: Optional[str] = None):
    jobs = job_manager.list_jobs(status)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(400, "Job cannot be cancelled")
    return {"cancelled": True}
