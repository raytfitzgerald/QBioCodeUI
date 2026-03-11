from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse

from ..services import dataset_service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("")
async def list_datasets():
    return {"datasets": dataset_service.list_datasets()}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, rows: Optional[int] = Query(None, description="Max rows to return (default 20, 0 = all)")):
    n = rows if rows is not None else 20
    data = dataset_service.get_dataset_preview(dataset_id, n_rows=n)
    if not data:
        raise HTTPException(404, "Dataset not found")
    return data


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    ds_id = dataset_service.save_uploaded_file(content, file.filename)
    meta = dataset_service.get_dataset(ds_id)
    return {"id": ds_id, "meta": meta}


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(404, "Dataset not found")
    return {"deleted": True}


@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    path = dataset_service.get_dataset_path(dataset_id)
    if not path or not path.exists():
        raise HTTPException(404, "Dataset file not found")
    return FileResponse(path, filename=path.name, media_type="text/csv")
