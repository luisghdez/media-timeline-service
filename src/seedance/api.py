from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from .client import SeedanceClient, SeedanceError
from .schema import GenerateRequest, GenerateResult, JobAccepted, JobStatus

def _timeout() -> float:
    raw = os.getenv("SEEDANCE_CLIENT_TIMEOUT")
    return float(raw) if raw else 600.0


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="Seedance Text-to-Video", version="0.1.0")
client = SeedanceClient(client_timeout=_timeout())


def _http_error(exc: SeedanceError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResult)
async def generate(request: GenerateRequest) -> GenerateResult:
    try:
        return await client.generate(request)
    except SeedanceError as exc:
        raise _http_error(exc) from exc


@app.post("/v1/jobs", response_model=JobAccepted)
async def submit_job(request: GenerateRequest) -> JobAccepted:
    try:
        return await client.submit(request)
    except SeedanceError as exc:
        raise _http_error(exc) from exc


@app.get("/v1/jobs/{request_id}", response_model=JobStatus)
async def job_status(request_id: str, logs: bool = Query(True)) -> JobStatus:
    try:
        return await client.status(request_id, with_logs=logs)
    except SeedanceError as exc:
        raise _http_error(exc) from exc


@app.get("/v1/jobs/{request_id}/result", response_model=GenerateResult)
async def job_result(request_id: str) -> GenerateResult:
    try:
        return await client.result(request_id)
    except SeedanceError as exc:
        raise _http_error(exc) from exc
