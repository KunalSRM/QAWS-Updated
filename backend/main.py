from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime, timezone
import json

from backend.config import load_config, ROOT, project_path
from backend.schemas import RunRequest
from backend.state import create_job, update_job, get_job
from engine.runner import run_inference

app = FastAPI(title="QAWS API", version="1.0.0",
              description="Quantization-Aware Weight Streaming research prototype")
FRONTEND = ROOT / "frontend"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")

@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "QAWS"}

@app.get("/api/config")
def config():
    cfg = load_config()
    return {
        "project": cfg["project"]["name"],
        "models": cfg["model"]["registry"],
        "default_model": cfg["model"]["default"],
        "available_precisions": ["FP16"],
        "implementation_note": "FP16 inference is implemented in this presentation build. Quantized modes, true weight streaming and MPI are research extensions."
    }

def _execute(job_id, request_data, model_id):
    update_job(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())
    try:
        result = run_inference(model_id, request_data["prompt"], request_data["max_new_tokens"])
        record = {
            "experiment_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_key": request_data["model_key"],
            "model_id": model_id,
            "requested_precision": request_data["precision"],
            "result": result,
            "execution_status": {
                "fp16_inference": True,
                "int8_int4_int3_int2": False,
                "true_weight_streaming": False,
                "mpi_execution": False
            }
        }
        path = RESULTS / f"{job_id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        update_job(job_id, status="completed", result=record)
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))

@app.post("/api/run")
def run(req: RunRequest, background: BackgroundTasks):
    cfg = load_config()
    registry = cfg["model"]["registry"]
    if req.model_key not in registry:
        raise HTTPException(400, "Unknown model_key")
    job_id = create_job(req.model_dump())
    background.add_task(_execute, job_id, req.model_dump(), registry[req.model_key]["name"])
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    item = get_job(job_id)
    if not item:
        raise HTTPException(404, "Job not found")
    return item

@app.get("/api/results")
def results():
    records = []
    for p in sorted(RESULTS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records[:30]
