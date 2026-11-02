from threading import Lock
from uuid import uuid4
from datetime import datetime, timezone

_lock = Lock()
_jobs = {}

def create_job(payload):
    job_id = str(uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": payload, "result": None, "error": None
        }
    return job_id

def update_job(job_id, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)

def get_job(job_id):
    with _lock:
        item = _jobs.get(job_id)
        return dict(item) if item else None
