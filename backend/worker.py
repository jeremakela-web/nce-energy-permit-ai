"""
Standalone ARQ worker entry point — for local debugging / future multi-service use.

In production (single-service mode), the worker starts automatically inside the
FastAPI process via @app.on_event("startup") in main.py. You do NOT need to run
this file separately.

To run standalone (e.g., for load testing a separate worker process):
    cd backend
    arq worker.WorkerSettings

Or with explicit Redis URL:
    REDIS_URL=redis://... arq worker.WorkerSettings
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "permit_ai"))

from arq.connections import RedisSettings
from main import arq_task_generate_permit

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class WorkerSettings:
    functions  = [arq_task_generate_permit]
    redis_settings = RedisSettings.from_dsn(_REDIS_URL)
    max_jobs   = 2
    handle_signals = True
    poll_delay = 0.5
