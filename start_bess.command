#!/bin/bash
cd ~/bess_tool
source venv/bin/activate 2>/dev/null || true
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
open http://localhost:8000
