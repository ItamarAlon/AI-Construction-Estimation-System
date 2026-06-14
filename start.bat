@echo off
start "Backend" cmd /k ".venv\Scripts\uvicorn server:app --reload"
start "Frontend" cmd /k "cd ui && npm run dev"
