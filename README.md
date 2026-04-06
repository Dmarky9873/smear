# smear

Local full-stack testing scaffold for a Smear card-game project.

## Layout

- `backend/`: FastAPI + Pydantic mock game API with in-memory storage
- `frontend/`: React + Vite + TypeScript debug UI

## Quick Start

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://127.0.0.1:8000` by default.
