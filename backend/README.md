# Backend

Minimal FastAPI scaffold for local Smear testing. The endpoint flow, models, and in-memory storage are real; the actual game rules and AI behavior are intentionally placeholder-only.

## Requirements

- Python 3.11+

## Create a Virtual Environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the API Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful endpoints:

- `GET /health`
- `POST /game/new`
- `GET /game/{game_id}`
- `GET /game/{game_id}/legal-actions`
- `POST /game/{game_id}/action`
- `POST /game/{game_id}/reset`
- `POST /game/{game_id}/ai-move`
- `GET /game/{game_id}/debug`

Most of the game-specific TODO hooks live in [app/mock_logic.py](/Users/danielmarkusson/Documents/GitHub/smear/backend/app/mock_logic.py).
