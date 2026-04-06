# Frontend

Minimal React + Vite + TypeScript debug UI for the Smear backend scaffold. The UI is intentionally simple and directly mirrors backend state so you can replace the placeholder backend logic later.

## Requirements

- Node.js 18+

## Install Dependencies

```bash
cd frontend
npm install
```

## Run the Vite Dev Server

```bash
npm run dev
```

The frontend runs at [http://127.0.0.1:5173](http://127.0.0.1:5173) by default and expects the FastAPI backend at [http://127.0.0.1:8000](http://127.0.0.1:8000).

If you want a different backend URL, set `VITE_API_BASE_URL` before starting Vite.
