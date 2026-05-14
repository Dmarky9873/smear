from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .api import router
except ImportError:
    from api import router


def load_cors_origins() -> list[str]:
    configured_origins = os.getenv("SMEAR_CORS_ORIGINS")
    if configured_origins:
        origins = [
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
        if origins:
            return origins

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]


def create_app() -> FastAPI:
    app = FastAPI(title="Smear API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=load_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
