from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


class GameEventHub:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[session_id].add(websocket)

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            session_connections = self._connections.get(session_id)
            if session_connections is None:
                return
            session_connections.discard(websocket)
            if not session_connections:
                self._connections.pop(session_id, None)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        async with self._lock:
            connections = list(self._connections.get(session_id, set()))

        stale_connections: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale_connections.append(websocket)

        if stale_connections:
            async with self._lock:
                session_connections = self._connections.get(session_id)
                if session_connections is None:
                    return
                for websocket in stale_connections:
                    session_connections.discard(websocket)
                if not session_connections:
                    self._connections.pop(session_id, None)


game_events = GameEventHub()
