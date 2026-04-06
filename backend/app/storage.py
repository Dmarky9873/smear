from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.mock_logic import create_mock_game_state
from app.models import GameState, NewGameRequest, ResetGameRequest


@dataclass
class StoredGame:
    request: NewGameRequest
    state: GameState
    created_at: str
    updated_at: str
    reset_count: int = 0


class GameNotFoundError(KeyError):
    """Raised when a requested in-memory game does not exist."""


_GAMES: dict[str, StoredGame] = {}


def create_game(request: NewGameRequest) -> GameState:
    game_id = uuid4().hex[:8]
    state = create_mock_game_state(game_id, request)
    timestamp = _utc_now()
    _GAMES[game_id] = StoredGame(
        request=request.model_copy(deep=True),
        state=state.model_copy(deep=True),
        created_at=timestamp,
        updated_at=timestamp,
    )
    return state


def get_game(game_id: str) -> GameState:
    return _get_record(game_id).state.model_copy(deep=True)


def save_game(state: GameState) -> GameState:
    record = _get_record(state.game_id)
    record.state = state.model_copy(deep=True)
    record.updated_at = _utc_now()
    return record.state.model_copy(deep=True)


def reset_game(game_id: str, request: ResetGameRequest | None = None) -> GameState:
    record = _get_record(game_id)
    base_request = record.request.model_copy(deep=True)

    if request is not None and request.seed is not None:
        base_request.seed = request.seed
        record.request.seed = request.seed

    state = create_mock_game_state(game_id, base_request)
    state.debug["resets"] = record.reset_count + 1
    record.state = state.model_copy(deep=True)
    record.updated_at = _utc_now()
    record.reset_count += 1
    return state


def get_game_metadata(game_id: str) -> dict[str, object]:
    record = _get_record(game_id)
    return {
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "reset_count": record.reset_count,
        "initial_request": record.request.model_dump(),
        "in_memory_store_size": len(_GAMES),
    }


def _get_record(game_id: str) -> StoredGame:
    try:
        return _GAMES[game_id]
    except KeyError as error:
        raise GameNotFoundError(f"Game '{game_id}' was not found in memory.") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
