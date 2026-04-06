from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.engine import (
    apply_ai_move,
    apply_game_action,
    build_debug_response,
    fetch_game,
    fetch_legal_actions,
    reset_existing_game,
    start_new_game,
)
from app.models import (
    GameActionRequest,
    GameDebugResponse,
    GameState,
    HealthResponse,
    LegalActionsResponse,
    NewGameRequest,
    ResetGameRequest,
)
from app.storage import GameNotFoundError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/game/new", response_model=GameState)
def create_new_game(payload: NewGameRequest | None = None) -> GameState:
    return start_new_game(payload or NewGameRequest())


@router.get("/game/{game_id}", response_model=GameState)
def get_game_state(game_id: str) -> GameState:
    try:
        return fetch_game(game_id)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/game/{game_id}/legal-actions", response_model=LegalActionsResponse)
def get_legal_actions(game_id: str) -> LegalActionsResponse:
    try:
        return fetch_legal_actions(game_id)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/game/{game_id}/action", response_model=GameState)
def post_action(game_id: str, payload: GameActionRequest) -> GameState:
    try:
        return apply_game_action(game_id, payload)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/game/{game_id}/reset", response_model=GameState)
def post_reset(game_id: str, payload: ResetGameRequest | None = None) -> GameState:
    try:
        return reset_existing_game(game_id, payload)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/game/{game_id}/ai-move", response_model=GameState)
def post_ai_move(game_id: str) -> GameState:
    try:
        return apply_ai_move(game_id)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/game/{game_id}/debug", response_model=GameDebugResponse)
def get_debug(game_id: str) -> GameDebugResponse:
    try:
        return build_debug_response(game_id)
    except GameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
