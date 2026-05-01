from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

try:
    from .schemas import (
        GameStateResponse,
        HealthResponse,
        LegalActionsResponse,
        NewGameRequest,
        PlayCardRequest,
        RoundScoreResponse,
    )
    from .serializers import serialize_game, serialize_score_details, sort_cards
    from .store import GameNotInitializedError, RoundNotTerminalError, game_store
except ImportError:
    from schemas import (
        GameStateResponse,
        HealthResponse,
        LegalActionsResponse,
        NewGameRequest,
        PlayCardRequest,
        RoundScoreResponse,
    )
    from serializers import serialize_game, serialize_score_details, sort_cards
    from store import GameNotInitializedError, RoundNotTerminalError, game_store


router = APIRouter()


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@router.post(
    "/game/new",
    response_model=GameStateResponse,
    status_code=status.HTTP_201_CREATED,
)
def new_game(payload: NewGameRequest) -> dict:
    try:
        game = game_store.create_game(
            num_players=payload.num_players,
            player_names=payload.player_names,
            teams=payload.teams,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    return serialize_game(game)


@router.post("/game/reset", response_model=GameStateResponse)
def reset_game() -> dict:
    try:
        game = game_store.reset_round()
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    return serialize_game(game)


@router.get("/game/state", response_model=GameStateResponse)
def get_game_state() -> dict:
    try:
        game = game_store.get_state()
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    return serialize_game(game)


@router.get("/game/legal-actions", response_model=LegalActionsResponse)
def get_game_legal_actions() -> dict:
    try:
        cards = sort_cards(game_store.get_legal_actions())
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    return {
        "actions": [
            {"type": "play_card", "card_code": card.code}
            for card in cards
        ]
    }


@router.post("/game/play", response_model=GameStateResponse)
def play_card(payload: PlayCardRequest) -> dict:
    try:
        game = game_store.play_card(payload.card_code)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    return serialize_game(game)


@router.get("/game/score", response_model=RoundScoreResponse)
def get_game_score() -> dict:
    try:
        return serialize_score_details(game_store.get_score())
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except RoundNotTerminalError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
