from __future__ import annotations

from app.mock_logic import apply_action_to_state, choose_placeholder_ai_action, compute_legal_actions
from app.models import GameActionRequest, GameDebugResponse, GameState, LegalActionsResponse, NewGameRequest, ResetGameRequest
from app.storage import create_game, get_game, get_game_metadata, reset_game, save_game


def start_new_game(request: NewGameRequest) -> GameState:
    return create_game(request)


def fetch_game(game_id: str) -> GameState:
    state = get_game(game_id)
    state.legal_actions = compute_legal_actions(state)
    return save_game(state)


def fetch_legal_actions(game_id: str) -> LegalActionsResponse:
    state = fetch_game(game_id)
    return LegalActionsResponse(
        game_id=game_id,
        current_player_id=state.current_player_id,
        legal_actions=state.legal_actions,
    )


def apply_game_action(game_id: str, action_request: GameActionRequest) -> GameState:
    state = get_game(game_id)
    updated_state = apply_action_to_state(state, action_request)
    return save_game(updated_state)


def reset_existing_game(game_id: str, reset_request: ResetGameRequest | None = None) -> GameState:
    return reset_game(game_id, reset_request)


def apply_ai_move(game_id: str) -> GameState:
    state = get_game(game_id)
    action = choose_placeholder_ai_action(state)
    if action is None:
        raise ValueError("No legal actions are available for the placeholder AI.")

    state.logs.append(f"Placeholder AI selected action '{action.type}'.")
    updated_state = apply_action_to_state(state, GameActionRequest(**action.model_dump()))
    return save_game(updated_state)


def build_debug_response(game_id: str) -> GameDebugResponse:
    state = fetch_game(game_id)
    metadata = get_game_metadata(game_id)
    return GameDebugResponse(game_id=game_id, state=state, metadata=metadata)
