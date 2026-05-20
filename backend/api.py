from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

try:
    from .donations import (
        DonationCheckoutError,
        DonationConfigurationError,
        create_donation_checkout_session,
    )
    from .lobbies import (
        LobbyNotFoundError,
        LobbyPermissionError,
        lobby_store,
    )
    from .realtime import game_events, lobby_events
    from .schemas import (
        AddLobbyBotRequest,
        BidRequest,
        BotProgressResponse,
        CreateLobbyRequest,
        DonationCheckoutRequest,
        DonationCheckoutResponse,
        GameStateResponse,
        HealthResponse,
        JoinLobbyRequest,
        LearnChallengeResponse,
        LegalActionsResponse,
        LobbyActionRequest,
        LobbyBidRequest,
        LobbyPlayCardRequest,
        LobbyStateResponse,
        NewGameRequest,
        PlayCardRequest,
        ReadyBotListResponse,
        RemoveLobbyBotRequest,
        RoundScoreResponse,
        StartLobbyRequest,
    )
    from .serializers import (
        serialize_game,
        serialize_game_for_player,
        serialize_score_details,
    )
    from .store import GameNotInitializedError, RoundNotTerminalError, game_store
    from .bots.registry import get_ready_bot_spec, list_ready_bot_metadata
    from .learn import generate_learn_challenge
except ImportError:
    from donations import (
        DonationCheckoutError,
        DonationConfigurationError,
        create_donation_checkout_session,
    )
    from lobbies import (
        LobbyNotFoundError,
        LobbyPermissionError,
        lobby_store,
    )
    from realtime import game_events, lobby_events
    from schemas import (
        AddLobbyBotRequest,
        BidRequest,
        BotProgressResponse,
        CreateLobbyRequest,
        DonationCheckoutRequest,
        DonationCheckoutResponse,
        GameStateResponse,
        HealthResponse,
        JoinLobbyRequest,
        LearnChallengeResponse,
        LegalActionsResponse,
        LobbyActionRequest,
        LobbyBidRequest,
        LobbyPlayCardRequest,
        LobbyStateResponse,
        NewGameRequest,
        PlayCardRequest,
        ReadyBotListResponse,
        RemoveLobbyBotRequest,
        RoundScoreResponse,
        StartLobbyRequest,
    )
    from serializers import (
        serialize_game,
        serialize_game_for_player,
        serialize_score_details,
    )
    from store import GameNotInitializedError, RoundNotTerminalError, game_store
    from bots.registry import get_ready_bot_spec, list_ready_bot_metadata
    from learn import generate_learn_challenge


router = APIRouter()
DEFAULT_SESSION_ID = "default"


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def normalize_session_id(session_id: str | None) -> str:
    resolved = (session_id or DEFAULT_SESSION_ID).strip()
    if not resolved:
        raise ValueError("session id must not be blank")
    if len(resolved) > 128:
        raise ValueError("session id must be 128 characters or fewer")
    return resolved


def resolve_session_id(
    x_smear_session_id: str | None = Header(
        default=None,
        alias="X-Smear-Session-Id",
    ),
    session_id: str | None = Query(default=None),
) -> str:
    try:
        return normalize_session_id(x_smear_session_id or session_id)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc


async def _broadcast_game_state(session_id: str, state_payload: dict | None) -> None:
    await game_events.broadcast(
        session_id,
        {
            "type": "game_state",
            "revision": game_store.get_revision(session_id),
            "state": state_payload,
        },
    )


def _is_lobby_player_turn(session, player_name: str) -> bool:
    if session.phase == "auction":
        return session.auction.current_bidder_name == player_name
    if session.phase == "play":
        return session.game.curr_player.name == player_name
    return False


def _serialize_lobby_seat(seat) -> dict:
    bot_label = None
    if seat.bot_id is not None:
        bot_label = get_ready_bot_spec(seat.bot_id).label

    return {
        "index": seat.index,
        "player_name": seat.player_name,
        "is_occupied": seat.is_occupied,
        "is_bot": seat.is_bot,
        "bot_id": seat.bot_id,
        "bot_label": bot_label,
        "is_host": seat.is_host,
    }


def _serialize_lobby(lobby, player_token: str | None = None) -> dict:
    seat = lobby.find_seat_by_token(player_token) if player_token else None
    game_state = None
    legal_actions: list[dict] = []
    score = None

    if lobby.status == "active" and seat is not None and seat.player_name is not None:
        session = game_store.get_state(lobby.session_id)
        game_state = serialize_game_for_player(session, seat.player_name)
        if _is_lobby_player_turn(session, seat.player_name):
            legal_actions = game_store.get_legal_actions(lobby.session_id)
        try:
            score = serialize_score_details(game_store.get_score(lobby.session_id))
        except RoundNotTerminalError:
            score = None
        except GameNotInitializedError:
            score = None

    return {
        "code": lobby.code,
        "status": lobby.status,
        "num_players": lobby.num_players,
        "seats": [_serialize_lobby_seat(seat_item) for seat_item in lobby.seats],
        "teams": lobby.teams,
        "is_full": lobby.is_full,
        "you": (
            {
                "player_token": player_token,
                "player_name": seat.player_name,
                "seat_index": seat.index,
                "is_host": seat.is_host,
            }
            if seat is not None and seat.player_name is not None and player_token
            else None
        ),
        "game_state": game_state,
        "legal_actions": legal_actions,
        "score": score,
    }


def _require_lobby_player(lobby_code: str, player_token: str | None):
    try:
        return lobby_store.require_player(lobby_code, player_token)
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except LobbyPermissionError as exc:
        raise _forbidden(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc


def _ensure_lobby_player_turn(lobby, seat) -> None:
    session = game_store.get_state(lobby.session_id)
    if seat.player_name is None or not _is_lobby_player_turn(session, seat.player_name):
        raise _forbidden("It is not your turn.")


async def _broadcast_lobby_state(lobby_code: str) -> None:
    try:
        lobby = lobby_store.get_lobby(lobby_code)
    except LobbyNotFoundError:
        return

    await lobby_events.broadcast(
        lobby.code,
        lambda player_token: {
            "type": "lobby_state",
            "revision": lobby.revision,
            "lobby": _serialize_lobby(lobby, player_token),
        },
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@router.get("/bots", response_model=ReadyBotListResponse)
def get_ready_bots() -> dict:
    return {"bots": list_ready_bot_metadata()}


@router.get("/learn/challenge", response_model=LearnChallengeResponse)
def get_learn_challenge(
    phase: str | None = Query(default=None, pattern="^(auction|play)$"),
    bot_id: str = Query(default="optimal-bot"),
) -> dict:
    try:
        return generate_learn_challenge(preferred_phase=phase, bot_id=bot_id)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/donations/checkout-session",
    response_model=DonationCheckoutResponse,
)
def create_donation_checkout(
    payload: DonationCheckoutRequest,
    request: Request,
) -> dict:
    try:
        checkout_url = create_donation_checkout_session(
            payload.amount_cents,
            origin=request.headers.get("origin"),
        )
    except DonationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DonationCheckoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return {"url": checkout_url}


@router.post(
    "/lobbies",
    response_model=LobbyStateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_lobby(payload: CreateLobbyRequest) -> dict:
    try:
        lobby, player_token = lobby_store.create_lobby(
            host_name=payload.host_name,
            num_players=payload.num_players,
            teams=payload.teams,
            host_seat_index=payload.host_seat_index,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.get("/lobbies/{lobby_code}", response_model=LobbyStateResponse)
def get_lobby(
    lobby_code: str,
    player_token: str | None = Query(default=None),
) -> dict:
    try:
        lobby = lobby_store.get_lobby(lobby_code)
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    if player_token is not None and lobby.find_seat_by_token(player_token) is None:
        raise _forbidden("This player is not part of the lobby.")

    return _serialize_lobby(lobby, player_token)


@router.post("/lobbies/{lobby_code}/join", response_model=LobbyStateResponse)
async def join_lobby(lobby_code: str, payload: JoinLobbyRequest) -> dict:
    try:
        lobby, player_token = lobby_store.join_lobby(
            code=lobby_code,
            player_name=payload.player_name,
            seat_index=payload.seat_index,
        )
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post("/lobbies/{lobby_code}/bots", response_model=LobbyStateResponse)
async def add_lobby_bot(lobby_code: str, payload: AddLobbyBotRequest) -> dict:
    try:
        lobby = lobby_store.add_bot(
            code=lobby_code,
            player_token=payload.player_token,
            seat_index=payload.seat_index,
            bot_id=payload.bot_id,
            player_name=payload.player_name,
        )
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except LobbyPermissionError as exc:
        raise _forbidden(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post("/lobbies/{lobby_code}/bots/remove", response_model=LobbyStateResponse)
async def remove_lobby_bot(
    lobby_code: str,
    payload: RemoveLobbyBotRequest,
) -> dict:
    try:
        lobby = lobby_store.remove_bot(
            code=lobby_code,
            player_token=payload.player_token,
            seat_index=payload.seat_index,
        )
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except LobbyPermissionError as exc:
        raise _forbidden(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post("/lobbies/{lobby_code}/start", response_model=LobbyStateResponse)
async def start_lobby(lobby_code: str, payload: StartLobbyRequest) -> dict:
    try:
        lobby, _seat = lobby_store.require_host(lobby_code, payload.player_token)
        if not lobby.is_full:
            raise ValueError("Every seat must be filled before starting.")
        if lobby.status != "active":
            game_store.create_game(
                session_id=lobby.session_id,
                num_players=lobby.num_players,
                player_names=lobby.player_names,
                teams=lobby.named_teams,
                player_bots=lobby.player_bot_ids,
                auto_run_bots=True,
            )
            lobby = lobby_store.mark_started(lobby.code)
    except LobbyNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except LobbyPermissionError as exc:
        raise _forbidden(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post(
    "/lobbies/{lobby_code}/auction/bid",
    response_model=LobbyStateResponse,
)
async def place_lobby_bid(lobby_code: str, payload: LobbyBidRequest) -> dict:
    lobby, seat = _require_lobby_player(lobby_code, payload.player_token)
    if lobby.status != "active":
        raise _bad_request("This lobby has not started.")

    try:
        _ensure_lobby_player_turn(lobby, seat)
        game_store.place_bid(
            lobby.session_id,
            payload.amount,
            auto_run_bots=True,
        )
        lobby = lobby_store.touch_lobby(lobby.code)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post(
    "/lobbies/{lobby_code}/auction/pass",
    response_model=LobbyStateResponse,
)
async def pass_lobby_auction(
    lobby_code: str,
    payload: LobbyActionRequest,
) -> dict:
    lobby, seat = _require_lobby_player(lobby_code, payload.player_token)
    if lobby.status != "active":
        raise _bad_request("This lobby has not started.")

    try:
        _ensure_lobby_player_turn(lobby, seat)
        game_store.pass_auction(lobby.session_id, auto_run_bots=True)
        lobby = lobby_store.touch_lobby(lobby.code)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post("/lobbies/{lobby_code}/play", response_model=LobbyStateResponse)
async def play_lobby_card(lobby_code: str, payload: LobbyPlayCardRequest) -> dict:
    lobby, seat = _require_lobby_player(lobby_code, payload.player_token)
    if lobby.status != "active":
        raise _bad_request("This lobby has not started.")

    try:
        _ensure_lobby_player_turn(lobby, seat)
        game_store.play_card(
            lobby.session_id,
            payload.card_code,
            auto_run_bots=True,
        )
        lobby = lobby_store.touch_lobby(lobby.code)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post("/lobbies/{lobby_code}/next-round", response_model=LobbyStateResponse)
async def next_lobby_round(lobby_code: str, payload: LobbyActionRequest) -> dict:
    lobby, _seat = _require_lobby_player(lobby_code, payload.player_token)
    if lobby.status != "active":
        raise _bad_request("This lobby has not started.")

    try:
        game_store.next_round(lobby.session_id, auto_run_bots=True)
        lobby = lobby_store.touch_lobby(lobby.code)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = _serialize_lobby(lobby, payload.player_token)
    await _broadcast_lobby_state(lobby.code)
    return response


@router.post(
    "/game/new",
    response_model=GameStateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def new_game(
    payload: NewGameRequest,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        game = game_store.create_game(
            session_id=session_id,
            num_players=payload.num_players,
            player_names=payload.player_names,
            teams=payload.teams,
            player_bots=payload.player_bots,
            auto_run_bots=payload.auto_run_bots,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(game)
    await _broadcast_game_state(session_id, response)
    return response


@router.post("/game/auction/bid", response_model=GameStateResponse)
async def place_bid(
    payload: BidRequest,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        session = game_store.place_bid(
            session_id,
            payload.amount,
            auto_run_bots=payload.auto_run_bots,
        )
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(session)
    await _broadcast_game_state(session_id, response)
    return response


@router.post("/game/auction/pass", response_model=GameStateResponse)
async def pass_auction(
    auto_run_bots: bool = True,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        session = game_store.pass_auction(
            session_id,
            auto_run_bots=auto_run_bots,
        )
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(session)
    await _broadcast_game_state(session_id, response)
    return response


@router.post("/game/reset", response_model=GameStateResponse)
async def reset_game(
    auto_run_bots: bool = True,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        game = game_store.reset_round(
            session_id,
            auto_run_bots=auto_run_bots,
        )
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    response = serialize_game(game)
    await _broadcast_game_state(session_id, response)
    return response


@router.post("/game/next-round", response_model=GameStateResponse)
async def next_round(
    auto_run_bots: bool = True,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        session = game_store.next_round(
            session_id,
            auto_run_bots=auto_run_bots,
        )
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(session)
    await _broadcast_game_state(session_id, response)
    return response


@router.get("/game/state", response_model=GameStateResponse)
def get_game_state(session_id: str = Depends(resolve_session_id)) -> dict:
    try:
        game = game_store.get_state(session_id)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    return serialize_game(game)


@router.get("/game/legal-actions", response_model=LegalActionsResponse)
def get_game_legal_actions(session_id: str = Depends(resolve_session_id)) -> dict:
    try:
        actions = game_store.get_legal_actions(session_id)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc

    return {"actions": actions}


@router.post("/game/play", response_model=GameStateResponse)
async def play_card(
    payload: PlayCardRequest,
    session_id: str = Depends(resolve_session_id),
) -> dict:
    try:
        game = game_store.play_card(
            session_id,
            payload.card_code,
            auto_run_bots=payload.auto_run_bots,
        )
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(game)
    await _broadcast_game_state(session_id, response)
    return response


@router.post("/game/bots/step", response_model=GameStateResponse)
async def step_bot_turn(session_id: str = Depends(resolve_session_id)) -> dict:
    try:
        session = game_store.advance_bot_turn(session_id)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    response = serialize_game(session)
    await _broadcast_game_state(session_id, response)
    return response


@router.get("/game/bots/progress", response_model=BotProgressResponse)
def get_bot_progress(session_id: str = Depends(resolve_session_id)) -> dict:
    try:
        return game_store.get_bot_progress(session_id)
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc


@router.get("/game/score", response_model=RoundScoreResponse)
def get_game_score(session_id: str = Depends(resolve_session_id)) -> dict:
    try:
        return serialize_score_details(game_store.get_score(session_id))
    except GameNotInitializedError as exc:
        raise _not_found(str(exc)) from exc
    except RoundNotTerminalError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.websocket("/game/ws")
async def game_websocket(
    websocket: WebSocket,
    session_id: str | None = Query(default=None),
) -> None:
    try:
        resolved_session_id = normalize_session_id(session_id)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await game_events.connect(resolved_session_id, websocket)
    try:
        try:
            current_state = serialize_game(game_store.get_state(resolved_session_id))
        except GameNotInitializedError:
            current_state = None

        await websocket.send_json(
            {
                "type": "game_state",
                "revision": game_store.get_revision(resolved_session_id),
                "state": current_state,
            }
        )

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await game_events.disconnect(resolved_session_id, websocket)


@router.websocket("/lobbies/{lobby_code}/ws")
async def lobby_websocket(
    websocket: WebSocket,
    lobby_code: str,
    player_token: str | None = Query(default=None),
) -> None:
    try:
        lobby, _seat = lobby_store.require_player(lobby_code, player_token)
    except (LobbyNotFoundError, LobbyPermissionError, ValueError):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    resolved_player_token = player_token or ""
    await lobby_events.connect(lobby.code, resolved_player_token, websocket)
    try:
        await websocket.send_json(
            {
                "type": "lobby_state",
                "revision": lobby.revision,
                "lobby": _serialize_lobby(lobby, resolved_player_token),
            }
        )

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await lobby_events.disconnect(lobby.code, websocket)
