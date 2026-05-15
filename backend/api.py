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
    from .realtime import game_events
    from .schemas import (
        BidRequest,
        BotProgressResponse,
        DonationCheckoutRequest,
        DonationCheckoutResponse,
        GameStateResponse,
        HealthResponse,
        LearnChallengeResponse,
        LegalActionsResponse,
        NewGameRequest,
        PlayCardRequest,
        ReadyBotListResponse,
        RoundScoreResponse,
    )
    from .serializers import serialize_game, serialize_score_details
    from .store import GameNotInitializedError, RoundNotTerminalError, game_store
    from .bots.registry import list_ready_bot_metadata
    from .learn import generate_learn_challenge
except ImportError:
    from donations import (
        DonationCheckoutError,
        DonationConfigurationError,
        create_donation_checkout_session,
    )
    from realtime import game_events
    from schemas import (
        BidRequest,
        BotProgressResponse,
        DonationCheckoutRequest,
        DonationCheckoutResponse,
        GameStateResponse,
        HealthResponse,
        LearnChallengeResponse,
        LegalActionsResponse,
        NewGameRequest,
        PlayCardRequest,
        ReadyBotListResponse,
        RoundScoreResponse,
    )
    from serializers import serialize_game, serialize_score_details
    from store import GameNotInitializedError, RoundNotTerminalError, game_store
    from bots.registry import list_ready_bot_metadata
    from learn import generate_learn_challenge


router = APIRouter()
DEFAULT_SESSION_ID = "default"


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


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


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@router.get("/bots", response_model=ReadyBotListResponse)
def get_ready_bots() -> dict:
    return {"bots": list_ready_bot_metadata()}


@router.get("/learn/challenge", response_model=LearnChallengeResponse)
def get_learn_challenge(
    phase: str | None = Query(default=None, pattern="^(auction|play)$"),
) -> dict:
    try:
        return generate_learn_challenge(preferred_phase=phase)
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
