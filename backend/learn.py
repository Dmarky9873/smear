from __future__ import annotations

from copy import deepcopy
from random import Random
from uuid import uuid4

try:
    from .bots.optimal_bot import OptimalBotPlayer
    from .engine import get_legal_actions
    from .gameplay import GameSession, MatchController, serialize_auction_event
    from .models import AuctionEvent, Card
    from .serializers import serialize_game
except ImportError:
    from bots.optimal_bot import OptimalBotPlayer
    from engine import get_legal_actions
    from gameplay import GameSession, MatchController, serialize_auction_event
    from models import AuctionEvent, Card
    from serializers import serialize_game


LEARN_PLAYER_NAME = "You"
LEARN_BOT_ID = "optimal-bot"
LEARN_BOT_LABEL = "Optimal Bot"


def _action_label(action: dict) -> str:
    if action["type"] == "bid":
        return f"Bid {action['amount']}"
    if action["type"] == "pass":
        return "Pass"
    if action["type"] == "play_card":
        return f"Play {action['card_code']}"
    return "Unknown action"


def _decorate_action(action: dict) -> dict:
    return {
        **action,
        "label": _action_label(action),
    }


def _serialize_card_action(card: Card) -> dict:
    return _decorate_action({"type": "play_card", "card_code": card.code})


def _serialize_auction_action(action: AuctionEvent) -> dict:
    return _decorate_action(serialize_auction_event(action))


def _current_actor_name(session: GameSession) -> str:
    if session.phase == "auction":
        return session.auction.current_bidder_name
    return session.game.curr_player.name


def _sync_bot_context(bot: OptimalBotPlayer, session: GameSession) -> None:
    actor = session.game.get_player_by_name(_current_actor_name(session))
    bot._cards = set(actor.cards)
    bot.set_match_context(
        player_names=session.player_names,
        teams=session.teams,
        match_scores=session.match_scores,
        target_score=session.target_score,
        auction_state=session.auction.state,
        round_state=session.game.round_state,
    )


def _choose_best_action(session: GameSession) -> dict:
    bot = OptimalBotPlayer(_current_actor_name(session))
    _sync_bot_context(bot, session)

    if session.phase == "auction":
        action = bot.choose_auction_action(deepcopy(session.auction.state))
        return _serialize_auction_action(action)

    card = bot.choose_card(session.game.round_state)
    return _serialize_card_action(card)


def _legal_options(session: GameSession) -> list[dict]:
    if session.phase == "auction":
        return [
            _serialize_auction_action(action)
            for action in session.auction.legal_actions()
        ]

    return [
        _serialize_card_action(card)
        for card in sorted(
            get_legal_actions(session.game.round_state),
            key=lambda candidate: candidate.code,
        )
    ]


def _make_controller(player_names: list[str]) -> MatchController:
    return MatchController.create(
        num_players=len(player_names),
        player_names=player_names,
        teams=None,
        player_bot_ids=[None for _ in player_names],
        auto_run_bots=False,
    )


def _apply_random_auction_action(
    controller: MatchController,
    rng: Random,
) -> None:
    legal_actions = controller.session.auction.legal_actions()
    action = rng.choice(legal_actions)
    if action.action == "bid":
        if action.amount is None:
            raise ValueError("random bid action is missing an amount")
        controller.place_bid(action.amount, auto_run_bots=False)
        return
    controller.pass_auction(auto_run_bots=False)


def _apply_random_play_action(
    controller: MatchController,
    rng: Random,
) -> None:
    legal_cards = sorted(
        get_legal_actions(controller.session.game.round_state),
        key=lambda card: card.code,
    )
    controller.play_card(rng.choice(legal_cards).code, auto_run_bots=False)


def _build_auction_position(rng: Random) -> GameSession:
    names_by_actor_index = [
        [LEARN_PLAYER_NAME, "North", "East"],
        ["North", LEARN_PLAYER_NAME, "East"],
        ["North", "East", LEARN_PLAYER_NAME],
    ]
    actor_index = rng.randrange(len(names_by_actor_index))
    controller = _make_controller(names_by_actor_index[actor_index])

    for _ in range(actor_index):
        _apply_random_auction_action(controller, rng)

    if (
        controller.session.phase != "auction"
        or controller.session.auction.current_bidder_name != LEARN_PLAYER_NAME
    ):
        raise ValueError("could not build an auction learning position")

    return controller.session


def _complete_random_auction(
    controller: MatchController,
    rng: Random,
) -> None:
    for _ in range(len(controller.session.player_names)):
        if controller.session.phase != "auction":
            return
        _apply_random_auction_action(controller, rng)
    if controller.session.phase == "auction":
        raise ValueError("auction did not complete")


def _build_play_position(rng: Random) -> GameSession:
    controller = _make_controller([LEARN_PLAYER_NAME, "North", "East"])
    _complete_random_auction(controller, rng)
    human_turns_to_skip = rng.randrange(4)
    skipped_human_turns = 0

    for _ in range(64):
        if controller.session.phase != "play":
            raise ValueError("round ended before a learning play position was found")
        if controller.session.game.curr_player.name == LEARN_PLAYER_NAME:
            if skipped_human_turns >= human_turns_to_skip:
                return controller.session
            skipped_human_turns += 1
        _apply_random_play_action(controller, rng)

    raise ValueError("could not find a learning play position")


def _build_position(rng: Random, preferred_phase: str | None) -> GameSession:
    phase = preferred_phase or rng.choice(["auction", "play", "play"])
    if phase == "auction":
        return _build_auction_position(rng)
    if phase == "play":
        return _build_play_position(rng)
    raise ValueError(f"unsupported learning phase: {phase}")


def _challenge_prompt(session: GameSession) -> str:
    if session.phase == "auction":
        highest_bid = session.auction.state.highest_bid
        highest_bidder = session.auction.state.highest_bidder_name
        if highest_bid is None:
            return "You are bidding. No one has bid yet."
        return f"You are bidding. The current high bid is {highest_bid} by {highest_bidder}."

    trick = session.game.round_state.current_trick
    if not trick.plays:
        return "You are leading the trick."
    return f"You are playing to a trick led by {trick.leader.name}."


def generate_learn_challenge(
    *,
    seed: int | None = None,
    preferred_phase: str | None = None,
) -> dict:
    rng = Random(seed)
    last_error: Exception | None = None
    phases_to_try = (
        [preferred_phase] if preferred_phase is not None else [None, "play", "auction"]
    )

    for phase in phases_to_try:
        for _ in range(16):
            try:
                session = _build_position(rng, phase)
                if _current_actor_name(session) != LEARN_PLAYER_NAME:
                    raise ValueError("learning position is not on the learner turn")
                if session.phase not in {"auction", "play"}:
                    raise ValueError("learning position is not actionable")

                options = _legal_options(session)
                if len(options) < 2:
                    raise ValueError("learning position needs at least two legal options")

                return {
                    "id": uuid4().hex,
                    "phase": session.phase,
                    "actor_name": LEARN_PLAYER_NAME,
                    "prompt": _challenge_prompt(session),
                    "state": serialize_game(session),
                    "options": options,
                    "best_bot_id": LEARN_BOT_ID,
                    "best_bot_label": LEARN_BOT_LABEL,
                    "best_action": _choose_best_action(session),
                }
            except Exception as exc:  # pragma: no cover - retried by design
                last_error = exc

    raise RuntimeError("could not generate a learning challenge") from last_error
