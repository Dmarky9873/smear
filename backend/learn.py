from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from uuid import uuid4

try:
    from .bots.base import BotPlayer
    from .bots.greedy_bot import GreedyPlayer
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .constants import GAME_VALUES, RANK_ORDER
    from .engine import get_legal_actions
    from .gameplay import GameSession, MatchController, serialize_auction_event
    from .models import AuctionEvent, Card, get_cards_value, would_win
    from .serializers import serialize_game
except ImportError:
    from bots.base import BotPlayer
    from bots.greedy_bot import GreedyPlayer
    from bots.registry import build_ready_bot, get_ready_bot_spec
    from constants import GAME_VALUES, RANK_ORDER
    from engine import get_legal_actions
    from gameplay import GameSession, MatchController, serialize_auction_event
    from models import AuctionEvent, Card, get_cards_value, would_win
    from serializers import serialize_game


LEARN_PLAYER_NAME = "You"
DEFAULT_LEARN_BOT_ID = "optimal-bot"
TEACHING_SUITS = ("H", "D", "C", "S")


@dataclass(frozen=True)
class AuctionSuitProfile:
    suit: str
    ceiling: int
    has_ace: bool
    has_low: bool
    has_jack: bool
    joker_count: int
    suit_card_count: int
    game_points: int


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


def _sync_bot_context(bot: BotPlayer, session: GameSession) -> None:
    actor = session.game.get_player_by_name(_current_actor_name(session))
    bot._cards = set(actor.cards)
    if hasattr(bot, "set_match_context"):
        bot.set_match_context(
            player_names=session.player_names,
            teams=session.teams,
            match_scores=session.match_scores,
            target_score=session.target_score,
            auction_state=session.auction.state,
            round_state=session.game.round_state,
        )


def _choose_best_action(session: GameSession, bot_id: str) -> dict:
    bot = build_ready_bot(bot_id, _current_actor_name(session))
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


def _apply_teaching_auction_action(
    controller: MatchController,
) -> None:
    bot = GreedyPlayer(
        controller.session.auction.current_bidder_name,
        use_rollout_auction=False,
    )
    _sync_bot_context(bot, controller.session)
    action = bot.choose_auction_action(deepcopy(controller.session.auction.state))
    if action.action == "bid":
        if action.amount is None:
            raise ValueError("teaching bid action is missing an amount")
        controller.place_bid(action.amount, auto_run_bots=False)
        return
    controller.pass_auction(auto_run_bots=False)


def _apply_teaching_play_action(
    controller: MatchController,
) -> None:
    bot = GreedyPlayer(
        controller.session.game.curr_player.name,
        use_rollout_auction=False,
    )
    _sync_bot_context(bot, controller.session)
    card = bot.choose_card(controller.session.game.round_state)
    controller.play_card(card.code, auto_run_bots=False)


def _build_auction_position(rng: Random) -> GameSession:
    names_by_actor_index = [
        [LEARN_PLAYER_NAME, "North", "East"],
        ["North", LEARN_PLAYER_NAME, "East"],
        ["North", "East", LEARN_PLAYER_NAME],
    ]
    actor_index = rng.randrange(len(names_by_actor_index))
    controller = _make_controller(names_by_actor_index[actor_index])

    for _ in range(actor_index):
        _apply_teaching_auction_action(controller)

    if (
        controller.session.phase != "auction"
        or controller.session.auction.current_bidder_name != LEARN_PLAYER_NAME
    ):
        raise ValueError("could not build an auction learning position")

    return controller.session


def _complete_teaching_auction(
    controller: MatchController,
) -> None:
    for _ in range(len(controller.session.player_names)):
        if controller.session.phase != "auction":
            return
        _apply_teaching_auction_action(controller)
    if controller.session.phase == "auction":
        raise ValueError("auction did not complete")


def _build_play_position(rng: Random) -> GameSession:
    controller = _make_controller([LEARN_PLAYER_NAME, "North", "East"])
    _complete_teaching_auction(controller)
    human_turns_to_skip = rng.randrange(4)
    skipped_human_turns = 0

    for _ in range(64):
        if controller.session.phase != "play":
            raise ValueError("round ended before a learning play position was found")
        if controller.session.game.curr_player.name == LEARN_PLAYER_NAME:
            if skipped_human_turns >= human_turns_to_skip:
                return controller.session
            skipped_human_turns += 1
        _apply_teaching_play_action(controller)

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


def _action_key(action: dict) -> tuple:
    return (
        action["type"],
        action.get("amount"),
        action.get("card_code"),
    )


def _actor_cards(session: GameSession) -> set[Card]:
    return set(session.game.get_player_by_name(_current_actor_name(session)).cards)


def _card_sort_key(card: Card) -> tuple:
    if card.is_joker:
        return (1, "", 99, card.code)
    return (0, card.suit or "", RANK_ORDER.get(card.rank, 0), card.code)


def _format_card_codes(cards: list[Card] | set[Card], *, limit: int = 4) -> str:
    ordered = sorted(cards, key=_card_sort_key, reverse=True)
    codes = [card.code for card in ordered[:limit]]
    if len(ordered) > limit:
        codes.append(f"{len(ordered) - limit} more")
    if not codes:
        return "none"
    if len(codes) == 1:
        return codes[0]
    return f"{', '.join(codes[:-1])}, and {codes[-1]}"


def _build_auction_profile(
    cards: set[Card],
    suit: str,
    *,
    deck_low: str,
) -> AuctionSuitProfile:
    suit_cards = [
        card
        for card in cards
        if not card.is_joker and card.suit == suit
    ]
    joker_count = sum(1 for card in cards if card.is_joker)
    has_ace = any(card.rank == "A" for card in suit_cards)
    has_low = any(card.rank == deck_low for card in suit_cards)
    has_jack = any(card.rank == "J" for card in suit_cards)
    high_controls = sum(
        1
        for card in suit_cards
        if card.rank in {"A", "K", "Q", "J"}
    )
    game_points = sum(
        GAME_VALUES.get(card.rank, 0)
        for card in cards
        if not card.is_joker
    )
    valuable_card_count = sum(
        1
        for card in cards
        if not card.is_joker and GAME_VALUES.get(card.rank, 0) > 0
    )
    ceiling = (
        int(has_ace)
        + int(has_low)
        + int(has_jack)
        + min(joker_count, 2)
        + int(game_points >= 16 or valuable_card_count >= 4)
        + int(len(suit_cards) >= 3 and high_controls >= 2)
    )
    if not has_ace:
        ceiling = min(ceiling, 3)
    if has_ace and len(suit_cards) == 1 and joker_count == 0:
        ceiling = min(ceiling, 2)
    return AuctionSuitProfile(
        suit=suit,
        ceiling=max(0, min(6, ceiling)),
        has_ace=has_ace,
        has_low=has_low,
        has_jack=has_jack,
        joker_count=joker_count,
        suit_card_count=len(suit_cards),
        game_points=game_points,
    )


def _best_auction_profile(session: GameSession) -> AuctionSuitProfile:
    cards = _actor_cards(session)
    deck_low = session.game.low
    return max(
        (
            _build_auction_profile(cards, suit, deck_low=deck_low)
            for suit in TEACHING_SUITS
        ),
        key=lambda profile: (
            profile.ceiling,
            profile.has_ace,
            profile.has_low,
            profile.has_jack,
            profile.suit_card_count,
            profile.game_points,
        ),
    )


def _auction_profile_features(profile: AuctionSuitProfile) -> str:
    features: list[str] = []
    if profile.has_ace:
        features.append("the ace")
    if profile.has_low:
        features.append("the low")
    if profile.has_jack:
        features.append("the jack")
    if profile.joker_count == 1:
        features.append("a joker")
    elif profile.joker_count > 1:
        features.append(f"{profile.joker_count} jokers")
    if profile.game_points >= 16:
        features.append("strong game cards")
    if profile.suit_card_count >= 3:
        features.append(f"{profile.suit_card_count} cards in suit")

    if not features:
        return f"{profile.suit} has no clear guaranteed point"
    return f"{profile.suit} has {', '.join(features)}"


def _is_good_auction_recommendation(
    session: GameSession,
    action: dict,
) -> bool:
    if action["type"] != "bid":
        return True

    amount = action.get("amount")
    if not isinstance(amount, int):
        return False

    profile = _best_auction_profile(session)
    if amount >= 6 and not profile.has_ace:
        return False
    if amount >= 5 and profile.suit_card_count < 2 and profile.joker_count == 0:
        return False
    return amount <= max(1, profile.ceiling)


def _is_good_teaching_position(session: GameSession, best_action: dict) -> bool:
    if session.phase == "auction":
        return _is_good_auction_recommendation(session, best_action)
    return True


def _explain_auction_action(
    session: GameSession,
    action: dict,
    bot_label: str,
) -> str:
    profile = _best_auction_profile(session)
    next_bid = (
        1
        if session.auction.state.highest_bid is None
        else session.auction.state.highest_bid + 1
    )
    profile_summary = _auction_profile_features(profile)

    if action["type"] == "pass":
        if session.auction.state.highest_bid is None:
            return (
                f"{bot_label} passes because the hand does not have a clear trump "
                f"commitment. The best candidate is {profile_summary}, which is "
                "not enough to open voluntarily."
            )
        support_relation = (
            "the very top of"
            if next_bid <= max(1, profile.ceiling)
            else "above"
        )
        return (
            f"{bot_label} passes because overcalling would require at least "
            f"{next_bid}, while the hand's best trump candidate is {profile_summary}. "
            f"That is {support_relation} the hand's estimated support."
        )

    amount = action.get("amount")
    supported_ceiling = max(1, profile.ceiling)
    return (
        f"{bot_label} bids {amount} because {profile_summary}. The bid is within "
        f"the hand's estimated ceiling of {supported_ceiling}, so it applies pressure "
        "without jumping past the points the hand can reasonably chase."
    )


def _explain_play_action(
    session: GameSession,
    action: dict,
    bot_label: str,
) -> str:
    card_code = action.get("card_code")
    if not isinstance(card_code, str):
        return f"{bot_label} chose this card because it was the strongest legal play."

    round_state = session.game.round_state
    card = next(
        (
            candidate
            for candidate in round_state.current_player.cards
            if candidate.code == card_code
        ),
        None,
    )
    if card is None:
        return f"{bot_label} chose {card_code} because it was the strongest legal play."

    trick = round_state.current_trick
    legal_cards = get_legal_actions(round_state)
    if not trick.plays:
        if round_state.trump is None and not card.is_joker:
            profile = _build_auction_profile(
                set(round_state.current_player.cards),
                card.suit or "",
                deck_low=session.game.low,
            )
            return (
                f"{bot_label} leads {card.code} to set trump to {card.suit}. "
                f"That suit is attractive because {_auction_profile_features(profile)}."
            )
        if card.is_joker or card.suit == round_state.trump:
            return (
                f"{bot_label} leads {card.code} to put a trump threat on the table "
                "before opponents can discard around it."
            )
        return (
            f"{bot_label} leads {card.code} because it keeps higher trump and point "
            "cards back for a more valuable trick."
        )

    trick_cards = {play.card for play in trick.plays}
    trick_points = get_cards_value(trick_cards | {card})
    wins_now = would_win(card, trick)
    card_points = get_cards_value({card})
    lowest_point_cost = min(get_cards_value({candidate}) for candidate in legal_cards)

    if wins_now and trick_points > 0:
        return (
            f"{bot_label} plays {card.code} because it is currently winning the trick "
            f"and would collect {trick_points} point card{'s' if trick_points != 1 else ''}."
        )
    if wins_now:
        return (
            f"{bot_label} plays {card.code} because it takes control of the trick "
            "without spending a more valuable card."
        )
    if card_points == lowest_point_cost:
        return (
            f"{bot_label} plays {card.code} because it cannot profitably win this "
            "trick, so it sheds one of the lowest-cost legal cards."
        )
    return (
        f"{bot_label} plays {card.code} because the lead limits the legal choices "
        f"to {_format_card_codes(legal_cards)}; this is the best tradeoff among them."
    )


def _explain_best_action(
    session: GameSession,
    action: dict,
    bot_label: str,
) -> str:
    if session.phase == "auction":
        return _explain_auction_action(session, action, bot_label)
    return _explain_play_action(session, action, bot_label)


def generate_learn_challenge(
    *,
    seed: int | None = None,
    preferred_phase: str | None = None,
    bot_id: str = DEFAULT_LEARN_BOT_ID,
) -> dict:
    bot_spec = get_ready_bot_spec(bot_id)
    rng = Random(seed)
    last_error: Exception | None = None
    phases_to_try = (
        [preferred_phase] if preferred_phase is not None else [None, "play", "auction"]
    )

    for phase in phases_to_try:
        for _ in range(32):
            try:
                session = _build_position(rng, phase)
                if _current_actor_name(session) != LEARN_PLAYER_NAME:
                    raise ValueError("learning position is not on the learner turn")
                if session.phase not in {"auction", "play"}:
                    raise ValueError("learning position is not actionable")

                options = _legal_options(session)
                if len(options) < 2:
                    raise ValueError("learning position needs at least two legal options")
                option_keys = {_action_key(option) for option in options}
                best_action = _choose_best_action(session, bot_id)
                if _action_key(best_action) not in option_keys:
                    raise ValueError("bot chose an action outside the legal options")
                if not _is_good_teaching_position(session, best_action):
                    raise ValueError("bot recommendation is not a good teaching position")

                return {
                    "id": uuid4().hex,
                    "phase": session.phase,
                    "actor_name": LEARN_PLAYER_NAME,
                    "prompt": _challenge_prompt(session),
                    "state": serialize_game(session),
                    "options": options,
                    "best_bot_id": bot_spec.id,
                    "best_bot_label": bot_spec.label,
                    "best_action": best_action,
                    "best_action_explanation": _explain_best_action(
                        session,
                        best_action,
                        bot_spec.label,
                    ),
                }
            except Exception as exc:  # pragma: no cover - retried by design
                last_error = exc

    raise RuntimeError("could not generate a learning challenge") from last_error
