from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:
    from .bots.base import BotPlayer
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .constants import RANK_ORDER
    from .engine import Auction, Game, get_legal_actions, score_round_details
    from .models import AuctionEvent, Card, Play
except ImportError:
    from bots.base import BotPlayer
    from bots.registry import build_ready_bot, get_ready_bot_spec
    from constants import RANK_ORDER
    from engine import Auction, Game, get_legal_actions, score_round_details
    from models import AuctionEvent, Card, Play


MAX_BID = 6
SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3}


class RoundNotTerminalError(RuntimeError):
    """Raised when score is requested before the round is complete."""


@dataclass
class GameSession:
    game: Game
    player_names: list[str]
    teams: list[tuple[str, ...]]
    dealer_index: int
    auction: Auction
    player_bot_ids: dict[str, str | None]
    match_scores: dict[str, int]
    bots: dict[str, BotPlayer] = field(default_factory=dict)
    target_score: int = 21
    round_number: int = 1
    last_round_score: dict | None = None

    @property
    def score_unit_names(self) -> list[str]:
        return [" / ".join(team) for team in self.teams]

    @property
    def match_winner_names(self) -> list[str]:
        if not self.match_scores:
            return []
        max_score = max(self.match_scores.values())
        if max_score < self.target_score:
            return []
        return [
            unit_name
            for unit_name, score in self.match_scores.items()
            if score == max_score
        ]

    @property
    def is_match_complete(self) -> bool:
        return len(self.match_winner_names) > 0

    @property
    def phase(self) -> str:
        if not self.auction.state.is_complete:
            return "auction"
        if self.game.round_state.is_terminal:
            if self.is_match_complete:
                return "match_complete"
            return "round_complete"
        return "play"


@dataclass(frozen=True)
class MatchResult:
    rounds_played: int
    is_draw: bool
    winner_names: list[str]
    final_scores: dict[str, int]


def sort_cards(cards: list[Card] | set[Card]) -> list[Card]:
    return sorted(
        cards,
        key=lambda card: (
            1 if card.is_joker else 0,
            SUIT_ORDER.get(card.suit or "", 99),
            RANK_ORDER.get(card.rank or "", 99),
            card.code,
        ),
    )


def serialize_auction_event(event: AuctionEvent) -> dict:
    payload = {"type": event.action}
    if event.amount is not None:
        payload["amount"] = event.amount
    return payload


def _normalize_teams(
    player_names: list[str],
    teams: Sequence[Sequence[str]] | None,
) -> list[tuple[str, ...]]:
    if teams is None:
        return [(name,) for name in player_names]

    seen_players: set[str] = set()
    normalized: list[tuple[str, ...]] = []
    expected_players = set(player_names)

    for raw_team in teams:
        cleaned_team = tuple(name.strip() for name in raw_team if name.strip())
        if not cleaned_team:
            raise ValueError("teams cannot contain empty groups")

        for name in cleaned_team:
            if name not in expected_players:
                raise ValueError(f"unknown player in teams: {name}")
            if name in seen_players:
                raise ValueError(f"player listed multiple times in teams: {name}")
            seen_players.add(name)

        normalized.append(cleaned_team)

    if seen_players != expected_players:
        missing = sorted(expected_players - seen_players)
        raise ValueError(
            f"every player must appear in exactly one team; missing: {', '.join(missing)}"
        )

    return normalized


def _normalize_player_bots(
    player_names: list[str],
    player_bots: list[str | None] | dict[str, str | None] | None,
) -> dict[str, str | None]:
    if player_bots is None:
        return {name: None for name in player_names}

    if isinstance(player_bots, dict):
        unknown_players = sorted(set(player_bots) - set(player_names))
        if unknown_players:
            raise ValueError(
                f"unknown players in player_bots: {', '.join(unknown_players)}"
            )
        raw_mapping = {
            player_name: player_bots.get(player_name)
            for player_name in player_names
        }
    else:
        if len(player_bots) != len(player_names):
            raise ValueError("player_bots must match the number of player_names")
        raw_mapping = dict(zip(player_names, player_bots))

    normalized: dict[str, str | None] = {}
    for player_name, bot_id in raw_mapping.items():
        if bot_id is None:
            normalized[player_name] = None
            continue

        cleaned_bot_id = bot_id.strip()
        if not cleaned_bot_id:
            normalized[player_name] = None
            continue

        get_ready_bot_spec(cleaned_bot_id)
        normalized[player_name] = cleaned_bot_id

    return normalized


class MatchController:
    def __init__(self, session: GameSession):
        self.session = session

    @classmethod
    def create(
        cls,
        num_players: int,
        player_names: list[str],
        teams: Sequence[Sequence[str]] | None,
        player_bot_ids: list[str | None] | dict[str, str | None] | None = None,
        bots: dict[str, BotPlayer] | None = None,
        target_score: int = 21,
        auto_run_bots: bool = False,
    ) -> "MatchController":
        normalized_names = [name.strip() for name in player_names]
        if len(normalized_names) != num_players:
            raise ValueError("num_players must match the number of player_names")
        if any(not name for name in normalized_names):
            raise ValueError("player_names cannot contain blank values")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("player_names must be unique")

        normalized_teams = _normalize_teams(normalized_names, teams)
        normalized_player_bot_ids = _normalize_player_bots(
            normalized_names,
            player_bot_ids,
        )
        normalized_bots = cls._normalize_explicit_bots(normalized_names, bots or {})
        dealer_index = len(normalized_names) - 1
        game = Game(num_players, normalized_names, normalized_teams)
        resolved_bots = cls._build_ready_bot_controllers(normalized_player_bot_ids)
        resolved_bots.update(normalized_bots)
        session = GameSession(
            game=game,
            player_names=normalized_names,
            teams=normalized_teams,
            dealer_index=dealer_index,
            auction=cls._build_auction_state(normalized_names, dealer_index),
            player_bot_ids=normalized_player_bot_ids,
            bots=resolved_bots,
            match_scores=cls._build_initial_match_scores(normalized_teams),
            target_score=target_score,
        )
        controller = cls(session)
        controller._sync_all_bot_hands()
        if auto_run_bots:
            controller.run_bot_turns()
        return controller

    def _sync_bot_hand(self, player_name: str) -> None:
        bot = self.session.bots[player_name]
        game_player = self.session.game.get_player_by_name(player_name)
        bot._cards = set(game_player.cards)

    def _sync_all_bot_hands(self) -> None:
        for player_name in self.session.bots:
            self._sync_bot_hand(player_name)

    @staticmethod
    def _build_auction_state(
        player_names: list[str],
        dealer_index: int,
    ) -> Auction:
        return Auction(
            player_names=player_names,
            dealer=player_names[dealer_index],
            max_bid=MAX_BID,
        )

    @staticmethod
    def _build_initial_match_scores(
        teams: list[tuple[str, ...]],
    ) -> dict[str, int]:
        return {" / ".join(team): 0 for team in teams}

    @staticmethod
    def _build_ready_bot_controllers(
        player_bot_ids: dict[str, str | None],
    ) -> dict[str, BotPlayer]:
        return {
            player_name: build_ready_bot(bot_id, player_name)
            for player_name, bot_id in player_bot_ids.items()
            if bot_id is not None
        }

    @staticmethod
    def _normalize_explicit_bots(
        player_names: list[str],
        bots: dict[str, BotPlayer],
    ) -> dict[str, BotPlayer]:
        unknown_players = sorted(set(bots) - set(player_names))
        if unknown_players:
            raise ValueError(
                f"unknown players in bots: {', '.join(unknown_players)}"
            )
        for player_name, bot in bots.items():
            if not isinstance(bot, BotPlayer):
                raise TypeError(
                    f"controller for '{player_name}' must be a BotPlayer instance"
                )
        return dict(bots)

    def _current_bidder_name(self) -> str:
        return self.session.auction.current_bidder_name

    def _current_bot_name(self) -> str | None:
        if self.session.phase == "auction":
            return self.session.auction.current_bidder_name
        if self.session.phase == "play":
            return self.session.game.curr_player.name
        return None

    def _run_bot_auction_turn(self, player_name: str) -> None:
        self._sync_bot_hand(player_name)
        bot = self.session.bots[player_name]
        action = bot.choose_auction_action(self.session.auction.state)
        legal_actions = self.session.auction.legal_actions()
        if action not in legal_actions:
            raise ValueError(
                f"bot '{player_name}' selected illegal auction action {action}; legal actions are {legal_actions}"
            )

        if action.action not in {"bid", "pass"}:
            raise ValueError(f"unsupported bot auction action: {action}")

        self.session.auction.apply_event(action)
        self._advance_or_finalize_auction()

    def _run_bot_play_turn(self, player_name: str) -> None:
        self._sync_bot_hand(player_name)
        bot = self.session.bots[player_name]
        legal_cards = get_legal_actions(self.session.game.round_state)
        card = bot.choose_card(self.session.game.round_state)
        if card not in legal_cards:
            legal_codes = sorted(legal_card.code for legal_card in legal_cards)
            raise ValueError(
                f"bot '{player_name}' selected illegal card {card.code}; legal cards are {legal_codes}"
            )

        self.session.game.apply_trick_action(Play(self.session.game.curr_player, card))
        self._score_terminal_round_if_needed()

    def run_bot_turns(self, max_steps: int = 512) -> None:
        for _ in range(max_steps):
            if self.session.phase in {"round_complete", "match_complete"}:
                return

            player_name = self._current_bot_name()
            if player_name is None or player_name not in self.session.bots:
                return

            if self.session.phase == "auction":
                self._run_bot_auction_turn(player_name)
                continue

            if self.session.phase == "play":
                self._run_bot_play_turn(player_name)
                continue

            return

        raise ValueError("bot turn loop exceeded the step limit")

    def _finalize_auction(self) -> None:
        auction_state = self.session.auction.state
        if auction_state.highest_bidder_name is None:
            raise ValueError("auction cannot complete without a winning bid")
        self.session.game.set_starting_player(
            self.session.game.get_player_by_name(auction_state.highest_bidder_name)
        )

    def _advance_or_finalize_auction(self) -> None:
        if self.session.auction.state.is_complete:
            self._finalize_auction()

    def _apply_bid_penalty(self, round_score: dict) -> dict:
        for result in round_score["results"]:
            result["match_delta"] = result["total_points"]
            result["bid_amount"] = None
            result["made_bid"] = None

        auction_state = self.session.auction.state
        bidder_name = auction_state.highest_bidder_name
        bid_amount = auction_state.highest_bid
        bid_summary = {
            "bidder_name": bidder_name,
            "unit_name": None,
            "amount": bid_amount,
            "points_won": None,
            "made_bid": None,
            "match_delta": None,
        }

        if bidder_name is None or bid_amount is None:
            round_score["bid_summary"] = bid_summary
            return round_score

        bidder_result = next(
            (
                result
                for result in round_score["results"]
                if bidder_name in result["member_names"]
            ),
            None,
        )
        if bidder_result is None:
            raise ValueError("could not map highest bidder to a scoring unit")

        made_bid = bidder_result["total_points"] >= bid_amount
        bidder_result["bid_amount"] = bid_amount
        bidder_result["made_bid"] = made_bid
        bidder_result["match_delta"] = (
            bidder_result["total_points"] if made_bid else -bid_amount
        )

        bid_summary.update(
            {
                "unit_name": bidder_result["name"],
                "points_won": bidder_result["total_points"],
                "made_bid": made_bid,
                "match_delta": bidder_result["match_delta"],
            }
        )
        round_score["bid_summary"] = bid_summary
        return round_score

    def _apply_match_score_updates(self, round_score: dict) -> None:
        auction_winning_unit_name = round_score["bid_summary"]["unit_name"]
        non_bidder_win_cap = self.session.target_score - 1

        for result in round_score["results"]:
            current_score = self.session.match_scores[result["name"]]
            next_score = current_score + result["match_delta"]

            if (
                result["name"] != auction_winning_unit_name
                and next_score > non_bidder_win_cap
            ):
                next_score = non_bidder_win_cap
                result["match_delta"] = next_score - current_score

            self.session.match_scores[result["name"]] = next_score

    def _score_terminal_round_if_needed(self) -> None:
        if (
            not self.session.game.round_state.is_terminal
            or self.session.last_round_score is not None
        ):
            return

        round_score = self._apply_bid_penalty(
            score_round_details(self.session.game.round_state),
        )
        self._apply_match_score_updates(round_score)
        self.session.last_round_score = round_score

    def get_state(self) -> GameSession:
        self._score_terminal_round_if_needed()
        return self.session

    def get_legal_actions(self) -> list[dict]:
        self._score_terminal_round_if_needed()
        if self.session.phase == "auction":
            return [
                serialize_auction_event(action)
                for action in self.session.auction.legal_actions()
            ]
        if self.session.phase in {"round_complete", "match_complete"}:
            return []
        return [
            {"type": "play_card", "card_code": card.code}
            for card in sort_cards(get_legal_actions(self.session.game.round_state))
        ]

    def reset_round(self, auto_run_bots: bool = False) -> GameSession:
        self.session.game.reset_round()
        self.session.last_round_score = None
        self.session.auction = self._build_auction_state(
            self.session.player_names,
            self.session.dealer_index,
        )
        self._sync_all_bot_hands()
        if auto_run_bots:
            self.run_bot_turns()
        return self.session

    def next_round(self, auto_run_bots: bool = False) -> GameSession:
        if not self.session.game.round_state.is_terminal:
            raise ValueError("the current round must be complete before starting the next round")

        self._score_terminal_round_if_needed()
        if self.session.is_match_complete:
            raise ValueError("the match is complete; start a new game to play again")

        next_dealer_index = (self.session.dealer_index + 1) % len(self.session.player_names)
        self.session.game.reset_round()
        self.session.dealer_index = next_dealer_index
        self.session.auction = self._build_auction_state(
            self.session.player_names,
            next_dealer_index,
        )
        self.session.round_number += 1
        self.session.last_round_score = None
        self._sync_all_bot_hands()
        if auto_run_bots:
            self.run_bot_turns()
        return self.session

    def place_bid(self, amount: int, auto_run_bots: bool = False) -> GameSession:
        if self.session.phase != "auction":
            raise ValueError("auction is not active")
        bidder_name = self._current_bidder_name()
        self.session.auction.apply_event(
            AuctionEvent(bidder_name=bidder_name, action="bid", amount=amount)
        )
        self._advance_or_finalize_auction()
        if auto_run_bots:
            self.run_bot_turns()
        return self.session

    def pass_auction(self, auto_run_bots: bool = False) -> GameSession:
        if self.session.phase != "auction":
            raise ValueError("auction is not active")
        bidder_name = self._current_bidder_name()
        self.session.auction.apply_event(
            AuctionEvent(bidder_name=bidder_name, action="pass")
        )
        self._advance_or_finalize_auction()
        if auto_run_bots:
            self.run_bot_turns()
        return self.session

    def play_card(self, card_code: str, auto_run_bots: bool = False) -> GameSession:
        if self.session.phase != "play":
            raise ValueError("the auction must complete before cards can be played")
        card = Card(card_code.upper().strip())
        play = Play(self.session.game.curr_player, card)
        self.session.game.apply_trick_action(play)
        self._score_terminal_round_if_needed()
        if auto_run_bots:
            self.run_bot_turns()
        return self.session

    def get_score(self) -> dict:
        if not self.session.game.round_state.is_terminal:
            raise RoundNotTerminalError("Round is not terminal yet.")
        self._score_terminal_round_if_needed()
        if self.session.last_round_score is None:
            raise RoundNotTerminalError("Round score is unavailable.")
        return self.session.last_round_score

    def run_match(self, alpha: int) -> MatchResult:
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for round_index in range(1, alpha + 1):
            self.run_bot_turns()
            if self.session.phase in {"auction", "play"}:
                waiting_on = self._current_bot_name()
                raise ValueError(
                    "match simulation requires bot-controlled players for all active turns; "
                    f"waiting on '{waiting_on}'"
                )

            if self.session.is_match_complete:
                return MatchResult(
                    rounds_played=round_index,
                    is_draw=False,
                    winner_names=self.session.match_winner_names,
                    final_scores=dict(self.session.match_scores),
                )

            if round_index < alpha:
                self.next_round(auto_run_bots=False)

        return MatchResult(
            rounds_played=alpha,
            is_draw=True,
            winner_names=[],
            final_scores=dict(self.session.match_scores),
        )
