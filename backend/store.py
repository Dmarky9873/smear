from __future__ import annotations

from dataclasses import dataclass, field

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


class GameNotInitializedError(RuntimeError):
    """Raised when a request needs a game but none exists."""


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


def _sort_cards(cards: list[Card] | set[Card]) -> list[Card]:
    return sorted(
        cards,
        key=lambda card: (
            1 if card.is_joker else 0,
            SUIT_ORDER.get(card.suit or "", 99),
            RANK_ORDER.get(card.rank or "", 99),
            card.code,
        ),
    )


def _normalize_teams(
    player_names: list[str],
    teams: list[list[str]] | None,
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
    player_bots: list[str | None] | None,
) -> dict[str, str | None]:
    if player_bots is None:
        return {name: None for name in player_names}

    if len(player_bots) != len(player_names):
        raise ValueError("player_bots must match the number of player_names")

    normalized: dict[str, str | None] = {}
    for player_name, bot_id in zip(player_names, player_bots):
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


class GameStore:
    def __init__(self):
        self._session: GameSession | None = None

    def require_session(self) -> GameSession:
        if self._session is None:
            raise GameNotInitializedError("No game exists yet. Create one first.")
        return self._session

    def _build_auction_state(
        self,
        player_names: list[str],
        dealer_index: int,
    ) -> Auction:
        return Auction(
            player_names=player_names,
            dealer=player_names[dealer_index],
            max_bid=MAX_BID,
        )

    def _build_initial_match_scores(
        self,
        teams: list[tuple[str, ...]],
    ) -> dict[str, int]:
        return {" / ".join(team): 0 for team in teams}

    def _build_bot_controllers(
        self,
        player_bot_ids: dict[str, str | None],
    ) -> dict[str, BotPlayer]:
        return {
            player_name: build_ready_bot(bot_id, player_name)
            for player_name, bot_id in player_bot_ids.items()
            if bot_id is not None
        }

    def _current_bidder_name(self, session: GameSession) -> str:
        return session.auction.current_bidder_name

    def _current_bot_name(self, session: GameSession) -> str | None:
        if session.phase == "auction":
            return session.auction.current_bidder_name
        if session.phase == "play":
            return session.game.curr_player.name
        return None

    def _run_bot_auction_turn(self, session: GameSession, player_name: str) -> None:
        bot = session.bots[player_name]
        action = bot.choose_auction_action(session.auction.state)
        legal_actions = session.auction.legal_actions()
        if action not in legal_actions:
            raise ValueError(
                f"bot '{player_name}' selected illegal auction action {action}; legal actions are {legal_actions}"
            )

        if action["type"] == "bid":
            session.auction.apply_event(
                AuctionEvent(
                    bidder_name=player_name,
                    action="bid",
                    amount=action["amount"],
                )
            )
        elif action["type"] == "pass":
            session.auction.apply_event(
                AuctionEvent(bidder_name=player_name, action="pass")
            )
        else:
            raise ValueError(f"unsupported bot auction action: {action}")

        self._advance_or_finalize_auction(session)

    def _run_bot_play_turn(self, session: GameSession, player_name: str) -> None:
        bot = session.bots[player_name]
        legal_cards = get_legal_actions(session.game.round_state)
        card = bot.choose_card(session.game.round_state)
        if card not in legal_cards:
            legal_codes = sorted(legal_card.code for legal_card in legal_cards)
            raise ValueError(
                f"bot '{player_name}' selected illegal card {card.code}; legal cards are {legal_codes}"
            )

        session.game.apply_trick_action(Play(session.game.curr_player, card))
        self._score_terminal_round_if_needed(session)

    def _run_bot_turns(self, session: GameSession, max_steps: int = 512) -> None:
        for _ in range(max_steps):
            if session.phase in {"round_complete", "match_complete"}:
                return

            player_name = self._current_bot_name(session)
            if player_name is None:
                return

            if player_name not in session.bots:
                return

            if session.phase == "auction":
                self._run_bot_auction_turn(session, player_name)
                continue

            if session.phase == "play":
                self._run_bot_play_turn(session, player_name)
                continue

            return

        raise ValueError("bot turn loop exceeded the step limit")

    def _finalize_auction(self, session: GameSession) -> None:
        auction_state = session.auction.state
        if auction_state.highest_bidder_name is None:
            raise ValueError("auction cannot complete without a winning bid")
        session.game.set_starting_player(
            session.game.get_player_by_name(auction_state.highest_bidder_name)
        )

    def _advance_or_finalize_auction(self, session: GameSession) -> None:
        if session.auction.state.is_complete:
            self._finalize_auction(session)

    def _apply_bid_penalty(self, session: GameSession, round_score: dict) -> dict:
        for result in round_score["results"]:
            result["match_delta"] = result["total_points"]
            result["bid_amount"] = None
            result["made_bid"] = None

        auction_state = session.auction.state
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

    def _score_terminal_round_if_needed(self, session: GameSession) -> None:
        if not session.game.round_state.is_terminal or session.last_round_score is not None:
            return

        round_score = self._apply_bid_penalty(
            session,
            score_round_details(session.game.round_state),
        )
        session.last_round_score = round_score
        for result in round_score["results"]:
            session.match_scores[result["name"]] += result["match_delta"]

    def create_game(
        self,
        num_players: int,
        player_names: list[str],
        teams: list[list[str]] | None,
        player_bots: list[str | None] | None = None,
    ) -> GameSession:
        normalized_names = [name.strip() for name in player_names]
        if len(normalized_names) != num_players:
            raise ValueError("num_players must match the number of player_names")
        if any(not name for name in normalized_names):
            raise ValueError("player_names cannot contain blank values")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("player_names must be unique")

        normalized_teams = _normalize_teams(normalized_names, teams)
        normalized_player_bots = _normalize_player_bots(
            normalized_names,
            player_bots,
        )
        dealer_index = len(normalized_names) - 1
        game = Game(num_players, normalized_names, normalized_teams)
        self._session = GameSession(
            game=game,
            player_names=normalized_names,
            teams=normalized_teams,
            dealer_index=dealer_index,
            auction=self._build_auction_state(normalized_names, dealer_index),
            player_bot_ids=normalized_player_bots,
            bots=self._build_bot_controllers(normalized_player_bots),
            match_scores=self._build_initial_match_scores(normalized_teams),
        )
        self._run_bot_turns(self._session)
        return self._session

    def reset_round(self) -> GameSession:
        session = self.require_session()
        session.game.reset_round()
        session.last_round_score = None
        session.auction = self._build_auction_state(
            session.player_names,
            session.dealer_index,
        )
        self._run_bot_turns(session)
        return session

    def next_round(self) -> GameSession:
        session = self.require_session()
        if not session.game.round_state.is_terminal:
            raise ValueError("the current round must be complete before starting the next round")

        self._score_terminal_round_if_needed(session)
        if session.is_match_complete:
            raise ValueError("the match is complete; start a new game to play again")

        next_dealer_index = (session.dealer_index + 1) % len(session.player_names)
        session.game.reset_round()
        session.dealer_index = next_dealer_index
        session.auction = self._build_auction_state(
            session.player_names,
            next_dealer_index,
        )
        session.round_number += 1
        session.last_round_score = None
        self._run_bot_turns(session)
        return session

    def get_state(self) -> GameSession:
        session = self.require_session()
        self._score_terminal_round_if_needed(session)
        return session

    def get_legal_actions(self) -> list[dict]:
        session = self.require_session()
        self._score_terminal_round_if_needed(session)
        if session.phase == "auction":
            return session.auction.legal_actions()
        if session.phase in {"round_complete", "match_complete"}:
            return []
        return [
            {"type": "play_card", "card_code": card.code}
            for card in _sort_cards(get_legal_actions(session.game.round_state))
        ]

    def place_bid(self, amount: int) -> GameSession:
        session = self.require_session()
        if session.phase != "auction":
            raise ValueError("auction is not active")
        bidder_name = self._current_bidder_name(session)
        session.auction.apply_event(
            AuctionEvent(bidder_name=bidder_name, action="bid", amount=amount)
        )
        self._advance_or_finalize_auction(session)
        self._run_bot_turns(session)
        return session

    def pass_auction(self) -> GameSession:
        session = self.require_session()
        if session.phase != "auction":
            raise ValueError("auction is not active")
        bidder_name = self._current_bidder_name(session)
        session.auction.apply_event(AuctionEvent(bidder_name=bidder_name, action="pass"))
        self._advance_or_finalize_auction(session)
        self._run_bot_turns(session)
        return session

    def play_card(self, card_code: str) -> GameSession:
        session = self.require_session()
        if session.phase != "play":
            raise ValueError("the auction must complete before cards can be played")
        game = session.game
        card = Card(card_code.upper().strip())
        play = Play(game.curr_player, card)
        game.apply_trick_action(play)
        self._score_terminal_round_if_needed(session)
        self._run_bot_turns(session)
        return session

    def get_score(self) -> dict:
        session = self.require_session()
        game = session.game
        if not game.round_state.is_terminal:
            raise RoundNotTerminalError("Round is not terminal yet.")
        self._score_terminal_round_if_needed(session)
        if session.last_round_score is None:
            raise RoundNotTerminalError("Round score is unavailable.")
        return session.last_round_score


game_store = GameStore()
