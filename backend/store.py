from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .constants import RANK_ORDER
    from .engine import Game, get_legal_actions, score_round_details
    from .models import Card, Play
except ImportError:
    from constants import RANK_ORDER
    from engine import Game, get_legal_actions, score_round_details
    from models import Card, Play


MAX_BID = 6
SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3}


class GameNotInitializedError(RuntimeError):
    """Raised when a request needs a game but none exists."""


class RoundNotTerminalError(RuntimeError):
    """Raised when score is requested before the round is complete."""


@dataclass
class AuctionEvent:
    bidder_name: str
    action: str
    amount: int | None = None


@dataclass
class AuctionState:
    dealer_index: int
    current_bidder_index: int
    highest_bid: int | None = None
    highest_bidder_name: str | None = None
    passed_player_names: set[str] = field(default_factory=set)
    bid_history: list[AuctionEvent] = field(default_factory=list)
    is_complete: bool = False


@dataclass
class GameSession:
    game: Game
    player_names: list[str]
    teams: list[tuple[str, ...]]
    dealer_index: int
    auction: AuctionState

    @property
    def phase(self) -> str:
        if not self.auction.is_complete:
            return "auction"
        if self.game.round_state.is_terminal:
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
    ) -> AuctionState:
        return AuctionState(
            dealer_index=dealer_index,
            current_bidder_index=(dealer_index + 1) % len(player_names),
        )

    def _current_bidder_name(self, session: GameSession) -> str:
        return session.player_names[session.auction.current_bidder_index]

    def _active_player_names(self, session: GameSession) -> list[str]:
        return [
            name
            for name in session.player_names
            if name not in session.auction.passed_player_names
        ]

    def _legal_bid_amounts(self, session: GameSession) -> list[int]:
        auction = session.auction
        if auction.is_complete:
            return []
        minimum_bid = 1 if auction.highest_bid is None else auction.highest_bid + 1
        if minimum_bid > MAX_BID:
            return []
        return list(range(minimum_bid, MAX_BID + 1))

    def _can_pass(self, session: GameSession) -> bool:
        auction = session.auction
        if auction.is_complete:
            return False
        if auction.highest_bidder_name is not None:
            return True
        return len(self._active_player_names(session)) > 1

    def _next_active_bidder_index(
        self,
        session: GameSession,
        start_index: int,
    ) -> int:
        total_players = len(session.player_names)
        for offset in range(1, total_players + 1):
            next_index = (start_index + offset) % total_players
            if session.player_names[next_index] not in session.auction.passed_player_names:
                return next_index
        raise ValueError("no active bidders remain in the auction")

    def _finalize_auction(self, session: GameSession) -> None:
        auction = session.auction
        if auction.highest_bidder_name is None:
            raise ValueError("auction cannot complete without a winning bid")
        auction.is_complete = True
        auction.current_bidder_index = session.player_names.index(
            auction.highest_bidder_name
        )
        session.game.set_starting_player(
            session.game.get_player_by_name(auction.highest_bidder_name)
        )

    def _advance_or_finalize_auction(self, session: GameSession) -> None:
        auction = session.auction
        active_player_names = self._active_player_names(session)

        if auction.highest_bidder_name is not None:
            if (
                auction.highest_bid == MAX_BID
                or active_player_names == [auction.highest_bidder_name]
            ):
                self._finalize_auction(session)
                return

        auction.current_bidder_index = self._next_active_bidder_index(
            session,
            auction.current_bidder_index,
        )

    def create_game(
        self,
        num_players: int,
        player_names: list[str],
        teams: list[list[str]] | None,
    ) -> GameSession:
        normalized_names = [name.strip() for name in player_names]
        if len(normalized_names) != num_players:
            raise ValueError("num_players must match the number of player_names")
        if any(not name for name in normalized_names):
            raise ValueError("player_names cannot contain blank values")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("player_names must be unique")

        normalized_teams = _normalize_teams(normalized_names, teams)
        dealer_index = len(normalized_names) - 1
        game = Game(num_players, normalized_names, normalized_teams)
        self._session = GameSession(
            game=game,
            player_names=normalized_names,
            teams=normalized_teams,
            dealer_index=dealer_index,
            auction=self._build_auction_state(normalized_names, dealer_index),
        )
        return self._session

    def reset_round(self) -> GameSession:
        session = self.require_session()
        next_dealer_index = (session.dealer_index + 1) % len(session.player_names)
        session.game.reset_round()
        session.dealer_index = next_dealer_index
        session.auction = self._build_auction_state(
            session.player_names,
            next_dealer_index,
        )
        return session

    def get_state(self) -> GameSession:
        return self.require_session()

    def get_legal_actions(self) -> list[dict]:
        session = self.require_session()
        if session.phase == "auction":
            actions = [
                {"type": "bid", "amount": amount}
                for amount in self._legal_bid_amounts(session)
            ]
            if self._can_pass(session):
                actions.append({"type": "pass"})
            return actions
        if session.phase == "round_complete":
            return []
        return [
            {"type": "play_card", "card_code": card.code}
            for card in _sort_cards(get_legal_actions(session.game.round_state))
        ]

    def place_bid(self, amount: int) -> GameSession:
        session = self.require_session()
        if session.phase != "auction":
            raise ValueError("auction is not active")
        legal_amounts = self._legal_bid_amounts(session)
        if amount not in legal_amounts:
            raise ValueError(
                f"illegal bid amount {amount}; legal bids are {legal_amounts}"
            )
        bidder_name = self._current_bidder_name(session)
        session.auction.highest_bid = amount
        session.auction.highest_bidder_name = bidder_name
        session.auction.bid_history.append(
            AuctionEvent(
                bidder_name=bidder_name,
                action="bid",
                amount=amount,
            )
        )
        self._advance_or_finalize_auction(session)
        return session

    def pass_auction(self) -> GameSession:
        session = self.require_session()
        if session.phase != "auction":
            raise ValueError("auction is not active")
        if not self._can_pass(session):
            raise ValueError("at least one player must bid before the auction can end")
        bidder_name = self._current_bidder_name(session)
        session.auction.passed_player_names.add(bidder_name)
        session.auction.bid_history.append(
            AuctionEvent(bidder_name=bidder_name, action="pass")
        )
        self._advance_or_finalize_auction(session)
        return session

    def play_card(self, card_code: str) -> GameSession:
        session = self.require_session()
        if session.phase != "play":
            raise ValueError("the auction must complete before cards can be played")
        game = session.game
        card = Card(card_code.upper().strip())
        play = Play(game.curr_player, card)
        game.apply_trick_action(play)
        return session

    def get_score(self) -> dict:
        session = self.require_session()
        game = session.game
        if not game.round_state.is_terminal:
            raise RoundNotTerminalError("Round is not terminal yet.")
        return score_round_details(game.round_state)


game_store = GameStore()
