from __future__ import annotations

from dataclasses import dataclass, field
import random

try:
    from .constants import HAND_SIZE, CARD_DICT, RANKS, RANK_ORDER
except ImportError:
    from constants import HAND_SIZE, CARD_DICT, RANKS, RANK_ORDER


# CARDS

def get_max_card(lst: list[Card] | set[Card]) -> Card:
    """Returns the card with the highest rank from a list of cards.

    Args:
        lst (list[Card]): A non-empty list of cards to evaluate.

    Returns:
        Card: The card with the highest rank according to RANK_ORDER.

    Raises:
        ValueError: If the list is empty.
    """
    if not lst:
        raise ValueError("lst must not be empty")
    return max(lst, key=lambda card: RANK_ORDER.get(card.rank, -1))


def get_cards_value(cards: list[Card] | set[Card]) -> int:
    """Returns the value of the cards passed. I.e., if passed a joker, jack, and an ace, return 3."""
    valuable_ranks = {'J', 'A'}
    count = 0
    for card in cards:
        if card.is_joker or card.rank in valuable_ranks:
            count += 1
    return count


def would_win(card: Card, trick_state: TrickState) -> bool:
    """Returns whether the given card would win in the passed trick state.

    This answers whether `card` would currently be the winning card if played
    next into `trick_state`. It mirrors the same precedence used by
    `get_trick_winner` in the engine:
    trump > first joker > lead suit, with rank breaking ties inside a suit.

    Args:
        card (Card): The card that would hypothetically be played.
        trick_state (TrickState): The trick that the card would be played in.

    Returns:
        bool: Whether or not the card would win in the trick
    """
    if not trick_state.plays:
        return True

    trump_high_rank = -1
    joker_played = False
    lead_suit = None
    lead_high_rank = -1
    for play in trick_state.plays:
        played_card = play.card
        if played_card.is_joker:
            joker_played = True
            continue
        played_rank = RANK_ORDER[played_card.rank]
        if played_card.suit == trick_state.trump:
            trump_high_rank = max(trump_high_rank, played_rank)
            continue
        if lead_suit is None:
            lead_suit = played_card.suit
        if played_card.suit == lead_suit:
            lead_high_rank = max(lead_high_rank, played_rank)

    if not card.is_joker and card.suit == trick_state.trump:
        return trump_high_rank < RANK_ORDER[card.rank]
    if trump_high_rank >= 0:
        return False

    if card.is_joker:
        return not joker_played
    if joker_played:
        return False

    if lead_suit is None:
        return True
    if card.suit != lead_suit:
        return False

    return RANK_ORDER[card.rank] > lead_high_rank


def get_rank_from_card_code(card_code: str) -> str:
    """Extract rank from card code (e.g. '2D' -> '2', '10H' -> '10', 'KS' -> 'K')"""
    return card_code[:-1]


@dataclass(frozen=True, slots=True)
class Card:
    code: str
    _is_joker: bool = field(init=False, repr=False, compare=False, hash=False)
    _rank: str | None = field(init=False, repr=False, compare=False, hash=False)
    _suit: str | None = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if self.code not in CARD_DICT:
            raise ValueError(f"Invalid card code: {self.code}")
        is_joker = self.code in {"J1", "J2"}
        object.__setattr__(self, "_is_joker", is_joker)
        object.__setattr__(self, "_rank", None if is_joker else self.code[:-1])
        object.__setattr__(self, "_suit", None if is_joker else self.code[-1])

    @property
    def is_joker(self) -> bool:
        return self._is_joker

    @property
    def rank(self) -> str | None:
        return self._rank

    @property
    def suit(self) -> str | None:
        return self._suit

    def __str__(self) -> str:
        return CARD_DICT[self.code]


class Deck:

    def __init__(self, low: str):
        if low not in RANKS:
            raise ValueError(f"low value of {low} is not valid")

        self._deck = []
        low_order = RANK_ORDER[low]
        for card_code in CARD_DICT.keys():
            rank = get_rank_from_card_code(card_code)
            rank_order = RANK_ORDER.get(rank)

            if rank_order is None or rank_order >= low_order:
                self._deck.append(Card(card_code))

    def shuffle(self) -> None:
        random.shuffle(self._deck)

    def get_copy(self) -> list[Card]:
        return self._deck.copy()

    def __str__(self) -> str:
        return f"deck: {self._deck}"

    @property
    def low(self) -> str:
        non_joker_ranks = [
            card.rank for card in self._deck if not card.is_joker]
        return min(non_joker_ranks, key=lambda rank: RANK_ORDER[rank])


# PLAYERS


class Player:

    def __init__(self, name: str, cards: set[Card] = None):
        if cards is None:
            cards = set()
        self.name = name
        self._cards = cards
        self._captured_plays = set()
        self.score = 0

    def __str__(self) -> str:
        return f"Player {self.name} has {self._cards} in their hands and {self.captured_cards} captured"

    def capture(self, play: Play):
        self._captured_plays.add(play)

    def play_card(self, card: Card):
        if card in self._cards:
            self._cards.remove(card)
        else:
            raise ValueError(f"{card} is not in {self.name}'s hand")

    def receive_new_hand(self, cards: set[Card]):
        if len(cards) != HAND_SIZE:
            raise ValueError(
                f"not enough cards were dealt, expected {HAND_SIZE} but got {len(cards)}")
        self._cards = cards
        self._captured_plays = set()
        self.score = 0

    @property
    def captured_plays(self) -> set[Play]:
        return self._captured_plays

    @property
    def captured_cards(self) -> set[Card]:
        return {play.card for play in self._captured_plays}

    @property
    def cards(self) -> set[Card]:
        return self._cards

    def is_out_of_cards(self) -> bool:
        return self._cards == set()


@dataclass
class Team:
    constituents: list[Player]
    captured_cards: set[Card]

    def __str__(self):
        return f"team with {[player.name for player in self.constituents]}"

    def __contains__(self, player: Player) -> bool:
        return player in self.constituents


@dataclass
class AuctionEvent:
    bidder_name: str
    action: str
    amount: int | None = None


@dataclass
class AuctionState:
    dealer_index: int
    current_bidder_index: int
    player_names: list[str]
    highest_bid: int | None = None
    highest_bidder_name: str | None = None
    passed_player_names: set[str] = field(default_factory=set)
    bid_history: list[AuctionEvent] = field(default_factory=list)
    is_complete: bool = False

    @property
    def dealer_name(self) -> str:
        return self.player_names[self.dealer_index]

    @property
    def current_bidder_name(self) -> str:
        return self.player_names[self.current_bidder_index]

    @property
    def active_player_names(self) -> list[str]:
        return [
            name for name in self.player_names
            if name not in self.passed_player_names
        ]


# GAMEPLAY


@dataclass(frozen=True)
class Play:
    """A player and a card
    """
    player: Player
    card: Card


@dataclass
class TrickState:
    leader: Player
    plays: list[Play]
    players: list[Player]
    trump: str | None

    @property
    def has_trump_played(self):
        for play in self.plays:
            if play.card.suit == self.trump:
                return True
        return False

    @property
    def has_joker_played(self):
        for play in self.plays:
            if play.card.is_joker:
                return True
        return False

    @property
    def trump_suit(self):
        if self.has_trump_played:
            return self.trump
        for play in self.plays:
            if not play.card.is_joker:
                return play.card.suit
        return None

    @property
    def is_terminal(self):
        return len(self.plays) == len(self.players)


@dataclass
class RoundState:
    players: list[Player]
    current_player: Player
    trump: str | None
    current_trick: TrickState
    hidden_cards: set[Card]
    trick_history: list[TrickState]
    teams: list[Team]
    deck: Deck

    @property
    def is_terminal(self):
        for player in self.players:
            if not player.is_out_of_cards():
                return False
        return len(self.trick_history) == HAND_SIZE and self.current_trick.plays == []
