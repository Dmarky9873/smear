from __future__ import annotations

from dataclasses import dataclass
import random

from constants import HAND_SIZE, CARD_DICT, RANKS, RANK_ORDER


# CARDS

def get_max_card(lst: list[Card]) -> Card:
    if not lst:
        raise ValueError("lst must not be empty")
    return max(lst, key=lambda card: RANK_ORDER.get(card.rank, -1))


def get_rank_from_card_code(card_code: str) -> str:
    """Extract rank from card code (e.g. '2D' -> '2', '10H' -> '10', 'KS' -> 'K')"""
    return card_code[:-1]


@dataclass(frozen=True)
class Card:
    code: str

    def __post_init__(self):
        if self.code not in CARD_DICT:
            raise ValueError(f"Invalid card code: {self.code}")

    @property
    def is_joker(self) -> bool:
        return self.code in {"J1", "J2"}

    @property
    def rank(self) -> str | None:
        return None if self.is_joker else self.code[:-1]

    @property
    def suit(self) -> str | None:
        return None if self.is_joker else self.code[-1]

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


# PLAYERS


class Player:

    def __init__(self, name: str, cards: set[Card] = None):
        if cards is None:
            cards = set()
        self.name = name
        self._cards = cards
        self._captured_plays = set()

    def __str__(self) -> str:
        return f"Player {self.name} has {self._cards} in their hands and {self._captured_cards} captured"

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

    @property
    def captured_cards(self) -> set[Card]:
        return self._captured_cards

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


# GAMEPLAY


@dataclass
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
        return self.current_trick.is_terminal
