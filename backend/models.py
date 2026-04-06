from dataclasses import dataclass
import random

from engine import Game

# CARDS


CARD_DICT = {
    "AD": "Ace of Diamonds",
    "2D": "Two of Diamonds",
    "3D": "Three of Diamonds",
    "4D": "Four of Diamonds",
    "5D": "Five of Diamonds",
    "6D": "Six of Diamonds",
    "7D": "Seven of Diamonds",
    "8D": "Eight of Diamonds",
    "9D": "Nine of Diamonds",
    "10D": "Ten of Diamonds",
    "JD": "Jack of Diamonds",
    "QD": "Queen of Diamonds",
    "KD": "King of Diamonds",
    "AH": "Ace of Hearts",
    "2H": "Two of Hearts",
    "3H": "Three of Hearts",
    "4H": "Four of Hearts",
    "5H": "Five of Hearts",
    "6H": "Six of Hearts",
    "7H": "Seven of Hearts",
    "8H": "Eight of Hearts",
    "9H": "Nine of Hearts",
    "10H": "Ten of Hearts",
    "JH": "Jack of Hearts",
    "QH": "Queen of Hearts",
    "KH": "King of Hearts",
    "AS": "Ace of Spades",
    "2S": "Two of Spades",
    "3S": "Three of Spades",
    "4S": "Four of Spades",
    "5S": "Five of Spades",
    "6S": "Six of Spades",
    "7S": "Seven of Spades",
    "8S": "Eight of Spades",
    "9S": "Nine of Spades",
    "10S": "Ten of Spades",
    "JS": "Jack of Spades",
    "QS": "Queen of Spades",
    "KS": "King of Spades",
    "AC": "Ace of Clubs",
    "2C": "Two of Clubs",
    "3C": "Three of Clubs",
    "4C": "Four of Clubs",
    "5C": "Five of Clubs",
    "6C": "Six of Clubs",
    "7C": "Seven of Clubs",
    "8C": "Eight of Clubs",
    "9C": "Nine of Clubs",
    "10C": "Ten of Clubs",
    "JC": "Jack of Clubs",
    "QC": "Queen of Clubs",
    "KC": "King of Clubs",
    "J1": "Joker 1",
    "J2": "Joker 2",
}

SUITS = {
    "H": "Hearts",
    "D": "Diamonds",
    "C": "Clubs",
    "S": "Spades"
}

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_ORDER = {rank: i for i, rank in enumerate(RANKS, start=2)}
GAME_VALUES = {
    "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0,
    "8": 0, "9": 0, "10": 10, "J": 1, "Q": 2, "K": 3, "A": 4,
}


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

    def __init__(self, low):
        if str(low) not in RANKS:
            raise ValueError(f"low value of {low} is not valid")

        self._deck = []
        for card in CARD_DICT.values():
            if card[:-1].isalpha():
                self._deck.append(card)
            elif int(card[:-1]) >= low:
                self._deck.append(card)

    def shuffle(self) -> None:
        random.shuffle(self._deck)

    @property
    def deck(self) -> list[Card]:
        return self._deck

    def __str__(self) -> str:
        return f"deck: {self._deck}"


# PLAYERS


class Player:
    name: str
    cards: set[Card]
    captured_cards: set[Card]

    def __init__(self, name: str, cards: set[Card]):
        if len(cards) != Game.hand_size:
            raise ValueError(
                f"player {name} was not dealt {Game.hand_size} cards, they were dealt {len(cards)} cards")
        self._cards = cards
        self._captured_cards = []

    def __str__(self) -> str:
        return f"Player {self.name} has {self.cards} in their hands and {self.captured_cards} captured"

    def capture(self, card: Card):
        self._captured_cards.add(card)

    def play_card(self, card: Card):
        if card in self._cards:
            self._cards.remove(card)
        else:
            raise ValueError(f"{card} is not in {self.name}'s hand")

    @property
    def captured_cards(self):
        return self._captured_cards

    @property
    def cards(self):
        return self._cards


@dataclass
class Team:
    constituents: list[Player]
    captured_cards: list[Card]

    def __str__(self):
        return f"team with {[player.name for player in self.constituents]}"
