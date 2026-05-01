from random import choice

from backend.engine import get_legal_actions, get_legal_auction_actions
from backend.models import AuctionState, RoundState, Card, would_win

try:
    from .base import BotPlayer
except ImportError:
    from base import BotPlayer


class GreedyPlayer(BotPlayer):
    def choose_card(self, round_state: RoundState) -> Card:
        """For the greedy player, always play the highest (most trumpy) card in legal moves.
        That is:
            - If trump in hand, pick the highest ranked trump card
            - If joker in hand, play a joker
            - If sub-round trump in hand, play the highest ranked sub-round trump card
            - Play the highest ranked card in hand
        """
        legal_cards = get_legal_actions(round_state, self.cards)
        return legal_cards[0]
