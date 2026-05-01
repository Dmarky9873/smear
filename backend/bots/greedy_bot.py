from random import choice

from backend.engine import get_legal_actions, get_legal_auction_actions
from backend.models import AuctionState, RoundState, Card, would_win, get_cards_value

try:
    from .base import BotPlayer
except ImportError:
    from base import BotPlayer


class GreedyPlayer(BotPlayer):
    def choose_card(self, round_state: RoundState) -> Card:
        """For the greedy player, play the card with the highest expected value
        """
        val_dict = dict()
        for card in get_legal_actions(round_state):
            val_dict[card] = 0
        for card in val_dict.keys():
            if would_win(card, round_state.current_trick):
                val_dict[card] += get_cards_value(
                    {play.card for play in round_state.current_trick.plays}.union({card}))
            else:
                val_dict[card] -= get_cards_value({card})
        max_value = max(val_dict.values())
        best_cards = [card for card, value in val_dict.items()
                      if value == max_value]
        return choice(best_cards)

    def choose_auction_action(self, auction_state):
        """For the greedy player, always bid one higher than the previous, if possible."""
        return auction_state.highest_bid + 1
