from random import choice

try:
    from backend.engine import get_legal_actions, get_legal_auction_actions
    from backend.models import AuctionEvent, AuctionState, Card, RoundState, would_win, get_cards_value
    from .base import BotPlayer
except ImportError:
    from engine import get_legal_actions, get_legal_auction_actions
    from models import AuctionEvent, AuctionState, Card, RoundState, would_win, get_cards_value
    from bots.base import BotPlayer

try:
    from .base import BotPlayer
except ImportError:
    from base import BotPlayer


class StupidBot(BotPlayer):
    def choose_card(self, round_state: RoundState) -> Card:
        """For the stupid player, play the card with the highest expected value
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

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        """For the stupid player, always bid one higher than the previous, if possible."""
        legal_actions = get_legal_auction_actions(auction_state)
        potential_bid = (
            1 if auction_state.highest_bid is None
            else auction_state.highest_bid + 1
        )
        for action in legal_actions:
            if action.amount == potential_bid:
                return action
        for action in legal_actions:
            if action.action == "pass":
                return action
        raise ValueError("greedy bot could not find a legal auction action")
