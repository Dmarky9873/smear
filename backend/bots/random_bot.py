from random import choice

try:
    from backend.engine import get_legal_actions, get_legal_auction_actions
    from backend.models import AuctionEvent, AuctionState, Card, RoundState
    from .base import BotPlayer
except ImportError:
    from engine import get_legal_actions, get_legal_auction_actions
    from models import AuctionEvent, AuctionState, Card, RoundState
    from bots.base import BotPlayer


class RandomPlayer(BotPlayer):
    def choose_card(self, round_state: RoundState) -> Card:
        return choice(list(get_legal_actions(round_state)))

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        return choice(get_legal_auction_actions(auction_state))
