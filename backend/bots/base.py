from __future__ import annotations

from backend.models import AuctionState, Player, RoundState, Card


class BotPlayer(Player):
    """Abstract base bot class"""

    def choose_card(self, round_state: RoundState) -> Card:
        raise NotImplementedError

    def choose_auction_action(self, auction_state: AuctionState):
        raise NotImplementedError
