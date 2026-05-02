try:
    from backend.constants import GAME_VALUES, RANK_ORDER
    from backend.engine import get_legal_actions, get_legal_auction_actions
    from backend.models import (
        AuctionEvent,
        AuctionState,
        Card,
        RoundState,
        get_cards_value,
        would_win,
    )
    from .base import BotPlayer
except ImportError:
    from constants import GAME_VALUES, RANK_ORDER
    from engine import get_legal_actions, get_legal_auction_actions
    from models import (
        AuctionEvent,
        AuctionState,
        Card,
        RoundState,
        get_cards_value,
        would_win,
    )
    from bots.base import BotPlayer


class GreedyPlayer(BotPlayer):
    preferred_suit = "H"
    _candidate_suits = ("H", "D", "C", "S")

    def _trump_cards(self, trump_suit: str) -> list[Card]:
        return [
            card for card in self.cards
            if not card.is_joker and card.suit == trump_suit
        ]

    def _count_jokers(self) -> int:
        return sum(1 for card in self.cards if card.is_joker)

    def _has_high_trump(self, trump_suit: str) -> bool:
        return any(card.rank == "A" for card in self._trump_cards(trump_suit))

    def _has_low_trump_candidate(self, trump_suit: str) -> bool:
        trump_cards = self._trump_cards(trump_suit)
        if not trump_cards:
            return False
        lowest_trump = min(trump_cards, key=lambda card: RANK_ORDER[card.rank])
        return RANK_ORDER[lowest_trump.rank] <= RANK_ORDER["10"]

    def _has_game_strength(self) -> bool:
        game_total = sum(
            GAME_VALUES.get(card.rank, 0)
            for card in self.cards
            if not card.is_joker
        )
        valuable_card_count = sum(
            1
            for card in self.cards
            if not card.is_joker and GAME_VALUES.get(card.rank, 0) > 0
        )
        return game_total >= 16 or valuable_card_count >= 4

    def _has_trump_control(self, trump_suit: str) -> bool:
        trump_cards = self._trump_cards(trump_suit)
        if len(trump_cards) >= 2:
            return True
        if not trump_cards:
            return False
        if self._count_jokers() > 0:
            return True
        return any(card.rank in {"A", "K", "Q", "J"} for card in trump_cards)

    def _estimate_hand_strength(self, trump_suit: str) -> int:
        estimate = 0
        if self._has_high_trump(trump_suit):
            estimate += 1
        if self._has_low_trump_candidate(trump_suit):
            estimate += 1
        estimate += self._count_jokers()
        if self._has_game_strength():
            estimate += 1
        if self._has_trump_control(trump_suit):
            estimate += 1
        return max(0, min(6, estimate))

    def _preferred_suit_key(self, trump_suit: str) -> tuple[int, int, int, int]:
        trump_cards = self._trump_cards(trump_suit)
        highest_rank = max(
            (RANK_ORDER[card.rank] for card in trump_cards),
            default=-1,
        )
        total_rank = sum(RANK_ORDER[card.rank] for card in trump_cards)
        return (
            self._estimate_hand_strength(trump_suit),
            highest_rank,
            len(trump_cards),
            total_rank,
        )

    def _select_preferred_suit(self) -> str:
        return max(self._candidate_suits, key=self._preferred_suit_key)

    def _card_preference_key(self, card: Card) -> tuple[bool, int, str]:
        rank_value = len(RANK_ORDER) + 1 if card.is_joker else RANK_ORDER[card.rank]
        return (
            card.suit == self.preferred_suit,
            rank_value,
            card.code,
        )

    def choose_card(self, round_state: RoundState) -> Card:
        """For the greedy player, play the card with the highest expected value
        """
        if round_state.trump is None:
            self.preferred_suit = self._select_preferred_suit()

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
        return max(best_cards, key=self._card_preference_key)

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        """Bid only when the next required bid fits the hand's best trump estimate."""
        self.preferred_suit = self._select_preferred_suit()
        legal_actions = get_legal_auction_actions(auction_state)
        next_bid = (
            1 if auction_state.highest_bid is None
            else auction_state.highest_bid + 1
        )
        best_estimated_strength = self._estimate_hand_strength(self.preferred_suit)

        if next_bid <= best_estimated_strength:
            for action in legal_actions:
                if action.action == "bid" and action.amount == next_bid:
                    return action

        for action in legal_actions:
            if action.action == "pass":
                return action

        for action in legal_actions:
            if action.action == "bid" and action.amount == next_bid:
                return action

        raise ValueError("greedy bot could not find a legal auction action")
