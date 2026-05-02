try:
    from backend.constants import GAME_VALUES, RANK_ORDER
    from backend.engine import (
        apply_trick_action_to_state,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
    )
    from backend.models import (
        AuctionEvent,
        AuctionState,
        Card,
        Play,
        RoundState,
        get_cards_value,
        TrickState,
    )
    from .base import BotPlayer
except ImportError:
    from constants import GAME_VALUES, RANK_ORDER
    from engine import (
        apply_trick_action_to_state,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
    )
    from models import (
        AuctionEvent,
        AuctionState,
        Card,
        Play,
        RoundState,
        get_cards_value,
        TrickState,
    )
    from bots.base import BotPlayer


class MinimaxOneTrickPlayer(BotPlayer):
    preferred_suit = "H"

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

    def _select_preferred_suit(self) -> str:
        candidate_suits = ("H", "D", "C", "S")
        return max(candidate_suits, key=self._preferred_suit_key)

    def _trump_cards(self, trump_suit: str) -> list[Card]:
        return [
            card for card in self.cards
            if not card.is_joker and card.suit == trump_suit
        ]

    def _has_high_trump(self, trump_suit: str) -> bool:
        return any(card.rank == "A" for card in self._trump_cards(trump_suit))

    def _has_low_trump_candidate(self, trump_suit: str) -> bool:
        trump_cards = self._trump_cards(trump_suit)
        if not trump_cards:
            return False
        lowest_trump = min(trump_cards, key=lambda card: RANK_ORDER[card.rank])
        return RANK_ORDER[lowest_trump.rank] <= RANK_ORDER["10"]

    def _count_jokers(self) -> int:
        return sum(1 for card in self.cards if card.is_joker)

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

    def _apply_action(self, round_state: RoundState, action: Play) -> RoundState:
        """Takes a round state and action and returns a new roundstate with that action applied

        Args:
            round_state (RoundState): The original round state
            action (Play): The action to be applied

        Returns:
            RoundState: The new round state with that action applied
        """
        return apply_trick_action_to_state(round_state, action)

    def _team_member_names(self, round_state: RoundState) -> set[str]:
        for team in round_state.teams:
            member_names = {player.name for player in team.constituents}
            if self.name in member_names:
                return member_names
        return {self.name}

    def _evaluate_trick(self, trick: TrickState, team_member_names: set[str]) -> int:
        """Return the signed value of a completed trick from this bot's perspective."""
        winner_name = get_trick_winner(trick).name
        trick_value = get_cards_value({play.card for play in trick.plays})
        if winner_name in team_member_names:
            return trick_value
        return -trick_value

    def _evaluate_state(
        self,
        round_state: RoundState,
        starting_trick_count: int,
        team_member_names: set[str],
    ) -> int:
        """Return the current trick value once the searched trick completes.

        Args:
            round_state (RoundState): The current round state after search actions.
            starting_trick_count (int): Trick count before the search started.
            team_member_names (set[str]): The bot's team by player name.

        Returns:
            int: Signed trick value for this bot's team.
        """
        if len(round_state.trick_history) <= starting_trick_count:
            raise ValueError("searched trick has not completed")
        return self._evaluate_trick(
            round_state.trick_history[-1],
            team_member_names,
        )

    def choose_card(self, round_state: RoundState) -> Card:
        """Choose the legal card with the best minimax value for the current trick.
        """
        team_member_names = self._team_member_names(round_state)
        starting_trick_count = len(round_state.trick_history)

        def _choose_card_helper(r: RoundState) -> int:
            if len(r.trick_history) > starting_trick_count:
                return self._evaluate_state(
                    r,
                    starting_trick_count,
                    team_member_names,
                )

            scores = []
            for card in get_legal_actions(r):
                play = Play(r.current_player, card)
                scores.append(_choose_card_helper(self._apply_action(r, play)))

            if r.current_player.name in team_member_names:
                return max(scores)
            return min(scores)

        legal_moves = get_legal_actions(round_state)
        move_val_dict: dict[Play, int] = {}

        for card in legal_moves:
            play = Play(round_state.current_player, card)
            move_val_dict[play] = _choose_card_helper(
                self._apply_action(round_state, play)
            )

        max_card, max_score = None, float("-inf")
        for move, score in move_val_dict.items():
            if score > max_score or (
                score == max_score and (
                    max_card is None or move.card.code > max_card.code)
            ):
                max_score = score
                max_card = move.card

        return max_card

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        """Bid only when the next required bid fits the hand's best trump estimate."""
        preferred_suit = self._select_preferred_suit()
        legal_actions = get_legal_auction_actions(auction_state)
        next_bid = (
            1 if auction_state.highest_bid is None
            else auction_state.highest_bid + 1
        )
        best_estimated_strength = self._estimate_hand_strength(
            preferred_suit)

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
