try:
    from backend.constants import GAME_VALUES, RANK_ORDER
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
        undo_trick_action_for_search,
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
    from .greedy_bot import GreedyPlayer
    from .base import BotPlayer
except ImportError:
    from constants import GAME_VALUES, RANK_ORDER
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
        undo_trick_action_for_search,
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
    from .greedy_bot import GreedyPlayer
    from .base import BotPlayer


class OMNISCIENT_MinimaxOneTrickPlayer(BotPlayer):
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
        team_member_names = self._team_member_names(round_state)
        starting_trick_count = len(round_state.trick_history)

        def _choose_card_helper(r: RoundState, alpha: float, beta: float) -> int:
            if len(r.trick_history) > starting_trick_count:
                return self._evaluate_state(
                    r,
                    starting_trick_count,
                    team_member_names,
                )

            maximizing = r.current_player.name in team_member_names

            if maximizing:
                value = float("-inf")
                for card in get_legal_actions(r):
                    play = Play(r.current_player, card)
                    undo = apply_trick_action_for_search(r, play)
                    try:
                        value = max(value, _choose_card_helper(r, alpha, beta))
                    finally:
                        undo_trick_action_for_search(r, undo)
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                return value

            value = float("inf")
            for card in get_legal_actions(r):
                play = Play(r.current_player, card)
                undo = apply_trick_action_for_search(r, play)
                try:
                    value = min(value, _choose_card_helper(r, alpha, beta))
                finally:
                    undo_trick_action_for_search(r, undo)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

        best_card = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        legal_actions = list(get_legal_actions(round_state))
        opening_suit_override = getattr(self, "_opening_suit_override", None)
        if round_state.trump is None and opening_suit_override is not None:
            suited_actions = [
                card for card in legal_actions if card.suit == opening_suit_override
            ]
            if suited_actions:
                legal_actions = suited_actions
        else:
            self._opening_suit_override = None

        for card in legal_actions:
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                score = _choose_card_helper(round_state, alpha, beta)
            finally:
                undo_trick_action_for_search(round_state, undo)

            if score > best_score:
                best_score = score
                best_card = card

            alpha = max(alpha, best_score)

        return best_card

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        rollout_proxy = GreedyPlayer(self.name)
        rollout_proxy._cards = set(self.cards)
        rollout_proxy._opening_suit_override = getattr(
            self,
            "_opening_suit_override",
            None,
        )
        if hasattr(self, "_context_player_names"):
            rollout_proxy._context_player_names = list(
                self._context_player_names)
        if hasattr(self, "_context_teams"):
            rollout_proxy._context_teams = list(self._context_teams)
        if hasattr(self, "_context_match_scores") and self._context_match_scores is not None:
            rollout_proxy._context_match_scores = dict(
                self._context_match_scores)
        if hasattr(self, "_context_target_score"):
            rollout_proxy._context_target_score = self._context_target_score

        action = rollout_proxy.choose_auction_action(auction_state)
        self._opening_suit_override = rollout_proxy._opening_suit_override
        return action
