from __future__ import annotations

try:
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_trick_winner,
        undo_trick_action_for_search,
    )
    from backend.models import (
        AuctionEvent,
        AuctionState,
        Card,
        Play,
        RoundState,
        TrickState,
        get_cards_value,
    )
    from .base import BotPlayer
    from .greedy_bot import GreedyPlayer
except ImportError:
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_trick_winner,
        undo_trick_action_for_search,
    )
    from models import (
        AuctionEvent,
        AuctionState,
        Card,
        Play,
        RoundState,
        TrickState,
        get_cards_value,
    )
    from .base import BotPlayer
    from .greedy_bot import GreedyPlayer


class OmniscientMinimaxOneTrickPlayer(BotPlayer):
    def _team_member_names(self, round_state: RoundState) -> set[str]:
        for team in round_state.teams:
            member_names = {player.name for player in team.constituents}
            if self.name in member_names:
                return member_names
        return {self.name}

    def _evaluate_trick(self, trick: TrickState, team_member_names: set[str]) -> int:
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
        if len(round_state.trick_history) <= starting_trick_count:
            raise ValueError("searched trick has not completed")
        return self._evaluate_trick(
            round_state.trick_history[-1],
            team_member_names,
        )

    def _search_one_trick_value(
        self,
        round_state: RoundState,
        starting_trick_count: int,
        team_member_names: set[str],
        alpha: float,
        beta: float,
    ) -> int:
        if len(round_state.trick_history) > starting_trick_count:
            return self._evaluate_state(
                round_state,
                starting_trick_count,
                team_member_names,
            )

        maximizing = round_state.current_player.name in team_member_names

        if maximizing:
            value = float("-inf")
            for card in get_legal_actions(round_state):
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    value = max(
                        value,
                        self._search_one_trick_value(
                            round_state,
                            starting_trick_count,
                            team_member_names,
                            alpha,
                            beta,
                        ),
                    )
                finally:
                    undo_trick_action_for_search(round_state, undo)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = float("inf")
        for card in get_legal_actions(round_state):
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                value = min(
                    value,
                    self._search_one_trick_value(
                        round_state,
                        starting_trick_count,
                        team_member_names,
                        alpha,
                        beta,
                    ),
                )
            finally:
                undo_trick_action_for_search(round_state, undo)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def choose_card(self, round_state: RoundState) -> Card:
        team_member_names = self._team_member_names(round_state)
        starting_trick_count = len(round_state.trick_history)
        legal_actions = sorted(get_legal_actions(round_state), key=lambda card: card.code)
        opening_suit_override = getattr(self, "_opening_suit_override", None)
        if round_state.trump is None and opening_suit_override is not None:
            suited_actions = [
                card for card in legal_actions if card.suit == opening_suit_override
            ]
            if suited_actions:
                legal_actions = suited_actions
        else:
            self._opening_suit_override = None

        best_card = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        self.begin_progress(
            label="Searching current trick",
            total_units=len(legal_actions),
        )

        try:
            for index, card in enumerate(legal_actions, start=1):
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    score = self._search_one_trick_value(
                        round_state,
                        starting_trick_count,
                        team_member_names,
                        alpha,
                        beta,
                    )
                finally:
                    undo_trick_action_for_search(round_state, undo)

                if score > best_score:
                    best_score = score
                    best_card = card

                alpha = max(alpha, best_score)
                self.update_progress(
                    completed_units=index,
                    detail=f"Evaluated {card.code}",
                )
        finally:
            self.clear_progress()

        if best_card is None:
            raise ValueError("omniscient minimax bot could not find a legal card")
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
            rollout_proxy._context_player_names = list(self._context_player_names)
        if hasattr(self, "_context_teams"):
            rollout_proxy._context_teams = list(self._context_teams)
        if hasattr(self, "_context_match_scores") and self._context_match_scores is not None:
            rollout_proxy._context_match_scores = dict(self._context_match_scores)
        if hasattr(self, "_context_target_score"):
            rollout_proxy._context_target_score = self._context_target_score

        action = rollout_proxy.choose_auction_action(auction_state)
        self._opening_suit_override = rollout_proxy._opening_suit_override
        return action


OMNISCIENT_MinimaxOneTrickPlayer = OmniscientMinimaxOneTrickPlayer
