from __future__ import annotations

try:
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from backend.models import Card, Play, RoundState
    from .omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
except ImportError:
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from models import Card, Play, RoundState
    from .omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer


class OmniscientMinimaxNTrickPlayer(OmniscientMinimaxOneTrickPlayer):
    """Perfect-information minimax over the next N completed tricks."""

    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
        *,
        depth: int = 2,
    ):
        super().__init__(name, cards)
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth

    def _evaluate_state(
        self,
        round_state: RoundState,
        starting_trick_count: int,
        team_member_names: set[str],
    ) -> int:
        if len(round_state.trick_history) <= starting_trick_count:
            raise ValueError("searched depth has not completed any new tricks")
        return sum(
            self._evaluate_trick(trick, team_member_names)
            for trick in round_state.trick_history[starting_trick_count:]
        )

    def _search_n_trick_value(
        self,
        round_state: RoundState,
        starting_trick_count: int,
        target_trick_count: int,
        team_member_names: set[str],
        alpha: float,
        beta: float,
    ) -> int:
        if (
            len(round_state.trick_history) >= target_trick_count
            or round_state.is_terminal
        ):
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
                        self._search_n_trick_value(
                            round_state,
                            starting_trick_count,
                            target_trick_count,
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
                    self._search_n_trick_value(
                        round_state,
                        starting_trick_count,
                        target_trick_count,
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
        if round_state.current_player.name != self.name:
            raise ValueError(
                f"{self.name} cannot choose a card for {round_state.current_player.name}"
            )

        team_member_names = self._team_member_names(round_state)
        starting_trick_count = len(round_state.trick_history)
        target_trick_count = starting_trick_count + self.depth
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
            label=f"Searching {self.depth} tricks ahead",
            total_units=len(legal_actions),
        )

        try:
            for index, card in enumerate(legal_actions, start=1):
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    score = self._search_n_trick_value(
                        round_state,
                        starting_trick_count,
                        target_trick_count,
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


OMNISCIENT_MinimaxNTrickPlayer = OmniscientMinimaxNTrickPlayer
