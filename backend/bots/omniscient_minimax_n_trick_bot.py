from __future__ import annotations

try:
    from backend.constants import GAME_VALUES, RANK_ORDER
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from backend.models import Card, Play, RoundState, get_cards_value, would_win
    from .omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
except ImportError:
    from constants import GAME_VALUES, RANK_ORDER
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from models import Card, Play, RoundState, get_cards_value, would_win
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

    def _card_order_key(self, round_state: RoundState, card: Card) -> tuple:
        trick_cards = {play.card for play in round_state.current_trick.plays}
        wins_trick = would_win(card, round_state.current_trick)
        trick_value = get_cards_value(trick_cards.union({card}))
        card_value = get_cards_value({card})
        is_trump = int(
            card.is_joker
            or (
                round_state.trump is not None
                and not card.is_joker
                and card.suit == round_state.trump
            )
        )
        game_value = GAME_VALUES.get(card.rank, 0) if not card.is_joker else 5
        rank_value = len(RANK_ORDER) + 2 if card.is_joker else RANK_ORDER[card.rank]
        return (
            int(wins_trick),
            trick_value if wins_trick else -card_value,
            is_trump,
            game_value,
            rank_value,
            card.code,
        )

    def _ordered_legal_actions(self, round_state: RoundState) -> list[Card]:
        return sorted(
            get_legal_actions(round_state),
            key=lambda card: self._card_order_key(round_state, card),
            reverse=True,
        )

    def _transposition_key(
        self,
        round_state: RoundState,
        remaining_tricks: int,
    ) -> tuple:
        return (
            remaining_tricks,
            round_state.current_player.name,
            round_state.current_trick.leader.name,
            round_state.trump,
            len(round_state.trick_history),
            tuple(
                (play.player.name, play.card.code)
                for play in round_state.current_trick.plays
            ),
            tuple(
                (player.name, tuple(sorted(card.code for card in player.cards)))
                for player in round_state.players
            ),
        )

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
        remaining_tricks: int,
        team_member_names: set[str],
        alpha: float,
        beta: float,
        transposition_table: dict[tuple, int],
    ) -> tuple[int, bool]:
        if remaining_tricks <= 0 or round_state.is_terminal:
            return 0, True

        state_key = self._transposition_key(round_state, remaining_tricks)
        cached_value = transposition_table.get(state_key)
        if cached_value is not None:
            return cached_value, True

        maximizing = round_state.current_player.name in team_member_names
        trick_count_before = len(round_state.trick_history)
        legal_actions = self._ordered_legal_actions(round_state)
        if not legal_actions:
            return 0, True

        if maximizing:
            value = float("-inf")
            exact_value = True
            for card in legal_actions:
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    trick_completed = len(round_state.trick_history) > trick_count_before
                    immediate_value = (
                        self._evaluate_trick(
                            round_state.trick_history[-1],
                            team_member_names,
                        )
                        if trick_completed
                        else 0
                    )
                    child_value, child_exact = self._search_n_trick_value(
                        round_state,
                        remaining_tricks - 1 if trick_completed else remaining_tricks,
                        team_member_names,
                        alpha,
                        beta,
                        transposition_table,
                    )
                    value = max(
                        value,
                        immediate_value + child_value,
                    )
                    exact_value = exact_value and child_exact
                finally:
                    undo_trick_action_for_search(round_state, undo)
                alpha = max(alpha, value)
                if beta <= alpha:
                    exact_value = False
                    break
            if exact_value:
                transposition_table[state_key] = int(value)
            return int(value), exact_value

        value = float("inf")
        exact_value = True
        for card in legal_actions:
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                immediate_value = (
                    self._evaluate_trick(
                        round_state.trick_history[-1],
                        team_member_names,
                    )
                    if trick_completed
                    else 0
                )
                child_value, child_exact = self._search_n_trick_value(
                    round_state,
                    remaining_tricks - 1 if trick_completed else remaining_tricks,
                    team_member_names,
                    alpha,
                    beta,
                    transposition_table,
                )
                value = min(
                    value,
                    immediate_value + child_value,
                )
                exact_value = exact_value and child_exact
            finally:
                undo_trick_action_for_search(round_state, undo)
            beta = min(beta, value)
            if beta <= alpha:
                exact_value = False
                break
        if exact_value:
            transposition_table[state_key] = int(value)
        return int(value), exact_value

    def choose_card(self, round_state: RoundState) -> Card:
        if round_state.current_player.name != self.name:
            raise ValueError(
                f"{self.name} cannot choose a card for {round_state.current_player.name}"
            )

        team_member_names = self._team_member_names(round_state)
        legal_actions = self._ordered_legal_actions(round_state)
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
        transposition_table: dict[tuple, int] = {}
        self.begin_progress(
            label=f"Searching {self.depth} tricks ahead",
            total_units=len(legal_actions),
        )

        try:
            for index, card in enumerate(legal_actions, start=1):
                trick_count_before = len(round_state.trick_history)
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    trick_completed = len(round_state.trick_history) > trick_count_before
                    immediate_value = (
                        self._evaluate_trick(
                            round_state.trick_history[-1],
                            team_member_names,
                        )
                        if trick_completed
                        else 0
                    )
                    child_value, _ = self._search_n_trick_value(
                        round_state,
                        self.depth - 1 if trick_completed else self.depth,
                        team_member_names,
                        alpha,
                        beta,
                        transposition_table,
                    )
                    score = immediate_value + child_value
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
