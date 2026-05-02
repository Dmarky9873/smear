from __future__ import annotations

from hashlib import sha256
from random import Random

try:
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from backend.models import Card, Play, RoundState, Team, TrickState
    from .o_minimax_one_trick_bot import OMNISCIENT_MinimaxOneTrickPlayer
except ImportError:
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        undo_trick_action_for_search,
    )
    from models import Card, Play, RoundState, Team, TrickState
    from .o_minimax_one_trick_bot import OMNISCIENT_MinimaxOneTrickPlayer


class MinimaxOneTrickPlayer(OMNISCIENT_MinimaxOneTrickPlayer):
    """One-trick minimax that reasons from public information only.

    The search never reads opponent hands from the live round state. Instead it
    samples determinizations of the unseen cards that are consistent with the
    cards this player can actually observe, then averages the one-trick minimax
    value across those sampled worlds.
    """

    DETERMINIZATION_SAMPLES = 12

    def set_match_context(
        self,
        *,
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
    ) -> None:
        self._context_player_names = list(player_names)
        self._context_teams = [tuple(team) for team in teams]
        self._context_match_scores = dict(match_scores)
        self._context_target_score = target_score

    def _public_visible_cards(
        self,
        round_state: RoundState,
        known_hand: set[Card],
    ) -> set[Card]:
        return (
            set(known_hand)
            | {
                play.card
                for trick in round_state.trick_history
                for play in trick.plays
            }
            | {play.card for play in round_state.current_trick.plays}
        )

    def _determinization_seed_payload(
        self,
        round_state: RoundState,
        known_hand: set[Card],
    ) -> str:
        completed_tricks = ";".join(
            ",".join(f"{play.player.name}:{play.card.code}" for play in trick.plays)
            for trick in round_state.trick_history
        )
        current_trick = ",".join(
            f"{play.player.name}:{play.card.code}"
            for play in round_state.current_trick.plays
        )
        remaining_counts = ",".join(
            f"{player.name}:{len(player.cards)}" for player in round_state.players
        )
        team_signature = ";".join(
            ",".join(player.name for player in team.constituents)
            for team in round_state.teams
        )
        return "|".join(
            [
                self.name,
                ",".join(sorted(card.code for card in known_hand)),
                round_state.current_player.name,
                round_state.current_trick.leader.name,
                str(round_state.trump),
                remaining_counts,
                team_signature,
                completed_tricks,
                current_trick,
            ]
        )

    def _clone_trick(
        self,
        trick: TrickState,
        cloned_players_by_name: dict[str, object],
    ) -> TrickState:
        return TrickState(
            leader=cloned_players_by_name[trick.leader.name],
            plays=[
                Play(cloned_players_by_name[play.player.name], play.card)
                for play in trick.plays
            ],
            players=[
                cloned_players_by_name[player.name] for player in trick.players
            ],
            trump=trick.trump,
        )

    def _build_determinized_states(
        self,
        round_state: RoundState,
        known_hand: set[Card],
    ) -> list[RoundState]:
        deck_cards = round_state.deck.get_copy()
        visible_cards = self._public_visible_cards(round_state, known_hand)
        unseen_cards = [card for card in deck_cards if card not in visible_cards]
        unseen_count = sum(
            len(player.cards)
            for player in round_state.players
            if player.name != self.name
        )
        hidden_count = len(unseen_cards) - unseen_count
        if hidden_count < 0:
            raise ValueError(
                "public state is inconsistent with the deck and remaining card counts"
            )

        seed_bytes = sha256(
            self._determinization_seed_payload(round_state, known_hand).encode("utf-8")
        ).digest()
        base_seed = int.from_bytes(seed_bytes[:8], "big")
        determinizations: list[RoundState] = []

        for sample_index in range(self.DETERMINIZATION_SAMPLES):
            rng = Random(base_seed + sample_index)
            shuffled_cards = unseen_cards.copy()
            rng.shuffle(shuffled_cards)
            cursor = 0
            cloned_players_by_name = {}

            for player in round_state.players:
                if player.name == self.name:
                    hand = set(known_hand)
                else:
                    hand_size = len(player.cards)
                    hand = set(shuffled_cards[cursor: cursor + hand_size])
                    cursor += hand_size
                cloned_players_by_name[player.name] = player.__class__(
                    player.name,
                    hand,
                )

            cloned_players = [
                cloned_players_by_name[player.name] for player in round_state.players
            ]
            cloned_current_trick = self._clone_trick(
                round_state.current_trick,
                cloned_players_by_name,
            )
            cloned_trick_history = [
                self._clone_trick(trick, cloned_players_by_name)
                for trick in round_state.trick_history
            ]
            cloned_teams = [
                Team(
                    [
                        cloned_players_by_name[player.name]
                        for player in team.constituents
                    ],
                    set(),
                )
                for team in round_state.teams
            ]

            determinizations.append(
                RoundState(
                    players=cloned_players,
                    current_player=cloned_players_by_name[
                        round_state.current_player.name
                    ],
                    trump=round_state.trump,
                    current_trick=cloned_current_trick,
                    hidden_cards=set(shuffled_cards[cursor: cursor + hidden_count]),
                    trick_history=cloned_trick_history,
                    teams=cloned_teams,
                    deck=round_state.deck,
                )
            )

        return determinizations

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
        if round_state.current_player.name != self.name:
            raise ValueError(
                f"{self.name} cannot choose a card for {round_state.current_player.name}"
            )

        known_hand = set(round_state.current_player.cards)
        team_member_names = self._team_member_names(round_state)
        starting_trick_count = len(round_state.trick_history)
        determinizations = self._build_determinized_states(round_state, known_hand)

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

        for card in legal_actions:
            total_score = 0.0
            for determinized_state in determinizations:
                play = Play(determinized_state.current_player, card)
                undo = apply_trick_action_for_search(determinized_state, play)
                try:
                    total_score += self._search_one_trick_value(
                        determinized_state,
                        starting_trick_count,
                        team_member_names,
                        float("-inf"),
                        float("inf"),
                    )
                finally:
                    undo_trick_action_for_search(determinized_state, undo)

            average_score = total_score / len(determinizations)
            if average_score > best_score:
                best_score = average_score
                best_card = card

        if best_card is None:
            raise ValueError("minimax bot could not find a legal card to play")
        return best_card
