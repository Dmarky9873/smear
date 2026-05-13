from __future__ import annotations

from hashlib import sha256
from random import Random

try:
    from backend.engine import (
        apply_trick_action_for_search,
        undo_trick_action_for_search,
    )
    from backend.models import Card, Play, RoundState, Team, TrickState
    from .omniscient_minimax_n_trick_bot import (
        OmniscientMinimaxNTrickPlayer,
        SearchTranspositionEntry,
    )
    from .search_eval import ensure_captured_plays_synchronized
except ImportError:
    from engine import (
        apply_trick_action_for_search,
        undo_trick_action_for_search,
    )
    from models import Card, Play, RoundState, Team, TrickState
    from .omniscient_minimax_n_trick_bot import (
        OmniscientMinimaxNTrickPlayer,
        SearchTranspositionEntry,
    )
    from .search_eval import ensure_captured_plays_synchronized


class HumanInformationMinimaxNTrickPlayer(OmniscientMinimaxNTrickPlayer):
    """Sampled hidden-information minimax over the next N completed tricks."""

    DETERMINIZATION_SAMPLES = 12
    MIN_DETERMINIZATION_SAMPLES = 2
    AUCTION_DETERMINIZATION_SAMPLES = 6
    THREE_PLAYER_AUCTION_DETERMINIZATION_SAMPLES = 7
    _HIDDEN_SLOT = "__hidden__"

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

    def _infer_void_constraints(
        self,
        round_state: RoundState,
    ) -> tuple[dict[str, set[str]], set[str]]:
        suit_voids = {
            player.name: set()
            for player in round_state.players
            if player.name != self.name
        }
        players_without_joker: set[str] = set()

        for trick in [*round_state.trick_history, round_state.current_trick]:
            if len(trick.plays) < 2:
                continue

            prior_plays: list[Play] = []
            for play in trick.plays:
                if not prior_plays:
                    prior_plays.append(play)
                    continue

                player_name = play.player.name
                if player_name == self.name:
                    prior_plays.append(play)
                    continue

                lead_card = prior_plays[0].card
                if (
                    not lead_card.is_joker
                    and not play.card.is_joker
                    and play.card.suit != lead_card.suit
                    and play.card.suit != trick.trump
                ):
                    suit_voids[player_name].add(lead_card.suit)
                    if lead_card.suit == trick.trump:
                        players_without_joker.add(player_name)

                prior_plays.append(play)

        return suit_voids, players_without_joker

    def _card_allowed_for_player(
        self,
        card: Card,
        player_name: str,
        *,
        suit_voids: dict[str, set[str]],
        players_without_joker: set[str],
    ) -> bool:
        if card.is_joker:
            return player_name not in players_without_joker
        return card.suit not in suit_voids.get(player_name, set())

    def _sample_constrained_unseen_cards(
        self,
        *,
        unseen_cards: list[Card],
        player_hand_sizes: dict[str, int],
        hidden_count: int,
        suit_voids: dict[str, set[str]],
        players_without_joker: set[str],
        rng: Random,
    ) -> tuple[dict[str, set[Card]], set[Card]]:
        recipient_capacities = {
            player_name: hand_size
            for player_name, hand_size in player_hand_sizes.items()
            if hand_size > 0
        }
        if hidden_count > 0:
            recipient_capacities[self._HIDDEN_SLOT] = hidden_count

        base_eligibility: dict[Card, tuple[str, ...]] = {}
        for card in unseen_cards:
            eligible_recipients = [
                recipient
                for recipient in recipient_capacities
                if recipient == self._HIDDEN_SLOT
                or self._card_allowed_for_player(
                    card,
                    recipient,
                    suit_voids=suit_voids,
                    players_without_joker=players_without_joker,
                )
            ]
            if not eligible_recipients:
                raise ValueError(
                    "no legal determinization assignment exists for unseen cards"
                )
            base_eligibility[card] = tuple(eligible_recipients)

        ordered_cards = list(unseen_cards)
        rng.shuffle(ordered_cards)
        ordered_cards.sort(
            key=lambda card: (len(base_eligibility[card]), card.code),
        )

        remaining_capacities = dict(recipient_capacities)
        assignments = {
            recipient: []
            for recipient in recipient_capacities
        }

        def is_feasible(start_index: int) -> bool:
            for recipient, remaining_capacity in remaining_capacities.items():
                if remaining_capacity <= 0:
                    continue
                eligible_remaining = sum(
                    1
                    for card in ordered_cards[start_index:]
                    if recipient in base_eligibility[card]
                )
                if eligible_remaining < remaining_capacity:
                    return False
            return True

        def backtrack(card_index: int) -> bool:
            if card_index == len(ordered_cards):
                return True
            if not is_feasible(card_index):
                return False

            card = ordered_cards[card_index]
            candidate_recipients = [
                recipient
                for recipient in base_eligibility[card]
                if remaining_capacities[recipient] > 0
            ]
            rng.shuffle(candidate_recipients)

            for recipient in candidate_recipients:
                remaining_capacities[recipient] -= 1
                assignments[recipient].append(card)
                if backtrack(card_index + 1):
                    return True
                assignments[recipient].pop()
                remaining_capacities[recipient] += 1

            return False

        if not backtrack(0):
            raise ValueError(
                "could not build a determinized world consistent with public constraints"
            )

        return (
            {
                player_name: set(assignments.get(player_name, []))
                for player_name in player_hand_sizes
            },
            set(assignments.get(self._HIDDEN_SLOT, [])),
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

        suit_voids, players_without_joker = self._infer_void_constraints(round_state)
        seed_bytes = sha256(
            self._determinization_seed_payload(round_state, known_hand).encode("utf-8")
        ).digest()
        base_seed = int.from_bytes(seed_bytes[:8], "big")
        determinizations: list[RoundState] = []
        sample_count = self._determinization_sample_count(
            player_count=len(round_state.players),
        )
        opponent_hand_sizes = {
            player.name: len(player.cards)
            for player in round_state.players
            if player.name != self.name
        }

        for sample_index in range(sample_count):
            rng = Random(base_seed + sample_index)
            sampled_hands, sampled_hidden_cards = self._sample_constrained_unseen_cards(
                unseen_cards=unseen_cards,
                player_hand_sizes=opponent_hand_sizes,
                hidden_count=hidden_count,
                suit_voids=suit_voids,
                players_without_joker=players_without_joker,
                rng=rng,
            )
            cloned_players_by_name = {}

            for player in round_state.players:
                if player.name == self.name:
                    hand = set(known_hand)
                else:
                    hand = sampled_hands[player.name]
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

            determinized_state = RoundState(
                players=cloned_players,
                current_player=cloned_players_by_name[
                    round_state.current_player.name
                ],
                trump=round_state.trump,
                current_trick=cloned_current_trick,
                hidden_cards=sampled_hidden_cards,
                trick_history=cloned_trick_history,
                teams=cloned_teams,
                deck=round_state.deck,
            )
            ensure_captured_plays_synchronized(determinized_state)
            determinizations.append(determinized_state)

        return determinizations

    def _determinization_sample_count(self, *, player_count: int | None = None) -> int:
        if player_count == 3:
            if self.depth <= 2:
                return self.DETERMINIZATION_SAMPLES
            if self.depth == 3:
                return max(
                    self.MIN_DETERMINIZATION_SAMPLES,
                    (self.DETERMINIZATION_SAMPLES * 7) // 12,
                )
            if self.depth == 4:
                return max(
                    self.MIN_DETERMINIZATION_SAMPLES,
                    self.DETERMINIZATION_SAMPLES // 3,
                )
            return max(
                self.MIN_DETERMINIZATION_SAMPLES,
                self.DETERMINIZATION_SAMPLES // 5,
            )

        if self.depth <= 2:
            return self.DETERMINIZATION_SAMPLES
        if self.depth == 3:
            return max(
                self.MIN_DETERMINIZATION_SAMPLES,
                self.DETERMINIZATION_SAMPLES // 2,
            )
        if self.depth == 4:
            return max(
                self.MIN_DETERMINIZATION_SAMPLES,
                self.DETERMINIZATION_SAMPLES // 4,
            )
        return self.MIN_DETERMINIZATION_SAMPLES

    def _build_auction_worlds(self, auction_state):
        return self._build_sampled_auction_worlds(
            auction_state,
            sample_count=self._auction_determinization_sample_count(auction_state),
        )

    def _auction_determinization_sample_count(self, auction_state) -> int:
        if len(auction_state.player_names) == 3:
            return self.THREE_PLAYER_AUCTION_DETERMINIZATION_SAMPLES
        return self.AUCTION_DETERMINIZATION_SAMPLES

    def score_card_candidates(
        self,
        round_state: RoundState,
        *,
        show_progress: bool = False,
    ) -> list[tuple[Card, float]]:
        if round_state.current_player.name != self.name:
            raise ValueError(
                f"{self.name} cannot score cards for {round_state.current_player.name}"
            )

        known_hand = set(round_state.current_player.cards)
        team_member_names = self._team_member_names(round_state)
        determinizations = self._build_determinized_states(round_state, known_hand)
        legal_actions = self._ordered_legal_actions(
            round_state,
            maximizing=True,
        )
        opening_suit_override = getattr(self, "_opening_suit_override", None)
        if round_state.trump is None and opening_suit_override is not None:
            suited_actions = [
                card for card in legal_actions if card.suit == opening_suit_override
            ]
            if suited_actions:
                legal_actions = suited_actions
        else:
            self._opening_suit_override = None

        total_units = len(legal_actions) * len(determinizations)
        completed_units = 0
        transposition_table: dict[tuple, SearchTranspositionEntry] = {}
        auction_state = self._play_auction_state()
        scored_cards: list[tuple[Card, float]] = []
        if show_progress:
            self.begin_progress(
                label=f"Searching {self.depth} tricks ahead",
                total_units=total_units,
            )

        try:
            for card in legal_actions:
                total_score = 0.0
                for sample_index, determinized_state in enumerate(
                    determinizations,
                    start=1,
                ):
                    trick_count_before = len(determinized_state.trick_history)
                    play = Play(determinized_state.current_player, card)
                    undo = apply_trick_action_for_search(
                        determinized_state,
                        play,
                        validate_legal=False,
                    )
                    try:
                        trick_completed = (
                            len(determinized_state.trick_history) > trick_count_before
                        )
                        child_value, _ = self._search_n_trick_value(
                            determinized_state,
                            self.depth - 1 if trick_completed else self.depth,
                            team_member_names,
                            float("-inf"),
                            float("inf"),
                            transposition_table,
                            auction_state,
                        )
                        total_score += child_value
                    finally:
                        undo_trick_action_for_search(determinized_state, undo)

                    completed_units += 1
                    if show_progress:
                        self.update_progress(
                            completed_units=completed_units,
                            detail=f"{card.code} in world {sample_index}/{len(determinizations)}",
                        )

                scored_cards.append((card, total_score / len(determinizations)))
        finally:
            if show_progress:
                self.clear_progress()

        return scored_cards

    def choose_card(self, round_state: RoundState) -> Card:
        scored_cards = self.score_card_candidates(
            round_state,
            show_progress=True,
        )
        return self.select_best_scored_card(scored_cards)


MinimaxNTrickPlayer = HumanInformationMinimaxNTrickPlayer
