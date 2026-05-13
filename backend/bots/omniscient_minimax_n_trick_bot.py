from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from random import Random

try:
    from backend.constants import CARD_DICT, GAME_VALUES, HAND_SIZE, RANK_ORDER
    from backend.engine import (
        apply_auction_action_for_search,
        apply_trick_action_for_search,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
        undo_auction_action_for_search,
        undo_trick_action_for_search,
    )
    from backend.models import AuctionEvent, AuctionState, Card, Deck, Play, RoundState, get_cards_value, would_win
    from .base import BotPlayer
    from .search_eval import (
        build_round_state_for_world,
        calculate_functional_deck_low,
        default_teams,
        ensure_captured_plays_synchronized,
        rollout_round_to_utility,
    )
except ImportError:
    from constants import CARD_DICT, GAME_VALUES, HAND_SIZE, RANK_ORDER
    from engine import (
        apply_auction_action_for_search,
        apply_trick_action_for_search,
        get_legal_actions,
        get_legal_auction_actions,
        get_trick_winner,
        undo_auction_action_for_search,
        undo_trick_action_for_search,
    )
    from models import AuctionEvent, AuctionState, Card, Deck, Play, RoundState, get_cards_value, would_win
    from .base import BotPlayer
    from .search_eval import (
        build_round_state_for_world,
        calculate_functional_deck_low,
        default_teams,
        ensure_captured_plays_synchronized,
        rollout_round_to_utility,
    )

@dataclass(frozen=True)
class SearchTranspositionEntry:
    value: float
    flag: str
    best_action: Card | AuctionEvent | None = None


class OmniscientMinimaxNTrickPlayer(BotPlayer):
    """Searches card play by projected match utility and searches the auction directly."""

    AUCTION_DETERMINIZATION_SAMPLES = 1
    MAX_BID = 6
    CARD_SIGNATURES = {
        code: 1 << index
        for index, code in enumerate(sorted(CARD_DICT.keys()))
    }
    TT_EXACT = "exact"
    TT_LOWER = "lower"
    TT_UPPER = "upper"

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
        self._context_player_names: list[str] | None = None
        self._context_teams: list[tuple[str, ...]] | None = None
        self._context_match_scores: dict[str, int] | None = None
        self._context_target_score = 21
        self._context_auction_state: AuctionState | None = None
        self._context_round_state: RoundState | None = None

    def set_match_context(
        self,
        *,
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
        auction_state: AuctionState | None = None,
        round_state: RoundState | None = None,
    ) -> None:
        self._context_player_names = list(player_names)
        self._context_teams = [tuple(team) for team in teams]
        self._context_match_scores = dict(match_scores)
        self._context_target_score = target_score
        self._context_auction_state = deepcopy(auction_state) if auction_state is not None else None
        self._context_round_state = round_state

    def _teams_for_players(self, player_names: list[str]) -> list[tuple[str, ...]]:
        if self._context_teams is not None:
            return [tuple(team) for team in self._context_teams]
        return default_teams(player_names)

    def _match_scores_for_teams(
        self,
        teams: list[tuple[str, ...]],
    ) -> dict[str, int]:
        if self._context_match_scores is not None:
            return dict(self._context_match_scores)
        return {" / ".join(team): 0 for team in teams}

    def _team_member_names(self, round_state: RoundState) -> set[str]:
        for team in round_state.teams:
            member_names = {player.name for player in team.constituents}
            if self.name in member_names:
                return member_names
        return {self.name}

    def _auction_team_member_names(self, auction_state: AuctionState) -> set[str]:
        for team in self._teams_for_players(auction_state.player_names):
            if self.name in team:
                return set(team)
        return {self.name}

    def _evaluate_trick(self, trick, team_member_names: set[str]) -> int:
        winner_name = get_trick_winner(trick).name
        trick_value = get_cards_value({play.card for play in trick.plays})
        if winner_name in team_member_names:
            return trick_value
        return -trick_value

    def _card_order_key(self, round_state: RoundState, card: Card) -> tuple:
        trick_cards = {play.card for play in round_state.current_trick.plays}
        wins_trick = would_win(card, round_state.current_trick)
        card_value = GAME_VALUES.get(card.rank, 0) if not card.is_joker else 5
        is_trump = int(
            card.is_joker
            or (
                round_state.trump is not None
                and not card.is_joker
                and card.suit == round_state.trump
            )
        )
        rank_value = len(RANK_ORDER) + 2 if card.is_joker else RANK_ORDER[card.rank]
        return (
            int(wins_trick),
            is_trump,
            card_value,
            rank_value,
            -len(trick_cards),
            card.code,
        )

    def _auction_candidate_suits(self) -> list[str]:
        candidate_suits = {
            card.suit
            for card in self.cards
            if not card.is_joker and card.suit is not None
        }
        if not candidate_suits:
            return ["H", "D", "C", "S"]
        return sorted(candidate_suits)

    def _guaranteed_opening_bid_commitment(
        self,
        auction_state: AuctionState,
    ) -> tuple[int, str] | None:
        if auction_state.highest_bid is not None:
            return None

        deck_low = calculate_functional_deck_low(len(auction_state.player_names))
        best_suit: str | None = None
        best_points = 0
        best_key: tuple[int, int, str] | None = None

        for suit in self._auction_candidate_suits():
            has_high = any(
                not card.is_joker and card.suit == suit and card.rank == "A"
                for card in self.cards
            )
            has_low = any(
                not card.is_joker and card.suit == suit and card.rank == deck_low
                for card in self.cards
            )
            guaranteed_points = int(has_high) + int(has_low)
            suit_cards = [
                card for card in self.cards
                if not card.is_joker and card.suit == suit
            ]
            suit_key = (
                len(suit_cards),
                max((RANK_ORDER[card.rank] for card in suit_cards), default=-1),
                suit,
            )
            if (
                guaranteed_points > best_points
                or (
                    guaranteed_points == best_points
                    and (best_key is None or suit_key > best_key)
                )
            ):
                best_points = guaranteed_points
                best_suit = suit
                best_key = suit_key

        if best_suit is None or best_points <= 0:
            return None
        return best_points, best_suit

    def _preferred_action_first(
        self,
        actions: list[Card] | list[AuctionEvent],
        preferred_action: Card | AuctionEvent | None,
    ) -> list[Card] | list[AuctionEvent]:
        if preferred_action is None or len(actions) < 2:
            return actions
        try:
            action_index = actions.index(preferred_action)
        except ValueError:
            return actions
        if action_index == 0:
            return actions
        actions.insert(0, actions.pop(action_index))
        return actions

    def _ordered_legal_actions(
        self,
        round_state: RoundState,
        *,
        maximizing: bool,
        preferred_action: Card | None = None,
    ) -> list[Card]:
        legal_actions = list(get_legal_actions(round_state))
        if len(legal_actions) < 2:
            return legal_actions
        legal_actions.sort(
            key=lambda card: self._card_order_key(round_state, card),
            reverse=True,
        )
        return self._preferred_action_first(legal_actions, preferred_action)

    def _tt_flag(
        self,
        *,
        maximizing: bool,
        value: float,
        original_alpha: float,
        original_beta: float,
        exact_value: bool,
    ) -> str:
        if exact_value:
            return self.TT_EXACT
        if value >= original_beta:
            return self.TT_LOWER
        if value <= original_alpha:
            return self.TT_UPPER
        return self.TT_LOWER if maximizing else self.TT_UPPER

    def _store_tt_entry(
        self,
        transposition_table: dict[tuple, SearchTranspositionEntry],
        state_key: tuple,
        *,
        maximizing: bool,
        value: float,
        original_alpha: float,
        original_beta: float,
        exact_value: bool,
        best_action: Card | AuctionEvent | None,
    ) -> None:
        transposition_table[state_key] = SearchTranspositionEntry(
            value=value,
            flag=self._tt_flag(
                maximizing=maximizing,
                value=value,
                original_alpha=original_alpha,
                original_beta=original_beta,
                exact_value=exact_value,
            ),
            best_action=best_action,
        )

    def _ordered_auction_actions(
        self,
        auction_state: AuctionState,
        *,
        maximizing: bool,
        preferred_action: AuctionEvent | None = None,
    ) -> list[AuctionEvent]:
        legal_actions = get_legal_auction_actions(auction_state, max_bid=self.MAX_BID)
        if len(legal_actions) < 2:
            return legal_actions
        legal_actions.sort(
            key=lambda action: (
                action.action == "bid",
                action.amount or 0,
            ),
            reverse=True,
        )
        return self._preferred_action_first(legal_actions, preferred_action)

    def _transposition_key(
        self,
        round_state: RoundState,
        remaining_tricks: int,
    ) -> tuple:
        def _cards_signature(cards: set[Card]) -> int:
            signature = 0
            for card in cards:
                signature |= self.CARD_SIGNATURES[card.code]
            return signature

        player_names = tuple(player.name for player in round_state.players)
        player_indexes = {
            player_name: index
            for index, player_name in enumerate(player_names)
        }
        captured_signatures = []
        for player in round_state.players:
            captured_by_origin = [0] * len(player_names)
            for play in player.captured_plays:
                captured_by_origin[player_indexes[play.player.name]] |= self.CARD_SIGNATURES[
                    play.card.code
                ]
            captured_signatures.append(
                (player.name, tuple(captured_by_origin))
            )

        return (
            remaining_tricks,
            round_state.current_player.name,
            round_state.current_trick.leader.name,
            round_state.trump,
            tuple(
                (play.player.name, play.card.code)
                for play in round_state.current_trick.plays
            ),
            _cards_signature(round_state.hidden_cards),
            tuple(
                (player.name, _cards_signature(player.cards))
                for player in round_state.players
            ),
            tuple(captured_signatures),
        )

    def _auction_transposition_key(self, auction_state: AuctionState) -> tuple:
        return (
            auction_state.current_bidder_index,
            auction_state.highest_bid,
            auction_state.highest_bidder_name,
            tuple(
                player_name in auction_state.passed_player_names
                for player_name in auction_state.player_names
            ),
            tuple(
                (event.bidder_name, event.action, event.amount)
                for event in auction_state.bid_history
            ),
        )

    def _play_auction_state(self) -> AuctionState | None:
        if self._context_auction_state is None:
            return None
        return deepcopy(self._context_auction_state)

    def _leaf_state_utility(
        self,
        round_state: RoundState,
        auction_state: AuctionState | None,
    ) -> float:
        teams = self._teams_for_players([player.name for player in round_state.players])
        return rollout_round_to_utility(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=self._match_scores_for_teams(teams),
            teams=teams,
            target_score=self._context_target_score,
            player_name=self.name,
            hybrid_cutoff=self.depth >= 4,
            exact_rollout_action_threshold=len(round_state.players) * 2,
        )

    def _search_n_trick_value(
        self,
        round_state: RoundState,
        remaining_tricks: int,
        team_member_names: set[str],
        alpha: float,
        beta: float,
        transposition_table: dict[tuple, SearchTranspositionEntry],
        auction_state: AuctionState | None,
    ) -> tuple[float, bool]:
        original_alpha = alpha
        original_beta = beta
        state_key = self._transposition_key(round_state, remaining_tricks)
        maximizing = round_state.current_player.name in team_member_names
        cached_entry = transposition_table.get(state_key)
        preferred_action: Card | None = None
        if cached_entry is not None and isinstance(cached_entry.best_action, Card):
            preferred_action = cached_entry.best_action
        if cached_entry is not None:
            if cached_entry.flag == self.TT_EXACT:
                return cached_entry.value, True
            if cached_entry.flag == self.TT_LOWER:
                alpha = max(alpha, cached_entry.value)
            else:
                beta = min(beta, cached_entry.value)
            if beta <= alpha:
                return cached_entry.value, False

        if round_state.is_terminal or remaining_tricks <= 0:
            value = self._leaf_state_utility(round_state, auction_state)
            self._store_tt_entry(
                transposition_table,
                state_key,
                maximizing=maximizing,
                value=value,
                original_alpha=original_alpha,
                original_beta=original_beta,
                exact_value=True,
                best_action=None,
            )
            return value, True

        trick_count_before = len(round_state.trick_history)
        legal_actions = self._ordered_legal_actions(
            round_state,
            maximizing=maximizing,
            preferred_action=preferred_action,
        )
        if not legal_actions:
            value = self._leaf_state_utility(round_state, auction_state)
            self._store_tt_entry(
                transposition_table,
                state_key,
                maximizing=maximizing,
                value=value,
                original_alpha=original_alpha,
                original_beta=original_beta,
                exact_value=True,
                best_action=None,
            )
            return value, True

        if maximizing:
            value = float("-inf")
            exact_value = True
            best_action: Card | None = None
            for card in legal_actions:
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(
                    round_state,
                    play,
                    validate_legal=False,
                )
                try:
                    trick_completed = len(round_state.trick_history) > trick_count_before
                    child_value, child_exact = self._search_n_trick_value(
                        round_state,
                        remaining_tricks - 1 if trick_completed else remaining_tricks,
                        team_member_names,
                        alpha,
                        beta,
                        transposition_table,
                        auction_state,
                    )
                    if child_value > value:
                        value = child_value
                        best_action = card
                    exact_value = exact_value and child_exact
                finally:
                    undo_trick_action_for_search(round_state, undo)
                alpha = max(alpha, value)
                if beta <= alpha:
                    exact_value = False
                    break
            self._store_tt_entry(
                transposition_table,
                state_key,
                maximizing=maximizing,
                value=value,
                original_alpha=original_alpha,
                original_beta=original_beta,
                exact_value=exact_value,
                best_action=best_action,
            )
            return value, exact_value

        value = float("inf")
        exact_value = True
        best_action = None
        for card in legal_actions:
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(
                round_state,
                play,
                validate_legal=False,
            )
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                child_value, child_exact = self._search_n_trick_value(
                    round_state,
                    remaining_tricks - 1 if trick_completed else remaining_tricks,
                    team_member_names,
                    alpha,
                    beta,
                    transposition_table,
                    auction_state,
                )
                if child_value < value:
                    value = child_value
                    best_action = card
                exact_value = exact_value and child_exact
            finally:
                undo_trick_action_for_search(round_state, undo)
            beta = min(beta, value)
            if beta <= alpha:
                exact_value = False
                break
        self._store_tt_entry(
            transposition_table,
            state_key,
            maximizing=maximizing,
            value=value,
            original_alpha=original_alpha,
            original_beta=original_beta,
            exact_value=exact_value,
            best_action=best_action,
        )
        return value, exact_value

    def select_best_scored_card(
        self,
        scored_cards: list[tuple[Card, float]],
    ) -> Card:
        best_card = None
        best_score = float("-inf")
        for card, score in scored_cards:
            if score > best_score:
                best_score = score
                best_card = card
        if best_card is None:
            raise ValueError("omniscient minimax bot could not find a legal card")
        return best_card

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

        ensure_captured_plays_synchronized(round_state)
        team_member_names = self._team_member_names(round_state)
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

        alpha = float("-inf")
        beta = float("inf")
        transposition_table: dict[tuple, SearchTranspositionEntry] = {}
        auction_state = self._play_auction_state()
        scored_cards: list[tuple[Card, float]] = []
        if show_progress:
            self.begin_progress(
                label=f"Searching {self.depth} tricks ahead",
                total_units=len(legal_actions),
            )

        try:
            for index, card in enumerate(legal_actions, start=1):
                trick_count_before = len(round_state.trick_history)
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(
                    round_state,
                    play,
                    validate_legal=False,
                )
                try:
                    trick_completed = len(round_state.trick_history) > trick_count_before
                    score, _ = self._search_n_trick_value(
                        round_state,
                        self.depth - 1 if trick_completed else self.depth,
                        team_member_names,
                        alpha,
                        beta,
                        transposition_table,
                        auction_state,
                    )
                finally:
                    undo_trick_action_for_search(round_state, undo)

                scored_cards.append((card, score))
                alpha = max(alpha, score)
                if show_progress:
                    self.update_progress(
                        completed_units=index,
                        detail=f"Evaluated {card.code}",
                    )
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

    def _auction_seed_payload(
        self,
        auction_state: AuctionState,
    ) -> str:
        bid_history = ",".join(
            f"{event.bidder_name}:{event.action}:{event.amount}"
            for event in auction_state.bid_history
        )
        cards = ",".join(sorted(card.code for card in self.cards))
        return "|".join(
            [
                self.name,
                cards,
                str(auction_state.dealer_index),
                str(auction_state.current_bidder_index),
                str(auction_state.highest_bid),
                str(auction_state.highest_bidder_name),
                bid_history,
            ]
        )

    def _build_sampled_auction_worlds(
        self,
        auction_state: AuctionState,
        *,
        sample_count: int,
    ) -> list[dict]:
        player_names = list(auction_state.player_names)
        low = calculate_functional_deck_low(len(player_names))
        deck_cards = Deck(low).get_copy()
        known_cards = set(self.cards)
        unknown_cards = [card for card in deck_cards if card not in known_cards]
        seed_bytes = sha256(self._auction_seed_payload(auction_state).encode("utf-8")).digest()
        base_seed = int.from_bytes(seed_bytes[:8], "big")

        worlds: list[dict] = []
        for sample_index in range(sample_count):
            rng = Random(base_seed + sample_index)
            shuffled_cards = unknown_cards.copy()
            rng.shuffle(shuffled_cards)
            cursor = 0
            hands: dict[str, set[Card]] = {self.name: set(self.cards)}

            for player_name in player_names:
                if player_name == self.name:
                    continue
                hands[player_name] = set(
                    shuffled_cards[cursor: cursor + HAND_SIZE]
                )
                cursor += HAND_SIZE

            worlds.append(
                {
                    "hands": hands,
                    "hidden_cards": set(shuffled_cards[cursor:]),
                    "low": low,
                }
            )

        return worlds

    def _build_auction_worlds(self, auction_state: AuctionState) -> list[dict]:
        if self._context_round_state is not None:
            return [
                {
                    "hands": {
                        player.name: set(player.cards)
                        for player in self._context_round_state.players
                    },
                    "hidden_cards": set(self._context_round_state.hidden_cards),
                    "low": self._context_round_state.deck.low,
                }
            ]

        return self._build_sampled_auction_worlds(
            auction_state,
            sample_count=self.AUCTION_DETERMINIZATION_SAMPLES,
        )

    def _auction_leaf_utility(
        self,
        auction_state: AuctionState,
        world: dict,
    ) -> float:
        if auction_state.highest_bidder_name is None:
            raise ValueError("auction leaf requires a winning bidder")

        teams = self._teams_for_players(auction_state.player_names)
        round_state = build_round_state_for_world(
            player_names=list(auction_state.player_names),
            teams=teams,
            hands=world["hands"],
            hidden_cards=world["hidden_cards"],
            low=world["low"],
            starting_player_name=auction_state.highest_bidder_name,
        )
        return rollout_round_to_utility(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=self._match_scores_for_teams(teams),
            teams=teams,
            target_score=self._context_target_score,
            player_name=self.name,
        )

    def _search_auction_value(
        self,
        auction_state: AuctionState,
        world: dict,
        alpha: float,
        beta: float,
        transposition_table: dict[tuple, SearchTranspositionEntry],
    ) -> tuple[float, bool]:
        if auction_state.is_complete:
            return self._auction_leaf_utility(auction_state, world), True

        original_alpha = alpha
        original_beta = beta
        state_key = self._auction_transposition_key(auction_state)
        team_member_names = self._auction_team_member_names(auction_state)
        maximizing = auction_state.current_bidder_name in team_member_names
        cached_entry = transposition_table.get(state_key)
        preferred_action: AuctionEvent | None = None
        if cached_entry is not None and isinstance(cached_entry.best_action, AuctionEvent):
            preferred_action = cached_entry.best_action
        if cached_entry is not None:
            if cached_entry.flag == self.TT_EXACT:
                return cached_entry.value, True
            if cached_entry.flag == self.TT_LOWER:
                alpha = max(alpha, cached_entry.value)
            else:
                beta = min(beta, cached_entry.value)
            if beta <= alpha:
                return cached_entry.value, False

        legal_actions = self._ordered_auction_actions(
            auction_state,
            maximizing=maximizing,
            preferred_action=preferred_action,
        )
        if not legal_actions:
            return self._auction_leaf_utility(auction_state, world), True

        if maximizing:
            value = float("-inf")
            exact_value = True
            best_action: AuctionEvent | None = None
            for action in legal_actions:
                undo = apply_auction_action_for_search(
                    auction_state,
                    action,
                    max_bid=self.MAX_BID,
                    validate_legal=False,
                )
                try:
                    child_value, child_exact = self._search_auction_value(
                        auction_state,
                        world,
                        alpha,
                        beta,
                        transposition_table,
                    )
                    if child_value > value:
                        value = child_value
                        best_action = action
                    exact_value = exact_value and child_exact
                    alpha = max(alpha, value)
                finally:
                    undo_auction_action_for_search(auction_state, undo)
                if beta <= alpha:
                    exact_value = False
                    break
            self._store_tt_entry(
                transposition_table,
                state_key,
                maximizing=maximizing,
                value=value,
                original_alpha=original_alpha,
                original_beta=original_beta,
                exact_value=exact_value,
                best_action=best_action,
            )
            return value, exact_value

        value = float("inf")
        exact_value = True
        best_action = None
        for action in legal_actions:
            undo = apply_auction_action_for_search(
                auction_state,
                action,
                max_bid=self.MAX_BID,
                validate_legal=False,
            )
            try:
                child_value, child_exact = self._search_auction_value(
                    auction_state,
                    world,
                    alpha,
                    beta,
                    transposition_table,
                )
                if child_value < value:
                    value = child_value
                    best_action = action
                exact_value = exact_value and child_exact
                beta = min(beta, value)
            finally:
                undo_auction_action_for_search(auction_state, undo)
            if beta <= alpha:
                exact_value = False
                break
        self._store_tt_entry(
            transposition_table,
            state_key,
            maximizing=maximizing,
            value=value,
            original_alpha=original_alpha,
            original_beta=original_beta,
            exact_value=exact_value,
            best_action=best_action,
        )
        return value, exact_value

    def _prefer_auction_action(
        self,
        candidate: AuctionEvent,
        incumbent: AuctionEvent | None,
    ) -> bool:
        if incumbent is None:
            return True
        if candidate.action != incumbent.action:
            return candidate.action == "pass"
        if candidate.action == "bid":
            return (candidate.amount or self.MAX_BID) < (incumbent.amount or self.MAX_BID)
        return False

    def select_best_scored_auction_action(
        self,
        scored_actions: list[tuple[AuctionEvent, float]],
    ) -> AuctionEvent:
        best_action: AuctionEvent | None = None
        best_value = float("-inf")
        for action, value in scored_actions:
            if (
                value > best_value
                or (
                    value == best_value
                    and self._prefer_auction_action(action, best_action)
                )
            ):
                best_value = value
                best_action = action
        if best_action is None:
            raise ValueError("omniscient minimax bot could not find a legal auction action")
        return best_action

    def score_auction_candidates(
        self,
        auction_state: AuctionState,
        *,
        show_progress: bool = False,
    ) -> list[tuple[AuctionEvent, float]]:
        opening_commitment = self._guaranteed_opening_bid_commitment(auction_state)
        legal_actions = self._ordered_auction_actions(
            auction_state,
            maximizing=True,
        )
        if opening_commitment is not None:
            opening_bid, _opening_suit = opening_commitment
            return [
                (
                    action,
                    1.0 if action.action == "bid" and action.amount == opening_bid else 0.0,
                )
                for action in legal_actions
            ]

        worlds = self._build_auction_worlds(auction_state)
        scored_actions: list[tuple[AuctionEvent, float]] = []
        total_units = len(legal_actions) * len(worlds)
        completed_units = 0
        if show_progress:
            self.begin_progress(
                label="Searching auction",
                total_units=total_units,
            )

        try:
            for action in legal_actions:
                total_value = 0.0
                for world_index, world in enumerate(worlds, start=1):
                    undo = apply_auction_action_for_search(
                        auction_state,
                        action,
                        max_bid=self.MAX_BID,
                        validate_legal=False,
                    )
                    try:
                        value, _ = self._search_auction_value(
                            auction_state,
                            world,
                            float("-inf"),
                            float("inf"),
                            {},
                        )
                        total_value += value
                    finally:
                        undo_auction_action_for_search(auction_state, undo)
                    completed_units += 1
                    if show_progress:
                        self.update_progress(
                            completed_units=completed_units,
                            detail=(
                                f"{action.action}"
                                f"{'' if action.amount is None else f' {action.amount}'}"
                                f" in world {world_index}/{len(worlds)}"
                            ),
                        )

                scored_actions.append((action, total_value / len(worlds)))
        finally:
            if show_progress:
                self.clear_progress()

        return scored_actions

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        self._opening_suit_override = None
        opening_commitment = self._guaranteed_opening_bid_commitment(auction_state)
        if opening_commitment is not None:
            opening_bid, opening_suit = opening_commitment
            for action in self._ordered_auction_actions(
                auction_state,
                maximizing=True,
            ):
                if action.action == "bid" and action.amount == opening_bid:
                    self._opening_suit_override = opening_suit
                    return action
        scored_actions = self.score_auction_candidates(
            auction_state,
            show_progress=True,
        )
        return self.select_best_scored_auction_action(scored_actions)


OMNISCIENT_MinimaxNTrickPlayer = OmniscientMinimaxNTrickPlayer
