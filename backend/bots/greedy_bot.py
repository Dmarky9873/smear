from copy import deepcopy
from hashlib import sha256
from random import Random

try:
    from backend.constants import GAME_VALUES, HAND_SIZE, RANKS, RANK_ORDER
    from backend.engine import Auction, get_legal_actions, get_legal_auction_actions
    from backend.models import (
        AuctionEvent,
        AuctionState,
        Card,
        Deck,
        RoundState,
        get_cards_value,
        would_win,
    )
    from .base import BotPlayer
except ImportError:
    from constants import GAME_VALUES, HAND_SIZE, RANKS, RANK_ORDER
    from engine import Auction, get_legal_actions, get_legal_auction_actions
    from models import (
        AuctionEvent,
        AuctionState,
        Card,
        Deck,
        RoundState,
        get_cards_value,
        would_win,
    )
    from bots.base import BotPlayer


class GreedyPlayer(BotPlayer):
    AUCTION_ROLLOUTS = 6
    MAX_AUCTION_CANDIDATE_SUITS = 2
    preferred_suit = "H"
    _candidate_suits = ("H", "D", "C", "S")

    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
        *,
        use_rollout_auction: bool = True,
    ):
        super().__init__(name, cards)
        self._use_rollout_auction = use_rollout_auction
        self._opening_suit_override: str | None = None
        self._context_player_names: list[str] | None = None
        self._context_teams: list[tuple[str, ...]] | None = None
        self._context_match_scores: dict[str, int] | None = None
        self._context_target_score = 21

    def set_match_context(
        self,
        *,
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
        auction_state=None,
        round_state=None,
    ) -> None:
        self._context_player_names = list(player_names)
        self._context_teams = [tuple(team) for team in teams]
        self._context_match_scores = dict(match_scores)
        self._context_target_score = target_score

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
            if self._opening_suit_override is not None:
                self.preferred_suit = self._opening_suit_override
            else:
                self.preferred_suit = self._select_preferred_suit()
        else:
            self._opening_suit_override = None

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

    def _choose_threshold_auction_action(
        self,
        auction_state: AuctionState,
    ) -> AuctionEvent:
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

    def _default_teams(self, player_names: list[str]) -> list[tuple[str, ...]]:
        return [(player_name,) for player_name in player_names]

    def _rollout_teams(self, auction_state: AuctionState) -> list[tuple[str, ...]]:
        if self._context_teams is not None:
            return [tuple(team) for team in self._context_teams]
        return self._default_teams(auction_state.player_names)

    def _my_unit_name(self, auction_state: AuctionState) -> str:
        for team in self._rollout_teams(auction_state):
            if self.name in team:
                return " / ".join(team)
        return self.name

    def _rollout_match_scores(self, auction_state: AuctionState) -> dict[str, int]:
        if self._context_match_scores is not None:
            return dict(self._context_match_scores)
        return {
            " / ".join(team): 0
            for team in self._rollout_teams(auction_state)
        }

    def _calculate_low(self, num_players: int) -> str:
        dealt = HAND_SIZE * num_players
        best_low = None
        best_diff = float("inf")

        for i, rank in enumerate(RANKS):
            remaining_ranks = RANKS[i:]
            deck_size = 4 * len(remaining_ranks) + 2
            hiding = deck_size - dealt

            if hiding <= 0:
                continue

            diff = abs(hiding - 2)
            if diff < best_diff:
                best_diff = diff
                best_low = rank

        if best_low is None:
            raise ValueError(f"could not determine a functional deck low for {num_players} players")
        return best_low

    def _candidate_opening_suits(self) -> list[str]:
        candidate_suits = {
            card.suit
            for card in self.cards
            if not card.is_joker and card.suit is not None
        }
        if not candidate_suits:
            return list(self._candidate_suits[: self.MAX_AUCTION_CANDIDATE_SUITS])
        ordered_suits = sorted(
            candidate_suits,
            key=lambda suit: (
                self._preferred_suit_key(suit),
                -self._candidate_suits.index(suit),
            ),
            reverse=True,
        )
        return ordered_suits[: self.MAX_AUCTION_CANDIDATE_SUITS]

    def _rollout_seed_payload(
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

    def _build_rollout_worlds(self, auction_state: AuctionState) -> list[dict]:
        player_names = list(auction_state.player_names)
        low = self._calculate_low(len(player_names))
        deck_cards = Deck(low).get_copy()
        known_cards = set(self.cards)
        unknown_cards = [card for card in deck_cards if card not in known_cards]
        seed_bytes = sha256(self._rollout_seed_payload(auction_state).encode("utf-8")).digest()
        base_seed = int.from_bytes(seed_bytes[:8], "big")

        worlds: list[dict] = []
        for rollout_index in range(self.AUCTION_ROLLOUTS):
            rng = Random(base_seed + rollout_index)
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

    def _rollout_utility(self, session) -> float:
        my_unit_name = self._my_unit_name(session.auction.state)
        scores = session.match_scores
        my_score = scores[my_unit_name]
        opponent_scores = [
            score for unit_name, score in scores.items() if unit_name != my_unit_name
        ]
        utility = float(my_score - max(opponent_scores, default=0))

        if my_unit_name in session.match_winner_names:
            utility += session.target_score
        elif session.match_winner_names:
            utility -= session.target_score

        return utility

    def _simulate_candidate(
        self,
        auction_state: AuctionState,
        action: AuctionEvent,
        opening_suit: str | None,
        world: dict,
    ) -> float:
        try:
            from backend.gameplay import MatchController
        except ImportError:
            from gameplay import MatchController

        player_names = list(auction_state.player_names)
        teams = self._rollout_teams(auction_state)
        bots = {
            player_name: GreedyPlayer(
                player_name,
                use_rollout_auction=False,
            )
            for player_name in player_names
        }
        if action.action == "bid" and action.bidder_name == self.name:
            bots[self.name]._opening_suit_override = opening_suit

        controller = MatchController.create(
            num_players=len(player_names),
            player_names=player_names,
            teams=teams,
            bots=bots,
            target_score=self._context_target_score,
            auto_run_bots=False,
        )
        controller.session.match_scores = self._rollout_match_scores(auction_state)

        round_state = controller.session.game.round_state
        for player in round_state.players:
            player.receive_new_hand(set(world["hands"][player.name]))
        round_state.hidden_cards = set(world["hidden_cards"])
        round_state.deck = Deck(world["low"])
        controller._sync_all_bot_hands()

        rollout_auction = Auction.from_state(deepcopy(auction_state))
        rollout_auction.apply_event(action)
        controller.session.auction = rollout_auction

        if controller.session.auction.state.is_complete:
            controller._finalize_auction()

        controller.run_bot_turns()
        return self._rollout_utility(controller.session)

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        """Choose the legal auction action with the best rollout-estimated utility."""
        legal_actions = get_legal_auction_actions(auction_state)
        if not self._use_rollout_auction:
            return self._choose_threshold_auction_action(auction_state)

        pass_action = next(
            (action for action in legal_actions if action.action == "pass"),
            None,
        )
        bid_actions = [action for action in legal_actions if action.action == "bid"]
        candidate_suits = self._candidate_opening_suits()
        rollout_worlds = self._build_rollout_worlds(auction_state)

        best_action: AuctionEvent | None = None
        best_value = float("-inf")
        best_suit: str | None = None

        if pass_action is not None:
            pass_value = sum(
                self._simulate_candidate(
                    auction_state,
                    pass_action,
                    None,
                    world,
                )
                for world in rollout_worlds
            ) / len(rollout_worlds)
            best_action = pass_action
            best_value = pass_value
            best_suit = None

        for action in bid_actions:
            action_best_value = float("-inf")
            action_best_suit: str | None = None

            for suit in candidate_suits:
                suit_value = sum(
                    self._simulate_candidate(
                        auction_state,
                        action,
                        suit,
                        world,
                    )
                    for world in rollout_worlds
                ) / len(rollout_worlds)

                if suit_value > action_best_value:
                    action_best_value = suit_value
                    action_best_suit = suit

            if (
                best_action is None
                or action_best_value > best_value
                or (
                    action_best_value == best_value
                    and best_action.action == "bid"
                    and action.amount < (best_action.amount or action.amount)
                )
            ):
                best_action = action
                best_value = action_best_value
                best_suit = action_best_suit

        if best_action is None:
            raise ValueError("greedy bot could not find a legal auction action")

        self._opening_suit_override = best_suit if best_action.action == "bid" else None
        if self._opening_suit_override is not None:
            self.preferred_suit = self._opening_suit_override
        return best_action
