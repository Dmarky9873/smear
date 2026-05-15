from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

try:
    from backend.engine import apply_auction_action_for_search, apply_trick_action_to_state
    from backend.models import AuctionEvent, AuctionState, Card, Play, RoundState
    from .base import BotPlayer
    from .greedy_bot import GreedyPlayer
    from .neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        is_supported_three_player_singleton_context,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from .neural_model import ScalarMLP
    from .search_eval import evaluate_terminal_round_utility
except ImportError:
    from engine import apply_auction_action_for_search, apply_trick_action_to_state
    from models import AuctionEvent, AuctionState, Card, Play, RoundState
    from bots.base import BotPlayer
    from bots.greedy_bot import GreedyPlayer
    from bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        is_supported_three_player_singleton_context,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from bots.neural_model import ScalarMLP
    from bots.search_eval import evaluate_terminal_round_utility


class NeuralThreePlayerBot(BotPlayer):
    """A lightweight dependency-free neural bot for 3-player singleton smear."""

    MODEL_DIR = Path(__file__).with_name("models")
    MODEL_FILE_V1 = MODEL_DIR / "neural_3p_v1.json"
    MODEL_FILE_V2 = MODEL_DIR / "neural_3p_v2.json"
    MODEL_FILE_V3 = MODEL_DIR / "neural_3p_v3.json"
    MODEL_FILE_V4 = MODEL_DIR / "neural_3p_v4.json"
    MODEL_FILE_V5 = MODEL_DIR / "neural_3p_v5.json"
    MODEL_FILE = MODEL_FILE_V2
    DEFAULT_PLAY_ROLLOUT_DEPTH = 3
    DEFAULT_AUCTION_ROLLOUT_DEPTH = 1
    _model_cache: dict[Path, dict] = {}

    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
        *,
        model_path: str | Path | None = None,
        model_bundle: dict | None = None,
        fallback_to_greedy: bool = True,
    ):
        super().__init__(name, cards)
        if model_path is not None and model_bundle is not None:
            raise ValueError("provide either model_path or model_bundle, not both")

        self._model_path = Path(model_path) if model_path is not None else self.MODEL_FILE
        bundle = (
            self._validate_model_bundle(model_bundle, source="<memory>")
            if model_bundle is not None
            else self._load_model_bundle(self._model_path)
        )
        self._configure_from_bundle(bundle)
        self._fallback_to_greedy = fallback_to_greedy
        self._fallback_bot: GreedyPlayer | None = None
        self._context_player_names: list[str] | None = None
        self._context_teams: list[tuple[str, ...]] | None = None
        self._context_match_scores: dict[str, int] | None = None
        self._context_target_score = 21
        self._context_auction_state: AuctionState | None = None

    @classmethod
    def _validate_model_bundle(
        cls,
        payload: dict,
        *,
        source: str,
    ) -> dict:
        version = int(payload.get("version", 0))
        if version not in {1, 2, 3}:
            raise ValueError(f"unsupported neural bot model version in {source}")
        return payload

    @classmethod
    def _load_model_bundle(cls, model_path: Path) -> dict:
        resolved_path = model_path.resolve()
        cached_bundle = cls._model_cache.get(resolved_path)
        if cached_bundle is not None:
            return cached_bundle
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        bundle = cls._validate_model_bundle(payload, source=str(resolved_path))
        cls._model_cache[resolved_path] = bundle
        return bundle

    def _configure_from_bundle(self, bundle: dict) -> None:
        self._bundle_version = int(bundle.get("version", 1))
        self._bundle_bot_id = str(bundle.get("bot_id", "neural-3p"))
        self._play_model = ScalarMLP.from_dict(bundle["play_model"])
        self._auction_model = ScalarMLP.from_dict(bundle["auction_model"])

        if self._bundle_version >= 2 and "play_value_model" in bundle:
            self._play_value_model = ScalarMLP.from_dict(bundle["play_value_model"])
            self._auction_value_model = ScalarMLP.from_dict(bundle["auction_value_model"])
        else:
            self._play_value_model = None
            self._auction_value_model = None

        inference = bundle.get("inference", {})
        self._play_policy_weight = float(inference.get("play_policy_weight", 1.0))
        self._play_value_weight = float(
            inference.get("play_value_weight", 0.0 if self._play_value_model is None else 1.0)
        )
        self._auction_policy_weight = float(inference.get("auction_policy_weight", 1.0))
        self._auction_value_weight = float(
            inference.get(
                "auction_value_weight",
                0.0 if self._auction_value_model is None else 1.0,
            )
        )
        self._play_rollout_depth = max(
            int(inference.get("play_rollout_depth", self.DEFAULT_PLAY_ROLLOUT_DEPTH)),
            1,
        )
        self._auction_rollout_depth = max(
            int(inference.get("auction_rollout_depth", self.DEFAULT_AUCTION_ROLLOUT_DEPTH)),
            1,
        )

    def estimate_play_state_value(
        self,
        round_state: RoundState,
        *,
        perspective_player_name: str | None = None,
    ) -> float:
        return self._estimate_leaf_play_value(
            round_state=round_state,
            perspective_player_name=perspective_player_name or self.name,
        )

    def estimate_auction_state_value(
        self,
        auction_state: AuctionState,
        *,
        perspective_player_name: str | None = None,
    ) -> float:
        return self._estimate_leaf_auction_value(
            auction_state=auction_state,
            perspective_player_name=perspective_player_name or self.name,
        )

    def score_play_candidates(
        self,
        round_state: RoundState,
        *,
        perspective_player_name: str | None = None,
    ) -> list[tuple[Card, float, list[float]]]:
        acting_player_name = perspective_player_name or self.name
        legal_cards = ordered_legal_cards(round_state)
        scored_candidates: list[tuple[Card, float, list[float]]] = []
        for card in legal_cards:
            features = encode_play_candidate(
                round_state=round_state,
                acting_player_name=acting_player_name,
                candidate_card=card,
                match_scores=self._context_match_scores,
                target_score=self._context_target_score,
                auction_state=self._context_auction_state,
            )
            score = (
                (self._play_policy_weight * self._play_model.score(features))
                + (
                    self._play_value_weight
                    * (
                        self._rollout_play_value(
                            round_state=apply_trick_action_to_state(
                                round_state,
                                Play(round_state.current_player, card),
                            ),
                            perspective_player_name=acting_player_name,
                            remaining_depth=self._play_rollout_depth - 1,
                        )
                        if self._play_rollout_depth > 1
                        else self._estimate_play_successor_value(
                            round_state=round_state,
                            perspective_player_name=acting_player_name,
                            candidate_card=card,
                        )
                    )
                )
            )
            scored_candidates.append((card, score, features))
        return scored_candidates

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
        self._context_auction_state = auction_state

    def _is_supported_context(
        self,
        *,
        player_names: list[str],
        teams: list[tuple[str, ...]] | None,
    ) -> bool:
        return is_supported_three_player_singleton_context(player_names, teams)

    def _ensure_fallback_bot(self) -> GreedyPlayer:
        if self._fallback_bot is None:
            self._fallback_bot = GreedyPlayer(self.name)
        self._fallback_bot._cards = set(self.cards)
        if (
            self._context_player_names is not None
            and self._context_teams is not None
            and self._context_match_scores is not None
        ):
            self._fallback_bot.set_match_context(
                player_names=self._context_player_names,
                teams=self._context_teams,
                match_scores=self._context_match_scores,
                target_score=self._context_target_score,
                auction_state=self._context_auction_state,
                round_state=None,
            )
        return self._fallback_bot

    def _normalized_utility(self, utility: float) -> float:
        scale = max(float(self._context_target_score) * 2.0, 1.0)
        return utility / scale

    def _context_teams_for_round(self, round_state: RoundState) -> list[tuple[str, ...]]:
        if self._context_teams is not None:
            return self._context_teams
        return [
            tuple(player.name for player in team.constituents)
            for team in round_state.teams
        ]

    def _context_match_scores_for_round(self, round_state: RoundState) -> dict[str, int]:
        if self._context_match_scores is not None:
            return self._context_match_scores
        return {
            player.name: 0
            for player in round_state.players
        }

    def _estimate_play_successor_value(
        self,
        *,
        round_state: RoundState,
        perspective_player_name: str,
        candidate_card: Card,
    ) -> float:
        if self._play_value_model is None:
            return 0.0

        successor_round = apply_trick_action_to_state(
            round_state,
            Play(round_state.current_player, candidate_card),
        )
        if successor_round.is_terminal:
            utility = evaluate_terminal_round_utility(
                round_state=successor_round,
                auction_state=self._context_auction_state,
                match_scores=self._context_match_scores_for_round(round_state),
                teams=self._context_teams_for_round(round_state),
                target_score=self._context_target_score,
                player_name=perspective_player_name,
            )
            return self._normalized_utility(utility)

        features = encode_play_state(
            round_state=successor_round,
            perspective_player_name=perspective_player_name,
            match_scores=self._context_match_scores_for_round(round_state),
            target_score=self._context_target_score,
            auction_state=self._context_auction_state,
        )
        return self._play_value_model.score(features)

    def _estimate_leaf_play_value(
        self,
        *,
        round_state: RoundState,
        perspective_player_name: str,
    ) -> float:
        if round_state.is_terminal:
            utility = evaluate_terminal_round_utility(
                round_state=round_state,
                auction_state=self._context_auction_state,
                match_scores=self._context_match_scores_for_round(round_state),
                teams=self._context_teams_for_round(round_state),
                target_score=self._context_target_score,
                player_name=perspective_player_name,
            )
            return self._normalized_utility(utility)
        if self._play_value_model is None:
            return 0.0
        return self._play_value_model.score(
            encode_play_state(
                round_state=round_state,
                perspective_player_name=perspective_player_name,
                match_scores=self._context_match_scores_for_round(round_state),
                target_score=self._context_target_score,
                auction_state=self._context_auction_state,
            )
        )

    def _score_play_action_for_player(
        self,
        *,
        round_state: RoundState,
        perspective_player_name: str,
        candidate_card: Card,
    ) -> float:
        features = encode_play_candidate(
            round_state=round_state,
            acting_player_name=perspective_player_name,
            candidate_card=candidate_card,
            match_scores=self._context_match_scores,
            target_score=self._context_target_score,
            auction_state=self._context_auction_state,
        )
        return (
            self._play_policy_weight * self._play_model.score(features)
            + self._play_value_weight
            * self._estimate_play_successor_value(
                round_state=round_state,
                perspective_player_name=perspective_player_name,
                candidate_card=candidate_card,
            )
        )

    def _rollout_play_value(
        self,
        *,
        round_state: RoundState,
        perspective_player_name: str,
        remaining_depth: int,
    ) -> float:
        if round_state.is_terminal or remaining_depth <= 0:
            return self._estimate_leaf_play_value(
                round_state=round_state,
                perspective_player_name=perspective_player_name,
            )

        acting_player_name = round_state.current_player.name
        legal_cards = ordered_legal_cards(round_state)
        if not legal_cards:
            return self._estimate_leaf_play_value(
                round_state=round_state,
                perspective_player_name=perspective_player_name,
            )

        chosen_card = max(
            legal_cards,
            key=lambda card: (
                self._score_play_action_for_player(
                    round_state=round_state,
                    perspective_player_name=acting_player_name,
                    candidate_card=card,
                ),
                card.code,
            ),
        )
        successor_round = apply_trick_action_to_state(
            round_state,
            Play(round_state.current_player, chosen_card),
        )
        return self._rollout_play_value(
            round_state=successor_round,
            perspective_player_name=perspective_player_name,
            remaining_depth=remaining_depth - 1,
        )

    def _choose_card_with_model(self, round_state: RoundState) -> Card:
        scored_candidates = self.score_play_candidates(round_state)
        if not scored_candidates:
            raise ValueError("neural bot could not find a legal card")

        best_card, _, _ = max(
            scored_candidates,
            key=lambda item: (
                item[1],
                item[0].code,
            ),
        )
        return best_card

    def choose_card(self, round_state: RoundState) -> Card:
        if round_state.current_player.name != self.name:
            raise ValueError(
                f"{self.name} cannot choose a card for {round_state.current_player.name}"
            )

        player_names = [player.name for player in round_state.players]
        teams = [tuple(player.name for player in team.constituents) for team in round_state.teams]
        if self._is_supported_context(player_names=player_names, teams=teams):
            return self._choose_card_with_model(round_state)
        if self._fallback_to_greedy:
            return self._ensure_fallback_bot().choose_card(round_state)
        raise ValueError(f"{self._bundle_bot_id} only supports 3-player singleton games")

    def _estimate_auction_successor_value(
        self,
        *,
        auction_state: AuctionState,
        perspective_player_name: str,
        candidate_action: AuctionEvent,
    ) -> float:
        if self._auction_value_model is None:
            return 0.0

        successor_auction = deepcopy(auction_state)
        apply_auction_action_for_search(successor_auction, candidate_action)
        features = encode_auction_state(
            auction_state=successor_auction,
            perspective_player_name=perspective_player_name,
            hand=set(self.cards),
            match_scores=self._context_match_scores,
            target_score=self._context_target_score,
        )
        return self._auction_value_model.score(features)

    def _estimate_leaf_auction_value(
        self,
        *,
        auction_state: AuctionState,
        perspective_player_name: str,
    ) -> float:
        if self._auction_value_model is None:
            return 0.0
        return self._auction_value_model.score(
            encode_auction_state(
                auction_state=auction_state,
                perspective_player_name=perspective_player_name,
                hand=set(self.cards),
                match_scores=self._context_match_scores,
                target_score=self._context_target_score,
            )
        )

    def _successor_auction_state(
        self,
        *,
        auction_state: AuctionState,
        candidate_action: AuctionEvent,
    ) -> AuctionState:
        successor_auction = deepcopy(auction_state)
        apply_auction_action_for_search(successor_auction, candidate_action)
        return successor_auction

    def _score_auction_action_for_player(
        self,
        *,
        auction_state: AuctionState,
        perspective_player_name: str,
        candidate_action: AuctionEvent,
    ) -> float:
        features = encode_auction_candidate(
            auction_state=auction_state,
            acting_player_name=perspective_player_name,
            hand=set(self.cards),
            candidate_action=candidate_action,
            match_scores=self._context_match_scores,
            target_score=self._context_target_score,
        )
        successor_value = self._estimate_auction_successor_value(
            auction_state=auction_state,
            perspective_player_name=perspective_player_name,
            candidate_action=candidate_action,
        )
        return (
            self._auction_policy_weight * self._auction_model.score(features)
            + self._auction_value_weight * successor_value
        )

    def score_auction_candidates(
        self,
        auction_state: AuctionState,
        *,
        perspective_player_name: str | None = None,
    ) -> list[tuple[AuctionEvent, float, list[float]]]:
        acting_player_name = perspective_player_name or self.name
        legal_actions = ordered_legal_auction_actions(auction_state)
        scored_candidates: list[tuple[AuctionEvent, float, list[float]]] = []
        for action in legal_actions:
            features = encode_auction_candidate(
                auction_state=auction_state,
                acting_player_name=acting_player_name,
                hand=set(self.cards),
                candidate_action=action,
                match_scores=self._context_match_scores,
                target_score=self._context_target_score,
            )
            successor_value = (
                self._rollout_auction_value(
                    auction_state=self._successor_auction_state(
                        auction_state=auction_state,
                        candidate_action=action,
                    ),
                    perspective_player_name=acting_player_name,
                    remaining_depth=self._auction_rollout_depth - 1,
                )
                if self._auction_rollout_depth > 1
                else self._estimate_auction_successor_value(
                    auction_state=auction_state,
                    perspective_player_name=acting_player_name,
                    candidate_action=action,
                )
            )
            score = (
                (self._auction_policy_weight * self._auction_model.score(features))
                + (self._auction_value_weight * successor_value)
            )
            scored_candidates.append((action, score, features))
        return scored_candidates

    def _rollout_auction_value(
        self,
        *,
        auction_state: AuctionState,
        perspective_player_name: str,
        remaining_depth: int,
    ) -> float:
        if auction_state.is_complete or remaining_depth <= 0:
            return self._estimate_leaf_auction_value(
                auction_state=auction_state,
                perspective_player_name=perspective_player_name,
            )

        acting_player_name = auction_state.current_bidder_name
        legal_actions = ordered_legal_auction_actions(auction_state)
        if not legal_actions:
            return self._estimate_leaf_auction_value(
                auction_state=auction_state,
                perspective_player_name=perspective_player_name,
            )

        chosen_action = max(
            legal_actions,
            key=lambda action: (
                self._score_auction_action_for_player(
                    auction_state=auction_state,
                    perspective_player_name=acting_player_name,
                    candidate_action=action,
                ),
                -(action.amount or 0),
                action.action == "bid",
            ),
        )
        successor_auction = self._successor_auction_state(
            auction_state=auction_state,
            candidate_action=chosen_action,
        )
        return self._rollout_auction_value(
            auction_state=successor_auction,
            perspective_player_name=perspective_player_name,
            remaining_depth=remaining_depth - 1,
        )

    def _choose_auction_action_with_model(
        self,
        auction_state: AuctionState,
    ) -> AuctionEvent:
        scored_candidates = self.score_auction_candidates(auction_state)
        if not scored_candidates:
            raise ValueError("neural bot could not find a legal auction action")

        best_action, _, _ = max(
            scored_candidates,
            key=lambda item: (
                item[1],
                -(item[0].amount or 0),
                item[0].action == "bid",
            ),
        )
        return best_action

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        if auction_state.current_bidder_name != self.name:
            raise ValueError(
                f"{self.name} cannot choose an auction action for {auction_state.current_bidder_name}"
            )

        teams = self._context_teams
        if self._is_supported_context(
            player_names=list(auction_state.player_names),
            teams=teams,
        ):
            return self._choose_auction_action_with_model(auction_state)
        if self._fallback_to_greedy:
            return self._ensure_fallback_bot().choose_auction_action(auction_state)
        raise ValueError(f"{self._bundle_bot_id} only supports 3-player singleton games")


class NeuralThreePlayerV1Bot(NeuralThreePlayerBot):
    MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V1


class NeuralThreePlayerV3Bot(NeuralThreePlayerBot):
    MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V3


class NeuralThreePlayerV4Bot(NeuralThreePlayerBot):
    MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V4
    FALLBACK_MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V3

    @classmethod
    def _load_model_bundle(cls, model_path: Path) -> dict:
        if not model_path.exists() and model_path == cls.MODEL_FILE:
            fallback_path = cls.FALLBACK_MODEL_FILE
            if fallback_path.exists():
                fallback_bundle = deepcopy(super()._load_model_bundle(fallback_path))
                fallback_bundle["bot_id"] = "neural-3p-v4"
                fallback_bundle.setdefault("seed_bot_id", "neural-3p-v3")
                fallback_bundle.setdefault("training_mode", "alternating_seed")
                return fallback_bundle
        return super()._load_model_bundle(model_path)


class NeuralThreePlayerV5Bot(NeuralThreePlayerBot):
    MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V5
    FALLBACK_MODEL_FILE = NeuralThreePlayerBot.MODEL_FILE_V4

    @classmethod
    def _load_model_bundle(cls, model_path: Path) -> dict:
        if not model_path.exists() and model_path == cls.MODEL_FILE:
            fallback_path = cls.FALLBACK_MODEL_FILE
            if fallback_path.exists():
                fallback_bundle = deepcopy(super()._load_model_bundle(fallback_path))
                fallback_bundle["bot_id"] = "neural-3p-v5"
                fallback_bundle.setdefault("seed_bot_id", "neural-3p-v4")
                fallback_bundle.setdefault("training_mode", "offline_policy_improvement_seed")
                return fallback_bundle
        return super()._load_model_bundle(model_path)
