from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
import random
from pathlib import Path
import sys

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot, NeuralThreePlayerV1Bot
    from .bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from .bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .bots.search_eval import evaluate_terminal_round_utility
    from .engine import apply_auction_action_for_search, apply_trick_action_to_state
    from .gameplay import MatchController
    from .models import AuctionEvent, Card, Play
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot, NeuralThreePlayerV1Bot
    from bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from bots.registry import build_ready_bot, get_ready_bot_spec
    from bots.search_eval import evaluate_terminal_round_utility
    from engine import apply_auction_action_for_search, apply_trick_action_to_state
    from gameplay import MatchController
    from models import AuctionEvent, Card, Play


DEFAULT_OUTPUT = NeuralThreePlayerBot.MODEL_FILE
DEFAULT_PLAY_HIDDEN_DIM = 48
DEFAULT_AUCTION_HIDDEN_DIM = 32
DEFAULT_PLAY_VALUE_HIDDEN_DIM = 48
DEFAULT_AUCTION_VALUE_HIDDEN_DIM = 32
DEFAULT_TEACHER_POOL = "optimal-bot:4,1-trick-minmax:2,greedy:1"
DEFAULT_PLAY_VALUE_WEIGHT = 0.8
DEFAULT_AUCTION_VALUE_WEIGHT = 0.55
THREE_PLAYER_NAMES = ["Player 1", "Player 2", "Player 3"]


@dataclass(frozen=True)
class TeacherSpec:
    bot_id: str
    weight: float = 1.0


@dataclass(frozen=True)
class PendingValueExample:
    player_name: str
    features: list[float]


@dataclass
class TrainingDataset:
    play_policy_examples: list[ChoiceExample] = field(default_factory=list)
    auction_policy_examples: list[ChoiceExample] = field(default_factory=list)
    play_value_examples: list[RegressionExample] = field(default_factory=list)
    auction_value_examples: list[RegressionExample] = field(default_factory=list)

    def extend(self, other: "TrainingDataset") -> None:
        self.play_policy_examples.extend(other.play_policy_examples)
        self.auction_policy_examples.extend(other.auction_policy_examples)
        self.play_value_examples.extend(other.play_value_examples)
        self.auction_value_examples.extend(other.auction_value_examples)


class TeacherOracle:
    def __init__(self, teacher_specs: list[TeacherSpec], seed: int):
        if not teacher_specs:
            raise ValueError("teacher_specs must not be empty")
        self._teacher_specs = list(teacher_specs)
        self._rng = random.Random(seed)
        self._bots = {}

    def _sample_teacher_spec(self) -> TeacherSpec:
        total_weight = sum(spec.weight for spec in self._teacher_specs)
        threshold = self._rng.random() * total_weight
        running_total = 0.0
        for spec in self._teacher_specs:
            running_total += spec.weight
            if threshold <= running_total:
                return spec
        return self._teacher_specs[-1]

    def _get_bot(self, bot_id: str, player_name: str):
        cache_key = (bot_id, player_name)
        bot = self._bots.get(cache_key)
        if bot is None:
            bot = build_ready_bot(bot_id, player_name)
            self._bots[cache_key] = bot
        return bot

    def choose_auction_action(
        self,
        *,
        auction_state,
        player_name: str,
        hand: set[Card],
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
    ) -> tuple[AuctionEvent, str]:
        teacher_spec = self._sample_teacher_spec()
        bot = self._get_bot(teacher_spec.bot_id, player_name)
        bot._cards = set(hand)
        if hasattr(bot, "set_match_context"):
            bot.set_match_context(
                player_names=player_names,
                teams=teams,
                match_scores=match_scores,
                target_score=target_score,
                auction_state=auction_state,
                round_state=None,
            )
        return bot.choose_auction_action(auction_state), teacher_spec.bot_id

    def choose_card(
        self,
        *,
        round_state,
        player_name: str,
        auction_state,
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
    ) -> tuple[Card, str]:
        teacher_spec = self._sample_teacher_spec()
        bot = self._get_bot(teacher_spec.bot_id, player_name)
        bot._cards = set(round_state.current_player.cards)
        if hasattr(bot, "set_match_context"):
            bot.set_match_context(
                player_names=player_names,
                teams=teams,
                match_scores=match_scores,
                target_score=target_score,
                auction_state=auction_state,
                round_state=round_state,
            )
        return bot.choose_card(round_state), teacher_spec.bot_id


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _format_teacher_pool(teacher_specs: list[TeacherSpec]) -> str:
    return ", ".join(
        f"{teacher_spec.bot_id}x{teacher_spec.weight:g}"
        for teacher_spec in teacher_specs
    )


def _format_dataset_counts(dataset: TrainingDataset) -> str:
    return (
        f"play_policy={len(dataset.play_policy_examples)} "
        f"auction_policy={len(dataset.auction_policy_examples)} "
        f"play_value={len(dataset.play_value_examples)} "
        f"auction_value={len(dataset.auction_value_examples)}"
    )


def _format_training_metrics(training_report: dict) -> str:
    return (
        f"play_acc={training_report['play_history'][-1]['accuracy']:.3f} "
        f"auction_acc={training_report['auction_history'][-1]['accuracy']:.3f} "
        f"play_value_mse={training_report['play_value_history'][-1]['mse']:.4f} "
        f"auction_value_mse={training_report['auction_value_history'][-1]['mse']:.4f}"
    )


def normalize_utility(utility: float, target_score: int) -> float:
    scale = max(float(target_score) * 2.0, 1.0)
    return utility / scale


def parse_teacher_specs(
    teacher_pool: str | None = None,
    *,
    fallback_teacher_bot_id: str | None = None,
) -> list[TeacherSpec]:
    raw_spec = teacher_pool.strip() if teacher_pool is not None else ""
    if not raw_spec:
        if fallback_teacher_bot_id is None:
            raise ValueError("teacher_pool cannot be empty")
        raw_spec = fallback_teacher_bot_id

    specs: list[TeacherSpec] = []
    for item in raw_spec.split(","):
        token = item.strip()
        if not token:
            continue
        bot_id, separator, raw_weight = token.partition(":")
        cleaned_bot_id = bot_id.strip()
        if not cleaned_bot_id:
            raise ValueError(f"invalid teacher token: {token!r}")
        get_ready_bot_spec(cleaned_bot_id)
        if separator:
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise ValueError(f"invalid teacher weight in {token!r}") from exc
        else:
            weight = 1.0
        if weight <= 0:
            raise ValueError(f"teacher weights must be positive: {token!r}")
        specs.append(TeacherSpec(cleaned_bot_id, weight))

    if not specs:
        raise ValueError("teacher_pool cannot be empty")
    return specs


def _serialize_teacher_specs(teacher_specs: list[TeacherSpec]) -> list[dict]:
    return [
        {"bot_id": teacher_spec.bot_id, "weight": teacher_spec.weight}
        for teacher_spec in teacher_specs
    ]


def load_model_bundle(model_path: str | Path) -> dict:
    return json.loads(Path(model_path).read_text(encoding="utf-8"))


def _build_student_bots(model_bundle: dict) -> dict[str, NeuralThreePlayerBot]:
    return {
        player_name: NeuralThreePlayerBot(player_name, model_bundle=model_bundle)
        for player_name in THREE_PLAYER_NAMES
    }


def _build_match_controller_for_actor(
    *,
    actor_mode: str,
    teacher_specs: list[TeacherSpec],
    student_bundle: dict | None,
    seat_rng: random.Random,
) -> MatchController:
    if actor_mode == "teacher":
        player_bot_ids = [
            seat_rng.choices(
                [spec.bot_id for spec in teacher_specs],
                weights=[spec.weight for spec in teacher_specs],
                k=1,
            )[0]
            for _ in THREE_PLAYER_NAMES
        ]
        return MatchController.create(
            num_players=3,
            player_names=THREE_PLAYER_NAMES,
            teams=None,
            player_bot_ids=player_bot_ids,
            auto_run_bots=False,
        )

    if student_bundle is None:
        raise ValueError("student_bundle is required for student rollouts")
    return MatchController.create(
        num_players=3,
        player_names=THREE_PLAYER_NAMES,
        teams=None,
        bots=_build_student_bots(student_bundle),
        auto_run_bots=False,
    )


def _resolve_round_value_targets(
    *,
    controller: MatchController,
    round_start_match_scores: dict[str, int],
) -> dict[str, float]:
    round_state = controller.session.game.round_state
    auction_state = controller.session.auction.state
    teams = controller.session.teams
    target_score = controller.session.target_score
    return {
        player_name: normalize_utility(
            evaluate_terminal_round_utility(
                round_state=round_state,
                auction_state=auction_state,
                match_scores=round_start_match_scores,
                teams=teams,
                target_score=target_score,
                player_name=player_name,
            ),
            target_score,
        )
        for player_name in controller.session.player_names
    }


def _collect_rollout_examples(
    *,
    actor_mode: str,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    student_bundle: dict | None = None,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    if actor_mode not in {"teacher", "student"}:
        raise ValueError("actor_mode must be 'teacher' or 'student'")
    if match_count <= 0:
        raise ValueError("match_count must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    dataset = TrainingDataset()
    teacher_oracle = TeacherOracle(teacher_specs, seed + 10_000)
    seat_rng = random.Random(seed)
    teacher_label_counts: dict[str, int] = {}
    acting_bot_counts: dict[str, int] = {}
    progress_interval = max(1, match_count // 10)

    _log(
        verbose,
        (
            f"[collect:{actor_mode}] start matches={match_count} alpha={alpha} "
            f"seed={seed} teacher_pool={_format_teacher_pool(teacher_specs)}"
        ),
    )

    for match_index in range(match_count):
        random.seed(seed + match_index)
        controller = _build_match_controller_for_actor(
            actor_mode=actor_mode,
            teacher_specs=teacher_specs,
            student_bundle=student_bundle,
            seat_rng=seat_rng,
        )
        round_start_match_scores = dict(controller.session.match_scores)
        pending_play_values: list[PendingValueExample] = []
        pending_auction_values: list[PendingValueExample] = []

        for round_index in range(1, alpha + 1):
            while controller.session.phase in {"auction", "play"}:
                player_name = controller._current_bot_name()
                if player_name is None:
                    raise ValueError("match controller had no active bot turn")

                controller._sync_bot_hand(player_name)
                bot = controller.session.bots[player_name]

                if controller.session.phase == "auction":
                    auction_state = controller.session.auction.state
                    legal_actions = ordered_legal_auction_actions(auction_state)
                    acting_hand = set(bot.cards)
                    environment_action = bot.choose_auction_action(auction_state)
                    if environment_action not in legal_actions:
                        raise ValueError(
                            f"actor selected illegal auction action {environment_action}"
                        )

                    if actor_mode == "teacher":
                        teacher_action = environment_action
                        teacher_bot_id = controller.session.player_bot_ids[player_name]
                        if teacher_bot_id is None:
                            raise ValueError("teacher rollout was missing a configured bot id")
                    else:
                        teacher_action, teacher_bot_id = teacher_oracle.choose_auction_action(
                            auction_state=auction_state,
                            player_name=player_name,
                            hand=acting_hand,
                            player_names=controller.session.player_names,
                            teams=controller.session.teams,
                            match_scores=controller.session.match_scores,
                            target_score=controller.session.target_score,
                        )
                        if teacher_action not in legal_actions:
                            raise ValueError(
                                f"teacher oracle selected illegal auction action {teacher_action}"
                            )

                    dataset.auction_policy_examples.append(
                        ChoiceExample(
                            candidate_features=[
                                encode_auction_candidate(
                                    auction_state=auction_state,
                                    acting_player_name=player_name,
                                    hand=acting_hand,
                                    candidate_action=legal_action,
                                    match_scores=controller.session.match_scores,
                                    target_score=controller.session.target_score,
                                )
                                for legal_action in legal_actions
                            ],
                            chosen_index=legal_actions.index(teacher_action),
                        )
                    )
                    teacher_label_counts[teacher_bot_id] = (
                        teacher_label_counts.get(teacher_bot_id, 0) + 1
                    )
                    acting_key = (
                        controller.session.player_bot_ids.get(player_name)
                        if actor_mode == "teacher"
                        else controller.session.bots[player_name]._bundle_bot_id
                    )
                    acting_bot_counts[acting_key] = acting_bot_counts.get(acting_key, 0) + 1

                    successor_auction = deepcopy(auction_state)
                    apply_auction_action_for_search(successor_auction, environment_action)
                    pending_auction_values.append(
                        PendingValueExample(
                            player_name=player_name,
                            features=encode_auction_state(
                                auction_state=successor_auction,
                                perspective_player_name=player_name,
                                hand=acting_hand,
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                            ),
                        )
                    )

                    controller.session.auction.apply_event(environment_action)
                    controller._advance_or_finalize_auction()
                    continue

                round_state = controller.session.game.round_state
                legal_cards = ordered_legal_cards(round_state)
                environment_card = bot.choose_card(round_state)
                if environment_card not in legal_cards:
                    raise ValueError(f"actor selected illegal card {environment_card.code}")

                if actor_mode == "teacher":
                    teacher_card = environment_card
                    teacher_bot_id = controller.session.player_bot_ids[player_name]
                    if teacher_bot_id is None:
                        raise ValueError("teacher rollout was missing a configured bot id")
                else:
                    teacher_card, teacher_bot_id = teacher_oracle.choose_card(
                        round_state=round_state,
                        player_name=player_name,
                        auction_state=controller.session.auction.state,
                        player_names=controller.session.player_names,
                        teams=controller.session.teams,
                        match_scores=controller.session.match_scores,
                        target_score=controller.session.target_score,
                    )
                    if teacher_card not in legal_cards:
                        raise ValueError(
                            f"teacher oracle selected illegal card {teacher_card.code}"
                        )

                dataset.play_policy_examples.append(
                    ChoiceExample(
                        candidate_features=[
                            encode_play_candidate(
                                round_state=round_state,
                                acting_player_name=player_name,
                                candidate_card=legal_card,
                                match_scores=controller.session.match_scores,
                                target_score=controller.session.target_score,
                                auction_state=controller.session.auction.state,
                            )
                            for legal_card in legal_cards
                        ],
                        chosen_index=legal_cards.index(teacher_card),
                    )
                )
                teacher_label_counts[teacher_bot_id] = (
                    teacher_label_counts.get(teacher_bot_id, 0) + 1
                )
                acting_key = (
                    controller.session.player_bot_ids.get(player_name)
                    if actor_mode == "teacher"
                    else controller.session.bots[player_name]._bundle_bot_id
                )
                acting_bot_counts[acting_key] = acting_bot_counts.get(acting_key, 0) + 1

                successor_round = apply_trick_action_to_state(
                    round_state,
                    Play(round_state.current_player, environment_card),
                )
                if not successor_round.is_terminal:
                    pending_play_values.append(
                        PendingValueExample(
                            player_name=player_name,
                            features=encode_play_state(
                                round_state=successor_round,
                                perspective_player_name=player_name,
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                                auction_state=controller.session.auction.state,
                            ),
                        )
                    )

                controller.session.game.apply_trick_action(
                    Play(controller.session.game.curr_player, environment_card)
                )
                controller._score_terminal_round_if_needed()

            if controller.session.game.round_state.is_terminal:
                round_targets = _resolve_round_value_targets(
                    controller=controller,
                    round_start_match_scores=round_start_match_scores,
                )
                dataset.play_value_examples.extend(
                    RegressionExample(example.features, round_targets[example.player_name])
                    for example in pending_play_values
                )
                dataset.auction_value_examples.extend(
                    RegressionExample(example.features, round_targets[example.player_name])
                    for example in pending_auction_values
                )
                pending_play_values = []
                pending_auction_values = []

            if controller.session.is_match_complete:
                break

            if round_index < alpha:
                controller.next_round(auto_run_bots=False)
                round_start_match_scores = dict(controller.session.match_scores)

        if (
            verbose
            and (
                match_index == 0
                or match_index + 1 == match_count
                or (match_index + 1) % progress_interval == 0
            )
        ):
            _log(
                True,
                (
                    f"[collect:{actor_mode}] matches={match_index + 1}/{match_count} "
                    f"{_format_dataset_counts(dataset)}"
                ),
            )

    report = {
        "actor_mode": actor_mode,
        "matches": match_count,
        "alpha": alpha,
        "teacher_label_counts": teacher_label_counts,
        "acting_bot_counts": acting_bot_counts,
        "play_policy_examples": len(dataset.play_policy_examples),
        "auction_policy_examples": len(dataset.auction_policy_examples),
        "play_value_examples": len(dataset.play_value_examples),
        "auction_value_examples": len(dataset.auction_value_examples),
    }
    _log(
        verbose,
        (
            f"[collect:{actor_mode}] done "
            f"{_format_dataset_counts(dataset)} labels={report['teacher_label_counts']}"
        ),
    )
    return dataset, report


def collect_teacher_examples(
    *,
    teacher_bot_id: str,
    match_count: int,
    alpha: int,
    seed: int,
) -> tuple[list[ChoiceExample], list[ChoiceExample]]:
    dataset, _ = _collect_rollout_examples(
        actor_mode="teacher",
        teacher_specs=parse_teacher_specs(fallback_teacher_bot_id=teacher_bot_id),
        match_count=match_count,
        alpha=alpha,
        seed=seed,
    )
    return dataset.play_policy_examples, dataset.auction_policy_examples


def collect_teacher_training_dataset(
    *,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    return _collect_rollout_examples(
        actor_mode="teacher",
        teacher_specs=teacher_specs,
        match_count=match_count,
        alpha=alpha,
        seed=seed,
        verbose=verbose,
    )


def collect_dagger_training_dataset(
    *,
    student_bundle: dict,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    return _collect_rollout_examples(
        actor_mode="student",
        teacher_specs=teacher_specs,
        match_count=match_count,
        alpha=alpha,
        seed=seed,
        student_bundle=student_bundle,
        verbose=verbose,
    )


def _initialize_model(
    *,
    initial_bundle: dict | None,
    bundle_key: str,
    input_dim: int,
    hidden_dim: int,
    seed: int,
) -> ScalarMLP:
    if initial_bundle is not None and bundle_key in initial_bundle:
        model = ScalarMLP.from_dict(initial_bundle[bundle_key]).copy()
        if model.input_dim != input_dim:
            raise ValueError(
                f"warm-started model {bundle_key} expected input_dim={model.input_dim}, got {input_dim}"
            )
        return model
    return ScalarMLP.initialize(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )


def train_neural_3p_bundle(
    *,
    play_examples: list[ChoiceExample],
    auction_examples: list[ChoiceExample],
    play_value_examples: list[RegressionExample],
    auction_value_examples: list[RegressionExample],
    play_hidden_dim: int,
    auction_hidden_dim: int,
    play_value_hidden_dim: int,
    auction_value_hidden_dim: int,
    play_epochs: int,
    auction_epochs: int,
    play_value_epochs: int,
    auction_value_epochs: int,
    play_learning_rate: float,
    auction_learning_rate: float,
    play_value_learning_rate: float,
    auction_value_learning_rate: float,
    l2: float,
    seed: int,
    teacher_specs: list[TeacherSpec],
    initial_bundle: dict | None = None,
    play_value_weight: float = DEFAULT_PLAY_VALUE_WEIGHT,
    auction_value_weight: float = DEFAULT_AUCTION_VALUE_WEIGHT,
    bot_id: str = "neural-3p-v2",
    verbose: bool = False,
) -> tuple[dict, dict]:
    if not play_examples:
        raise ValueError("play_examples must not be empty")
    if not auction_examples:
        raise ValueError("auction_examples must not be empty")
    if not play_value_examples:
        raise ValueError("play_value_examples must not be empty")
    if not auction_value_examples:
        raise ValueError("auction_value_examples must not be empty")

    play_input_dim = len(play_examples[0].candidate_features[0])
    auction_input_dim = len(auction_examples[0].candidate_features[0])
    play_value_input_dim = len(play_value_examples[0].features)
    auction_value_input_dim = len(auction_value_examples[0].features)

    _log(
        verbose,
        (
            "[train] start "
            f"play_examples={len(play_examples)} auction_examples={len(auction_examples)} "
            f"play_value_examples={len(play_value_examples)} "
            f"auction_value_examples={len(auction_value_examples)} "
            f"seed={seed}"
        ),
    )

    play_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="play_model",
        input_dim=play_input_dim,
        hidden_dim=play_hidden_dim,
        seed=seed,
    )
    auction_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="auction_model",
        input_dim=auction_input_dim,
        hidden_dim=auction_hidden_dim,
        seed=seed + 1,
    )
    play_value_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="play_value_model",
        input_dim=play_value_input_dim,
        hidden_dim=play_value_hidden_dim,
        seed=seed + 2,
    )
    auction_value_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="auction_value_model",
        input_dim=auction_value_input_dim,
        hidden_dim=auction_value_hidden_dim,
        seed=seed + 3,
    )

    play_history = play_model.train_choice_examples(
        play_examples,
        epochs=play_epochs,
        learning_rate=play_learning_rate,
        l2=l2,
        seed=seed,
    )
    auction_history = auction_model.train_choice_examples(
        auction_examples,
        epochs=auction_epochs,
        learning_rate=auction_learning_rate,
        l2=l2,
        seed=seed + 1,
    )
    play_value_history = play_value_model.train_regression_examples(
        play_value_examples,
        epochs=play_value_epochs,
        learning_rate=play_value_learning_rate,
        l2=l2,
        seed=seed + 2,
    )
    auction_value_history = auction_value_model.train_regression_examples(
        auction_value_examples,
        epochs=auction_value_epochs,
        learning_rate=auction_value_learning_rate,
        l2=l2,
        seed=seed + 3,
    )

    bundle = {
        "version": 2,
        "bot_id": bot_id,
        "teacher_pool": _serialize_teacher_specs(teacher_specs),
        "play_model": play_model.to_dict(),
        "auction_model": auction_model.to_dict(),
        "play_value_model": play_value_model.to_dict(),
        "auction_value_model": auction_value_model.to_dict(),
        "inference": {
            "play_policy_weight": 1.0,
            "play_value_weight": play_value_weight,
            "auction_policy_weight": 1.0,
            "auction_value_weight": auction_value_weight,
        },
    }
    training_report = {
        "play_history": play_history,
        "auction_history": auction_history,
        "play_value_history": play_value_history,
        "auction_value_history": auction_value_history,
        "play_examples": len(play_examples),
        "auction_examples": len(auction_examples),
        "play_value_examples": len(play_value_examples),
        "auction_value_examples": len(auction_value_examples),
    }
    _log(verbose, f"[train] done {_format_training_metrics(training_report)}")
    return bundle, training_report


def train_with_dagger(
    *,
    teacher_specs: list[TeacherSpec],
    bootstrap_matches: int,
    dagger_matches: int,
    alpha: int,
    dagger_iterations: int,
    seed: int,
    play_hidden_dim: int,
    auction_hidden_dim: int,
    play_value_hidden_dim: int,
    auction_value_hidden_dim: int,
    play_epochs: int,
    auction_epochs: int,
    play_value_epochs: int,
    auction_value_epochs: int,
    play_learning_rate: float,
    auction_learning_rate: float,
    play_value_learning_rate: float,
    auction_value_learning_rate: float,
    l2: float,
    initial_bundle: dict | None = None,
    play_value_weight: float = DEFAULT_PLAY_VALUE_WEIGHT,
    auction_value_weight: float = DEFAULT_AUCTION_VALUE_WEIGHT,
    bot_id: str = "neural-3p-v2",
    verbose: bool = False,
) -> tuple[dict, dict]:
    _log(
        verbose,
        (
            f"[dagger] bootstrap_matches={bootstrap_matches} dagger_matches={dagger_matches} "
            f"dagger_iterations={dagger_iterations} teacher_pool={_format_teacher_pool(teacher_specs)}"
        ),
    )
    aggregate_dataset, bootstrap_report = collect_teacher_training_dataset(
        teacher_specs=teacher_specs,
        match_count=bootstrap_matches,
        alpha=alpha,
        seed=seed,
        verbose=verbose,
    )
    _log(verbose, f"[dagger] bootstrap dataset {_format_dataset_counts(aggregate_dataset)}")

    bundle, training_report = train_neural_3p_bundle(
        play_examples=aggregate_dataset.play_policy_examples,
        auction_examples=aggregate_dataset.auction_policy_examples,
        play_value_examples=aggregate_dataset.play_value_examples,
        auction_value_examples=aggregate_dataset.auction_value_examples,
        play_hidden_dim=play_hidden_dim,
        auction_hidden_dim=auction_hidden_dim,
        play_value_hidden_dim=play_value_hidden_dim,
        auction_value_hidden_dim=auction_value_hidden_dim,
        play_epochs=play_epochs,
        auction_epochs=auction_epochs,
        play_value_epochs=play_value_epochs,
        auction_value_epochs=auction_value_epochs,
        play_learning_rate=play_learning_rate,
        auction_learning_rate=auction_learning_rate,
        play_value_learning_rate=play_value_learning_rate,
        auction_value_learning_rate=auction_value_learning_rate,
        l2=l2,
        seed=seed,
        teacher_specs=teacher_specs,
        initial_bundle=initial_bundle,
        play_value_weight=play_value_weight,
        auction_value_weight=auction_value_weight,
        bot_id=bot_id,
        verbose=verbose,
    )
    _log(verbose, f"[dagger] post-bootstrap {_format_training_metrics(training_report)}")

    dagger_reports: list[dict] = []
    for dagger_iteration in range(dagger_iterations):
        _log(verbose, f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} collecting student rollouts")
        dagger_dataset, dagger_report = collect_dagger_training_dataset(
            student_bundle=bundle,
            teacher_specs=teacher_specs,
            match_count=dagger_matches,
            alpha=alpha,
            seed=seed + 1_000 + dagger_iteration,
            verbose=verbose,
        )
        aggregate_dataset.extend(dagger_dataset)
        _log(
            verbose,
            (
                f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} "
                f"aggregate {_format_dataset_counts(aggregate_dataset)}"
            ),
        )
        bundle, training_report = train_neural_3p_bundle(
            play_examples=aggregate_dataset.play_policy_examples,
            auction_examples=aggregate_dataset.auction_policy_examples,
            play_value_examples=aggregate_dataset.play_value_examples,
            auction_value_examples=aggregate_dataset.auction_value_examples,
            play_hidden_dim=play_hidden_dim,
            auction_hidden_dim=auction_hidden_dim,
            play_value_hidden_dim=play_value_hidden_dim,
            auction_value_hidden_dim=auction_value_hidden_dim,
            play_epochs=play_epochs,
            auction_epochs=auction_epochs,
            play_value_epochs=play_value_epochs,
            auction_value_epochs=auction_value_epochs,
            play_learning_rate=play_learning_rate,
            auction_learning_rate=auction_learning_rate,
            play_value_learning_rate=play_value_learning_rate,
            auction_value_learning_rate=auction_value_learning_rate,
            l2=l2,
            seed=seed + (dagger_iteration + 1) * 100,
            teacher_specs=teacher_specs,
            initial_bundle=bundle,
            play_value_weight=play_value_weight,
            auction_value_weight=auction_value_weight,
            bot_id=bot_id,
            verbose=verbose,
        )
        _log(
            verbose,
            f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} {_format_training_metrics(training_report)}",
        )
        dagger_reports.append(
            {
                "iteration": dagger_iteration + 1,
                **dagger_report,
                "aggregate_play_examples": len(aggregate_dataset.play_policy_examples),
                "aggregate_auction_examples": len(aggregate_dataset.auction_policy_examples),
                "aggregate_play_value_examples": len(aggregate_dataset.play_value_examples),
                "aggregate_auction_value_examples": len(aggregate_dataset.auction_value_examples),
            }
        )

    report = {
        "bootstrap": bootstrap_report,
        "dagger_iterations": dagger_reports,
        "training": training_report,
        "teacher_pool": _serialize_teacher_specs(teacher_specs),
    }
    return bundle, report


def save_model_bundle(bundle: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(bundle), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the dependency-free 3-player singleton neural smear bot."
    )
    parser.add_argument(
        "--teacher",
        default=None,
        help="legacy single-teacher bot id; used when --teacher-pool is omitted",
    )
    parser.add_argument(
        "--teacher-pool",
        default=DEFAULT_TEACHER_POOL,
        help=(
            "comma-separated teacher mixture using bot_id[:weight], for example "
            "'optimal-bot:4,1-trick-minmax:2,greedy:1'"
        ),
    )
    parser.add_argument(
        "--bootstrap-matches",
        type=int,
        default=36,
        help="number of mixed-teacher rollout matches for the initial supervised pass",
    )
    parser.add_argument(
        "--dagger-matches",
        type=int,
        default=18,
        help="number of student rollout matches per DAgger iteration",
    )
    parser.add_argument(
        "--dagger-iterations",
        type=int,
        default=2,
        help="number of DAgger relabel-and-retrain iterations",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=12,
        help="maximum rounds per collected match",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for data collection and initialization",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="optional starting checkpoint to warm-start from",
    )
    parser.add_argument("--play-hidden-dim", type=int, default=DEFAULT_PLAY_HIDDEN_DIM)
    parser.add_argument("--auction-hidden-dim", type=int, default=DEFAULT_AUCTION_HIDDEN_DIM)
    parser.add_argument(
        "--play-value-hidden-dim",
        type=int,
        default=DEFAULT_PLAY_VALUE_HIDDEN_DIM,
    )
    parser.add_argument(
        "--auction-value-hidden-dim",
        type=int,
        default=DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
    )
    parser.add_argument("--play-epochs", type=int, default=16)
    parser.add_argument("--auction-epochs", type=int, default=16)
    parser.add_argument("--play-value-epochs", type=int, default=18)
    parser.add_argument("--auction-value-epochs", type=int, default=18)
    parser.add_argument("--play-learning-rate", type=float, default=0.028)
    parser.add_argument("--auction-learning-rate", type=float, default=0.034)
    parser.add_argument("--play-value-learning-rate", type=float, default=0.024)
    parser.add_argument("--auction-value-learning-rate", type=float, default=0.028)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--play-value-weight",
        type=float,
        default=DEFAULT_PLAY_VALUE_WEIGHT,
        help="runtime weight for the play value head during one-ply lookahead",
    )
    parser.add_argument(
        "--auction-value-weight",
        type=float,
        default=DEFAULT_AUCTION_VALUE_WEIGHT,
        help="runtime weight for the auction value head during one-ply lookahead",
    )
    parser.add_argument(
        "--bot-id",
        default="neural-3p-v2",
        help="bot id to store in the checkpoint metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the trained model bundle",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress live training progress logs and only print the final JSON summary",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    teacher_specs = parse_teacher_specs(
        args.teacher_pool,
        fallback_teacher_bot_id=args.teacher,
    )
    initial_bundle = (
        load_model_bundle(args.initial_model)
        if args.initial_model is not None
        else None
    )
    bundle, report = train_with_dagger(
        teacher_specs=teacher_specs,
        bootstrap_matches=args.bootstrap_matches,
        dagger_matches=args.dagger_matches,
        alpha=args.alpha,
        dagger_iterations=args.dagger_iterations,
        seed=args.seed,
        play_hidden_dim=args.play_hidden_dim,
        auction_hidden_dim=args.auction_hidden_dim,
        play_value_hidden_dim=args.play_value_hidden_dim,
        auction_value_hidden_dim=args.auction_value_hidden_dim,
        play_epochs=args.play_epochs,
        auction_epochs=args.auction_epochs,
        play_value_epochs=args.play_value_epochs,
        auction_value_epochs=args.auction_value_epochs,
        play_learning_rate=args.play_learning_rate,
        auction_learning_rate=args.auction_learning_rate,
        play_value_learning_rate=args.play_value_learning_rate,
        auction_value_learning_rate=args.auction_value_learning_rate,
        l2=args.l2,
        initial_bundle=initial_bundle,
        play_value_weight=args.play_value_weight,
        auction_value_weight=args.auction_value_weight,
        bot_id=args.bot_id,
        verbose=not args.quiet,
    )
    save_model_bundle(bundle, args.output)
    _log(
        not args.quiet,
        (
            f"[train] wrote {args.output} "
            f"{_format_training_metrics(report['training'])}"
        ),
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "teacher_pool": _serialize_teacher_specs(teacher_specs),
                "final_play_accuracy": report["training"]["play_history"][-1]["accuracy"],
                "final_auction_accuracy": report["training"]["auction_history"][-1]["accuracy"],
                "final_play_value_mse": report["training"]["play_value_history"][-1]["mse"],
                "final_auction_value_mse": report["training"]["auction_value_history"][-1]["mse"],
                **report,
            }
        )
    )


if __name__ == "__main__":
    main()
