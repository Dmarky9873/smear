from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot
    from .bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from .bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from .bots.registry import get_ready_bot_spec
    from .engine import apply_auction_action_for_search, apply_trick_action_to_state
    from .gameplay import MatchController
    from .models import AuctionEvent, Card, Play
    from .self_train_neural_3p_v2 import (
        DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
        DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
        DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
        _render_promotion_outcome_block,
        evaluate_candidate,
    )
    from .train_neural_3p_bot import (
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_WEIGHT,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_WORKERS,
        THREE_PLAYER_NAMES,
        TRAINING_BACKEND_CHOICES,
        TeacherOracle,
        TeacherSpec,
        TrainingBotProvider,
        TrainingDataset,
        _create_parallel_executor,
        _format_dataset_counts,
        _format_seconds,
        _format_training_metrics,
        _merge_count_dict,
        _policy_example_weight,
        _render_compact_block,
        _resolve_round_value_targets,
        _resolve_worker_count,
        _serialize_teacher_specs,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_neural_3p_bundle,
    )
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot
    from bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from bots.registry import get_ready_bot_spec
    from engine import apply_auction_action_for_search, apply_trick_action_to_state
    from gameplay import MatchController
    from models import AuctionEvent, Card, Play
    from self_train_neural_3p_v2 import (
        DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
        DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
        DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
        _render_promotion_outcome_block,
        evaluate_candidate,
    )
    from train_neural_3p_bot import (
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_WEIGHT,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_WORKERS,
        THREE_PLAYER_NAMES,
        TRAINING_BACKEND_CHOICES,
        TeacherOracle,
        TeacherSpec,
        TrainingBotProvider,
        TrainingDataset,
        _create_parallel_executor,
        _format_dataset_counts,
        _format_seconds,
        _format_training_metrics,
        _merge_count_dict,
        _policy_example_weight,
        _render_compact_block,
        _resolve_round_value_targets,
        _resolve_worker_count,
        _serialize_teacher_specs,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_neural_3p_bundle,
    )


V5_BOT_ID = "neural-3p-v5"
INCUMBENT_ACTOR_ID = "incumbent"
DEFAULT_OUTPUT = NeuralThreePlayerBot.MODEL_FILE_V5
DEFAULT_RUN_ROOT = NeuralThreePlayerBot.MODEL_DIR / "self_train_runs_v5"
DEFAULT_ACTOR_POOL = "incumbent:3,greedy:2,1-trick-minmax:2,optimal-bot:1,random:1"
DEFAULT_TEACHER_POOL = "optimal-bot:5,1-trick-minmax:3,greedy:1"
DEFAULT_TRAIN_MATCHES = 360
DEFAULT_VALIDATION_MATCHES = 72
DEFAULT_ALPHA = 40
DEFAULT_PLAY_HIDDEN_DIM = 96
DEFAULT_AUCTION_HIDDEN_DIM = 64
DEFAULT_PLAY_VALUE_HIDDEN_DIM = 96
DEFAULT_AUCTION_VALUE_HIDDEN_DIM = 64
DEFAULT_PLAY_EPOCHS = 16
DEFAULT_AUCTION_EPOCHS = 16
DEFAULT_PLAY_VALUE_EPOCHS = 18
DEFAULT_AUCTION_VALUE_EPOCHS = 18
DEFAULT_PLAY_LEARNING_RATE = 0.010
DEFAULT_AUCTION_LEARNING_RATE = 0.012
DEFAULT_PLAY_VALUE_LEARNING_RATE = 0.008
DEFAULT_AUCTION_VALUE_LEARNING_RATE = 0.010
DEFAULT_L2 = 1e-4
DEFAULT_ACTOR_SAMPLE_SCALE = 0.5
DEFAULT_DISAGREEMENT_WEIGHT = 2.5
DEFAULT_MIN_VALIDATION_SCORE_GAIN = 0.0
DEFAULT_EVAL_GAMES = 72
DEFAULT_EVAL_GAMES_VS_OPTIMAL = 48
DEFAULT_PRECHECK_GAMES = 24


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def parse_actor_specs(actor_pool: str | None) -> list[TeacherSpec]:
    raw_spec = actor_pool.strip() if actor_pool is not None else ""
    if not raw_spec:
        raise ValueError("actor_pool cannot be empty")

    specs: list[TeacherSpec] = []
    for item in raw_spec.split(","):
        token = item.strip()
        if not token:
            continue
        bot_id, separator, raw_weight = token.partition(":")
        cleaned_bot_id = bot_id.strip()
        if not cleaned_bot_id:
            raise ValueError(f"invalid actor token: {token!r}")
        if cleaned_bot_id != INCUMBENT_ACTOR_ID:
            get_ready_bot_spec(cleaned_bot_id)
        if separator:
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise ValueError(f"invalid actor weight in {token!r}") from exc
        else:
            weight = 1.0
        if weight <= 0:
            raise ValueError(f"actor weights must be positive: {token!r}")
        specs.append(TeacherSpec(cleaned_bot_id, weight))

    if not specs:
        raise ValueError("actor_pool cannot be empty")
    return specs


def _format_spec_pool(specs: list[TeacherSpec]) -> str:
    return ", ".join(f"{spec.bot_id}x{spec.weight:g}" for spec in specs)


def _actor_bot(
    *,
    actor_bot_id: str,
    player_name: str,
    incumbent_bundle: dict,
    actor_provider: TrainingBotProvider,
):
    if actor_bot_id == INCUMBENT_ACTOR_ID:
        return NeuralThreePlayerBot(player_name, model_bundle=incumbent_bundle)
    return actor_provider.get_bot(bot_id=actor_bot_id, player_name=player_name)


def _build_actor_match_controller(
    *,
    actor_specs: list[TeacherSpec],
    incumbent_bundle: dict,
    seat_rng: random.Random,
    actor_provider: TrainingBotProvider,
) -> MatchController:
    sampled_bot_ids = {
        player_name: seat_rng.choices(
            [spec.bot_id for spec in actor_specs],
            weights=[spec.weight for spec in actor_specs],
            k=1,
        )[0]
        for player_name in THREE_PLAYER_NAMES
    }
    controller = MatchController.create(
        num_players=3,
        player_names=THREE_PLAYER_NAMES,
        teams=None,
        player_bot_ids={player_name: None for player_name in THREE_PLAYER_NAMES},
        bots={
            player_name: _actor_bot(
                actor_bot_id=actor_bot_id,
                player_name=player_name,
                incumbent_bundle=incumbent_bundle,
                actor_provider=actor_provider,
            )
            for player_name, actor_bot_id in sampled_bot_ids.items()
        },
        auto_run_bots=False,
    )
    controller.session.player_bot_ids = sampled_bot_ids
    return controller


def _policy_actions_for_decision(
    *,
    teacher_action: Card | AuctionEvent,
    teacher_candidate_actions: list[Card | AuctionEvent] | None,
    teacher_target_distribution: list[float] | None,
    legal_actions: list[Card | AuctionEvent],
) -> tuple[list[Card | AuctionEvent], list[float] | None]:
    if (
        teacher_candidate_actions is None
        or teacher_target_distribution is None
        or teacher_action not in teacher_candidate_actions
        or len(teacher_candidate_actions) != len(teacher_target_distribution)
    ):
        return list(legal_actions), None
    return list(teacher_candidate_actions), list(teacher_target_distribution)


def _collect_offline_policy_dataset_sequential(
    *,
    actor_specs: list[TeacherSpec],
    teacher_specs: list[TeacherSpec],
    incumbent_bundle: dict,
    match_count: int,
    alpha: int,
    seed: int,
    actor_sample_scale: float = DEFAULT_ACTOR_SAMPLE_SCALE,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    disagreement_weight: float = DEFAULT_DISAGREEMENT_WEIGHT,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    if match_count <= 0:
        raise ValueError("match_count must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if disagreement_weight <= 0:
        raise ValueError("disagreement_weight must be positive")

    dataset = TrainingDataset()
    actor_provider = TrainingBotProvider(teacher_sample_scale=actor_sample_scale)
    teacher_provider = TrainingBotProvider(teacher_sample_scale=teacher_sample_scale)
    teacher_oracle = TeacherOracle(
        teacher_specs,
        seed + 10_000,
        bot_provider=teacher_provider,
        teacher_target_temperature=teacher_target_temperature,
    )
    seat_rng = random.Random(seed)
    teacher_weight_map = {
        teacher_spec.bot_id: teacher_spec.weight
        for teacher_spec in teacher_specs
    }
    actor_counts: dict[str, int] = {}
    teacher_label_counts: dict[str, int] = {}
    forced_auction_actions = 0
    forced_play_actions = 0
    teacher_queries = 0
    policy_disagreements = 0
    soft_policy_examples = 0
    started_at = time.perf_counter()

    _log(
        verbose,
        (
            f"[v5:collect] start matches={match_count} alpha={alpha} seed={seed} "
            f"actors={_format_spec_pool(actor_specs)} teachers={_format_spec_pool(teacher_specs)}"
        ),
    )

    for match_index in range(match_count):
        random.seed(seed + match_index)
        controller = _build_actor_match_controller(
            actor_specs=actor_specs,
            incumbent_bundle=incumbent_bundle,
            seat_rng=seat_rng,
            actor_provider=actor_provider,
        )
        for actor_bot_id in controller.session.player_bot_ids.values():
            actor_counts[actor_bot_id] = actor_counts.get(actor_bot_id, 0) + 1

        round_start_match_scores = dict(controller.session.match_scores)
        pending_play_values: list[tuple[str, list[float], float]] = []
        pending_auction_values: list[tuple[str, list[float], float]] = []

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
                    if len(legal_actions) == 1:
                        forced_auction_actions += 1
                        environment_action = legal_actions[0]
                    else:
                        environment_action = bot.choose_auction_action(auction_state)
                        if environment_action not in legal_actions:
                            raise ValueError(
                                f"actor selected illegal auction action {environment_action}"
                            )
                        teacher_queries += 1
                        teacher_decision = teacher_oracle.choose_auction_action(
                            auction_state=auction_state,
                            player_name=player_name,
                            hand=acting_hand,
                            player_names=controller.session.player_names,
                            teams=controller.session.teams,
                            match_scores=controller.session.match_scores,
                            target_score=controller.session.target_score,
                        )
                        teacher_action = teacher_decision.action
                        if teacher_action not in legal_actions:
                            raise ValueError(
                                f"teacher selected illegal auction action {teacher_action}"
                            )
                        teacher_label_counts[teacher_decision.bot_id] = (
                            teacher_label_counts.get(teacher_decision.bot_id, 0) + 1
                        )
                        if teacher_action != environment_action:
                            policy_disagreements += 1
                        policy_actions, target_distribution = _policy_actions_for_decision(
                            teacher_action=teacher_action,
                            teacher_candidate_actions=teacher_decision.candidate_actions,
                            teacher_target_distribution=teacher_decision.target_distribution,
                            legal_actions=legal_actions,
                        )
                        if target_distribution is not None:
                            soft_policy_examples += 1
                        example_weight = _policy_example_weight(
                            actor_mode="student",
                            teacher_weight=teacher_weight_map.get(teacher_decision.bot_id, 1.0),
                            teacher_matches_environment=(teacher_action == environment_action),
                        )
                        if teacher_action != environment_action:
                            example_weight *= disagreement_weight / DEFAULT_DISAGREEMENT_WEIGHT
                        dataset.auction_policy_examples.append(
                            ChoiceExample(
                                candidate_features=[
                                    encode_auction_candidate(
                                        auction_state=auction_state,
                                        acting_player_name=player_name,
                                        hand=acting_hand,
                                        candidate_action=policy_action,
                                        match_scores=controller.session.match_scores,
                                        target_score=controller.session.target_score,
                                    )
                                    for policy_action in policy_actions
                                ],
                                chosen_index=policy_actions.index(teacher_action),
                                weight=example_weight,
                                target_distribution=target_distribution,
                            )
                        )

                    successor_auction = deepcopy(auction_state)
                    apply_auction_action_for_search(successor_auction, environment_action)
                    pending_auction_values.append(
                        (
                            player_name,
                            encode_auction_state(
                                auction_state=successor_auction,
                                perspective_player_name=player_name,
                                hand=acting_hand,
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                            ),
                            1.0,
                        )
                    )
                    controller.session.auction.apply_event(environment_action)
                    controller._advance_or_finalize_auction()
                    continue

                round_state = controller.session.game.round_state
                legal_cards = ordered_legal_cards(round_state)
                if len(legal_cards) == 1:
                    forced_play_actions += 1
                    environment_card = legal_cards[0]
                else:
                    environment_card = bot.choose_card(round_state)
                    if environment_card not in legal_cards:
                        raise ValueError(f"actor selected illegal card {environment_card.code}")
                    teacher_queries += 1
                    teacher_decision = teacher_oracle.choose_card(
                        round_state=round_state,
                        player_name=player_name,
                        auction_state=controller.session.auction.state,
                        player_names=controller.session.player_names,
                        teams=controller.session.teams,
                        match_scores=controller.session.match_scores,
                        target_score=controller.session.target_score,
                    )
                    teacher_card = teacher_decision.action
                    if teacher_card not in legal_cards:
                        raise ValueError(f"teacher selected illegal card {teacher_card.code}")
                    teacher_label_counts[teacher_decision.bot_id] = (
                        teacher_label_counts.get(teacher_decision.bot_id, 0) + 1
                    )
                    if teacher_card != environment_card:
                        policy_disagreements += 1
                    policy_cards, target_distribution = _policy_actions_for_decision(
                        teacher_action=teacher_card,
                        teacher_candidate_actions=teacher_decision.candidate_actions,
                        teacher_target_distribution=teacher_decision.target_distribution,
                        legal_actions=legal_cards,
                    )
                    if target_distribution is not None:
                        soft_policy_examples += 1
                    example_weight = _policy_example_weight(
                        actor_mode="student",
                        teacher_weight=teacher_weight_map.get(teacher_decision.bot_id, 1.0),
                        teacher_matches_environment=(teacher_card == environment_card),
                    )
                    if teacher_card != environment_card:
                        example_weight *= disagreement_weight / DEFAULT_DISAGREEMENT_WEIGHT
                    dataset.play_policy_examples.append(
                        ChoiceExample(
                            candidate_features=[
                                encode_play_candidate(
                                    round_state=round_state,
                                    acting_player_name=player_name,
                                    candidate_card=policy_card,
                                    match_scores=controller.session.match_scores,
                                    target_score=controller.session.target_score,
                                    auction_state=controller.session.auction.state,
                                )
                                for policy_card in policy_cards
                            ],
                            chosen_index=policy_cards.index(teacher_card),
                            weight=example_weight,
                            target_distribution=target_distribution,
                        )
                    )

                successor_round = apply_trick_action_to_state(
                    round_state,
                    Play(round_state.current_player, environment_card),
                )
                if not successor_round.is_terminal:
                    pending_play_values.append(
                        (
                            player_name,
                            encode_play_state(
                                round_state=successor_round,
                                perspective_player_name=player_name,
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                                auction_state=controller.session.auction.state,
                            ),
                            1.0,
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
                    RegressionExample(
                        features=features,
                        target=round_targets[player_name],
                        weight=weight,
                    )
                    for player_name, features, weight in pending_play_values
                )
                dataset.auction_value_examples.extend(
                    RegressionExample(
                        features=features,
                        target=round_targets[player_name],
                        weight=weight,
                    )
                    for player_name, features, weight in pending_auction_values
                )
                pending_play_values = []
                pending_auction_values = []

            if controller.session.is_match_complete:
                break

            if round_index < alpha:
                controller.next_round(auto_run_bots=False)
                round_start_match_scores = dict(controller.session.match_scores)

    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "matches": match_count,
        "alpha": alpha,
        "actor_counts": actor_counts,
        "teacher_label_counts": teacher_label_counts,
        "play_policy_examples": len(dataset.play_policy_examples),
        "auction_policy_examples": len(dataset.auction_policy_examples),
        "play_value_examples": len(dataset.play_value_examples),
        "auction_value_examples": len(dataset.auction_value_examples),
        "forced_auction_actions": forced_auction_actions,
        "forced_play_actions": forced_play_actions,
        "teacher_queries": teacher_queries,
        "policy_disagreements": policy_disagreements,
        "soft_policy_examples": soft_policy_examples,
        "elapsed_seconds": elapsed_seconds,
    }
    _log(
        verbose,
        (
            f"[v5:collect] done {_format_dataset_counts(dataset)} "
            f"elapsed={_format_seconds(elapsed_seconds)} "
            f"forced_auction={forced_auction_actions} forced_play={forced_play_actions} "
            f"teacher_queries={teacher_queries} disagreements={policy_disagreements} "
            f"soft={soft_policy_examples}"
        ),
    )
    return dataset, report


def _collect_offline_policy_dataset_worker(task: dict) -> tuple[TrainingDataset, dict]:
    return _collect_offline_policy_dataset_sequential(**task)


def _merge_collection_reports(
    *,
    match_count: int,
    alpha: int,
    worker_count: int,
    aggregate_dataset: TrainingDataset,
    shard_reports: list[dict],
    elapsed_seconds: float,
) -> dict:
    actor_counts: dict[str, int] = {}
    teacher_label_counts: dict[str, int] = {}
    forced_auction_actions = 0
    forced_play_actions = 0
    teacher_queries = 0
    policy_disagreements = 0
    soft_policy_examples = 0
    worker_elapsed_seconds = 0.0
    for shard_report in shard_reports:
        _merge_count_dict(actor_counts, shard_report["actor_counts"])
        _merge_count_dict(teacher_label_counts, shard_report["teacher_label_counts"])
        forced_auction_actions += shard_report["forced_auction_actions"]
        forced_play_actions += shard_report["forced_play_actions"]
        teacher_queries += shard_report["teacher_queries"]
        policy_disagreements += shard_report["policy_disagreements"]
        soft_policy_examples += shard_report["soft_policy_examples"]
        worker_elapsed_seconds += shard_report["elapsed_seconds"]

    return {
        "matches": match_count,
        "alpha": alpha,
        "actor_counts": actor_counts,
        "teacher_label_counts": teacher_label_counts,
        "play_policy_examples": len(aggregate_dataset.play_policy_examples),
        "auction_policy_examples": len(aggregate_dataset.auction_policy_examples),
        "play_value_examples": len(aggregate_dataset.play_value_examples),
        "auction_value_examples": len(aggregate_dataset.auction_value_examples),
        "forced_auction_actions": forced_auction_actions,
        "forced_play_actions": forced_play_actions,
        "teacher_queries": teacher_queries,
        "policy_disagreements": policy_disagreements,
        "soft_policy_examples": soft_policy_examples,
        "elapsed_seconds": elapsed_seconds,
        "worker_elapsed_seconds": worker_elapsed_seconds,
        "workers": worker_count,
    }


def collect_offline_policy_dataset(
    *,
    actor_specs: list[TeacherSpec],
    teacher_specs: list[TeacherSpec],
    incumbent_bundle: dict,
    match_count: int,
    alpha: int,
    seed: int,
    actor_sample_scale: float = DEFAULT_ACTOR_SAMPLE_SCALE,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    disagreement_weight: float = DEFAULT_DISAGREEMENT_WEIGHT,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    if workers <= 1 or match_count <= 1:
        return _collect_offline_policy_dataset_sequential(
            actor_specs=actor_specs,
            teacher_specs=teacher_specs,
            incumbent_bundle=incumbent_bundle,
            match_count=match_count,
            alpha=alpha,
            seed=seed,
            actor_sample_scale=actor_sample_scale,
            teacher_sample_scale=teacher_sample_scale,
            teacher_target_temperature=teacher_target_temperature,
            disagreement_weight=disagreement_weight,
            verbose=verbose,
        )

    resolved_workers = _resolve_worker_count(workers=workers, work_items=match_count)
    aggregate_dataset = TrainingDataset()
    shard_reports: list[dict] = []
    started_at = time.perf_counter()
    _log(
        verbose,
        (
            f"[v5:collect] start matches={match_count} alpha={alpha} seed={seed} "
            f"workers={resolved_workers} actors={_format_spec_pool(actor_specs)} "
            f"teachers={_format_spec_pool(teacher_specs)}"
        ),
    )
    tasks = [
        {
            "actor_specs": actor_specs,
            "teacher_specs": teacher_specs,
            "incumbent_bundle": incumbent_bundle,
            "match_count": 1,
            "alpha": alpha,
            "seed": seed + (match_index * 1_000),
            "actor_sample_scale": actor_sample_scale,
            "teacher_sample_scale": teacher_sample_scale,
            "teacher_target_temperature": teacher_target_temperature,
            "disagreement_weight": disagreement_weight,
            "verbose": False,
        }
        for match_index in range(match_count)
    ]

    completed_matches = 0
    executor, executor_kind = _create_parallel_executor(max_workers=resolved_workers)
    if executor_kind != "process":
        _log(verbose, f"[v5:collect] falling back to {executor_kind} workers")
    with executor:
        futures = [executor.submit(_collect_offline_policy_dataset_worker, task) for task in tasks]
        for future in as_completed(futures):
            shard_dataset, shard_report = future.result()
            aggregate_dataset.extend(shard_dataset)
            shard_reports.append(shard_report)
            completed_matches += shard_report["matches"]
            if verbose and (completed_matches == match_count or completed_matches % resolved_workers == 0):
                _log(
                    verbose,
                    (
                        f"[v5:collect] {completed_matches}/{match_count} "
                        f"{_format_dataset_counts(aggregate_dataset)}"
                    ),
                )

    elapsed_seconds = time.perf_counter() - started_at
    report = _merge_collection_reports(
        match_count=match_count,
        alpha=alpha,
        worker_count=resolved_workers,
        aggregate_dataset=aggregate_dataset,
        shard_reports=shard_reports,
        elapsed_seconds=elapsed_seconds,
    )
    _log(
        verbose,
        (
            f"[v5:collect] done {_format_dataset_counts(aggregate_dataset)} "
            f"elapsed={_format_seconds(elapsed_seconds)} workers={resolved_workers} "
            f"teacher_queries={report['teacher_queries']} "
            f"disagreements={report['policy_disagreements']} "
            f"soft={report['soft_policy_examples']}"
        ),
    )
    return aggregate_dataset, report


def _choice_metrics(model: ScalarMLP, examples: list[ChoiceExample]) -> dict:
    if not examples:
        return {"loss": None, "accuracy": None, "examples": 0}
    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    for example in examples:
        target_distribution = ScalarMLP._normalized_target_distribution(example)
        target_index = max(
            range(len(target_distribution)),
            key=lambda index: target_distribution[index],
        )
        scores = model.score_many(example.candidate_features)
        predicted_index = max(range(len(scores)), key=lambda index: scores[index])
        if predicted_index == target_index:
            correct += 1
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        denom = sum(exp_scores)
        probabilities = [value / denom for value in exp_scores]
        total_loss += example.weight * -sum(
            target_probability * math.log(max(probability, 1e-12))
            for probability, target_probability in zip(probabilities, target_distribution)
            if target_probability > 0.0
        )
        total_weight += example.weight
    return {
        "loss": total_loss / max(total_weight, 1e-12),
        "accuracy": correct / len(examples),
        "examples": len(examples),
    }


def _regression_metrics(model: ScalarMLP, examples: list[RegressionExample]) -> dict:
    if not examples:
        return {"mse": None, "examples": 0}
    total_weighted_error = 0.0
    total_weight = 0.0
    for example in examples:
        error = model.score(example.features) - example.target
        total_weighted_error += example.weight * error * error
        total_weight += example.weight
    return {
        "mse": total_weighted_error / max(total_weight, 1e-12),
        "examples": len(examples),
    }


def evaluate_bundle_on_dataset(bundle: dict, dataset: TrainingDataset) -> dict:
    play_policy = _choice_metrics(
        ScalarMLP.from_dict(bundle["play_model"]),
        dataset.play_policy_examples,
    )
    auction_policy = _choice_metrics(
        ScalarMLP.from_dict(bundle["auction_model"]),
        dataset.auction_policy_examples,
    )
    play_value = _regression_metrics(
        ScalarMLP.from_dict(bundle["play_value_model"]),
        dataset.play_value_examples,
    )
    auction_value = _regression_metrics(
        ScalarMLP.from_dict(bundle["auction_value_model"]),
        dataset.auction_value_examples,
    )
    play_accuracy = 0.0 if play_policy["accuracy"] is None else play_policy["accuracy"]
    auction_accuracy = 0.0 if auction_policy["accuracy"] is None else auction_policy["accuracy"]
    play_value_mse = 1.0 if play_value["mse"] is None else play_value["mse"]
    auction_value_mse = 1.0 if auction_value["mse"] is None else auction_value["mse"]
    validation_score = (
        (0.55 * play_accuracy)
        + (0.35 * auction_accuracy)
        - (0.05 * play_value_mse)
        - (0.05 * auction_value_mse)
    )
    return {
        "play_policy": play_policy,
        "auction_policy": auction_policy,
        "play_value": play_value,
        "auction_value": auction_value,
        "score": validation_score,
    }


def _format_validation_metrics(metrics: dict) -> str:
    def _format_optional(value: float | None, precision: int) -> str:
        if value is None:
            return "-"
        return f"{value:.{precision}f}"

    return (
        f"score={metrics['score']:.4f} "
        f"play_acc={_format_optional(metrics['play_policy']['accuracy'], 3)} "
        f"auction_acc={_format_optional(metrics['auction_policy']['accuracy'], 3)} "
        f"play_v_mse={_format_optional(metrics['play_value']['mse'], 4)} "
        f"auction_v_mse={_format_optional(metrics['auction_value']['mse'], 4)}"
    )


def _resolve_default_initial_model_path() -> Path:
    if NeuralThreePlayerBot.MODEL_FILE_V4.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V4
    if NeuralThreePlayerBot.MODEL_FILE_V3.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V3
    if NeuralThreePlayerBot.MODEL_FILE_V2.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V2
    return NeuralThreePlayerBot.MODEL_FILE_V1


def train_offline_policy_improvement(
    *,
    initial_bundle: dict,
    actor_specs: list[TeacherSpec],
    teacher_specs: list[TeacherSpec],
    train_matches: int,
    validation_matches: int,
    alpha: int,
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
    warm_start: bool,
    actor_sample_scale: float = DEFAULT_ACTOR_SAMPLE_SCALE,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    disagreement_weight: float = DEFAULT_DISAGREEMENT_WEIGHT,
    play_value_weight: float = DEFAULT_PLAY_VALUE_WEIGHT,
    auction_value_weight: float = DEFAULT_AUCTION_VALUE_WEIGHT,
    play_rollout_depth: int = DEFAULT_PLAY_ROLLOUT_DEPTH,
    auction_rollout_depth: int = DEFAULT_AUCTION_ROLLOUT_DEPTH,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
    trainer_backend: str = DEFAULT_TRAINING_BACKEND,
    trainer_batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
    min_validation_score_gain: float = DEFAULT_MIN_VALIDATION_SCORE_GAIN,
    eval_games: int = DEFAULT_EVAL_GAMES,
    eval_games_vs_optimal: int = DEFAULT_EVAL_GAMES_VS_OPTIMAL,
    precheck_games: int = DEFAULT_PRECHECK_GAMES,
    promotion_head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    promotion_greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    promotion_optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
    run_game_eval: bool = True,
    promote_on_validation_only: bool = False,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> tuple[dict, dict]:
    started_at = time.perf_counter()
    _log(
        verbose,
        _render_compact_block(
            prefix="[v5]",
            title="collect train corpus",
            rows=[
                ("matches", str(train_matches)),
                ("alpha", str(alpha)),
                ("actors", _format_spec_pool(actor_specs)),
                ("teachers", _format_spec_pool(teacher_specs)),
            ],
        ),
    )
    train_dataset, train_collection_report = collect_offline_policy_dataset(
        actor_specs=actor_specs,
        teacher_specs=teacher_specs,
        incumbent_bundle=initial_bundle,
        match_count=train_matches,
        alpha=alpha,
        seed=seed,
        actor_sample_scale=actor_sample_scale,
        teacher_sample_scale=teacher_sample_scale,
        teacher_target_temperature=teacher_target_temperature,
        disagreement_weight=disagreement_weight,
        workers=workers,
        verbose=verbose,
    )

    _log(
        verbose,
        _render_compact_block(
            prefix="[v5]",
            title="collect validation corpus",
            rows=[
                ("matches", str(validation_matches)),
                ("alpha", str(alpha)),
            ],
        ),
    )
    validation_dataset, validation_collection_report = collect_offline_policy_dataset(
        actor_specs=actor_specs,
        teacher_specs=teacher_specs,
        incumbent_bundle=initial_bundle,
        match_count=validation_matches,
        alpha=alpha,
        seed=seed + 1_000_000,
        actor_sample_scale=actor_sample_scale,
        teacher_sample_scale=teacher_sample_scale,
        teacher_target_temperature=teacher_target_temperature,
        disagreement_weight=disagreement_weight,
        workers=workers,
        verbose=verbose,
    )

    initial_validation = evaluate_bundle_on_dataset(initial_bundle, validation_dataset)
    _log(verbose, f"[v5] initial validation {_format_validation_metrics(initial_validation)}")

    candidate_bundle, training_report = train_neural_3p_bundle(
        play_examples=train_dataset.play_policy_examples,
        auction_examples=train_dataset.auction_policy_examples,
        play_value_examples=train_dataset.play_value_examples,
        auction_value_examples=train_dataset.auction_value_examples,
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
        seed=seed + 2_000_000,
        teacher_specs=teacher_specs,
        initial_bundle=initial_bundle if warm_start else None,
        play_value_weight=play_value_weight,
        auction_value_weight=auction_value_weight,
        play_rollout_depth=play_rollout_depth,
        auction_rollout_depth=auction_rollout_depth,
        gradient_clip=gradient_clip,
        trainer_backend=trainer_backend,
        trainer_batch_size=trainer_batch_size,
        bot_id=V5_BOT_ID,
        bundle_version=3,
        extra_metadata={
            "training_mode": "offline_policy_improvement",
            "seed_bot_id": initial_bundle.get("bot_id", "unknown"),
            "actor_pool": _serialize_teacher_specs(actor_specs),
            "teacher_pool": _serialize_teacher_specs(teacher_specs),
            "teacher_target_temperature": teacher_target_temperature,
        },
        workers=workers,
        verbose=verbose,
    )
    candidate_validation = evaluate_bundle_on_dataset(candidate_bundle, validation_dataset)
    validation_score_gain = candidate_validation["score"] - initial_validation["score"]
    validation_passed = validation_score_gain >= min_validation_score_gain
    _log(verbose, f"[v5] train {_format_training_metrics(training_report)}")
    _log(verbose, f"[v5] candidate validation {_format_validation_metrics(candidate_validation)}")
    _log(
        verbose,
        (
            f"[v5] validation gate "
            f"{'passed' if validation_passed else 'failed'} "
            f"gain={validation_score_gain:+.4f} required={min_validation_score_gain:+.4f}"
        ),
    )

    game_evaluation_report = None
    game_eval_allowed = validation_passed and run_game_eval
    if game_eval_allowed:
        game_evaluation_report = evaluate_candidate(
            candidate_bundle=candidate_bundle,
            incumbent_bundle=initial_bundle,
            alpha=alpha,
            eval_games=eval_games,
            eval_games_vs_optimal=eval_games_vs_optimal,
            precheck_games=min(precheck_games, eval_games),
            seed=seed + 3_000_000,
            incumbent_baselines=None,
            promotion_head_to_head_margin=promotion_head_to_head_margin,
            promotion_greedy_regression_tolerance=promotion_greedy_regression_tolerance,
            promotion_optimal_regression_tolerance=promotion_optimal_regression_tolerance,
            workers=workers,
            verbose=verbose,
        )
        _log(
            verbose,
            _render_promotion_outcome_block(
                prefix="[v5]",
                title="game evaluation",
                evaluation_report=game_evaluation_report,
                head_to_head_margin=promotion_head_to_head_margin,
                greedy_regression_tolerance=promotion_greedy_regression_tolerance,
                optimal_regression_tolerance=promotion_optimal_regression_tolerance,
            ),
        )
    elif run_game_eval:
        _log(verbose, "[v5] skipped game evaluation because validation gate failed")

    game_evaluation_accepted = (
        game_evaluation_report.get("accepted", False)
        if game_evaluation_report is not None
        else False
    )
    promoted = validation_passed and (
        promote_on_validation_only
        or (run_game_eval and game_evaluation_accepted)
    )
    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "accepted": promoted,
        "validation_passed": validation_passed,
        "validation_score_gain": validation_score_gain,
        "train_collection": train_collection_report,
        "validation_collection": validation_collection_report,
        "training": training_report,
        "initial_validation": initial_validation,
        "candidate_validation": candidate_validation,
        "game_evaluation": game_evaluation_report,
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "actor_pool": _serialize_teacher_specs(actor_specs),
            "teacher_pool": _serialize_teacher_specs(teacher_specs),
            "train_matches": train_matches,
            "validation_matches": validation_matches,
            "alpha": alpha,
            "seed": seed,
            "warm_start": warm_start,
            "min_validation_score_gain": min_validation_score_gain,
            "run_game_eval": run_game_eval,
            "promote_on_validation_only": promote_on_validation_only,
        },
    }
    return candidate_bundle, report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train neural-3p-v5 with offline policy improvement: fixed actor "
            "rollouts, teacher-scored legal actions, heldout validation, then optional game eval."
        )
    )
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--actor-pool", default=DEFAULT_ACTOR_POOL)
    parser.add_argument("--teacher-pool", default=DEFAULT_TEACHER_POOL)
    parser.add_argument("--train-matches", type=int, default=DEFAULT_TRAIN_MATCHES)
    parser.add_argument("--validation-matches", type=int, default=DEFAULT_VALIDATION_MATCHES)
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--play-hidden-dim", type=int, default=DEFAULT_PLAY_HIDDEN_DIM)
    parser.add_argument("--auction-hidden-dim", type=int, default=DEFAULT_AUCTION_HIDDEN_DIM)
    parser.add_argument("--play-value-hidden-dim", type=int, default=DEFAULT_PLAY_VALUE_HIDDEN_DIM)
    parser.add_argument("--auction-value-hidden-dim", type=int, default=DEFAULT_AUCTION_VALUE_HIDDEN_DIM)
    parser.add_argument("--play-epochs", type=int, default=DEFAULT_PLAY_EPOCHS)
    parser.add_argument("--auction-epochs", type=int, default=DEFAULT_AUCTION_EPOCHS)
    parser.add_argument("--play-value-epochs", type=int, default=DEFAULT_PLAY_VALUE_EPOCHS)
    parser.add_argument("--auction-value-epochs", type=int, default=DEFAULT_AUCTION_VALUE_EPOCHS)
    parser.add_argument("--play-learning-rate", type=float, default=DEFAULT_PLAY_LEARNING_RATE)
    parser.add_argument("--auction-learning-rate", type=float, default=DEFAULT_AUCTION_LEARNING_RATE)
    parser.add_argument("--play-value-learning-rate", type=float, default=DEFAULT_PLAY_VALUE_LEARNING_RATE)
    parser.add_argument("--auction-value-learning-rate", type=float, default=DEFAULT_AUCTION_VALUE_LEARNING_RATE)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--actor-sample-scale", type=float, default=DEFAULT_ACTOR_SAMPLE_SCALE)
    parser.add_argument("--teacher-sample-scale", type=float, default=DEFAULT_TEACHER_SAMPLE_SCALE)
    parser.add_argument("--teacher-target-temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--disagreement-weight", type=float, default=DEFAULT_DISAGREEMENT_WEIGHT)
    parser.add_argument("--play-value-weight", type=float, default=DEFAULT_PLAY_VALUE_WEIGHT)
    parser.add_argument("--auction-value-weight", type=float, default=DEFAULT_AUCTION_VALUE_WEIGHT)
    parser.add_argument("--play-rollout-depth", type=int, default=DEFAULT_PLAY_ROLLOUT_DEPTH)
    parser.add_argument("--auction-rollout-depth", type=int, default=DEFAULT_AUCTION_ROLLOUT_DEPTH)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument(
        "--trainer-backend",
        default=DEFAULT_TRAINING_BACKEND,
        choices=TRAINING_BACKEND_CHOICES,
    )
    parser.add_argument("--trainer-batch-size", type=int, default=DEFAULT_TRAINER_BATCH_SIZE)
    parser.add_argument("--min-validation-score-gain", type=float, default=DEFAULT_MIN_VALIDATION_SCORE_GAIN)
    parser.add_argument("--skip-game-eval", action="store_true")
    parser.add_argument(
        "--promote-on-validation-only",
        action="store_true",
        help="write --promote-to after validation passes even if game eval is skipped or fails",
    )
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES)
    parser.add_argument("--eval-games-vs-optimal", type=int, default=DEFAULT_EVAL_GAMES_VS_OPTIMAL)
    parser.add_argument("--precheck-games", type=int, default=DEFAULT_PRECHECK_GAMES)
    parser.add_argument("--promotion-head-to-head-margin", type=float, default=DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN)
    parser.add_argument(
        "--promotion-greedy-regression-tolerance",
        type=float,
        default=DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    )
    parser.add_argument(
        "--promotion-optimal-regression-tolerance",
        type=float,
        default=DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    initial_model_path = args.initial_model or _resolve_default_initial_model_path()
    initial_bundle = load_model_bundle(initial_model_path)
    actor_specs = parse_actor_specs(args.actor_pool)
    teacher_specs = parse_teacher_specs(args.teacher_pool)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / f"neural-3p-v5-train-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "candidate.json"
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "manifest.json"

    manifest = {
        "started_at": timestamp,
        "initial_model_path": str(initial_model_path),
        "promote_to": str(args.promote_to),
        "actor_pool": args.actor_pool,
        "teacher_pool": args.teacher_pool,
        "train_matches": args.train_matches,
        "validation_matches": args.validation_matches,
        "alpha": args.alpha,
        "seed": args.seed,
        "workers": args.workers,
        "trainer_backend": args.trainer_backend,
        "trainer_batch_size": args.trainer_batch_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(
        verbose,
        _render_compact_block(
            prefix="[v5]",
            title="run",
            rows=[
                ("run dir", str(run_dir)),
                ("initial", str(initial_model_path)),
                ("actors", args.actor_pool),
                ("teachers", args.teacher_pool),
                (
                    "budget",
                    (
                        f"train {args.train_matches} | validation {args.validation_matches} | "
                        f"alpha {args.alpha} | workers {args.workers}"
                    ),
                ),
                (
                    "model",
                    (
                        f"hidden {args.play_hidden_dim}/{args.auction_hidden_dim}/"
                        f"{args.play_value_hidden_dim}/{args.auction_value_hidden_dim} | "
                        f"{'warm' if args.warm_start else 'fresh'}"
                    ),
                ),
                (
                    "eval",
                    (
                        "skipped"
                        if args.skip_game_eval
                        else f"games {args.eval_games} | optimal {args.eval_games_vs_optimal}"
                    ),
                ),
            ],
        ),
    )

    candidate_bundle, report = train_offline_policy_improvement(
        initial_bundle=initial_bundle,
        actor_specs=actor_specs,
        teacher_specs=teacher_specs,
        train_matches=args.train_matches,
        validation_matches=args.validation_matches,
        alpha=args.alpha,
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
        warm_start=args.warm_start,
        actor_sample_scale=args.actor_sample_scale,
        teacher_sample_scale=args.teacher_sample_scale,
        teacher_target_temperature=args.teacher_target_temperature,
        disagreement_weight=args.disagreement_weight,
        play_value_weight=args.play_value_weight,
        auction_value_weight=args.auction_value_weight,
        play_rollout_depth=args.play_rollout_depth,
        auction_rollout_depth=args.auction_rollout_depth,
        gradient_clip=args.gradient_clip,
        trainer_backend=args.trainer_backend,
        trainer_batch_size=args.trainer_batch_size,
        min_validation_score_gain=args.min_validation_score_gain,
        eval_games=args.eval_games,
        eval_games_vs_optimal=args.eval_games_vs_optimal,
        precheck_games=args.precheck_games,
        promotion_head_to_head_margin=args.promotion_head_to_head_margin,
        promotion_greedy_regression_tolerance=args.promotion_greedy_regression_tolerance,
        promotion_optimal_regression_tolerance=args.promotion_optimal_regression_tolerance,
        run_game_eval=not args.skip_game_eval,
        promote_on_validation_only=args.promote_on_validation_only,
        workers=args.workers,
        verbose=verbose,
    )
    save_model_bundle(candidate_bundle, candidate_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["accepted"]:
        save_model_bundle(candidate_bundle, args.promote_to)
        _log(verbose, f"[v5] wrote promoted model to {args.promote_to}")
    else:
        _log(verbose, f"[v5] kept candidate at {candidate_path}")
    _log(verbose, f"[v5] wrote report to {report_path}")


if __name__ == "__main__":
    main()
