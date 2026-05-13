from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot
    from .self_play_neural_3p_v3 import (
        DEFAULT_AUCTION_TEMPERATURE,
        DEFAULT_EPSILON,
        DEFAULT_EPSILON_DECAY,
        DEFAULT_PERSISTED_REPLAY_LIMIT,
        DEFAULT_PLAY_TEMPERATURE,
        DEFAULT_POLICY_ADVANTAGE_SCALE,
        DEFAULT_POLICY_ADVANTAGE_THRESHOLD,
        DEFAULT_REPLAY_WINDOW,
        DEFAULT_SELF_PLAY_MATCHES,
        DEFAULT_TEMPERATURE_DECAY,
        DEFAULT_VALUE_WEIGHT_SCALE,
        DEFAULT_WINNER_POLICY_WEIGHT,
        SELF_PLAY_OUTER_METRICS,
        SelfPlayDatasetHistory,
        _build_self_play_outer_snapshot,
        load_persisted_replay_history,
        prune_replay_shards,
        reset_replay_state_after_promotion,
        save_replay_shard,
        train_with_self_play,
    )
    from .self_train_neural_3p_v2 import (
        DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
        DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
        DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
        _build_incumbent_baseline_cache_from_evaluation,
        _format_seconds,
        evaluate_candidate,
    )
    from .train_neural_3p_bot import (
        ComparisonMetric,
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_WEIGHT,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TEACHER_POOL,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_WARM_START_EPOCH_SCALE,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WORKERS,
        TRAINING_BACKEND_CHOICES,
        _render_compact_block,
        _render_iteration_comparison_table,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot
    from self_play_neural_3p_v3 import (
        DEFAULT_AUCTION_TEMPERATURE,
        DEFAULT_EPSILON,
        DEFAULT_EPSILON_DECAY,
        DEFAULT_PERSISTED_REPLAY_LIMIT,
        DEFAULT_PLAY_TEMPERATURE,
        DEFAULT_POLICY_ADVANTAGE_SCALE,
        DEFAULT_POLICY_ADVANTAGE_THRESHOLD,
        DEFAULT_REPLAY_WINDOW,
        DEFAULT_SELF_PLAY_MATCHES,
        DEFAULT_TEMPERATURE_DECAY,
        DEFAULT_VALUE_WEIGHT_SCALE,
        DEFAULT_WINNER_POLICY_WEIGHT,
        SELF_PLAY_OUTER_METRICS,
        SelfPlayDatasetHistory,
        _build_self_play_outer_snapshot,
        load_persisted_replay_history,
        prune_replay_shards,
        reset_replay_state_after_promotion,
        save_replay_shard,
        train_with_self_play,
    )
    from self_train_neural_3p_v2 import (
        DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
        DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
        DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
        _build_incumbent_baseline_cache_from_evaluation,
        _format_seconds,
        evaluate_candidate,
    )
    from train_neural_3p_bot import (
        ComparisonMetric,
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_WEIGHT,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TEACHER_POOL,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_WARM_START_EPOCH_SCALE,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WORKERS,
        TRAINING_BACKEND_CHOICES,
        _render_compact_block,
        _render_iteration_comparison_table,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )


V4_BOT_ID = "neural-3p-v4"
DEFAULT_OUTPUT = NeuralThreePlayerBot.MODEL_FILE_V4
DEFAULT_RUN_ROOT = NeuralThreePlayerBot.MODEL_DIR / "self_train_runs_v4"
DEFAULT_REPLAY_STORE_DIR = NeuralThreePlayerBot.MODEL_DIR / "self_play_replay_v4"
ALTERNATING_OUTER_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric(
        key="phase",
        label="Phase",
        kind="text",
    ),
    *SELF_PLAY_OUTER_METRICS,
)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _normalize_phase_name(phase: str) -> str:
    normalized = phase.strip().lower().replace("-", "_")
    if normalized not in {"imitation", "self_play"}:
        raise ValueError("phase must be one of: imitation, self-play")
    return normalized


def _display_phase_name(phase: str) -> str:
    return "self-play" if phase == "self_play" else phase


def _phase_for_iteration(*, start_phase: str, iteration_index: int) -> str:
    if iteration_index < 0:
        raise ValueError("iteration_index must be non-negative")
    if iteration_index % 2 == 0:
        return start_phase
    return "self_play" if start_phase == "imitation" else "imitation"


def resolve_initial_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if NeuralThreePlayerBot.MODEL_FILE_V4.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V4
    if NeuralThreePlayerBot.MODEL_FILE_V3.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V3
    if NeuralThreePlayerBot.MODEL_FILE_V2.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V2
    return NeuralThreePlayerBot.MODEL_FILE_V1


def train_with_alternating_phases(
    *,
    initial_bundle: dict,
    teacher_specs: list,
    alpha: int,
    bootstrap_matches: int,
    dagger_matches: int,
    dagger_iterations: int,
    self_play_matches: int,
    replay_window: int,
    persisted_replay_limit: int,
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
    play_value_weight: float = DEFAULT_PLAY_VALUE_WEIGHT,
    auction_value_weight: float = DEFAULT_AUCTION_VALUE_WEIGHT,
    play_rollout_depth: int = DEFAULT_PLAY_ROLLOUT_DEPTH,
    auction_rollout_depth: int = DEFAULT_AUCTION_ROLLOUT_DEPTH,
    warm_start_lr_scale: float = DEFAULT_WARM_START_LR_SCALE,
    warm_start_epoch_scale: float = DEFAULT_WARM_START_EPOCH_SCALE,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
    student_agreement_keep_prob: float = DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    play_temperature: float = DEFAULT_PLAY_TEMPERATURE,
    auction_temperature: float = DEFAULT_AUCTION_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
    temperature_decay: float = DEFAULT_TEMPERATURE_DECAY,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    policy_advantage_threshold: float = DEFAULT_POLICY_ADVANTAGE_THRESHOLD,
    policy_advantage_scale: float = DEFAULT_POLICY_ADVANTAGE_SCALE,
    value_weight_scale: float = DEFAULT_VALUE_WEIGHT_SCALE,
    winner_policy_weight: float = DEFAULT_WINNER_POLICY_WEIGHT,
    eval_games: int = 36,
    eval_games_vs_optimal: int = 18,
    precheck_games: int = 12,
    promotion_head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    promotion_greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    promotion_optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
    max_iterations: int | None = None,
    deadline: float | None = None,
    run_started_at: float | None = None,
    start_phase: str = "imitation",
    replay_history: SelfPlayDatasetHistory | None = None,
    checkpoint_dir: Path | None = None,
    best_path: Path | None = None,
    report_path: Path | None = None,
    replay_store_dir: Path | None = None,
    run_id: str | None = None,
    trainer_backend: str = DEFAULT_TRAINING_BACKEND,
    trainer_batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> tuple[dict, dict]:
    normalized_start_phase = _normalize_phase_name(start_phase)
    started_at = run_started_at if run_started_at is not None else time.time()
    current_best_bundle = initial_bundle
    current_replay_history = replay_history or SelfPlayDatasetHistory()
    incumbent_baseline_cache: dict | None = None
    previous_outer_snapshot: dict[str, object] | None = None
    self_play_phase_count = 0
    iteration_index = 1
    iteration_reports: list[dict] = []
    resolved_run_id = run_id or "neural-3p-v4"

    while True:
        if max_iterations is not None and iteration_index > max_iterations:
            break
        if deadline is not None and time.time() >= deadline and iteration_index > 1:
            break

        phase = _phase_for_iteration(
            start_phase=normalized_start_phase,
            iteration_index=iteration_index - 1,
        )
        phase_label = _display_phase_name(phase)
        iteration_seed = seed + (iteration_index * 10_000)
        iteration_started_at = time.perf_counter()
        _log(
            verbose,
            _render_compact_block(
                prefix="[v4]",
                title=f"iteration {iteration_index}",
                rows=[
                    ("phase", phase_label),
                    ("seed", str(iteration_seed)),
                    ("elapsed", f"{(time.time() - started_at) / 3600.0:.2f}h"),
                ],
            ),
        )

        checkpoint_path: Path | None = None
        replay_shard_path: Path | None = None

        if phase == "imitation":
            candidate_bundle, training_report = train_with_dagger(
                teacher_specs=teacher_specs,
                bootstrap_matches=bootstrap_matches,
                dagger_matches=dagger_matches,
                alpha=alpha,
                dagger_iterations=dagger_iterations,
                seed=iteration_seed,
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
                initial_bundle=current_best_bundle,
                play_value_weight=play_value_weight,
                auction_value_weight=auction_value_weight,
                play_rollout_depth=play_rollout_depth,
                auction_rollout_depth=auction_rollout_depth,
                warm_start_lr_scale=warm_start_lr_scale,
                warm_start_epoch_scale=warm_start_epoch_scale,
                gradient_clip=gradient_clip,
                student_agreement_keep_prob=student_agreement_keep_prob,
                teacher_sample_scale=teacher_sample_scale,
                teacher_target_temperature=teacher_target_temperature,
                trainer_backend=trainer_backend,
                trainer_batch_size=trainer_batch_size,
                workers=workers,
                bot_id=V4_BOT_ID,
                bundle_version=3,
                extra_metadata={
                    "training_mode": "alternating_imitation",
                    "seed_bot_id": current_best_bundle.get("bot_id", V4_BOT_ID),
                },
                verbose=verbose,
            )
            phase_training_report = training_report["training"]
        else:
            candidate_bundle, training_report = train_with_self_play(
                initial_bundle=current_best_bundle,
                self_play_matches=self_play_matches,
                alpha=alpha,
                self_play_iterations=1,
                replay_window=replay_window,
                seed=iteration_seed,
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
                play_value_weight=play_value_weight,
                auction_value_weight=auction_value_weight,
                play_rollout_depth=play_rollout_depth,
                auction_rollout_depth=auction_rollout_depth,
                play_temperature=play_temperature,
                auction_temperature=auction_temperature,
                epsilon=epsilon,
                temperature_decay=temperature_decay,
                epsilon_decay=epsilon_decay,
                policy_advantage_threshold=policy_advantage_threshold,
                policy_advantage_scale=policy_advantage_scale,
                value_weight_scale=value_weight_scale,
                winner_policy_weight=winner_policy_weight,
                gradient_clip=gradient_clip,
                trainer_backend=trainer_backend,
                trainer_batch_size=trainer_batch_size,
                replay_history=current_replay_history,
                iteration_offset=self_play_phase_count,
                workers=workers,
                bot_id=V4_BOT_ID,
                bundle_version=3,
                extra_metadata={
                    "training_mode": "alternating_self_play",
                    "seed_bot_id": current_best_bundle.get("bot_id", V4_BOT_ID),
                },
                verbose=verbose,
            )
            phase_training_report = training_report["self_play_iterations"][-1]["training"]
            self_play_phase_count += 1
            if current_replay_history.iterations and replay_store_dir is not None:
                replay_shard_path = save_replay_shard(
                    replay_store_dir=replay_store_dir,
                    dataset=current_replay_history.iterations[-1],
                    run_id=resolved_run_id,
                    iteration=iteration_index,
                    source_bot_id=current_best_bundle.get("bot_id", V4_BOT_ID),
                )
                prune_replay_shards(
                    replay_store_dir=replay_store_dir,
                    replay_window=replay_window,
                    persisted_replay_limit=persisted_replay_limit,
                    verbose=verbose,
                )

        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"iteration-{iteration_index:03d}.json"
            save_model_bundle(candidate_bundle, checkpoint_path)
            _log(
                verbose,
                _render_compact_block(
                    prefix="[v4]",
                    title=f"iteration {iteration_index} trained",
                    rows=[
                        ("phase", phase_label),
                        ("checkpoint", str(checkpoint_path)),
                        (
                            "replay",
                            str(replay_shard_path) if replay_shard_path is not None else "-",
                        ),
                        ("elapsed", _format_seconds(phase_training_report.get("elapsed_seconds", 0.0))),
                    ],
                ),
            )

        evaluation_report = evaluate_candidate(
            candidate_bundle=candidate_bundle,
            incumbent_bundle=current_best_bundle,
            alpha=alpha,
            eval_games=eval_games,
            eval_games_vs_optimal=eval_games_vs_optimal,
            precheck_games=min(precheck_games, eval_games),
            seed=iteration_seed,
            incumbent_baselines=incumbent_baseline_cache,
            promotion_head_to_head_margin=promotion_head_to_head_margin,
            promotion_greedy_regression_tolerance=promotion_greedy_regression_tolerance,
            promotion_optimal_regression_tolerance=promotion_optimal_regression_tolerance,
            workers=workers,
            verbose=verbose,
        )
        if (
            incumbent_baseline_cache is None
            and evaluation_report["incumbent_vs_greedy"] is not None
            and evaluation_report["incumbent_vs_optimal"] is not None
        ):
            incumbent_baseline_cache = _build_incumbent_baseline_cache_from_evaluation(
                evaluation_report,
                promoted_candidate=False,
            )

        replay_reset = False
        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
            incumbent_baseline_cache = _build_incumbent_baseline_cache_from_evaluation(
                evaluation_report,
                promoted_candidate=True,
            )
            if best_path is not None:
                save_model_bundle(current_best_bundle, best_path)
            if replay_store_dir is not None:
                current_replay_history = reset_replay_state_after_promotion(
                    replay_store_dir=replay_store_dir,
                    verbose=verbose,
                )
            else:
                current_replay_history = SelfPlayDatasetHistory()
            replay_reset = True
            _log(verbose, f"[v4] iteration {iteration_index} promoted candidate")
        else:
            _log(verbose, f"[v4] iteration {iteration_index} kept incumbent")

        iteration_elapsed_seconds = time.perf_counter() - iteration_started_at
        iteration_snapshot = _build_self_play_outer_snapshot(
            training_report=phase_training_report,
            evaluation_report=evaluation_report,
            iteration_elapsed_seconds=iteration_elapsed_seconds,
        )
        iteration_snapshot["phase"] = phase_label
        _log(
            verbose,
            _render_iteration_comparison_table(
                prefix="[v4]",
                title=f"iteration {iteration_index} comparison",
                current_snapshot=iteration_snapshot,
                previous_snapshot=previous_outer_snapshot,
                metrics=ALTERNATING_OUTER_METRICS,
                footer_note="compares this alternating v4 candidate attempt against the previous v4 iteration",
            ),
        )
        previous_outer_snapshot = iteration_snapshot

        iteration_report = {
            "iteration": iteration_index,
            "phase": phase_label,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "replay_shard_path": str(replay_shard_path) if replay_shard_path is not None else None,
            "accepted": evaluation_report["accepted"],
            "candidate_bot_id": candidate_bundle.get("bot_id"),
            "candidate_version": candidate_bundle.get("version"),
            "training": training_report,
            "evaluation": evaluation_report,
            "timing": {
                "iteration_elapsed_seconds": iteration_elapsed_seconds,
            },
            "replay_history_iterations": len(current_replay_history.iterations),
            "replay_reset": replay_reset,
        }
        iteration_reports.append(iteration_report)
        if report_path is not None:
            with report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(iteration_report) + "\n")
        iteration_index += 1

    return current_best_bundle, {
        "iterations": iteration_reports,
        "iterations_completed": iteration_index - 1,
        "elapsed_seconds": time.time() - started_at,
        "start_phase": _display_phase_name(normalized_start_phase),
        "remaining_replay_iterations": len(current_replay_history.iterations),
        "self_play_phases_completed": self_play_phase_count,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train neural-3p-v4 by alternating imitation refreshes and self-play promotion cycles."
    )
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--start-phase",
        default="imitation",
        choices=("imitation", "self-play"),
        help="which phase should run first in the alternating schedule",
    )
    parser.add_argument("--teacher-pool", default=DEFAULT_TEACHER_POOL)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--replay-store-dir", type=Path, default=DEFAULT_REPLAY_STORE_DIR)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=12)
    parser.add_argument("--bootstrap-matches", type=int, default=24)
    parser.add_argument("--dagger-matches", type=int, default=16)
    parser.add_argument("--dagger-iterations", type=int, default=1)
    parser.add_argument("--self-play-matches", type=int, default=DEFAULT_SELF_PLAY_MATCHES)
    parser.add_argument("--replay-window", type=int, default=DEFAULT_REPLAY_WINDOW)
    parser.add_argument(
        "--persisted-replay-limit",
        type=int,
        default=DEFAULT_PERSISTED_REPLAY_LIMIT,
    )
    parser.add_argument("--eval-games", type=int, default=36)
    parser.add_argument("--eval-games-vs-optimal", type=int, default=18)
    parser.add_argument("--precheck-games", type=int, default=12)
    parser.add_argument(
        "--promotion-head-to-head-margin",
        type=float,
        default=DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    )
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
    parser.add_argument("--play-hidden-dim", type=int, default=DEFAULT_PLAY_HIDDEN_DIM)
    parser.add_argument("--auction-hidden-dim", type=int, default=DEFAULT_AUCTION_HIDDEN_DIM)
    parser.add_argument("--play-value-hidden-dim", type=int, default=DEFAULT_PLAY_VALUE_HIDDEN_DIM)
    parser.add_argument("--auction-value-hidden-dim", type=int, default=DEFAULT_AUCTION_VALUE_HIDDEN_DIM)
    parser.add_argument("--play-epochs", type=int, default=10)
    parser.add_argument("--auction-epochs", type=int, default=10)
    parser.add_argument("--play-value-epochs", type=int, default=12)
    parser.add_argument("--auction-value-epochs", type=int, default=12)
    parser.add_argument("--play-learning-rate", type=float, default=0.022)
    parser.add_argument("--auction-learning-rate", type=float, default=0.026)
    parser.add_argument("--play-value-learning-rate", type=float, default=0.018)
    parser.add_argument("--auction-value-learning-rate", type=float, default=0.022)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--warm-start-lr-scale", type=float, default=DEFAULT_WARM_START_LR_SCALE)
    parser.add_argument("--warm-start-epoch-scale", type=float, default=DEFAULT_WARM_START_EPOCH_SCALE)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument(
        "--student-agreement-keep-prob",
        type=float,
        default=DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    )
    parser.add_argument("--teacher-sample-scale", type=float, default=DEFAULT_TEACHER_SAMPLE_SCALE)
    parser.add_argument("--teacher-target-temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--play-value-weight", type=float, default=DEFAULT_PLAY_VALUE_WEIGHT)
    parser.add_argument("--auction-value-weight", type=float, default=DEFAULT_AUCTION_VALUE_WEIGHT)
    parser.add_argument("--play-rollout-depth", type=int, default=DEFAULT_PLAY_ROLLOUT_DEPTH)
    parser.add_argument("--auction-rollout-depth", type=int, default=DEFAULT_AUCTION_ROLLOUT_DEPTH)
    parser.add_argument("--play-temperature", type=float, default=DEFAULT_PLAY_TEMPERATURE)
    parser.add_argument("--auction-temperature", type=float, default=DEFAULT_AUCTION_TEMPERATURE)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--temperature-decay", type=float, default=DEFAULT_TEMPERATURE_DECAY)
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY)
    parser.add_argument(
        "--policy-advantage-threshold",
        type=float,
        default=DEFAULT_POLICY_ADVANTAGE_THRESHOLD,
    )
    parser.add_argument(
        "--policy-advantage-scale",
        type=float,
        default=DEFAULT_POLICY_ADVANTAGE_SCALE,
    )
    parser.add_argument(
        "--value-weight-scale",
        type=float,
        default=DEFAULT_VALUE_WEIGHT_SCALE,
    )
    parser.add_argument(
        "--winner-policy-weight",
        type=float,
        default=DEFAULT_WINNER_POLICY_WEIGHT,
    )
    parser.add_argument(
        "--trainer-backend",
        default=DEFAULT_TRAINING_BACKEND,
        choices=TRAINING_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--trainer-batch-size",
        type=int,
        default=DEFAULT_TRAINER_BATCH_SIZE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    teacher_specs = parse_teacher_specs(args.teacher_pool)
    initial_model_path = resolve_initial_model_path(args.initial_model)
    current_best_bundle = load_model_bundle(initial_model_path)
    verbose = not args.quiet

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / f"neural-3p-v4-train-{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.json"
    report_path = run_dir / "report.jsonl"
    manifest_path = run_dir / "manifest.json"
    save_model_bundle(current_best_bundle, best_path)

    replay_history = load_persisted_replay_history(
        replay_store_dir=args.replay_store_dir,
        replay_window=args.replay_window,
        persisted_replay_limit=args.persisted_replay_limit,
        verbose=verbose,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "started_at": timestamp,
                "initial_model_path": str(initial_model_path),
                "teacher_pool": args.teacher_pool,
                "start_phase": args.start_phase,
                "replay_store_dir": str(args.replay_store_dir),
                "duration_hours": args.duration_hours,
                "alpha": args.alpha,
                "bootstrap_matches": args.bootstrap_matches,
                "dagger_matches": args.dagger_matches,
                "dagger_iterations": args.dagger_iterations,
                "self_play_matches": args.self_play_matches,
                "replay_window": args.replay_window,
                "persisted_replay_limit": args.persisted_replay_limit,
                "loaded_persisted_replay_shards": len(replay_history.iterations),
                "eval_games": args.eval_games,
                "eval_games_vs_optimal": args.eval_games_vs_optimal,
                "precheck_games": args.precheck_games,
                "promotion_head_to_head_margin": args.promotion_head_to_head_margin,
                "promotion_greedy_regression_tolerance": args.promotion_greedy_regression_tolerance,
                "promotion_optimal_regression_tolerance": args.promotion_optimal_regression_tolerance,
                "trainer_backend": args.trainer_backend,
                "trainer_batch_size": args.trainer_batch_size,
                "workers": args.workers,
                "seed": args.seed,
            }
        ),
        encoding="utf-8",
    )
    _log(
        verbose,
        _render_compact_block(
            prefix="[v4]",
            title="run",
            rows=[
                ("run dir", str(run_dir)),
                ("initial", str(initial_model_path)),
                ("phase order", f"{args.start_phase} -> {'self-play' if args.start_phase == 'imitation' else 'imitation'}"),
                ("teachers", args.teacher_pool),
                (
                    "budget",
                    (
                        f"{args.duration_hours:.2f}h | alpha {args.alpha} | workers {args.workers} | "
                        f"trainer {args.trainer_backend} | batch {args.trainer_batch_size}"
                    ),
                ),
                (
                    "schedule",
                    (
                        f"bootstrap {args.bootstrap_matches} | dagger {args.dagger_matches} x{args.dagger_iterations} | "
                        f"self-play {args.self_play_matches} | replay {args.replay_window}"
                    ),
                ),
                (
                    "eval",
                    (
                        f"games {args.eval_games} | optimal {args.eval_games_vs_optimal} | "
                        f"precheck {args.precheck_games}"
                    ),
                ),
            ],
        ),
    )

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    best_bundle, report = train_with_alternating_phases(
        initial_bundle=current_best_bundle,
        teacher_specs=teacher_specs,
        alpha=args.alpha,
        bootstrap_matches=args.bootstrap_matches,
        dagger_matches=args.dagger_matches,
        dagger_iterations=args.dagger_iterations,
        self_play_matches=args.self_play_matches,
        replay_window=args.replay_window,
        persisted_replay_limit=args.persisted_replay_limit,
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
        play_value_weight=args.play_value_weight,
        auction_value_weight=args.auction_value_weight,
        play_rollout_depth=args.play_rollout_depth,
        auction_rollout_depth=args.auction_rollout_depth,
        warm_start_lr_scale=args.warm_start_lr_scale,
        warm_start_epoch_scale=args.warm_start_epoch_scale,
        gradient_clip=args.gradient_clip,
        student_agreement_keep_prob=args.student_agreement_keep_prob,
        teacher_sample_scale=args.teacher_sample_scale,
        teacher_target_temperature=args.teacher_target_temperature,
        play_temperature=args.play_temperature,
        auction_temperature=args.auction_temperature,
        epsilon=args.epsilon,
        temperature_decay=args.temperature_decay,
        epsilon_decay=args.epsilon_decay,
        policy_advantage_threshold=args.policy_advantage_threshold,
        policy_advantage_scale=args.policy_advantage_scale,
        value_weight_scale=args.value_weight_scale,
        winner_policy_weight=args.winner_policy_weight,
        eval_games=args.eval_games,
        eval_games_vs_optimal=args.eval_games_vs_optimal,
        precheck_games=args.precheck_games,
        promotion_head_to_head_margin=args.promotion_head_to_head_margin,
        promotion_greedy_regression_tolerance=args.promotion_greedy_regression_tolerance,
        promotion_optimal_regression_tolerance=args.promotion_optimal_regression_tolerance,
        max_iterations=args.max_iterations,
        deadline=deadline,
        run_started_at=started_at,
        start_phase=args.start_phase,
        replay_history=replay_history,
        checkpoint_dir=checkpoint_dir,
        best_path=best_path,
        report_path=report_path,
        replay_store_dir=args.replay_store_dir,
        run_id=timestamp,
        trainer_backend=args.trainer_backend,
        trainer_batch_size=args.trainer_batch_size,
        workers=args.workers,
        verbose=verbose,
    )

    if args.promote_to is not None:
        save_model_bundle(best_bundle, args.promote_to)
        _log(verbose, f"[v4] wrote promoted best model to {args.promote_to}")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_model": str(best_path),
                "promoted_to": str(args.promote_to) if args.promote_to is not None else None,
                "iterations_completed": report["iterations_completed"],
            }
        )
    )


if __name__ == "__main__":
    main()
