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
        _format_league_pool,
        load_persisted_replay_history,
        parse_league_opponent_specs,
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
        _render_promotion_outcome_block,
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
        TeacherSpec,
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
        _format_league_pool,
        load_persisted_replay_history,
        parse_league_opponent_specs,
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
        _render_promotion_outcome_block,
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
        TeacherSpec,
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
DEFAULT_GUARD_FOCUS_MULTIPLIER = 2.0
DEFAULT_REJECTED_REPLAY_POLICY = "rollback"
REJECTED_REPLAY_POLICIES = ("rollback", "keep", "near-miss")
DEFAULT_REJECTED_REPLAY_MIN_HEAD_TO_HEAD = 50.0
DEFAULT_REJECTED_REPLAY_GUARD_TOLERANCE = 12.5
GREEDY_FOCUS_BOT_IDS = frozenset({"greedy", "1-trick-minmax"})
OPTIMAL_FOCUS_BOT_IDS = frozenset({"optimal-bot"})
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
    return _phase_for_iteration_with_burst(
        start_phase=start_phase,
        iteration_index=iteration_index,
        self_play_phases_per_cycle=1,
    )


def _phase_for_iteration_with_burst(
    *,
    start_phase: str,
    iteration_index: int,
    self_play_phases_per_cycle: int,
) -> str:
    if iteration_index < 0:
        raise ValueError("iteration_index must be non-negative")
    if self_play_phases_per_cycle <= 0:
        raise ValueError("self_play_phases_per_cycle must be positive")
    if start_phase == "imitation":
        cycle = ["imitation", *(["self_play"] * self_play_phases_per_cycle)]
    else:
        cycle = [*(["self_play"] * self_play_phases_per_cycle), "imitation"]
    return cycle[iteration_index % len(cycle)]


def _should_run_imitation_after_self_play(
    *,
    evaluation_report: dict,
    self_play_phases_since_imitation: int,
    failed_self_play_phases_since_imitation: int,
    self_play_phases_per_cycle: int,
    self_play_failures_before_imitation: int,
) -> bool:
    if evaluation_report["accepted"]:
        return False

    if self_play_phases_since_imitation < self_play_phases_per_cycle:
        return False

    if self_play_failures_before_imitation <= 0:
        return True

    return failed_self_play_phases_since_imitation >= self_play_failures_before_imitation


def _extract_guard_focus_bot_ids(
    evaluation_report: dict,
) -> tuple[set[str], set[str]]:
    diagnostics = evaluation_report.get("promotion_diagnostics", {})
    reasons = diagnostics.get("reasons", [])
    teacher_focus_bot_ids: set[str] = set()
    league_focus_bot_ids: set[str] = set()

    if any(reason.startswith("greedy guard failed:") for reason in reasons):
        teacher_focus_bot_ids.update(GREEDY_FOCUS_BOT_IDS)
        league_focus_bot_ids.update(GREEDY_FOCUS_BOT_IDS)
    if any(reason.startswith("optimal guard failed:") for reason in reasons):
        teacher_focus_bot_ids.update(OPTIMAL_FOCUS_BOT_IDS)
        league_focus_bot_ids.update(OPTIMAL_FOCUS_BOT_IDS)

    return teacher_focus_bot_ids, league_focus_bot_ids


def _apply_focus_to_specs(
    specs: list[TeacherSpec],
    *,
    focus_bot_ids: set[str],
    multiplier: float,
) -> list[TeacherSpec]:
    if not specs or not focus_bot_ids or multiplier <= 1.0:
        return specs
    return [
        TeacherSpec(
            spec.bot_id,
            spec.weight * multiplier if spec.bot_id in focus_bot_ids else spec.weight,
        )
        for spec in specs
    ]


def _evaluation_win_percentage(
    evaluation_report: dict,
    *,
    report_key: str,
    model_key: str,
) -> float | None:
    report = evaluation_report.get(report_key)
    if report is None:
        return None
    try:
        return float(report["models"][model_key]["win_percentage"])
    except (KeyError, TypeError, ValueError):
        return None


def _guard_deficits(
    evaluation_report: dict,
    *,
    greedy_regression_tolerance: float,
    optimal_regression_tolerance: float,
) -> list[float] | None:
    candidate_vs_greedy = _evaluation_win_percentage(
        evaluation_report,
        report_key="vs_greedy",
        model_key="candidate",
    )
    incumbent_vs_greedy = _evaluation_win_percentage(
        evaluation_report,
        report_key="incumbent_vs_greedy",
        model_key="incumbent",
    )
    candidate_vs_optimal = _evaluation_win_percentage(
        evaluation_report,
        report_key="vs_optimal_bot",
        model_key="candidate",
    )
    incumbent_vs_optimal = _evaluation_win_percentage(
        evaluation_report,
        report_key="incumbent_vs_optimal",
        model_key="incumbent",
    )
    if (
        candidate_vs_greedy is None
        or incumbent_vs_greedy is None
        or candidate_vs_optimal is None
        or incumbent_vs_optimal is None
    ):
        return None
    return [
        max(0.0, incumbent_vs_greedy - greedy_regression_tolerance - candidate_vs_greedy),
        max(0.0, incumbent_vs_optimal - optimal_regression_tolerance - candidate_vs_optimal),
    ]


def _should_keep_rejected_self_play_replay(
    evaluation_report: dict,
    *,
    rejected_replay_policy: str,
    rejected_replay_min_head_to_head: float,
    rejected_replay_guard_tolerance: float,
    greedy_regression_tolerance: float,
    optimal_regression_tolerance: float,
) -> tuple[bool, str]:
    if rejected_replay_policy not in REJECTED_REPLAY_POLICIES:
        raise ValueError(
            "rejected_replay_policy must be one of: "
            + ", ".join(REJECTED_REPLAY_POLICIES)
        )
    if rejected_replay_policy == "rollback":
        return False, "policy rollback"
    if rejected_replay_policy == "keep":
        return True, "policy keep"

    candidate_vs_incumbent = _evaluation_win_percentage(
        evaluation_report,
        report_key="vs_incumbent",
        model_key="candidate",
    )
    if candidate_vs_incumbent is None:
        return False, "no head-to-head result"
    if candidate_vs_incumbent < rejected_replay_min_head_to_head:
        return (
            False,
            (
                f"head-to-head {candidate_vs_incumbent:.1f}% "
                f"< {rejected_replay_min_head_to_head:.1f}%"
            ),
        )

    deficits = _guard_deficits(
        evaluation_report,
        greedy_regression_tolerance=greedy_regression_tolerance,
        optimal_regression_tolerance=optimal_regression_tolerance,
    )
    if deficits is None:
        return True, f"head-to-head {candidate_vs_incumbent:.1f}%"

    max_deficit = max(deficits)
    if max_deficit <= rejected_replay_guard_tolerance:
        return (
            True,
            (
                f"head-to-head {candidate_vs_incumbent:.1f}% "
                f"and guard deficit {max_deficit:.1f} pts"
            ),
        )
    return (
        False,
        (
            f"guard deficit {max_deficit:.1f} pts "
            f"> {rejected_replay_guard_tolerance:.1f} pts"
        ),
    )


def _resolve_default_initial_model_path() -> Path:
    if NeuralThreePlayerBot.MODEL_FILE_V4.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V4
    if NeuralThreePlayerBot.MODEL_FILE_V3.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V3
    if NeuralThreePlayerBot.MODEL_FILE_V2.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V2
    return NeuralThreePlayerBot.MODEL_FILE_V1


def resolve_initial_model_path(explicit_path: Path | None) -> tuple[Path, str | None]:
    if explicit_path is None:
        return _resolve_default_initial_model_path(), None

    requested_path = Path(explicit_path)
    if requested_path.exists():
        return requested_path, None

    requested_resolved = requested_path.resolve(strict=False)
    canonical_v4_path = NeuralThreePlayerBot.MODEL_FILE_V4.resolve(strict=False)
    if requested_resolved == canonical_v4_path:
        fallback_path = _resolve_default_initial_model_path()
        if fallback_path.resolve(strict=False) != canonical_v4_path:
            return (
                fallback_path,
                (
                    f"requested initial model {requested_path} was missing; "
                    f"using {fallback_path} instead"
                ),
            )

    raise FileNotFoundError(
        f"initial model not found: {requested_path}"
    )


def train_with_alternating_phases(
    *,
    initial_bundle: dict,
    teacher_specs: list,
    self_play_league_specs: list[TeacherSpec] | None = None,
    guard_focus_multiplier: float = DEFAULT_GUARD_FOCUS_MULTIPLIER,
    alpha: int,
    bootstrap_matches: int,
    repeat_imitation_bootstrap_matches: int = 0,
    dagger_matches: int,
    repeat_imitation_dagger_matches: int | None = None,
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
    self_play_phases_per_cycle: int = 1,
    self_play_failures_before_imitation: int = 0,
    retain_replay_iterations_after_promotion: int = 0,
    carry_rejected_candidates: bool = False,
    rejected_replay_policy: str = DEFAULT_REJECTED_REPLAY_POLICY,
    rejected_replay_min_head_to_head: float = DEFAULT_REJECTED_REPLAY_MIN_HEAD_TO_HEAD,
    rejected_replay_guard_tolerance: float = DEFAULT_REJECTED_REPLAY_GUARD_TOLERANCE,
    replay_history: SelfPlayDatasetHistory | None = None,
    checkpoint_dir: Path | None = None,
    best_path: Path | None = None,
    promote_to_path: Path | None = None,
    report_path: Path | None = None,
    replay_store_dir: Path | None = None,
    run_id: str | None = None,
    trainer_backend: str = DEFAULT_TRAINING_BACKEND,
    trainer_batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> tuple[dict, dict]:
    normalized_start_phase = _normalize_phase_name(start_phase)
    if repeat_imitation_bootstrap_matches < 0:
        raise ValueError("repeat_imitation_bootstrap_matches must be non-negative")
    if repeat_imitation_dagger_matches is not None and repeat_imitation_dagger_matches < 0:
        raise ValueError("repeat_imitation_dagger_matches must be non-negative")
    if self_play_phases_per_cycle <= 0:
        raise ValueError("self_play_phases_per_cycle must be positive")
    if self_play_failures_before_imitation < 0:
        raise ValueError("self_play_failures_before_imitation must be non-negative")
    if retain_replay_iterations_after_promotion < 0:
        raise ValueError("retain_replay_iterations_after_promotion must be non-negative")
    if guard_focus_multiplier <= 0:
        raise ValueError("guard_focus_multiplier must be positive")
    if rejected_replay_policy not in REJECTED_REPLAY_POLICIES:
        raise ValueError(
            "rejected_replay_policy must be one of: "
            + ", ".join(REJECTED_REPLAY_POLICIES)
        )
    if rejected_replay_min_head_to_head < 0 or rejected_replay_min_head_to_head > 100:
        raise ValueError("rejected_replay_min_head_to_head must be between 0 and 100")
    if rejected_replay_guard_tolerance < 0:
        raise ValueError("rejected_replay_guard_tolerance must be non-negative")
    started_at = run_started_at if run_started_at is not None else time.time()
    current_best_bundle = initial_bundle
    working_bundle = initial_bundle
    current_replay_history = replay_history or SelfPlayDatasetHistory()
    incumbent_baseline_cache: dict | None = None
    previous_outer_snapshot: dict[str, object] | None = None
    self_play_phase_count = 0
    imitation_phase_count = 0
    self_play_phases_since_imitation = 0
    failed_self_play_phases_since_imitation = 0
    iteration_index = 1
    iteration_reports: list[dict] = []
    resolved_run_id = run_id or "neural-3p-v4"
    adaptive_imitation_schedule = self_play_failures_before_imitation > 0
    next_phase = normalized_start_phase
    teacher_focus_bot_ids: set[str] = set()
    league_focus_bot_ids: set[str] = set()

    while True:
        if max_iterations is not None and iteration_index > max_iterations:
            break
        if deadline is not None and time.time() >= deadline and iteration_index > 1:
            break

        if adaptive_imitation_schedule:
            phase = next_phase
        else:
            phase = _phase_for_iteration(
                start_phase=normalized_start_phase,
                iteration_index=iteration_index - 1,
            )
            if self_play_phases_per_cycle != 1:
                phase = _phase_for_iteration_with_burst(
                    start_phase=normalized_start_phase,
                    iteration_index=iteration_index - 1,
                    self_play_phases_per_cycle=self_play_phases_per_cycle,
                )
        phase_label = _display_phase_name(phase)
        active_teacher_specs = _apply_focus_to_specs(
            teacher_specs,
            focus_bot_ids=teacher_focus_bot_ids,
            multiplier=guard_focus_multiplier,
        )
        active_league_specs = _apply_focus_to_specs(
            self_play_league_specs or [],
            focus_bot_ids=league_focus_bot_ids,
            multiplier=guard_focus_multiplier,
        )
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
                    (
                        "focus",
                        (
                            ", ".join(sorted(teacher_focus_bot_ids or league_focus_bot_ids))
                            if (teacher_focus_bot_ids or league_focus_bot_ids)
                            else "-"
                        ),
                    ),
                ],
            ),
        )

        checkpoint_path: Path | None = None
        replay_shard_path: Path | None = None
        replay_iterations_before_phase = list(current_replay_history.iterations)

        if phase == "imitation":
            phase_bootstrap_matches = (
                bootstrap_matches
                if imitation_phase_count == 0
                else repeat_imitation_bootstrap_matches
            )
            phase_dagger_matches = (
                dagger_matches
                if imitation_phase_count == 0 or repeat_imitation_dagger_matches is None
                else repeat_imitation_dagger_matches
            )
            candidate_bundle, training_report = train_with_dagger(
                teacher_specs=active_teacher_specs,
                bootstrap_matches=phase_bootstrap_matches,
                dagger_matches=phase_dagger_matches,
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
                initial_bundle=working_bundle,
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
                    "seed_bot_id": working_bundle.get("bot_id", V4_BOT_ID),
                },
                verbose=verbose,
            )
            phase_training_report = training_report["training"]
            imitation_phase_count += 1
        else:
            candidate_bundle, training_report = train_with_self_play(
                initial_bundle=working_bundle,
                incumbent_bundle=current_best_bundle,
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
                league_opponent_specs=active_league_specs,
                replay_history=current_replay_history,
                iteration_offset=self_play_phase_count,
                workers=workers,
                bot_id=V4_BOT_ID,
                bundle_version=3,
                extra_metadata={
                    "training_mode": "alternating_self_play",
                    "seed_bot_id": working_bundle.get("bot_id", V4_BOT_ID),
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
                    source_bot_id=working_bundle.get("bot_id", V4_BOT_ID),
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
        replay_rolled_back = False
        rejected_replay_kept = False
        rejected_replay_reason: str | None = None
        carried_rejected_candidate = False
        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
            working_bundle = candidate_bundle
            teacher_focus_bot_ids.clear()
            league_focus_bot_ids.clear()
            incumbent_baseline_cache = _build_incumbent_baseline_cache_from_evaluation(
                evaluation_report,
                promoted_candidate=True,
            )
            if best_path is not None:
                save_model_bundle(current_best_bundle, best_path)
            if promote_to_path is not None:
                save_model_bundle(current_best_bundle, promote_to_path)
            if replay_store_dir is not None:
                current_replay_history = reset_replay_state_after_promotion(
                    replay_store_dir=replay_store_dir,
                    replay_history=current_replay_history,
                    retain_recent_iterations=min(
                        retain_replay_iterations_after_promotion,
                        replay_window,
                    ),
                    verbose=verbose,
                )
            else:
                if retain_replay_iterations_after_promotion > 0:
                    current_replay_history = SelfPlayDatasetHistory(
                        iterations=list(
                            current_replay_history.iterations[
                                -min(retain_replay_iterations_after_promotion, replay_window) :
                            ]
                        )
                    )
                else:
                    current_replay_history = SelfPlayDatasetHistory()
            replay_reset = True
            _log(verbose, f"[v4] iteration {iteration_index} promoted candidate")
        else:
            if carry_rejected_candidates:
                working_bundle = candidate_bundle
                carried_rejected_candidate = True
                rejected_replay_kept = phase == "self_play"
                rejected_replay_reason = "carrying rejected candidate"
                if replay_store_dir is not None and replay_shard_path is not None:
                    prune_replay_shards(
                        replay_store_dir=replay_store_dir,
                        replay_window=replay_window,
                        persisted_replay_limit=persisted_replay_limit,
                        verbose=verbose,
                    )
            else:
                working_bundle = current_best_bundle
                if phase == "self_play":
                    (
                        rejected_replay_kept,
                        rejected_replay_reason,
                    ) = _should_keep_rejected_self_play_replay(
                        evaluation_report,
                        rejected_replay_policy=rejected_replay_policy,
                        rejected_replay_min_head_to_head=rejected_replay_min_head_to_head,
                        rejected_replay_guard_tolerance=rejected_replay_guard_tolerance,
                        greedy_regression_tolerance=promotion_greedy_regression_tolerance,
                        optimal_regression_tolerance=promotion_optimal_regression_tolerance,
                    )
                    if rejected_replay_kept:
                        if replay_store_dir is not None and replay_shard_path is not None:
                            prune_replay_shards(
                                replay_store_dir=replay_store_dir,
                                replay_window=replay_window,
                                persisted_replay_limit=persisted_replay_limit,
                                verbose=verbose,
                            )
                    else:
                        current_replay_history = SelfPlayDatasetHistory(
                            iterations=replay_iterations_before_phase,
                        )
                        replay_rolled_back = True
                        if replay_shard_path is not None:
                            replay_shard_path.unlink(missing_ok=True)
                            replay_shard_path = None
            _log(verbose, f"[v4] iteration {iteration_index} kept incumbent")
            if replay_rolled_back:
                _log(
                    verbose,
                    f"[v4] iteration {iteration_index} rolled back rejected self-play replay",
                )
            if rejected_replay_kept:
                _log(
                    verbose,
                    (
                        f"[v4] iteration {iteration_index} kept rejected self-play replay"
                        + (
                            f" ({rejected_replay_reason})"
                            if rejected_replay_reason is not None
                            else ""
                        )
                    ),
                )
            if carried_rejected_candidate:
                _log(
                    verbose,
                    f"[v4] iteration {iteration_index} carrying rejected candidate forward",
                )
            _log(
                verbose,
                _render_promotion_outcome_block(
                    prefix="[v4]",
                    title=f"iteration {iteration_index} promotion",
                    evaluation_report=evaluation_report,
                    head_to_head_margin=promotion_head_to_head_margin,
                    greedy_regression_tolerance=promotion_greedy_regression_tolerance,
                    optimal_regression_tolerance=promotion_optimal_regression_tolerance,
                ),
            )

        if adaptive_imitation_schedule:
            if phase == "imitation":
                self_play_phases_since_imitation = 0
                failed_self_play_phases_since_imitation = 0
                next_phase = "self_play"
            else:
                self_play_phases_since_imitation += 1
                if evaluation_report["accepted"]:
                    failed_self_play_phases_since_imitation = 0
                else:
                    failed_self_play_phases_since_imitation += 1
                    if (
                        not evaluation_report["rejected_after_precheck"]
                        and not evaluation_report["rejected_after_head_to_head"]
                    ):
                        (
                            teacher_focus_bot_ids,
                            league_focus_bot_ids,
                        ) = _extract_guard_focus_bot_ids(evaluation_report)
                    else:
                        teacher_focus_bot_ids.clear()
                        league_focus_bot_ids.clear()
                next_phase = (
                    "imitation"
                    if _should_run_imitation_after_self_play(
                        evaluation_report=evaluation_report,
                        self_play_phases_since_imitation=self_play_phases_since_imitation,
                        failed_self_play_phases_since_imitation=failed_self_play_phases_since_imitation,
                        self_play_phases_per_cycle=self_play_phases_per_cycle,
                        self_play_failures_before_imitation=self_play_failures_before_imitation,
                    )
                    else "self_play"
                )

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
            "working_bot_id": working_bundle.get("bot_id"),
            "incumbent_bot_id": current_best_bundle.get("bot_id"),
            "training": training_report,
            "evaluation": evaluation_report,
            "timing": {
                "iteration_elapsed_seconds": iteration_elapsed_seconds,
            },
            "replay_history_iterations": len(current_replay_history.iterations),
            "replay_reset": replay_reset,
            "replay_rolled_back": replay_rolled_back,
            "rejected_replay_kept": rejected_replay_kept,
            "rejected_replay_reason": rejected_replay_reason,
            "carried_rejected_candidate": carried_rejected_candidate,
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
    parser.add_argument(
        "--self-play-league-pool",
        default="",
        help=(
            "optional weighted opponent pool for league self-play; use ready bot ids "
            "and/or incumbent, e.g. 'incumbent:2,greedy:2,1-trick-minmax:2,optimal-bot:1'"
        ),
    )
    parser.add_argument(
        "--guard-focus-multiplier",
        type=float,
        default=DEFAULT_GUARD_FOCUS_MULTIPLIER,
        help="multiply focused teacher/league opponent weights after greedy or optimal guard failures",
    )
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--replay-store-dir", type=Path, default=DEFAULT_REPLAY_STORE_DIR)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=12)
    parser.add_argument("--bootstrap-matches", type=int, default=24)
    parser.add_argument(
        "--repeat-imitation-bootstrap-matches",
        type=int,
        default=0,
        help="bootstrap teacher matches for imitation phases after the first; use 0 to skip repeated bootstrap",
    )
    parser.add_argument(
        "--repeat-imitation-dagger-matches",
        type=int,
        default=8,
        help="student rollout matches for imitation phases after the first",
    )
    parser.add_argument("--dagger-matches", type=int, default=16)
    parser.add_argument("--dagger-iterations", type=int, default=1)
    parser.add_argument("--self-play-matches", type=int, default=DEFAULT_SELF_PLAY_MATCHES)
    parser.add_argument(
        "--self-play-phases-per-cycle",
        type=int,
        default=2,
        help="number of self-play phases to run between imitation phases, or the minimum self-play burst in adaptive mode",
    )
    parser.add_argument(
        "--self-play-failures-before-imitation",
        type=int,
        default=0,
        help="adaptive mode: run imitation only after this many consecutive rejected self-play phases; use 0 for fixed cadence",
    )
    parser.add_argument("--replay-window", type=int, default=DEFAULT_REPLAY_WINDOW)
    parser.add_argument(
        "--persisted-replay-limit",
        type=int,
        default=DEFAULT_PERSISTED_REPLAY_LIMIT,
    )
    parser.add_argument(
        "--retain-replay-iterations-after-promotion",
        type=int,
        default=1,
        help="how many freshest replay iterations to keep after a promotion",
    )
    parser.add_argument(
        "--carry-rejected-candidates",
        action="store_true",
        help=(
            "continue training from rejected candidates and keep their replay shards; "
            "by default v4 rolls back to the incumbent after failed promotion checks"
        ),
    )
    parser.add_argument(
        "--rejected-replay-policy",
        choices=REJECTED_REPLAY_POLICIES,
        default=DEFAULT_REJECTED_REPLAY_POLICY,
        help=(
            "what to do with self-play replay from rejected candidates when not carrying "
            "candidate weights: rollback discards it, keep retains all, near-miss keeps "
            "only rejected candidates that were competitive enough to be useful"
        ),
    )
    parser.add_argument(
        "--rejected-replay-min-head-to-head",
        type=float,
        default=DEFAULT_REJECTED_REPLAY_MIN_HEAD_TO_HEAD,
        help="near-miss replay requires this candidate win percentage vs incumbent",
    )
    parser.add_argument(
        "--rejected-replay-guard-tolerance",
        type=float,
        default=DEFAULT_REJECTED_REPLAY_GUARD_TOLERANCE,
        help="near-miss replay allows this maximum greedy/optimal guard deficit in points",
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
    self_play_league_specs = parse_league_opponent_specs(args.self_play_league_pool)
    initial_model_path, initial_model_resolution_note = resolve_initial_model_path(
        args.initial_model
    )
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
                "self_play_league_pool": args.self_play_league_pool,
                "guard_focus_multiplier": args.guard_focus_multiplier,
                "start_phase": args.start_phase,
                "replay_store_dir": str(args.replay_store_dir),
                "duration_hours": args.duration_hours,
                "alpha": args.alpha,
                "bootstrap_matches": args.bootstrap_matches,
                "repeat_imitation_bootstrap_matches": args.repeat_imitation_bootstrap_matches,
                "dagger_matches": args.dagger_matches,
                "repeat_imitation_dagger_matches": args.repeat_imitation_dagger_matches,
                "dagger_iterations": args.dagger_iterations,
                "self_play_matches": args.self_play_matches,
                "self_play_phases_per_cycle": args.self_play_phases_per_cycle,
                "self_play_failures_before_imitation": args.self_play_failures_before_imitation,
                "replay_window": args.replay_window,
                "persisted_replay_limit": args.persisted_replay_limit,
                "retain_replay_iterations_after_promotion": args.retain_replay_iterations_after_promotion,
                "carry_rejected_candidates": args.carry_rejected_candidates,
                "rejected_replay_policy": args.rejected_replay_policy,
                "rejected_replay_min_head_to_head": args.rejected_replay_min_head_to_head,
                "rejected_replay_guard_tolerance": args.rejected_replay_guard_tolerance,
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
                (
                    "phase order",
                    (
                        (
                            f"{args.start_phase} -> adaptive self-play/imitation"
                            if args.self_play_failures_before_imitation > 0
                            else (
                                f"{args.start_phase} -> self-play x{args.self_play_phases_per_cycle}"
                                if args.start_phase == "imitation"
                                else f"self-play x{args.self_play_phases_per_cycle} -> imitation"
                            )
                        )
                    ),
                ),
                ("teachers", args.teacher_pool),
                ("league", _format_league_pool(self_play_league_specs)),
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
                        (
                            f"bootstrap {args.bootstrap_matches}/{args.repeat_imitation_bootstrap_matches} | "
                            f"dagger {args.dagger_matches}/{args.repeat_imitation_dagger_matches} x{args.dagger_iterations} | "
                            f"self-play {args.self_play_matches} min-burst {args.self_play_phases_per_cycle} | "
                            f"imitate after {args.self_play_failures_before_imitation} self-play fails | "
                            f"replay {args.replay_window} keep {args.retain_replay_iterations_after_promotion} | "
                            f"rejected {'carry' if args.carry_rejected_candidates else args.rejected_replay_policy} | "
                            f"focus x{args.guard_focus_multiplier:.1f}"
                        )
                        if args.self_play_failures_before_imitation > 0
                        else (
                            f"bootstrap {args.bootstrap_matches}/{args.repeat_imitation_bootstrap_matches} | "
                            f"dagger {args.dagger_matches}/{args.repeat_imitation_dagger_matches} x{args.dagger_iterations} | "
                            f"self-play {args.self_play_matches} x{args.self_play_phases_per_cycle} | "
                            f"replay {args.replay_window} keep {args.retain_replay_iterations_after_promotion} | "
                            f"rejected {'carry' if args.carry_rejected_candidates else args.rejected_replay_policy} | "
                            f"focus x{args.guard_focus_multiplier:.1f}"
                        )
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
    if initial_model_resolution_note is not None:
        _log(verbose, f"[v4] note {initial_model_resolution_note}")

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    best_bundle, report = train_with_alternating_phases(
        initial_bundle=current_best_bundle,
        teacher_specs=teacher_specs,
        self_play_league_specs=self_play_league_specs,
        guard_focus_multiplier=args.guard_focus_multiplier,
        alpha=args.alpha,
        bootstrap_matches=args.bootstrap_matches,
        repeat_imitation_bootstrap_matches=args.repeat_imitation_bootstrap_matches,
        dagger_matches=args.dagger_matches,
        repeat_imitation_dagger_matches=args.repeat_imitation_dagger_matches,
        dagger_iterations=args.dagger_iterations,
        self_play_matches=args.self_play_matches,
        self_play_phases_per_cycle=args.self_play_phases_per_cycle,
        self_play_failures_before_imitation=args.self_play_failures_before_imitation,
        replay_window=args.replay_window,
        persisted_replay_limit=args.persisted_replay_limit,
        retain_replay_iterations_after_promotion=args.retain_replay_iterations_after_promotion,
        carry_rejected_candidates=args.carry_rejected_candidates,
        rejected_replay_policy=args.rejected_replay_policy,
        rejected_replay_min_head_to_head=args.rejected_replay_min_head_to_head,
        rejected_replay_guard_tolerance=args.rejected_replay_guard_tolerance,
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
        promote_to_path=args.promote_to,
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
