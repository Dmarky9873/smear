from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot
    from .simulator import compare_models_objectively
    from .train_neural_3p_bot import (
        ComparisonMetric,
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_TEACHER_POOL,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_WORKERS,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WARM_START_EPOCH_SCALE,
        LiveProgressDisplay,
        TRAINING_BACKEND_CHOICES,
        _create_parallel_executor,
        _resolve_worker_count,
        _render_compact_block,
        _render_iteration_comparison_table,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot
    from simulator import compare_models_objectively
    from train_neural_3p_bot import (
        ComparisonMetric,
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TRAINER_BATCH_SIZE,
        DEFAULT_TRAINING_BACKEND,
        DEFAULT_TEACHER_POOL,
        DEFAULT_TEACHER_SAMPLE_SCALE,
        DEFAULT_TEACHER_TARGET_TEMPERATURE,
        DEFAULT_WORKERS,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WARM_START_EPOCH_SCALE,
        LiveProgressDisplay,
        TRAINING_BACKEND_CHOICES,
        _create_parallel_executor,
        _resolve_worker_count,
        _render_compact_block,
        _render_iteration_comparison_table,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )


DEFAULT_RUN_ROOT = NeuralThreePlayerBot.MODEL_DIR / "self_train_runs"
DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN = 5.0
DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE = 0.0
DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE = 0.0
TRAINING_CYCLE_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric(
        key="play_accuracy",
        label="Play accuracy",
        kind="float",
        digits=3,
        higher_is_better=True,
        tolerance=5e-4,
    ),
    ComparisonMetric(
        key="auction_accuracy",
        label="Auction accuracy",
        kind="float",
        digits=3,
        higher_is_better=True,
        tolerance=5e-4,
    ),
    ComparisonMetric(
        key="play_value_mse",
        label="Play value MSE",
        kind="float",
        digits=4,
        higher_is_better=False,
        tolerance=5e-5,
    ),
    ComparisonMetric(
        key="auction_value_mse",
        label="Auction value MSE",
        kind="float",
        digits=4,
        higher_is_better=False,
        tolerance=5e-5,
    ),
    ComparisonMetric(
        key="candidate_vs_incumbent",
        label="Vs incumbent",
        kind="percent",
        digits=1,
        higher_is_better=True,
        tolerance=0.05,
    ),
    ComparisonMetric(
        key="candidate_vs_greedy",
        label="Vs greedy",
        kind="percent",
        digits=1,
        higher_is_better=True,
        tolerance=0.05,
    ),
    ComparisonMetric(
        key="candidate_vs_optimal",
        label="Vs optimal",
        kind="percent",
        digits=1,
        higher_is_better=True,
        tolerance=0.05,
    ),
    ComparisonMetric(
        key="training_elapsed_seconds",
        label="Train time",
        kind="duration",
        higher_is_better=False,
        tolerance=0.5,
    ),
    ComparisonMetric(
        key="evaluation_elapsed_seconds",
        label="Eval time",
        kind="duration",
        higher_is_better=False,
        tolerance=0.5,
    ),
    ComparisonMetric(
        key="cycle_elapsed_seconds",
        label="Cycle time",
        kind="duration",
        higher_is_better=False,
        tolerance=0.5,
    ),
    ComparisonMetric(
        key="decision",
        label="Decision",
        kind="text",
    ),
)


@dataclass(frozen=True)
class BundleBotFactory:
    label: str
    bundle: dict

    @property
    def __name__(self) -> str:
        return self.label

    def __call__(self, player_name: str):
        return NeuralThreePlayerBot(player_name, model_bundle=self.bundle)


def resolve_initial_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if NeuralThreePlayerBot.MODEL_FILE.exists():
        return NeuralThreePlayerBot.MODEL_FILE
    return NeuralThreePlayerBot.MODEL_FILE_V1


def make_named_factory(label: str, bundle: dict):
    return BundleBotFactory(label=label, bundle=bundle)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_eval_summary(evaluation_report: dict) -> str:
    vs_incumbent = evaluation_report["vs_incumbent"]["models"]
    if evaluation_report.get("rejected_after_precheck"):
        precheck = evaluation_report["vs_incumbent"]["models"]
        return (
            f"precheck_reject candidate_vs_incumbent="
            f"{precheck['candidate']['win_percentage']:.1f}%/"
            f"{precheck['incumbent']['win_percentage']:.1f}%"
        )
    if evaluation_report.get("rejected_after_head_to_head"):
        return (
            f"head_to_head_reject candidate_vs_incumbent="
            f"{vs_incumbent['candidate']['win_percentage']:.1f}%/"
            f"{vs_incumbent['incumbent']['win_percentage']:.1f}%"
        )
    vs_greedy = evaluation_report["vs_greedy"]["models"]
    vs_optimal = evaluation_report["vs_optimal_bot"]["models"]
    return (
        f"candidate_vs_incumbent={vs_incumbent['candidate']['win_percentage']:.1f}%/"
        f"{vs_incumbent['incumbent']['win_percentage']:.1f}% "
        f"candidate_vs_greedy={vs_greedy['candidate']['win_percentage']:.1f}% "
        f"candidate_vs_optimal={vs_optimal['candidate']['win_percentage']:.1f}%"
    )


def _compare_models_worker(task: dict) -> dict:
    return compare_models_objectively(
        task["n"],
        task["alpha"],
        task["model1"],
        task["model2"],
        show_progress=False,
        seed=task["seed"],
        three_player=True,
        progress_label=task["progress_label"],
    )


def _compare_models_maybe_parallel(
    *,
    n: int,
    alpha: int,
    model1,
    model2,
    seed: int,
    progress_label: str,
    workers: int,
    verbose: bool,
) -> dict:
    if workers <= 1 or n <= 1:
        return compare_models_objectively(
            n,
            alpha,
            model1,
            model2,
            show_progress=verbose,
            seed=seed,
            three_player=True,
            progress_label=progress_label,
        )

    resolved_workers = _resolve_worker_count(workers=workers, work_items=n)
    started_at = time.perf_counter()
    progress = LiveProgressDisplay(
        verbose=verbose,
        label=progress_label,
        total=n,
        started_at=started_at,
    )
    progress.start(detail=f"workers={resolved_workers}")
    tasks = [
        {
            "n": 1,
            "alpha": alpha,
            "model1": model1,
            "model2": model2,
            "seed": seed + (index * 1_000),
            "progress_label": progress_label,
        }
        for index in range(n)
    ]
    merged_report: dict | None = None
    completed_games = 0
    executor, executor_kind = _create_parallel_executor(max_workers=resolved_workers)
    if executor_kind != "process":
        _log(verbose, f"[{progress_label}] falling back to {executor_kind} workers")
    with executor:
        futures = [executor.submit(_compare_models_worker, task) for task in tasks]
        for future in as_completed(futures):
            shard_report = future.result()
            merged_report = (
                shard_report
                if merged_report is None
                else _merge_evaluation_reports(merged_report, shard_report)
            )
            completed_games += shard_report["games_played"]
            progress.update(
                completed=completed_games,
                detail=f"workers={resolved_workers}",
            )
    progress.stop(completed=n, detail=f"workers={resolved_workers}")
    if merged_report is None:
        raise ValueError("parallel evaluation produced no reports")
    wall_elapsed_seconds = time.perf_counter() - started_at
    total_games = merged_report["games_played"]
    total_rounds = merged_report["average_rounds_played"] * total_games
    merged_report = {
        **merged_report,
        "elapsed_seconds": wall_elapsed_seconds,
        "average_seconds_per_game": (
            wall_elapsed_seconds / total_games if total_games > 0 else None
        ),
        "average_seconds_per_round": (
            wall_elapsed_seconds / total_rounds if total_rounds > 0 else None
        ),
        "games_per_second": (
            total_games / wall_elapsed_seconds if wall_elapsed_seconds > 0 else None
        ),
        "rounds_per_second": (
            total_rounds / wall_elapsed_seconds if wall_elapsed_seconds > 0 else None
        ),
    }
    return merged_report


def _build_cycle_snapshot(
    *,
    training_report: dict,
    evaluation_report: dict,
    evaluation_elapsed_seconds: float,
    cycle_elapsed_seconds: float,
) -> dict[str, object]:
    training_metrics = training_report["training"]
    snapshot = {
        "play_accuracy": training_metrics["play_history"][-1]["accuracy"],
        "auction_accuracy": training_metrics["auction_history"][-1]["accuracy"],
        "play_value_mse": training_metrics["play_value_history"][-1]["mse"],
        "auction_value_mse": training_metrics["auction_value_history"][-1]["mse"],
        "candidate_vs_incumbent": evaluation_report["vs_incumbent"]["models"]["candidate"]["win_percentage"],
        "candidate_vs_greedy": None,
        "candidate_vs_optimal": None,
        "training_elapsed_seconds": training_metrics.get("elapsed_seconds"),
        "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
        "cycle_elapsed_seconds": cycle_elapsed_seconds,
        "decision": "promoted" if evaluation_report["accepted"] else "kept",
    }
    if evaluation_report.get("vs_greedy") is not None:
        snapshot["candidate_vs_greedy"] = evaluation_report["vs_greedy"]["models"]["candidate"]["win_percentage"]
    if evaluation_report.get("vs_optimal_bot") is not None:
        snapshot["candidate_vs_optimal"] = evaluation_report["vs_optimal_bot"]["models"]["candidate"]["win_percentage"]
    return snapshot


def _candidate_score(evaluation_report: dict) -> float:
    candidate_vs_incumbent = evaluation_report["vs_incumbent"]["models"]["candidate"]["win_percentage"]
    candidate_vs_greedy = evaluation_report["vs_greedy"]["models"]["candidate"]["win_percentage"]
    candidate_vs_optimal = evaluation_report["vs_optimal_bot"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_greedy = evaluation_report["incumbent_vs_greedy"]["models"]["incumbent"]["win_percentage"]
    incumbent_vs_optimal = evaluation_report["incumbent_vs_optimal"]["models"]["incumbent"]["win_percentage"]
    return (
        (1.5 * candidate_vs_incumbent)
        + candidate_vs_greedy
        + (1.25 * candidate_vs_optimal)
        - (0.5 * incumbent_vs_greedy)
        - (0.75 * incumbent_vs_optimal)
    )


def candidate_meets_promotion_criteria(
    evaluation_report: dict,
    *,
    head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
) -> bool:
    candidate_vs_incumbent = evaluation_report["vs_incumbent"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_candidate = evaluation_report["vs_incumbent"]["models"]["incumbent"]["win_percentage"]
    candidate_vs_greedy = evaluation_report["vs_greedy"]["models"]["candidate"]["win_percentage"]
    candidate_vs_optimal = evaluation_report["vs_optimal_bot"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_greedy = evaluation_report["incumbent_vs_greedy"]["models"]["incumbent"]["win_percentage"]
    incumbent_vs_optimal = evaluation_report["incumbent_vs_optimal"]["models"]["incumbent"]["win_percentage"]
    return (
        _candidate_score(evaluation_report) > 0.0
        and candidate_vs_incumbent >= (incumbent_vs_candidate + head_to_head_margin)
        and candidate_vs_greedy >= (incumbent_vs_greedy - greedy_regression_tolerance)
        and candidate_vs_optimal >= (incumbent_vs_optimal - optimal_regression_tolerance)
    )


def _build_promotion_diagnostics(
    evaluation_report: dict,
    *,
    head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
) -> dict[str, object]:
    if evaluation_report["accepted"]:
        return {
            "summary": "promoted",
            "reasons": [],
        }

    if evaluation_report.get("rejected_after_precheck"):
        candidate_wp = evaluation_report["precheck_vs_incumbent"]["models"]["candidate"]["win_percentage"]
        incumbent_wp = evaluation_report["precheck_vs_incumbent"]["models"]["incumbent"]["win_percentage"]
        return {
            "summary": "not promoted: precheck failed",
            "reasons": [
                (
                    f"candidate {candidate_wp:.1f}% <= incumbent {incumbent_wp:.1f}% "
                    "in precheck"
                ),
            ],
        }

    if evaluation_report.get("rejected_after_head_to_head"):
        candidate_wp = evaluation_report["vs_incumbent"]["models"]["candidate"]["win_percentage"]
        incumbent_wp = evaluation_report["vs_incumbent"]["models"]["incumbent"]["win_percentage"]
        return {
            "summary": "not promoted: head-to-head failed",
            "reasons": [
                (
                    f"candidate {candidate_wp:.1f}% <= incumbent {incumbent_wp:.1f}% "
                    "after full head-to-head"
                ),
            ],
        }

    candidate_vs_incumbent = evaluation_report["vs_incumbent"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_candidate = evaluation_report["vs_incumbent"]["models"]["incumbent"]["win_percentage"]
    candidate_vs_greedy = evaluation_report["vs_greedy"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_greedy = evaluation_report["incumbent_vs_greedy"]["models"]["incumbent"]["win_percentage"]
    candidate_vs_optimal = evaluation_report["vs_optimal_bot"]["models"]["candidate"]["win_percentage"]
    incumbent_vs_optimal = evaluation_report["incumbent_vs_optimal"]["models"]["incumbent"]["win_percentage"]
    promotion_score = _candidate_score(evaluation_report)

    reasons: list[str] = []
    if promotion_score <= 0.0:
        reasons.append(f"promotion score {promotion_score:.1f} <= 0.0")

    required_head_to_head = incumbent_vs_candidate + head_to_head_margin
    if candidate_vs_incumbent < required_head_to_head:
        reasons.append(
            (
                f"head-to-head margin failed: {candidate_vs_incumbent:.1f}% "
                f"< required {required_head_to_head:.1f}% "
                f"({incumbent_vs_candidate:.1f}% + margin {head_to_head_margin:.1f})"
            )
        )

    required_vs_greedy = incumbent_vs_greedy - greedy_regression_tolerance
    if candidate_vs_greedy < required_vs_greedy:
        reasons.append(
            (
                f"greedy guard failed: {candidate_vs_greedy:.1f}% "
                f"< required {required_vs_greedy:.1f}% "
                f"(incumbent {incumbent_vs_greedy:.1f}% - tol {greedy_regression_tolerance:.1f})"
            )
        )

    required_vs_optimal = incumbent_vs_optimal - optimal_regression_tolerance
    if candidate_vs_optimal < required_vs_optimal:
        reasons.append(
            (
                f"optimal guard failed: {candidate_vs_optimal:.1f}% "
                f"< required {required_vs_optimal:.1f}% "
                f"(incumbent {incumbent_vs_optimal:.1f}% - tol {optimal_regression_tolerance:.1f})"
            )
        )

    return {
        "summary": "not promoted: promotion guard failed",
        "reasons": reasons or ["promotion criteria not met"],
    }


def _render_promotion_outcome_block(
    *,
    prefix: str,
    title: str,
    evaluation_report: dict,
    head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
) -> str:
    diagnostics = _build_promotion_diagnostics(
        evaluation_report,
        head_to_head_margin=head_to_head_margin,
        greedy_regression_tolerance=greedy_regression_tolerance,
        optimal_regression_tolerance=optimal_regression_tolerance,
    )
    rows = [("summary", str(diagnostics["summary"]))]
    for index, reason in enumerate(diagnostics["reasons"], start=1):
        rows.append((f"why {index}", reason))
    return _render_compact_block(
        prefix=prefix,
        title=title,
        rows=rows,
    )


def _reject_report_from_precheck(precheck_report: dict) -> dict:
    return {
        "accepted": False,
        "rejected_after_precheck": True,
        "rejected_after_head_to_head": False,
        "precheck_vs_incumbent": precheck_report,
        "vs_incumbent": precheck_report,
        "vs_greedy": None,
        "vs_optimal_bot": None,
        "incumbent_vs_greedy": None,
        "incumbent_vs_optimal": None,
    }


def _reject_report_from_head_to_head(
    *,
    precheck_report: dict,
    vs_incumbent: dict,
) -> dict:
    return {
        "accepted": False,
        "rejected_after_precheck": False,
        "rejected_after_head_to_head": True,
        "precheck_vs_incumbent": precheck_report,
        "vs_incumbent": vs_incumbent,
        "vs_greedy": None,
        "vs_optimal_bot": None,
        "incumbent_vs_greedy": None,
        "incumbent_vs_optimal": None,
    }


def _rename_report_model_key(
    report: dict,
    *,
    source_key: str,
    target_key: str,
    target_label: str | None = None,
) -> dict:
    renamed_report = deepcopy(report)
    models = renamed_report.get("models", {})
    if source_key not in models:
        raise KeyError(source_key)
    replacement_label = target_label or target_key

    renamed_models = {}
    for model_key, stats in models.items():
        if model_key == source_key:
            renamed_models[target_key] = {
                **stats,
                "label": replacement_label,
            }
        else:
            renamed_models[model_key] = stats
    renamed_report["models"] = renamed_models
    return renamed_report


def _build_incumbent_baseline_cache_from_evaluation(
    evaluation_report: dict,
    *,
    promoted_candidate: bool,
) -> dict[str, dict]:
    if promoted_candidate:
        return {
            "vs_greedy": _rename_report_model_key(
                evaluation_report["vs_greedy"],
                source_key="candidate",
                target_key="incumbent",
                target_label="incumbent",
            ),
            "vs_optimal": _rename_report_model_key(
                evaluation_report["vs_optimal_bot"],
                source_key="candidate",
                target_key="incumbent",
                target_label="incumbent",
            ),
        }

    return {
        "vs_greedy": evaluation_report["incumbent_vs_greedy"],
        "vs_optimal": evaluation_report["incumbent_vs_optimal"],
    }


def _merge_evaluation_reports(base_report: dict, extra_report: dict) -> dict:
    if base_report["models"].keys() != extra_report["models"].keys():
        raise ValueError("cannot merge evaluation reports with different model keys")

    total_games = base_report["games_played"] + extra_report["games_played"]
    total_elapsed = base_report["elapsed_seconds"] + extra_report["elapsed_seconds"]
    base_rounds = base_report["average_rounds_played"] * base_report["games_played"]
    extra_rounds = extra_report["average_rounds_played"] * extra_report["games_played"]
    total_rounds = base_rounds + extra_rounds
    merged_draws = base_report["draws"] + extra_report["draws"]

    merged_models = {}
    for model_key in base_report["models"]:
        merged_games_won = (
            base_report["models"][model_key]["games_won"]
            + extra_report["models"][model_key]["games_won"]
        )
        merged_models[model_key] = {
            **base_report["models"][model_key],
            "games_won": merged_games_won,
            "win_percentage": (merged_games_won / total_games) * 100,
        }

    return {
        **base_report,
        "games_played": total_games,
        "elapsed_seconds": total_elapsed,
        "average_seconds_per_game": total_elapsed / total_games,
        "average_seconds_per_round": (
            total_elapsed / total_rounds if total_rounds > 0 else None
        ),
        "games_per_second": (total_games / total_elapsed) if total_elapsed > 0 else None,
        "rounds_per_second": (total_rounds / total_elapsed) if total_elapsed > 0 else None,
        "draws": merged_draws,
        "draw_percentage": (merged_draws / total_games) * 100,
        "average_rounds_played": total_rounds / total_games,
        "models": merged_models,
    }


def _run_incumbent_baselines(
    *,
    incumbent_bundle: dict,
    alpha: int,
    eval_games: int,
    eval_games_vs_optimal: int,
    seed: int,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> dict:
    incumbent_factory = make_named_factory("incumbent", incumbent_bundle)
    return {
        "vs_greedy": _compare_models_maybe_parallel(
            n=eval_games,
            alpha=alpha,
            model1=incumbent_factory,
            model2="greedy",
            seed=seed + 3_000,
            progress_label="eval:incumbent-vs-greedy",
            workers=workers,
            verbose=verbose,
        ),
        "vs_optimal": _compare_models_maybe_parallel(
            n=eval_games_vs_optimal,
            alpha=alpha,
            model1=incumbent_factory,
            model2="optimal-bot",
            seed=seed + 4_000,
            progress_label="eval:incumbent-vs-optimal",
            workers=workers,
            verbose=verbose,
        ),
    }


def evaluate_candidate(
    *,
    candidate_bundle: dict,
    incumbent_bundle: dict,
    alpha: int,
    eval_games: int,
    eval_games_vs_optimal: int,
    precheck_games: int,
    seed: int,
    incumbent_baselines: dict | None = None,
    promotion_head_to_head_margin: float = DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
    promotion_greedy_regression_tolerance: float = DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
    promotion_optimal_regression_tolerance: float = DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> dict:
    candidate_factory = make_named_factory("candidate", candidate_bundle)
    incumbent_factory = make_named_factory("incumbent", incumbent_bundle)

    _log(
        verbose,
        (
            f"[eval] precheck candidate_vs_incumbent games={precheck_games} "
            f"full_eval_games={eval_games}"
        ),
    )
    precheck_vs_incumbent = _compare_models_maybe_parallel(
        n=precheck_games,
        alpha=alpha,
        model1=candidate_factory,
        model2=incumbent_factory,
        seed=seed,
        progress_label="eval:precheck",
        workers=workers,
        verbose=verbose,
    )
    precheck_candidate_wp = precheck_vs_incumbent["models"]["candidate"]["win_percentage"]
    precheck_incumbent_wp = precheck_vs_incumbent["models"]["incumbent"]["win_percentage"]
    if precheck_candidate_wp <= precheck_incumbent_wp:
        rejection_report = _reject_report_from_precheck(precheck_vs_incumbent)
        rejection_report["promotion_diagnostics"] = _build_promotion_diagnostics(
            rejection_report,
            head_to_head_margin=promotion_head_to_head_margin,
            greedy_regression_tolerance=promotion_greedy_regression_tolerance,
            optimal_regression_tolerance=promotion_optimal_regression_tolerance,
        )
        return rejection_report

    _log(verbose, "[eval] full head-to-head candidate_vs_incumbent")
    if eval_games <= precheck_games:
        vs_incumbent = precheck_vs_incumbent
    else:
        remaining_vs_incumbent = _compare_models_maybe_parallel(
            n=eval_games - precheck_games,
            alpha=alpha,
            model1=candidate_factory,
            model2=incumbent_factory,
            seed=seed + 50_000,
            progress_label="eval:head-to-head",
            workers=workers,
            verbose=verbose,
        )
        vs_incumbent = _merge_evaluation_reports(
            precheck_vs_incumbent,
            remaining_vs_incumbent,
        )

    candidate_incumbent_wp = vs_incumbent["models"]["candidate"]["win_percentage"]
    incumbent_wp = vs_incumbent["models"]["incumbent"]["win_percentage"]
    if candidate_incumbent_wp <= incumbent_wp:
        rejection_report = _reject_report_from_head_to_head(
            precheck_report=precheck_vs_incumbent,
            vs_incumbent=vs_incumbent,
        )
        rejection_report["promotion_diagnostics"] = _build_promotion_diagnostics(
            rejection_report,
            head_to_head_margin=promotion_head_to_head_margin,
            greedy_regression_tolerance=promotion_greedy_regression_tolerance,
            optimal_regression_tolerance=promotion_optimal_regression_tolerance,
        )
        return rejection_report

    _log(verbose, "[eval] candidate baselines vs greedy and optimal-bot")
    vs_greedy = _compare_models_maybe_parallel(
        n=eval_games,
        alpha=alpha,
        model1=candidate_factory,
        model2="greedy",
        seed=seed + 1_000,
        progress_label="eval:candidate-vs-greedy",
        workers=workers,
        verbose=verbose,
    )
    vs_optimal = _compare_models_maybe_parallel(
        n=eval_games_vs_optimal,
        alpha=alpha,
        model1=candidate_factory,
        model2="optimal-bot",
        seed=seed + 2_000,
        progress_label="eval:candidate-vs-optimal",
        workers=workers,
        verbose=verbose,
    )
    if incumbent_baselines is None:
        _log(verbose, "[eval] building incumbent baseline cache")
        incumbent_baselines = _run_incumbent_baselines(
            incumbent_bundle=incumbent_bundle,
            alpha=alpha,
            eval_games=eval_games,
            eval_games_vs_optimal=eval_games_vs_optimal,
            seed=seed,
            workers=workers,
            verbose=verbose,
        )
    incumbent_vs_greedy = incumbent_baselines["vs_greedy"]
    incumbent_vs_optimal = incumbent_baselines["vs_optimal"]

    promotion_report = {
        "vs_incumbent": vs_incumbent,
        "vs_greedy": vs_greedy,
        "vs_optimal_bot": vs_optimal,
        "incumbent_vs_greedy": incumbent_vs_greedy,
        "incumbent_vs_optimal": incumbent_vs_optimal,
    }
    accepted = candidate_meets_promotion_criteria(
        promotion_report,
        head_to_head_margin=promotion_head_to_head_margin,
        greedy_regression_tolerance=promotion_greedy_regression_tolerance,
        optimal_regression_tolerance=promotion_optimal_regression_tolerance,
    )

    promotion_report_with_status = {
        "accepted": accepted,
        "rejected_after_precheck": False,
        "rejected_after_head_to_head": False,
        "precheck_vs_incumbent": precheck_vs_incumbent,
        "vs_incumbent": vs_incumbent,
        "vs_greedy": vs_greedy,
        "vs_optimal_bot": vs_optimal,
        "incumbent_vs_greedy": incumbent_vs_greedy,
        "incumbent_vs_optimal": incumbent_vs_optimal,
    }
    promotion_report_with_status["promotion_diagnostics"] = _build_promotion_diagnostics(
        promotion_report_with_status,
        head_to_head_margin=promotion_head_to_head_margin,
        greedy_regression_tolerance=promotion_greedy_regression_tolerance,
        optimal_regression_tolerance=promotion_optimal_regression_tolerance,
    )
    return promotion_report_with_status


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated mixed-teacher + DAgger cycles to train the v2 3-player neural smear bot."
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=8.0,
        help="wall-clock training budget in hours",
    )
    parser.add_argument(
        "--teacher-pool",
        default=DEFAULT_TEACHER_POOL,
        help="teacher mixture using bot_id[:weight], for example 'optimal-bot:4,1-trick-minmax:2,greedy:1'",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="optional initial checkpoint; defaults to neural_3p_v2.json when present, otherwise v1",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="directory where the v2 self-train run folder should be created",
    )
    parser.add_argument(
        "--promote-to",
        type=Path,
        default=None,
        help="optional checkpoint path to overwrite with the best promoted model at the end",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=12)
    parser.add_argument("--bootstrap-matches", type=int, default=24)
    parser.add_argument("--dagger-matches", type=int, default=16)
    parser.add_argument("--dagger-iterations", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=36)
    parser.add_argument("--eval-games-vs-optimal", type=int, default=18)
    parser.add_argument(
        "--precheck-games",
        type=int,
        default=12,
        help="cheap candidate-vs-incumbent gate before running the full evaluation suite",
    )
    parser.add_argument(
        "--promotion-head-to-head-margin",
        type=float,
        default=DEFAULT_PROMOTION_HEAD_TO_HEAD_MARGIN,
        help="minimum candidate lead in head-to-head win percentage points required for promotion",
    )
    parser.add_argument(
        "--promotion-greedy-regression-tolerance",
        type=float,
        default=DEFAULT_PROMOTION_GREEDY_REGRESSION_TOLERANCE,
        help="maximum allowed regression in candidate-vs-greedy win percentage points",
    )
    parser.add_argument(
        "--promotion-optimal-regression-tolerance",
        type=float,
        default=DEFAULT_PROMOTION_OPTIMAL_REGRESSION_TOLERANCE,
        help="maximum allowed regression in candidate-vs-optimal win percentage points",
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--play-hidden-dim", type=int, default=DEFAULT_PLAY_HIDDEN_DIM)
    parser.add_argument("--auction-hidden-dim", type=int, default=DEFAULT_AUCTION_HIDDEN_DIM)
    parser.add_argument("--play-value-hidden-dim", type=int, default=DEFAULT_PLAY_VALUE_HIDDEN_DIM)
    parser.add_argument("--auction-value-hidden-dim", type=int, default=DEFAULT_AUCTION_VALUE_HIDDEN_DIM)
    parser.add_argument("--play-epochs", type=int, default=12)
    parser.add_argument("--auction-epochs", type=int, default=12)
    parser.add_argument("--play-value-epochs", type=int, default=14)
    parser.add_argument("--auction-value-epochs", type=int, default=14)
    parser.add_argument("--play-learning-rate", type=float, default=0.026)
    parser.add_argument("--auction-learning-rate", type=float, default=0.032)
    parser.add_argument("--play-value-learning-rate", type=float, default=0.022)
    parser.add_argument("--auction-value-learning-rate", type=float, default=0.026)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--warm-start-lr-scale", type=float, default=DEFAULT_WARM_START_LR_SCALE)
    parser.add_argument(
        "--warm-start-epoch-scale",
        type=float,
        default=DEFAULT_WARM_START_EPOCH_SCALE,
    )
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument(
        "--student-agreement-keep-prob",
        type=float,
        default=DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    )
    parser.add_argument(
        "--teacher-sample-scale",
        type=float,
        default=DEFAULT_TEACHER_SAMPLE_SCALE,
        help="scale search-teacher determinization sample counts during training collection",
    )
    parser.add_argument(
        "--teacher-target-temperature",
        type=float,
        default=DEFAULT_TEACHER_TARGET_TEMPERATURE,
        help="softmax temperature used when distilling teacher search scores into policy targets",
    )
    parser.add_argument(
        "--trainer-backend",
        default=DEFAULT_TRAINING_BACKEND,
        choices=TRAINING_BACKEND_CHOICES,
        help="training backend to use for model fitting",
    )
    parser.add_argument(
        "--trainer-batch-size",
        type=int,
        default=DEFAULT_TRAINER_BATCH_SIZE,
        help="mini-batch size for the PyTorch training backend",
    )
    parser.add_argument("--play-value-weight", type=float, default=0.8)
    parser.add_argument("--auction-value-weight", type=float, default=0.55)
    parser.add_argument("--play-rollout-depth", type=int, default=DEFAULT_PLAY_ROLLOUT_DEPTH)
    parser.add_argument("--auction-rollout-depth", type=int, default=DEFAULT_AUCTION_ROLLOUT_DEPTH)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="number of worker processes for rollout collection, evaluation, and model training",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress live cycle logs and only print the final JSON summary",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    teacher_specs = parse_teacher_specs(args.teacher_pool)
    initial_model_path = resolve_initial_model_path(args.initial_model)
    current_best_bundle = load_model_bundle(initial_model_path)
    verbose = not args.quiet

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / f"neural-3p-train-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.json"
    report_path = run_dir / "report.jsonl"
    manifest_path = run_dir / "manifest.json"
    save_model_bundle(current_best_bundle, best_path)
    manifest_path.write_text(
        json.dumps(
            {
                "started_at": timestamp,
                "initial_model_path": str(initial_model_path),
                "teacher_pool": args.teacher_pool,
                "duration_hours": args.duration_hours,
                "alpha": args.alpha,
                "bootstrap_matches": args.bootstrap_matches,
                "dagger_matches": args.dagger_matches,
                "dagger_iterations": args.dagger_iterations,
                "eval_games": args.eval_games,
                "eval_games_vs_optimal": args.eval_games_vs_optimal,
                "precheck_games": args.precheck_games,
                "promotion_head_to_head_margin": args.promotion_head_to_head_margin,
                "promotion_greedy_regression_tolerance": args.promotion_greedy_regression_tolerance,
                "promotion_optimal_regression_tolerance": args.promotion_optimal_regression_tolerance,
                "teacher_sample_scale": args.teacher_sample_scale,
                "teacher_target_temperature": args.teacher_target_temperature,
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
            prefix="[self-train]",
            title="run",
            rows=[
                ("run dir", str(run_dir)),
                ("initial", str(initial_model_path)),
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
                        f"eval {args.eval_games} | optimal {args.eval_games_vs_optimal} | precheck {args.precheck_games}"
                    ),
                ),
                (
                    "promote",
                    (
                        f"margin {args.promotion_head_to_head_margin:.1f} | "
                        f"greedy tol {args.promotion_greedy_regression_tolerance:.1f} | "
                        f"optimal tol {args.promotion_optimal_regression_tolerance:.1f}"
                    ),
                ),
                (
                    "search",
                    (
                        f"sample scale {args.teacher_sample_scale:.2f} | "
                        f"target temp {args.teacher_target_temperature:.2f}"
                    ),
                ),
            ],
        ),
    )

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    cycle_index = 1
    incumbent_baseline_cache: dict | None = None
    previous_cycle_snapshot: dict[str, object] | None = None
    _log(verbose, "[self-train] incumbent baseline cache will be built lazily")

    while True:
        if args.max_cycles is not None and cycle_index > args.max_cycles:
            break
        if time.time() >= deadline and cycle_index > 1:
            break

        cycle_seed = args.seed + (cycle_index * 10_000)
        _log(
            verbose,
            _render_compact_block(
                prefix="[self-train]",
                title=f"cycle {cycle_index}",
                rows=[
                    ("state", "start"),
                    ("seed", str(cycle_seed)),
                    ("elapsed", f"{(time.time() - started_at) / 3600.0:.2f}h"),
                    ("left", _format_seconds(max(0.0, deadline - time.time()))),
                ],
            ),
        )
        cycle_started_at = time.perf_counter()
        try:
            candidate_bundle, training_report = train_with_dagger(
                teacher_specs=teacher_specs,
                bootstrap_matches=args.bootstrap_matches,
                dagger_matches=args.dagger_matches,
                alpha=args.alpha,
                dagger_iterations=args.dagger_iterations,
                seed=cycle_seed,
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
                initial_bundle=current_best_bundle,
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
                trainer_backend=args.trainer_backend,
                trainer_batch_size=args.trainer_batch_size,
                workers=args.workers,
                bot_id="neural-3p-v2",
                verbose=verbose,
            )
        except FloatingPointError as exc:
            _log(verbose, f"[self-train] cycle {cycle_index} aborted due to non-finite training: {exc}")
            cycle_report = {
                "cycle": cycle_index,
                "elapsed_hours": (time.time() - started_at) / 3600.0,
                "accepted": False,
                "error": str(exc),
            }
            with report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(cycle_report) + "\n")
            cycle_index += 1
            continue

        candidate_path = checkpoints_dir / f"cycle-{cycle_index:03d}.json"
        save_model_bundle(candidate_bundle, candidate_path)
        _log(
            verbose,
            _render_compact_block(
                prefix="[self-train]",
                title=f"cycle {cycle_index} trained",
                rows=[
                    ("checkpoint", str(candidate_path)),
                    (
                        "metrics",
                        (
                            f"play_acc={training_report['training']['play_history'][-1]['accuracy']:.3f} "
                            f"auction_acc={training_report['training']['auction_history'][-1]['accuracy']:.3f} "
                            f"play_value_mse={training_report['training']['play_value_history'][-1]['mse']:.4f} "
                            f"auction_value_mse={training_report['training']['auction_value_history'][-1]['mse']:.4f}"
                        ),
                    ),
                    (
                        "elapsed",
                        _format_seconds(training_report["training"].get("elapsed_seconds", 0.0)),
                    ),
                ],
            ),
        )
        evaluation_started_at = time.perf_counter()
        baseline_cache_was_empty = incumbent_baseline_cache is None
        evaluation_report = evaluate_candidate(
            candidate_bundle=candidate_bundle,
            incumbent_bundle=current_best_bundle,
            alpha=args.alpha,
            eval_games=args.eval_games,
            eval_games_vs_optimal=args.eval_games_vs_optimal,
            precheck_games=min(args.precheck_games, args.eval_games),
            seed=cycle_seed,
            incumbent_baselines=incumbent_baseline_cache,
            promotion_head_to_head_margin=args.promotion_head_to_head_margin,
            promotion_greedy_regression_tolerance=args.promotion_greedy_regression_tolerance,
            promotion_optimal_regression_tolerance=args.promotion_optimal_regression_tolerance,
            workers=args.workers,
            verbose=verbose,
        )
        evaluation_elapsed = time.perf_counter() - evaluation_started_at
        if (
            baseline_cache_was_empty
            and evaluation_report["incumbent_vs_greedy"] is not None
            and evaluation_report["incumbent_vs_optimal"] is not None
        ):
            incumbent_baseline_cache = _build_incumbent_baseline_cache_from_evaluation(
                evaluation_report,
                promoted_candidate=False,
            )
            _log(
                verbose,
                (
                    "[self-train] incumbent baseline cache populated "
                    f"incumbent_vs_greedy={incumbent_baseline_cache['vs_greedy']['models']['incumbent']['win_percentage']:.1f}% "
                    f"incumbent_vs_optimal={incumbent_baseline_cache['vs_optimal']['models']['incumbent']['win_percentage']:.1f}%"
                ),
            )
        _log(
            verbose,
            _render_compact_block(
                prefix="[self-train]",
                title=f"cycle {cycle_index} eval",
                rows=[
                    ("summary", _format_eval_summary(evaluation_report)),
                    ("elapsed", _format_seconds(evaluation_elapsed)),
                ],
            ),
        )

        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
            incumbent_baseline_cache = _build_incumbent_baseline_cache_from_evaluation(
                evaluation_report,
                promoted_candidate=True,
            )
            save_model_bundle(current_best_bundle, best_path)
            _log(
                verbose,
                f"[self-train] cycle {cycle_index} promoted candidate to {best_path}",
            )
        else:
            _log(verbose, f"[self-train] cycle {cycle_index} kept incumbent")
            _log(
                verbose,
                _render_promotion_outcome_block(
                    prefix="[self-train]",
                    title=f"cycle {cycle_index} promotion",
                    evaluation_report=evaluation_report,
                    head_to_head_margin=args.promotion_head_to_head_margin,
                    greedy_regression_tolerance=args.promotion_greedy_regression_tolerance,
                    optimal_regression_tolerance=args.promotion_optimal_regression_tolerance,
                ),
            )

        cycle_elapsed_seconds = time.perf_counter() - cycle_started_at
        cycle_snapshot = _build_cycle_snapshot(
            training_report=training_report,
            evaluation_report=evaluation_report,
            evaluation_elapsed_seconds=evaluation_elapsed,
            cycle_elapsed_seconds=cycle_elapsed_seconds,
        )
        _log(
            verbose,
            _render_iteration_comparison_table(
                prefix="[self-train]",
                title=f"cycle {cycle_index} comparison",
                current_snapshot=cycle_snapshot,
                previous_snapshot=previous_cycle_snapshot,
                metrics=TRAINING_CYCLE_METRICS,
                footer_note="compares this candidate cycle against the previous self-train cycle",
            ),
        )
        previous_cycle_snapshot = cycle_snapshot

        cycle_report = {
            "cycle": cycle_index,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
            "candidate_path": str(candidate_path),
            "accepted": evaluation_report["accepted"],
            "training": training_report,
            "evaluation": evaluation_report,
            "timing": {
                "cycle_elapsed_seconds": cycle_elapsed_seconds,
                "training_elapsed_seconds": training_report["training"].get("elapsed_seconds"),
                "evaluation_elapsed_seconds": evaluation_elapsed,
            },
        }
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_report) + "\n")

        cycle_index += 1

    if args.promote_to is not None:
        save_model_bundle(current_best_bundle, args.promote_to)
        _log(verbose, f"[self-train] wrote promoted best model to {args.promote_to}")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_model": str(best_path),
                "promoted_to": str(args.promote_to) if args.promote_to is not None else None,
                "cycles_completed": cycle_index - 1,
            }
        )
    )


if __name__ == "__main__":
    main()
