from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot
    from .simulator import compare_models_objectively
    from .train_neural_3p_bot import (
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TEACHER_POOL,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WARM_START_EPOCH_SCALE,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot
    from simulator import compare_models_objectively
    from train_neural_3p_bot import (
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_GRADIENT_CLIP,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        DEFAULT_TEACHER_POOL,
        DEFAULT_WARM_START_LR_SCALE,
        DEFAULT_WARM_START_EPOCH_SCALE,
        load_model_bundle,
        parse_teacher_specs,
        save_model_bundle,
        train_with_dagger,
    )


DEFAULT_RUN_ROOT = NeuralThreePlayerBot.MODEL_DIR / "self_train_runs"


def resolve_initial_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if NeuralThreePlayerBot.MODEL_FILE.exists():
        return NeuralThreePlayerBot.MODEL_FILE
    return NeuralThreePlayerBot.MODEL_FILE_V1


def make_named_factory(label: str, bundle: dict):
    def _factory(player_name: str):
        return NeuralThreePlayerBot(player_name, model_bundle=bundle)

    _factory.__name__ = label
    return _factory


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
    verbose: bool = False,
) -> dict:
    incumbent_factory = make_named_factory("incumbent", incumbent_bundle)
    return {
        "vs_greedy": compare_models_objectively(
            eval_games,
            alpha,
            incumbent_factory,
            "greedy",
            show_progress=verbose,
            seed=seed + 3_000,
            three_player=True,
            progress_label="eval:incumbent-vs-greedy",
        ),
        "vs_optimal": compare_models_objectively(
            eval_games_vs_optimal,
            alpha,
            incumbent_factory,
            "optimal-bot",
            show_progress=verbose,
            seed=seed + 4_000,
            three_player=True,
            progress_label="eval:incumbent-vs-optimal",
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
    precheck_vs_incumbent = compare_models_objectively(
        precheck_games,
        alpha,
        candidate_factory,
        incumbent_factory,
        show_progress=verbose,
        seed=seed,
        three_player=True,
        progress_label="eval:precheck",
    )
    precheck_candidate_wp = precheck_vs_incumbent["models"]["candidate"]["win_percentage"]
    precheck_incumbent_wp = precheck_vs_incumbent["models"]["incumbent"]["win_percentage"]
    if precheck_candidate_wp <= precheck_incumbent_wp:
        return _reject_report_from_precheck(precheck_vs_incumbent)

    _log(verbose, "[eval] full head-to-head candidate_vs_incumbent")
    if eval_games <= precheck_games:
        vs_incumbent = precheck_vs_incumbent
    else:
        remaining_vs_incumbent = compare_models_objectively(
            eval_games - precheck_games,
            alpha,
            candidate_factory,
            incumbent_factory,
            show_progress=verbose,
            seed=seed + 50_000,
            three_player=True,
            progress_label="eval:head-to-head",
        )
        vs_incumbent = _merge_evaluation_reports(
            precheck_vs_incumbent,
            remaining_vs_incumbent,
        )

    candidate_incumbent_wp = vs_incumbent["models"]["candidate"]["win_percentage"]
    incumbent_wp = vs_incumbent["models"]["incumbent"]["win_percentage"]
    if candidate_incumbent_wp <= incumbent_wp:
        return _reject_report_from_head_to_head(
            precheck_report=precheck_vs_incumbent,
            vs_incumbent=vs_incumbent,
        )

    _log(verbose, "[eval] candidate baselines vs greedy and optimal-bot")
    vs_greedy = compare_models_objectively(
        eval_games,
        alpha,
        candidate_factory,
        "greedy",
        show_progress=verbose,
        seed=seed + 1_000,
        three_player=True,
        progress_label="eval:candidate-vs-greedy",
    )
    vs_optimal = compare_models_objectively(
        eval_games_vs_optimal,
        alpha,
        candidate_factory,
        "optimal-bot",
        show_progress=verbose,
        seed=seed + 2_000,
        three_player=True,
        progress_label="eval:candidate-vs-optimal",
    )
    if incumbent_baselines is None:
        _log(verbose, "[eval] building incumbent baseline cache")
        incumbent_baselines = _run_incumbent_baselines(
            incumbent_bundle=incumbent_bundle,
            alpha=alpha,
            eval_games=eval_games,
            eval_games_vs_optimal=eval_games_vs_optimal,
            seed=seed,
            verbose=verbose,
        )
    incumbent_vs_greedy = incumbent_baselines["vs_greedy"]
    incumbent_vs_optimal = incumbent_baselines["vs_optimal"]

    candidate_greedy_wp = vs_greedy["models"]["candidate"]["win_percentage"]
    candidate_optimal_wp = vs_optimal["models"]["candidate"]["win_percentage"]
    incumbent_greedy_wp = incumbent_vs_greedy["models"]["incumbent"]["win_percentage"]
    incumbent_optimal_wp = incumbent_vs_optimal["models"]["incumbent"]["win_percentage"]
    accepted = (
        _candidate_score(
            {
                "vs_incumbent": vs_incumbent,
                "vs_greedy": vs_greedy,
                "vs_optimal_bot": vs_optimal,
                "incumbent_vs_greedy": incumbent_vs_greedy,
                "incumbent_vs_optimal": incumbent_vs_optimal,
            }
        )
        > 0.0
        and candidate_incumbent_wp > incumbent_wp
        and candidate_greedy_wp >= (incumbent_greedy_wp - 5.0)
        and candidate_optimal_wp >= (incumbent_optimal_wp - 5.0)
    )

    return {
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated mixed-teacher + DAgger cycles to self-train the 3-player neural smear bot."
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
        help="directory where the overnight run folder should be created",
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
    parser.add_argument("--play-value-weight", type=float, default=0.8)
    parser.add_argument("--auction-value-weight", type=float, default=0.55)
    parser.add_argument("--play-rollout-depth", type=int, default=DEFAULT_PLAY_ROLLOUT_DEPTH)
    parser.add_argument("--auction-rollout-depth", type=int, default=DEFAULT_AUCTION_ROLLOUT_DEPTH)
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
    run_dir = args.run_root / f"neural-3p-overnight-{timestamp}"
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
                "seed": args.seed,
            }
        ),
        encoding="utf-8",
    )
    _log(
        verbose,
        (
            f"[overnight] run_dir={run_dir} initial_model={initial_model_path} "
            f"teacher_pool={args.teacher_pool}"
        ),
    )
    _log(
        verbose,
        (
            f"[overnight] config duration={args.duration_hours:.2f}h alpha={args.alpha} "
            f"bootstrap_matches={args.bootstrap_matches} dagger_matches={args.dagger_matches} "
            f"dagger_iterations={args.dagger_iterations} eval_games={args.eval_games} "
            f"eval_games_vs_optimal={args.eval_games_vs_optimal} precheck_games={args.precheck_games}"
        ),
    )

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    cycle_index = 1
    incumbent_baseline_cache: dict | None = None
    _log(verbose, "[overnight] incumbent baseline cache will be built lazily")

    while True:
        if args.max_cycles is not None and cycle_index > args.max_cycles:
            break
        if time.time() >= deadline and cycle_index > 1:
            break

        cycle_seed = args.seed + (cycle_index * 10_000)
        _log(
            verbose,
            (
                f"[overnight] cycle {cycle_index} start seed={cycle_seed} "
                f"elapsed_hours={(time.time() - started_at) / 3600.0:.2f} "
                f"time_left={_format_seconds(max(0.0, deadline - time.time()))}"
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
                bot_id="neural-3p-v2",
                verbose=verbose,
            )
        except FloatingPointError as exc:
            _log(verbose, f"[overnight] cycle {cycle_index} aborted due to non-finite training: {exc}")
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
            (
                f"[overnight] cycle {cycle_index} trained checkpoint={candidate_path} "
                f"play_acc={training_report['training']['play_history'][-1]['accuracy']:.3f} "
                f"auction_acc={training_report['training']['auction_history'][-1]['accuracy']:.3f} "
                f"play_value_mse={training_report['training']['play_value_history'][-1]['mse']:.4f} "
                f"auction_value_mse={training_report['training']['auction_value_history'][-1]['mse']:.4f} "
                f"train_elapsed={_format_seconds(training_report['training'].get('elapsed_seconds', 0.0))}"
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
            verbose=verbose,
        )
        evaluation_elapsed = time.perf_counter() - evaluation_started_at
        if (
            baseline_cache_was_empty
            and evaluation_report["incumbent_vs_greedy"] is not None
            and evaluation_report["incumbent_vs_optimal"] is not None
        ):
            incumbent_baseline_cache = {
                "vs_greedy": evaluation_report["incumbent_vs_greedy"],
                "vs_optimal": evaluation_report["incumbent_vs_optimal"],
            }
            _log(
                verbose,
                (
                    "[overnight] incumbent baseline cache populated "
                    f"incumbent_vs_greedy={incumbent_baseline_cache['vs_greedy']['models']['incumbent']['win_percentage']:.1f}% "
                    f"incumbent_vs_optimal={incumbent_baseline_cache['vs_optimal']['models']['incumbent']['win_percentage']:.1f}%"
                ),
            )
        _log(
            verbose,
            (
                f"[overnight] cycle {cycle_index} eval {_format_eval_summary(evaluation_report)} "
                f"eval_elapsed={_format_seconds(evaluation_elapsed)}"
            ),
        )

        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
            incumbent_baseline_cache = {
                "vs_greedy": evaluation_report["vs_greedy"],
                "vs_optimal": evaluation_report["vs_optimal_bot"],
            }
            save_model_bundle(current_best_bundle, best_path)
            _log(
                verbose,
                f"[overnight] cycle {cycle_index} promoted candidate to {best_path}",
            )
        else:
            _log(verbose, f"[overnight] cycle {cycle_index} kept incumbent")

        cycle_report = {
            "cycle": cycle_index,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
            "candidate_path": str(candidate_path),
            "accepted": evaluation_report["accepted"],
            "training": training_report,
            "evaluation": evaluation_report,
            "timing": {
                "cycle_elapsed_seconds": time.perf_counter() - cycle_started_at,
                "training_elapsed_seconds": training_report["training"].get("elapsed_seconds"),
                "evaluation_elapsed_seconds": evaluation_elapsed,
            },
        }
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_report) + "\n")

        cycle_index += 1

    if args.promote_to is not None:
        save_model_bundle(current_best_bundle, args.promote_to)
        _log(verbose, f"[overnight] wrote promoted best model to {args.promote_to}")

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
