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
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_TEACHER_POOL,
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
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_TEACHER_POOL,
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


def _format_eval_summary(evaluation_report: dict) -> str:
    vs_incumbent = evaluation_report["vs_incumbent"]["models"]
    vs_greedy = evaluation_report["vs_greedy"]["models"]
    vs_optimal = evaluation_report["vs_optimal_bot"]["models"]
    return (
        f"candidate_vs_incumbent={vs_incumbent['candidate']['win_percentage']:.1f}%/"
        f"{vs_incumbent['incumbent']['win_percentage']:.1f}% "
        f"candidate_vs_greedy={vs_greedy['candidate']['win_percentage']:.1f}% "
        f"candidate_vs_optimal={vs_optimal['candidate']['win_percentage']:.1f}%"
    )


def evaluate_candidate(
    *,
    candidate_bundle: dict,
    incumbent_bundle: dict,
    alpha: int,
    eval_games: int,
    eval_games_vs_optimal: int,
    seed: int,
) -> dict:
    candidate_factory = make_named_factory("candidate", candidate_bundle)
    incumbent_factory = make_named_factory("incumbent", incumbent_bundle)

    vs_incumbent = compare_models_objectively(
        eval_games,
        alpha,
        candidate_factory,
        incumbent_factory,
        show_progress=False,
        seed=seed,
        three_player=True,
    )
    vs_greedy = compare_models_objectively(
        eval_games,
        alpha,
        candidate_factory,
        "greedy",
        show_progress=False,
        seed=seed + 1_000,
        three_player=True,
    )
    vs_optimal = compare_models_objectively(
        eval_games_vs_optimal,
        alpha,
        candidate_factory,
        "optimal-bot",
        show_progress=False,
        seed=seed + 2_000,
        three_player=True,
    )

    candidate_incumbent_wp = vs_incumbent["models"]["candidate"]["win_percentage"]
    incumbent_wp = vs_incumbent["models"]["incumbent"]["win_percentage"]
    accepted = candidate_incumbent_wp > incumbent_wp

    return {
        "accepted": accepted,
        "vs_incumbent": vs_incumbent,
        "vs_greedy": vs_greedy,
        "vs_optimal_bot": vs_optimal,
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
    parser.add_argument("--play-value-weight", type=float, default=0.8)
    parser.add_argument("--auction-value-weight", type=float, default=0.55)
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

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    cycle_index = 1

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
                f"elapsed_hours={(time.time() - started_at) / 3600.0:.2f}"
            ),
        )
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
            bot_id="neural-3p-v2",
            verbose=verbose,
        )

        candidate_path = checkpoints_dir / f"cycle-{cycle_index:03d}.json"
        save_model_bundle(candidate_bundle, candidate_path)
        _log(
            verbose,
            (
                f"[overnight] cycle {cycle_index} trained checkpoint={candidate_path} "
                f"play_acc={training_report['training']['play_history'][-1]['accuracy']:.3f} "
                f"auction_acc={training_report['training']['auction_history'][-1]['accuracy']:.3f} "
                f"play_value_mse={training_report['training']['play_value_history'][-1]['mse']:.4f} "
                f"auction_value_mse={training_report['training']['auction_value_history'][-1]['mse']:.4f}"
            ),
        )
        evaluation_report = evaluate_candidate(
            candidate_bundle=candidate_bundle,
            incumbent_bundle=current_best_bundle,
            alpha=args.alpha,
            eval_games=args.eval_games,
            eval_games_vs_optimal=args.eval_games_vs_optimal,
            seed=cycle_seed,
        )
        _log(
            verbose,
            f"[overnight] cycle {cycle_index} eval {_format_eval_summary(evaluation_report)}",
        )

        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
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
