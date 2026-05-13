from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

try:
    from .bots.neural_3p_bot import NeuralThreePlayerBot
    from .bots.neural_3p_features import (
        encode_auction_state,
        encode_play_state,
    )
    from .bots.neural_model import ChoiceExample, RegressionExample
    from .engine import apply_auction_action_for_search, apply_trick_action_to_state
    from .gameplay import MatchController
    from .models import Play
    from .self_train_neural_3p_overnight import evaluate_candidate
    from .train_neural_3p_bot import (
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_WEIGHT,
        LiveProgressDisplay,
        PendingValueExample,
        TrainingDataset,
        _format_seconds,
        _format_training_metrics,
        _resolve_round_value_targets,
        load_model_bundle,
        ordered_legal_auction_actions,
        ordered_legal_cards,
        save_model_bundle,
        train_neural_3p_bundle,
    )
except ImportError:
    from bots.neural_3p_bot import NeuralThreePlayerBot
    from bots.neural_3p_features import (
        encode_auction_state,
        encode_play_state,
    )
    from bots.neural_model import ChoiceExample, RegressionExample
    from engine import apply_auction_action_for_search, apply_trick_action_to_state
    from gameplay import MatchController
    from models import Play
    from self_train_neural_3p_overnight import evaluate_candidate
    from train_neural_3p_bot import (
        DEFAULT_AUCTION_HIDDEN_DIM,
        DEFAULT_AUCTION_ROLLOUT_DEPTH,
        DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
        DEFAULT_AUCTION_VALUE_WEIGHT,
        DEFAULT_PLAY_HIDDEN_DIM,
        DEFAULT_PLAY_ROLLOUT_DEPTH,
        DEFAULT_PLAY_VALUE_HIDDEN_DIM,
        DEFAULT_PLAY_VALUE_WEIGHT,
        LiveProgressDisplay,
        PendingValueExample,
        TrainingDataset,
        _format_seconds,
        _format_training_metrics,
        _resolve_round_value_targets,
        load_model_bundle,
        ordered_legal_auction_actions,
        ordered_legal_cards,
        save_model_bundle,
        train_neural_3p_bundle,
    )


DEFAULT_OUTPUT = NeuralThreePlayerBot.MODEL_FILE_V3
DEFAULT_RUN_ROOT = NeuralThreePlayerBot.MODEL_DIR / "self_play_runs"
DEFAULT_SELF_PLAY_MATCHES = 24
DEFAULT_SELF_PLAY_ITERATIONS = 4
DEFAULT_REPLAY_WINDOW = 4
DEFAULT_POLICY_ADVANTAGE_THRESHOLD = 0.02
DEFAULT_POLICY_ADVANTAGE_SCALE = 6.0
DEFAULT_VALUE_WEIGHT_SCALE = 2.0
DEFAULT_WINNER_POLICY_WEIGHT = 1.5
DEFAULT_PLAY_TEMPERATURE = 0.9
DEFAULT_AUCTION_TEMPERATURE = 0.75
DEFAULT_EPSILON = 0.08
DEFAULT_TEMPERATURE_DECAY = 0.96
DEFAULT_EPSILON_DECAY = 0.94
THREE_PLAYER_NAMES = ["Player 1", "Player 2", "Player 3"]


@dataclass(frozen=True)
class PendingPolicyExample:
    player_name: str
    candidate_features: list[list[float]]
    chosen_index: int
    baseline_value: float
    phase: str


@dataclass
class SelfPlayDatasetHistory:
    iterations: list[TrainingDataset] = field(default_factory=list)

    def append(self, dataset: TrainingDataset, *, replay_window: int) -> TrainingDataset:
        self.iterations.append(dataset)
        if replay_window > 0:
            self.iterations = self.iterations[-replay_window:]
        aggregate = TrainingDataset()
        for iteration_dataset in self.iterations:
            aggregate.extend(iteration_dataset)
        return aggregate


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _format_dataset_counts(dataset: TrainingDataset) -> str:
    return (
        f"play_policy={len(dataset.play_policy_examples)} "
        f"auction_policy={len(dataset.auction_policy_examples)} "
        f"play_value={len(dataset.play_value_examples)} "
        f"auction_value={len(dataset.auction_value_examples)}"
    )


def _resolve_initial_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if NeuralThreePlayerBot.MODEL_FILE_V3.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V3
    if NeuralThreePlayerBot.MODEL_FILE_V2.exists():
        return NeuralThreePlayerBot.MODEL_FILE_V2
    return NeuralThreePlayerBot.MODEL_FILE_V1


def _sample_index(
    *,
    scores: list[float],
    rng: random.Random,
    temperature: float,
    epsilon: float,
) -> int:
    if not scores:
        raise ValueError("scores must not be empty")
    if len(scores) == 1:
        return 0
    if epsilon > 0 and rng.random() < epsilon:
        return rng.randrange(len(scores))

    clamped_temperature = max(temperature, 1e-3)
    max_score = max(scores)
    exp_scores = [
        math.exp((score - max_score) / clamped_temperature)
        for score in scores
    ]
    total = sum(exp_scores)
    threshold = rng.random() * total
    running_total = 0.0
    for index, weight in enumerate(exp_scores):
        running_total += weight
        if threshold <= running_total:
            return index
    return len(scores) - 1


def _build_self_play_bots(model_bundle: dict) -> dict[str, NeuralThreePlayerBot]:
    return {
        player_name: NeuralThreePlayerBot(
            player_name,
            model_bundle=model_bundle,
        )
        for player_name in THREE_PLAYER_NAMES
    }


def collect_self_play_training_dataset(
    *,
    model_bundle: dict,
    match_count: int,
    alpha: int,
    seed: int,
    play_temperature: float,
    auction_temperature: float,
    epsilon: float,
    policy_advantage_threshold: float,
    policy_advantage_scale: float,
    value_weight_scale: float,
    winner_policy_weight: float,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    if match_count <= 0:
        raise ValueError("match_count must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    dataset = TrainingDataset()
    rng = random.Random(seed)
    forced_auction_actions = 0
    forced_play_actions = 0
    sampled_auction_actions = 0
    sampled_play_actions = 0
    started_at = time.perf_counter()
    progress = LiveProgressDisplay(
        verbose=verbose,
        label="collect:self-play",
        total=match_count,
        started_at=started_at,
    )

    _log(
        verbose,
        (
            f"[collect:self-play] start matches={match_count} alpha={alpha} "
            f"seed={seed} play_temp={play_temperature:.2f} "
            f"auction_temp={auction_temperature:.2f} epsilon={epsilon:.2f}"
        ),
    )
    progress.start(detail=_format_dataset_counts(dataset))

    for match_index in range(match_count):
        random.seed(seed + match_index)
        controller = MatchController.create(
            num_players=3,
            player_names=THREE_PLAYER_NAMES,
            teams=None,
            bots=_build_self_play_bots(model_bundle),
            auto_run_bots=False,
        )
        round_start_match_scores = dict(controller.session.match_scores)
        pending_play_values: list[PendingValueExample] = []
        pending_auction_values: list[PendingValueExample] = []
        pending_play_policy: list[PendingPolicyExample] = []
        pending_auction_policy: list[PendingPolicyExample] = []

        for round_index in range(1, alpha + 1):
            while controller.session.phase in {"auction", "play"}:
                player_name = controller._current_bot_name()
                if player_name is None:
                    raise ValueError("match controller had no active bot turn")
                controller._sync_bot_hand(player_name)
                bot = controller.session.bots[player_name]

                if controller.session.phase == "auction":
                    auction_state = controller.session.auction.state
                    scored_actions = bot.score_auction_candidates(auction_state)
                    legal_actions = [action for action, _, _ in scored_actions]
                    if not legal_actions:
                        raise ValueError("self-play found no legal auction actions")

                    if len(legal_actions) == 1:
                        forced_auction_actions += 1
                        chosen_index = 0
                    else:
                        sampled_auction_actions += 1
                        chosen_index = _sample_index(
                            scores=[score for _, score, _ in scored_actions],
                            rng=rng,
                            temperature=auction_temperature,
                            epsilon=epsilon,
                        )
                        pending_auction_policy.append(
                            PendingPolicyExample(
                                player_name=player_name,
                                candidate_features=[
                                    features for _, _, features in scored_actions
                                ],
                                chosen_index=chosen_index,
                                baseline_value=bot.estimate_auction_state_value(
                                    auction_state,
                                    perspective_player_name=player_name,
                                ),
                                phase="auction",
                            )
                        )

                    chosen_action = legal_actions[chosen_index]
                    successor_auction = deepcopy(auction_state)
                    apply_auction_action_for_search(successor_auction, chosen_action)
                    pending_auction_values.append(
                        PendingValueExample(
                            player_name=player_name,
                            features=encode_auction_state(
                                auction_state=successor_auction,
                                perspective_player_name=player_name,
                                hand=set(bot.cards),
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                            ),
                            weight=1.0,
                        )
                    )
                    controller.session.auction.apply_event(chosen_action)
                    controller._advance_or_finalize_auction()
                    continue

                round_state = controller.session.game.round_state
                scored_cards = bot.score_play_candidates(round_state)
                legal_cards = [card for card, _, _ in scored_cards]
                if not legal_cards:
                    raise ValueError("self-play found no legal cards")

                if len(legal_cards) == 1:
                    forced_play_actions += 1
                    chosen_index = 0
                else:
                    sampled_play_actions += 1
                    chosen_index = _sample_index(
                        scores=[score for _, score, _ in scored_cards],
                        rng=rng,
                        temperature=play_temperature,
                        epsilon=epsilon,
                    )
                    pending_play_policy.append(
                        PendingPolicyExample(
                            player_name=player_name,
                            candidate_features=[
                                features for _, _, features in scored_cards
                            ],
                            chosen_index=chosen_index,
                            baseline_value=bot.estimate_play_state_value(
                                round_state,
                                perspective_player_name=player_name,
                            ),
                            phase="play",
                        )
                    )

                chosen_card = legal_cards[chosen_index]
                successor_round = apply_trick_action_to_state(
                    round_state,
                    Play(round_state.current_player, chosen_card),
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
                            weight=1.0,
                        )
                    )
                controller.session.game.apply_trick_action(
                    Play(controller.session.game.curr_player, chosen_card)
                )
                controller._score_terminal_round_if_needed()

            if controller.session.game.round_state.is_terminal:
                round_targets = _resolve_round_value_targets(
                    controller=controller,
                    round_start_match_scores=round_start_match_scores,
                )
                dataset.play_value_examples.extend(
                    RegressionExample(
                        features=example.features,
                        target=round_targets[example.player_name],
                        weight=1.0 + (value_weight_scale * abs(round_targets[example.player_name])),
                    )
                    for example in pending_play_values
                )
                dataset.auction_value_examples.extend(
                    RegressionExample(
                        features=example.features,
                        target=round_targets[example.player_name],
                        weight=1.0 + (value_weight_scale * abs(round_targets[example.player_name])),
                    )
                    for example in pending_auction_values
                )

                for pending_example in pending_play_policy:
                    final_utility = round_targets[pending_example.player_name]
                    advantage = final_utility - pending_example.baseline_value
                    if (
                        advantage > policy_advantage_threshold
                        or final_utility > 0.0
                    ):
                        dataset.play_policy_examples.append(
                            ChoiceExample(
                                candidate_features=pending_example.candidate_features,
                                chosen_index=pending_example.chosen_index,
                                weight=(
                                    1.0
                                    + (policy_advantage_scale * max(advantage, 0.0))
                                    + (
                                        winner_policy_weight
                                        if final_utility > 0.0
                                        else 0.0
                                    )
                                ),
                            )
                        )
                for pending_example in pending_auction_policy:
                    final_utility = round_targets[pending_example.player_name]
                    advantage = final_utility - pending_example.baseline_value
                    if (
                        advantage > policy_advantage_threshold
                        or final_utility > 0.0
                    ):
                        dataset.auction_policy_examples.append(
                            ChoiceExample(
                                candidate_features=pending_example.candidate_features,
                                chosen_index=pending_example.chosen_index,
                                weight=(
                                    1.0
                                    + (policy_advantage_scale * max(advantage, 0.0))
                                    + (
                                        winner_policy_weight
                                        if final_utility > 0.0
                                        else 0.0
                                    )
                                ),
                            )
                        )

                pending_play_values = []
                pending_auction_values = []
                pending_play_policy = []
                pending_auction_policy = []

            if controller.session.is_match_complete:
                break

            if round_index < alpha:
                controller.next_round(auto_run_bots=False)
                round_start_match_scores = dict(controller.session.match_scores)

        progress.update(
            completed=match_index + 1,
            detail=_format_dataset_counts(dataset),
        )

    elapsed_seconds = time.perf_counter() - started_at
    progress.stop(
        completed=match_count,
        detail=_format_dataset_counts(dataset),
    )
    report = {
        "matches": match_count,
        "alpha": alpha,
        "play_policy_examples": len(dataset.play_policy_examples),
        "auction_policy_examples": len(dataset.auction_policy_examples),
        "play_value_examples": len(dataset.play_value_examples),
        "auction_value_examples": len(dataset.auction_value_examples),
        "forced_auction_actions": forced_auction_actions,
        "forced_play_actions": forced_play_actions,
        "sampled_auction_actions": sampled_auction_actions,
        "sampled_play_actions": sampled_play_actions,
        "elapsed_seconds": elapsed_seconds,
    }
    _log(
        verbose,
        (
            f"[collect:self-play] done {_format_dataset_counts(dataset)} "
            f"elapsed={_format_seconds(elapsed_seconds)} "
            f"forced_auction={forced_auction_actions} forced_play={forced_play_actions} "
            f"sampled_auction={sampled_auction_actions} sampled_play={sampled_play_actions}"
        ),
    )
    return dataset, report


def train_with_self_play(
    *,
    initial_bundle: dict,
    self_play_matches: int,
    alpha: int,
    self_play_iterations: int,
    replay_window: int,
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
    play_temperature: float = DEFAULT_PLAY_TEMPERATURE,
    auction_temperature: float = DEFAULT_AUCTION_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
    temperature_decay: float = DEFAULT_TEMPERATURE_DECAY,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    policy_advantage_threshold: float = DEFAULT_POLICY_ADVANTAGE_THRESHOLD,
    policy_advantage_scale: float = DEFAULT_POLICY_ADVANTAGE_SCALE,
    value_weight_scale: float = DEFAULT_VALUE_WEIGHT_SCALE,
    winner_policy_weight: float = DEFAULT_WINNER_POLICY_WEIGHT,
    gradient_clip: float = 3.0,
    verbose: bool = False,
) -> tuple[dict, dict]:
    current_bundle = initial_bundle
    replay_history = SelfPlayDatasetHistory()
    iteration_reports: list[dict] = []

    for iteration_index in range(self_play_iterations):
        iteration_temperature = max(
            0.2,
            play_temperature * (temperature_decay ** iteration_index),
        )
        iteration_auction_temperature = max(
            0.15,
            auction_temperature * (temperature_decay ** iteration_index),
        )
        iteration_epsilon = max(
            0.01,
            epsilon * (epsilon_decay ** iteration_index),
        )
        _log(
            verbose,
            (
                f"[self-play] iteration {iteration_index + 1}/{self_play_iterations} "
                f"matches={self_play_matches} play_temp={iteration_temperature:.2f} "
                f"auction_temp={iteration_auction_temperature:.2f} epsilon={iteration_epsilon:.2f}"
            ),
        )
        iteration_dataset, collection_report = collect_self_play_training_dataset(
            model_bundle=current_bundle,
            match_count=self_play_matches,
            alpha=alpha,
            seed=seed + (iteration_index * 10_000),
            play_temperature=iteration_temperature,
            auction_temperature=iteration_auction_temperature,
            epsilon=iteration_epsilon,
            policy_advantage_threshold=policy_advantage_threshold,
            policy_advantage_scale=policy_advantage_scale,
            value_weight_scale=value_weight_scale,
            winner_policy_weight=winner_policy_weight,
            verbose=verbose,
        )
        aggregate_dataset = replay_history.append(
            iteration_dataset,
            replay_window=replay_window,
        )
        _log(
            verbose,
            (
                f"[self-play] iteration {iteration_index + 1}/{self_play_iterations} "
                f"replay_window={min(replay_window, len(replay_history.iterations))} "
                f"aggregate {_format_dataset_counts(aggregate_dataset)}"
            ),
        )
        current_bundle, training_report = train_neural_3p_bundle(
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
            seed=seed + (iteration_index * 10_000) + 500,
            teacher_specs=[],
            initial_bundle=current_bundle,
            play_value_weight=play_value_weight,
            auction_value_weight=auction_value_weight,
            play_rollout_depth=play_rollout_depth,
            auction_rollout_depth=auction_rollout_depth,
            gradient_clip=gradient_clip,
            bot_id="neural-3p-v3",
            bundle_version=3,
            extra_metadata={
                "training_mode": "self_play",
                "seed_bot_id": initial_bundle.get("bot_id", "neural-3p-v2"),
            },
            verbose=verbose,
        )
        _log(
            verbose,
            (
                f"[self-play] iteration {iteration_index + 1}/{self_play_iterations} "
                f"{_format_training_metrics(training_report)}"
            ),
        )
        iteration_reports.append(
            {
                "iteration": iteration_index + 1,
                "temperature": iteration_temperature,
                "auction_temperature": iteration_auction_temperature,
                "epsilon": iteration_epsilon,
                "collection": collection_report,
                "training": training_report,
            }
        )

    return current_bundle, {"self_play_iterations": iteration_reports}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train neural-3p-v3 from v2 using advantage-weighted 3-player self-play.",
    )
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--self-play-matches", type=int, default=DEFAULT_SELF_PLAY_MATCHES)
    parser.add_argument("--alpha", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-window", type=int, default=DEFAULT_REPLAY_WINDOW)
    parser.add_argument("--eval-games", type=int, default=36)
    parser.add_argument("--eval-games-vs-optimal", type=int, default=18)
    parser.add_argument("--precheck-games", type=int, default=12)
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
    parser.add_argument("--gradient-clip", type=float, default=3.0)
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
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    initial_model_path = _resolve_initial_model_path(args.initial_model)
    current_best_bundle = load_model_bundle(initial_model_path)
    verbose = not args.quiet

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / f"neural-3p-self-play-{timestamp}"
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
                "self_play_matches": args.self_play_matches,
                "alpha": args.alpha,
                "replay_window": args.replay_window,
                "duration_hours": args.duration_hours,
                "eval_games": args.eval_games,
                "eval_games_vs_optimal": args.eval_games_vs_optimal,
                "precheck_games": args.precheck_games,
                "winner_policy_weight": args.winner_policy_weight,
                "seed": args.seed,
            }
        ),
        encoding="utf-8",
    )
    _log(
        verbose,
        (
            f"[self-play] run_dir={run_dir} initial_model={initial_model_path} "
            f"self_play_matches={args.self_play_matches}"
        ),
    )

    started_at = time.time()
    deadline = started_at + (args.duration_hours * 3600.0)
    iteration_index = 1
    incumbent_baseline_cache: dict | None = None

    while True:
        if args.max_iterations is not None and iteration_index > args.max_iterations:
            break
        if time.time() >= deadline and iteration_index > 1:
            break

        iteration_started_at = time.perf_counter()
        _log(
            verbose,
            (
                f"[self-play] iteration {iteration_index} start "
                f"elapsed_hours={(time.time() - started_at) / 3600.0:.2f}"
            ),
        )
        candidate_bundle, report = train_with_self_play(
            initial_bundle=current_best_bundle,
            self_play_matches=args.self_play_matches,
            alpha=args.alpha,
            self_play_iterations=1,
            replay_window=args.replay_window,
            seed=args.seed + (iteration_index * 10_000),
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
            play_temperature=args.play_temperature,
            auction_temperature=args.auction_temperature,
            epsilon=args.epsilon,
            temperature_decay=args.temperature_decay,
            epsilon_decay=args.epsilon_decay,
            policy_advantage_threshold=args.policy_advantage_threshold,
            policy_advantage_scale=args.policy_advantage_scale,
            value_weight_scale=args.value_weight_scale,
            winner_policy_weight=args.winner_policy_weight,
            gradient_clip=args.gradient_clip,
            verbose=verbose,
        )

        candidate_path = checkpoints_dir / f"iteration-{iteration_index:03d}.json"
        save_model_bundle(candidate_bundle, candidate_path)
        training_report = report["self_play_iterations"][-1]["training"]
        _log(
            verbose,
            (
                f"[self-play] iteration {iteration_index} trained checkpoint={candidate_path} "
                f"{_format_training_metrics(training_report)} "
                f"train_elapsed={_format_seconds(training_report.get('elapsed_seconds', 0.0))}"
            ),
        )

        evaluation_report = evaluate_candidate(
            candidate_bundle=candidate_bundle,
            incumbent_bundle=current_best_bundle,
            alpha=args.alpha,
            eval_games=args.eval_games,
            eval_games_vs_optimal=args.eval_games_vs_optimal,
            precheck_games=min(args.precheck_games, args.eval_games),
            seed=args.seed + (iteration_index * 10_000),
            incumbent_baselines=incumbent_baseline_cache,
            verbose=verbose,
        )
        if (
            incumbent_baseline_cache is None
            and evaluation_report["incumbent_vs_greedy"] is not None
            and evaluation_report["incumbent_vs_optimal"] is not None
        ):
            incumbent_baseline_cache = {
                "vs_greedy": evaluation_report["incumbent_vs_greedy"],
                "vs_optimal": evaluation_report["incumbent_vs_optimal"],
            }
        if evaluation_report["accepted"]:
            current_best_bundle = candidate_bundle
            incumbent_baseline_cache = {
                "vs_greedy": evaluation_report["vs_greedy"],
                "vs_optimal": evaluation_report["vs_optimal_bot"],
            }
            save_model_bundle(current_best_bundle, best_path)
            _log(verbose, f"[self-play] iteration {iteration_index} promoted candidate to {best_path}")
        else:
            _log(verbose, f"[self-play] iteration {iteration_index} kept incumbent")

        cycle_report = {
            "iteration": iteration_index,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
            "candidate_path": str(candidate_path),
            "accepted": evaluation_report["accepted"],
            "training": report,
            "evaluation": evaluation_report,
            "timing": {
                "iteration_elapsed_seconds": time.perf_counter() - iteration_started_at,
            },
        }
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_report) + "\n")
        iteration_index += 1

    if args.promote_to is not None:
        save_model_bundle(current_best_bundle, args.promote_to)
        _log(verbose, f"[self-play] wrote promoted best model to {args.promote_to}")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_model": str(best_path),
                "promoted_to": str(args.promote_to) if args.promote_to is not None else None,
                "iterations_completed": iteration_index - 1,
            }
        )
    )


if __name__ == "__main__":
    main()
