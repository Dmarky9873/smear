from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

try:
    from .bots.base import BotPlayer
    from .bots.human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .bots.human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
    )
    from .bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from .bots.omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .gameplay import MatchController, MatchResult
except ImportError:
    from bots.base import BotPlayer
    from bots.human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from bots.human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
    )
    from bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from bots.omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
    from bots.registry import build_ready_bot, get_ready_bot_spec
    from gameplay import MatchController, MatchResult


TARGET_SCORE = 21
MAX_PLAYERS = 8
MIN_PLAYERS = 3
BotFactory = Callable[[str], BotPlayer]
PlayerModelArg = str | type[BotPlayer] | BotFactory | BotPlayer


@dataclass(frozen=True)
class ResolvedModelSpec:
    key: str
    label: str
    factory: BotFactory


def _parse_minimax_bot_id(bot_id: str) -> tuple[str, int] | None:
    if bot_id == "one-trick-minmax":
        return ("human", 1)
    if bot_id == "o-one-trick-minmax":
        return ("omniscient", 1)

    omniscient = bot_id.startswith("o-")
    normalized_id = bot_id[2:] if omniscient else bot_id
    suffix = "-trick-minmax"
    if not normalized_id.endswith(suffix):
        return None

    depth_token = normalized_id[: -len(suffix)]
    if not depth_token.isdigit():
        return None

    family = "omniscient" if omniscient else "human"
    return family, int(depth_token)


def _build_minimax_resolved_spec(family: str, depth: int) -> ResolvedModelSpec:
    if depth <= 0:
        raise ValueError("depth must be positive")

    if family == "human":
        if depth == 1:
            return ResolvedModelSpec(
                key="one-trick-minmax",
                label="L-1 Minmax",
                factory=lambda player_name: HumanInformationMinimaxOneTrickPlayer(
                    player_name
                ),
            )
        return ResolvedModelSpec(
            key=f"{depth}-trick-minmax",
            label=f"L-{depth} Minmax",
            factory=lambda player_name, depth=depth: HumanInformationMinimaxNTrickPlayer(
                player_name,
                depth=depth,
            ),
        )

    if family == "omniscient":
        if depth == 1:
            return ResolvedModelSpec(
                key="o-one-trick-minmax",
                label="Omniscient L-1 Minmax",
                factory=lambda player_name: OmniscientMinimaxOneTrickPlayer(
                    player_name
                ),
            )
        return ResolvedModelSpec(
            key=f"o-{depth}-trick-minmax",
            label=f"Omniscient L-{depth} Minmax",
            factory=lambda player_name, depth=depth: OmniscientMinimaxNTrickPlayer(
                player_name,
                depth=depth,
            ),
        )

    raise ValueError(f"unsupported minimax family: {family}")


def _resolve_model_spec(
    model: PlayerModelArg,
    *,
    depth: int | None = None,
) -> ResolvedModelSpec:
    if isinstance(model, str):
        minimax_spec = _parse_minimax_bot_id(model)
        if minimax_spec is not None and depth is not None:
            family, _ = minimax_spec
            return _build_minimax_resolved_spec(family, depth)

        spec = get_ready_bot_spec(model)
        return ResolvedModelSpec(
            key=spec.id,
            label=spec.label,
            factory=lambda player_name, bot_id=spec.id: build_ready_bot(
                bot_id,
                player_name,
            ),
        )

    if isinstance(model, BotPlayer):
        model_class = type(model)
        return ResolvedModelSpec(
            key=model_class.__name__,
            label=model_class.__name__,
            factory=lambda player_name, cls=model_class: cls(player_name),
        )

    if inspect.isclass(model) and issubclass(model, BotPlayer):
        return ResolvedModelSpec(
            key=model.__name__,
            label=model.__name__,
            factory=lambda player_name, cls=model: cls(player_name),
        )

    if callable(model):
        label = getattr(model, "__name__", model.__class__.__name__)

        def _factory(player_name: str, fn: BotFactory = model) -> BotPlayer:
            controller = fn(player_name)
            if not isinstance(controller, BotPlayer):
                raise TypeError(
                    f"model factory '{label}' must return a BotPlayer instance"
                )
            return controller

        return ResolvedModelSpec(key=label, label=label, factory=_factory)

    raise TypeError(f"unsupported model type: {type(model)!r}")


def _resolve_model_specs(
    model1: PlayerModelArg,
    model2: PlayerModelArg,
    model3: PlayerModelArg | None = None,
    model4: PlayerModelArg | None = None,
    model5: PlayerModelArg | None = None,
    model6: PlayerModelArg | None = None,
    model7: PlayerModelArg | None = None,
    model8: PlayerModelArg | None = None,
    *,
    depth: int | None = None,
) -> list[ResolvedModelSpec]:
    provided_models = [
        model
        for model in [model1, model2, model3, model4, model5, model6, model7, model8]
        if model is not None
    ]

    if len(provided_models) < 2:
        raise ValueError("at least two models must be provided")
    if len(provided_models) > MAX_PLAYERS:
        raise ValueError(f"no more than {MAX_PLAYERS} models may be provided")

    return [_resolve_model_spec(model, depth=depth) for model in provided_models]


def _build_simulation_roster(
    base_model_specs: list[ResolvedModelSpec],
    team_size: int,
    *,
    three_player: bool = False,
) -> tuple[list[str], list[ResolvedModelSpec], list[tuple[str, ...]]]:
    if team_size <= 0:
        raise ValueError("team_size must be positive")
    if three_player and team_size != 1:
        raise ValueError("three-player mode requires team_size 1")
    if three_player and len(base_model_specs) > MIN_PLAYERS:
        raise ValueError("three-player mode accepts at most three models")

    requested_players = len(base_model_specs) * team_size
    if requested_players > MAX_PLAYERS:
        raise ValueError(
            f"team_size {team_size} with {len(base_model_specs)} models exceeds "
            f"the {MAX_PLAYERS}-player limit"
        )

    player_names: list[str] = []
    seat_model_specs: list[ResolvedModelSpec] = []
    teams: list[tuple[str, ...]] = []
    next_player_number = 1

    for model_spec in base_model_specs:
        team_members: list[str] = []
        for _ in range(team_size):
            player_name = f"Player {next_player_number}"
            next_player_number += 1
            player_names.append(player_name)
            seat_model_specs.append(model_spec)
            team_members.append(player_name)
        teams.append(tuple(team_members))

    filler_model = _resolve_model_spec("random")
    while len(seat_model_specs) < MIN_PLAYERS:
        player_name = f"Player {next_player_number}"
        next_player_number += 1
        player_names.append(player_name)
        seat_model_specs.append(filler_model)
        teams.append((player_name,))

    return player_names, seat_model_specs, teams


def _render_progress_bar(
    completed: int,
    total: int,
    width: int = 30,
    label: str = "Simulating games",
) -> None:
    if total <= 0:
        return
    ratio = completed / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(
        f"\r{label}: [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)"
    )
    if completed == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


class Simulator:
    def __init__(self, controller: MatchController):
        self._controller = controller

    def run_match(self, alpha: int) -> MatchResult:
        return self._controller.run_match(alpha)


def _build_team_model_specs(
    player_names: list[str],
    model_specs: list[ResolvedModelSpec],
    teams: list[tuple[str, ...]],
) -> list[ResolvedModelSpec]:
    spec_by_player = {
        player_name: model_spec
        for player_name, model_spec in zip(player_names, model_specs)
    }
    team_model_specs: list[ResolvedModelSpec] = []

    for team in teams:
        first_spec = spec_by_player[team[0]]
        if any(spec_by_player[player_name].key != first_spec.key for player_name in team):
            raise ValueError(
                "each simulated team must use a single model in fair mode")
        team_model_specs.append(first_spec)

    return team_model_specs


def _build_fair_team_assignments(
    team_model_specs: list[ResolvedModelSpec],
) -> list[tuple[ResolvedModelSpec, ...]]:
    if not team_model_specs:
        return []

    assignments: list[tuple[ResolvedModelSpec, ...]] = []
    seen_assignment_keys: set[tuple[str, ...]] = set()
    candidate_orders = [team_model_specs, list(reversed(team_model_specs))]

    for candidate_order in candidate_orders:
        for rotation in range(len(candidate_order)):
            rotated_specs = tuple(
                candidate_order[rotation:] + candidate_order[:rotation]
            )
            assignment_key = tuple(spec.key for spec in rotated_specs)
            if assignment_key in seen_assignment_keys:
                continue
            seen_assignment_keys.add(assignment_key)
            assignments.append(rotated_specs)

    return assignments


def _build_assignment_summary(
    player_names: list[str],
    teams: list[tuple[str, ...]],
    team_assignment: tuple[ResolvedModelSpec, ...],
) -> tuple[dict[str, str], list[dict], list[dict]]:
    if len(teams) != len(team_assignment):
        raise ValueError("team assignment count must match simulated teams")

    model_key_by_player: dict[str, str] = {}
    model_label_by_player: dict[str, str] = {}
    for team, model_spec in zip(teams, team_assignment):
        for player_name in team:
            model_key_by_player[player_name] = model_spec.key
            model_label_by_player[player_name] = model_spec.label

    seat_models = [
        {
            "player_name": player_name,
            "model_key": model_key_by_player[player_name],
            "model_label": model_label_by_player[player_name],
        }
        for player_name in player_names
    ]
    team_summaries = [
        {
            "team_name": " / ".join(team),
            "player_names": list(team),
            "model_keys": [model_key_by_player[player_name] for player_name in team],
            "model_labels": [model_label_by_player[player_name] for player_name in team],
        }
        for team in teams
    ]
    return model_key_by_player, seat_models, team_summaries


def _instantiate_assignment_bots(
    teams: list[tuple[str, ...]],
    team_assignment: tuple[ResolvedModelSpec, ...],
) -> dict[str, BotPlayer]:
    controllers: dict[str, BotPlayer] = {}
    for team, model_spec in zip(teams, team_assignment):
        for player_name in team:
            controllers[player_name] = model_spec.factory(player_name)
    return controllers


def benchmark_models(
    n: int,
    alpha: int,
    model1: PlayerModelArg,
    model2: PlayerModelArg,
    model3: PlayerModelArg | None = None,
    model4: PlayerModelArg | None = None,
    model5: PlayerModelArg | None = None,
    model6: PlayerModelArg | None = None,
    model7: PlayerModelArg | None = None,
    model8: PlayerModelArg | None = None,
    show_progress: bool = True,
    team_size: int = 1,
    depth: int | None = None,
    fair: bool = False,
    seed: int | None = None,
    three_player: bool = False,
    progress_label: str = "Simulating games",
) -> dict:
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if depth is not None and depth <= 0:
        raise ValueError("depth must be positive")

    base_model_specs = _resolve_model_specs(
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        depth=depth,
    )
    player_names, model_specs, teams = _build_simulation_roster(
        base_model_specs,
        team_size,
        three_player=three_player,
    )
    team_members_by_name = {" / ".join(team): team for team in teams}
    model_label_by_key = {
        model_spec.key: model_spec.label for model_spec in model_specs}
    seat_counts: dict[str, int] = {}
    for model_spec in model_specs:
        seat_counts[model_spec.key] = seat_counts.get(model_spec.key, 0) + 1

    model_results = {
        model_key: {
            "label": model_label_by_key[model_key],
            "seat_count": seat_counts[model_key],
            "games_won": 0.0,
            "win_percentage": 0.0,
        }
        for model_key in seat_counts
    }

    draws = 0
    total_rounds = 0
    team_model_specs = _build_team_model_specs(
        player_names, model_specs, teams)

    if fair:
        fair_assignments = _build_fair_team_assignments(team_model_specs)
        effective_seed = 0 if seed is None else seed
    else:
        fair_assignments = [tuple(team_model_specs)]
        effective_seed = seed

    assignment_game_counts = [0 for _ in fair_assignments]
    for simulation_index in range(n):
        assignment_game_counts[simulation_index % len(fair_assignments)] += 1

    assignment_summaries = []
    for assignment_index, team_assignment in enumerate(fair_assignments, start=1):
        _, assignment_seat_models, assignment_teams = _build_assignment_summary(
            player_names,
            teams,
            team_assignment,
        )
        assignment_summaries.append(
            {
                "assignment_index": assignment_index,
                "games_scheduled": assignment_game_counts[assignment_index - 1],
                "seat_models": assignment_seat_models,
                "teams": assignment_teams,
            }
        )

    _, canonical_seat_models, canonical_teams = _build_assignment_summary(
        player_names,
        teams,
        fair_assignments[0],
    )

    if show_progress:
        _render_progress_bar(0, n, label=progress_label)

    started_at = time.perf_counter()

    for simulation_index in range(n):
        assignment_index = simulation_index % len(fair_assignments)
        team_assignment = fair_assignments[assignment_index]
        if effective_seed is not None:
            if fair:
                game_seed = effective_seed + \
                    (simulation_index // len(fair_assignments))
            else:
                game_seed = effective_seed + simulation_index
            random.seed(game_seed)

        controllers = _instantiate_assignment_bots(teams, team_assignment)
        current_model_key_by_player, _, _ = _build_assignment_summary(
            player_names,
            teams,
            team_assignment,
        )
        controller = MatchController.create(
            num_players=len(model_specs),
            player_names=player_names,
            teams=teams,
            bots=controllers,
            target_score=TARGET_SCORE,
        )
        simulator = Simulator(controller)
        match_result = simulator.run_match(alpha)
        total_rounds += match_result.rounds_played

        if match_result.is_draw:
            draws += 1
        else:
            try:
                winning_player_names = {
                    player_name
                    for team_name in match_result.winner_names
                    for player_name in team_members_by_name[team_name]
                }
            except KeyError as exc:
                raise ValueError(
                    f"simulation returned an unknown winning team: {exc.args[0]}"
                ) from exc

            winning_model_keys = sorted(
                {
                    current_model_key_by_player[player_name]
                    for player_name in winning_player_names
                }
            )
            win_share = 1.0 / len(winning_model_keys)
            for model_key in winning_model_keys:
                model_results[model_key]["games_won"] += win_share

        if show_progress:
            _render_progress_bar(simulation_index + 1, n, label=progress_label)

    elapsed_seconds = time.perf_counter() - started_at
    games_per_second = n / elapsed_seconds if elapsed_seconds > 0 else None
    rounds_per_second = (
        total_rounds / elapsed_seconds if elapsed_seconds > 0 else None
    )

    for model_key, stats in model_results.items():
        stats["win_percentage"] = (stats["games_won"] / n) * 100

    return {
        "games_played": n,
        "alpha": alpha,
        "team_size": team_size,
        "minimax_depth": depth,
        "three_player": three_player,
        "comparison_mode": "fair" if fair else "standard",
        "seed": effective_seed,
        "players_per_game": len(model_specs),
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_game": elapsed_seconds / n,
        "average_seconds_per_round": (
            elapsed_seconds / total_rounds if total_rounds > 0 else None
        ),
        "games_per_second": games_per_second,
        "rounds_per_second": rounds_per_second,
        "seat_models": canonical_seat_models,
        "teams": canonical_teams,
        "fair_schedule": (
            {
                "strategy": "rotations_and_reversals",
                "assignment_count": len(fair_assignments),
                "fully_balanced": n % len(fair_assignments) == 0,
                "assignments": assignment_summaries,
            }
            if fair
            else None
        ),
        "draws": draws,
        "draw_percentage": (draws / n) * 100,
        "average_rounds_played": total_rounds / n,
        "models": model_results,
    }


def compare_models_objectively(
    n: int,
    alpha: int,
    model1: PlayerModelArg,
    model2: PlayerModelArg,
    model3: PlayerModelArg | None = None,
    model4: PlayerModelArg | None = None,
    model5: PlayerModelArg | None = None,
    model6: PlayerModelArg | None = None,
    model7: PlayerModelArg | None = None,
    model8: PlayerModelArg | None = None,
    *,
    show_progress: bool = True,
    team_size: int = 1,
    depth: int | None = None,
    seed: int | None = 0,
    three_player: bool = False,
    progress_label: str = "Simulating games",
) -> dict:
    return benchmark_models(
        n,
        alpha,
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        show_progress=show_progress,
        team_size=team_size,
        depth=depth,
        fair=True,
        seed=seed,
        three_player=three_player,
        progress_label=progress_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated all-bot Smear matches and summarize win rates.",
    )
    parser.add_argument("n", type=int, help="number of games to simulate")
    parser.add_argument(
        "alpha",
        type=int,
        help="maximum number of rounds before a draw is declared",
    )
    parser.add_argument(
        "--three-player",
        action="store_true",
        help=(
            "force an exactly three-player simulation; if only two models are "
            "supplied, the third seat is filled with random"
        ),
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=1,
        help=(
            "number of same-model seats to create per supplied model; "
            "for example, --team-size 2 with two models simulates a 2v2 match"
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "override minimax bot search depth for minimax model ids; "
            "for example, --depth 3 with one-trick-minmax uses the 3-trick variant"
        ),
    )
    parser.add_argument(
        "--fair",
        action="store_true",
        help=(
            "reduce seat-order variance by rotating and reversing seat assignments; "
            "when combined with --seed, each seat assignment in a batch shares the same deal seed"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "optional RNG seed for reproducible simulations; "
            "in --fair mode, identical batch seeds are paired across seat assignments"
        ),
    )
    parser.add_argument("model1", help="first ready bot id")
    parser.add_argument("model2", help="second ready bot id")
    parser.add_argument(
        "models",
        nargs="*",
        help="optional additional ready bot ids (up to six total extras)",
    )
    args = parser.parse_args()

    if len(args.models) > 6:
        raise SystemExit("no more than eight total models may be provided")

    result = benchmark_models(
        args.n,
        args.alpha,
        args.model1,
        args.model2,
        *args.models,
        team_size=args.team_size,
        depth=args.depth,
        fair=args.fair,
        seed=args.seed,
        three_player=args.three_player,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""python simulator.py --fair --seed 0 60 50 3-trick-minmax greedy"""
