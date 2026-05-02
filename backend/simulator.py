from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from typing import Callable

try:
    from .bots.base import BotPlayer
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .gameplay import MatchController, MatchResult
except ImportError:
    from bots.base import BotPlayer
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


def _resolve_model_spec(model: PlayerModelArg) -> ResolvedModelSpec:
    if isinstance(model, str):
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

    resolved_models = [_resolve_model_spec(model) for model in provided_models]
    while len(resolved_models) < MIN_PLAYERS:
        resolved_models.append(_resolve_model_spec("random"))
    return resolved_models


def _render_progress_bar(completed: int, total: int, width: int = 30) -> None:
    if total <= 0:
        return
    ratio = completed / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(
        f"\rSimulating games: [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)"
    )
    if completed == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


class Simulator:
    def __init__(self, controller: MatchController):
        self._controller = controller

    def run_match(self, alpha: int) -> MatchResult:
        return self._controller.run_match(alpha)


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
) -> dict:
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    model_specs = _resolve_model_specs(
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
    )
    player_names = [f"Player {index + 1}" for index in range(len(model_specs))]
    teams = [(player_name,) for player_name in player_names]
    model_key_by_player = {
        player_name: model_spec.key
        for player_name, model_spec in zip(player_names, model_specs)
    }
    model_label_by_key = {model_spec.key: model_spec.label for model_spec in model_specs}
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

    if show_progress:
        _render_progress_bar(0, n)

    for simulation_index in range(n):
        controllers = {
            player_name: model_spec.factory(player_name)
            for player_name, model_spec in zip(player_names, model_specs)
        }
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
            winning_model_keys = sorted(
                {
                    model_key_by_player[player_name]
                    for player_name in match_result.winner_names
                }
            )
            win_share = 1.0 / len(winning_model_keys)
            for model_key in winning_model_keys:
                model_results[model_key]["games_won"] += win_share

        if show_progress:
            _render_progress_bar(simulation_index + 1, n)

    for model_key, stats in model_results.items():
        stats["win_percentage"] = (stats["games_won"] / n) * 100

    return {
        "games_played": n,
        "alpha": alpha,
        "players_per_game": len(model_specs),
        "seat_models": [
            {
                "player_name": player_name,
                "model_key": model_spec.key,
                "model_label": model_spec.label,
            }
            for player_name, model_spec in zip(player_names, model_specs)
        ],
        "draws": draws,
        "draw_percentage": (draws / n) * 100,
        "average_rounds_played": total_rounds / n,
        "models": model_results,
    }


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
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
