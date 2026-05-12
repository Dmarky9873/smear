from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OptimalBotCandidate:
    id: str
    depth: int
    play_determinization_samples: int
    min_play_determinization_samples: int = 2
    auction_determinization_samples: int = 6
    three_player_auction_determinization_samples: int = 7


@dataclass(frozen=True)
class OptimalBotCandidateBenchmark:
    candidate: OptimalBotCandidate
    win_rate: float
    seconds_per_game: float


DEPTH_2_SPEED_CANDIDATE = OptimalBotCandidate(
    id="balanced-depth-2-det-10",
    depth=2,
    play_determinization_samples=10,
)

DEPTH_3_FAST_CANDIDATE = OptimalBotCandidate(
    id="balanced-depth-3-det-8",
    depth=3,
    play_determinization_samples=8,
)

OPTIMAL_BOT_CANDIDATE = OptimalBotCandidate(
    id="optimal-bot",
    depth=3,
    play_determinization_samples=10,
)

DEPTH_3_BASELINE_CANDIDATE = OptimalBotCandidate(
    id="balanced-depth-3-det-12",
    depth=3,
    play_determinization_samples=12,
)

DEPTH_4_SEARCH_CANDIDATE = OptimalBotCandidate(
    id="balanced-depth-4-det-10",
    depth=4,
    play_determinization_samples=10,
)

OPTIMAL_BOT_TUNING_CANDIDATES: tuple[OptimalBotCandidate, ...] = (
    DEPTH_2_SPEED_CANDIDATE,
    DEPTH_3_FAST_CANDIDATE,
    OPTIMAL_BOT_CANDIDATE,
    DEPTH_3_BASELINE_CANDIDATE,
    DEPTH_4_SEARCH_CANDIDATE,
)


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2 * left * right / (left + right)


def balance_score(
    benchmark: OptimalBotCandidateBenchmark,
    *,
    fastest_seconds_per_game: float,
) -> float:
    if benchmark.seconds_per_game <= 0 or fastest_seconds_per_game <= 0:
        raise ValueError("seconds_per_game must be positive")
    return _harmonic_mean(
        benchmark.win_rate,
        fastest_seconds_per_game / benchmark.seconds_per_game,
    )


def select_optimal_bot_candidate(
    benchmarks: Iterable[OptimalBotCandidateBenchmark],
) -> OptimalBotCandidateBenchmark:
    ranked_benchmarks = tuple(benchmarks)
    if not ranked_benchmarks:
        raise ValueError("at least one benchmark result is required")

    fastest_seconds_per_game = min(
        benchmark.seconds_per_game for benchmark in ranked_benchmarks
    )
    return max(
        ranked_benchmarks,
        key=lambda benchmark: (
            balance_score(
                benchmark,
                fastest_seconds_per_game=fastest_seconds_per_game,
            ),
            benchmark.win_rate,
            -benchmark.seconds_per_game,
            -benchmark.candidate.depth,
            -benchmark.candidate.play_determinization_samples,
        ),
    )
