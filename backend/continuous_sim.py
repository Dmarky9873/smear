from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .bots.registry import (
        HIDDEN_BOTS,
        READY_BOTS,
        build_ready_bot,
        get_ready_bot_rating_fingerprint,
        get_ready_bot_spec,
    )
    from .gameplay import MatchController
    from .simulator import Simulator, TARGET_SCORE
except ImportError:
    from bots.registry import (
        HIDDEN_BOTS,
        READY_BOTS,
        build_ready_bot,
        get_ready_bot_rating_fingerprint,
        get_ready_bot_spec,
    )
    from gameplay import MatchController
    from simulator import Simulator, TARGET_SCORE


DEFAULT_ALPHA = 50
DEFAULT_INITIAL_ELO = 1500.0
DEFAULT_K_FACTOR = 32.0
DEFAULT_PLAYERS_PER_GAME = 3
DEFAULT_ELO_FILE = Path("continuous-sim-elo.json")
DEFAULT_SCHEDULE_MODE = "balanced"
MAX_PLAYERS = 8
MIN_PLAYERS = 3
MIN_RATED_BOTS = 2
FILLER_BOT_ID = "random"
FILLER_SEAT_KEY = "random-filler"
ELO_FILE_VERSION = 1
PAIRWISE_ELO_Q = math.log(10.0) / 400.0
PAIRWISE_CI_Z = 1.959963984540054


@dataclass(frozen=True)
class SeatAssignment:
    seat_key: str
    bot_id: str
    is_rated: bool = True


@dataclass(frozen=True)
class MatchTask:
    match_index: int
    participants: tuple[SeatAssignment, ...]
    alpha: int
    seed: int | None


@dataclass(frozen=True)
class MatchOutcome:
    match_index: int
    participants: tuple[SeatAssignment, ...]
    rounds_played: int
    is_draw: bool
    winner_seat_keys: tuple[str, ...]
    scores_by_seat: tuple[tuple[str, int], ...]
    scores_by_bot: tuple[tuple[str, int], ...]
    elapsed_seconds: float
    seed: int | None


@dataclass
class EloEntry:
    rating: float
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    pairwise_games: int = 0
    information: float = 0.0


@dataclass
class ContinuousSimResult:
    games_completed: int
    started_at: float
    ended_at: float
    ratings: dict[str, EloEntry]
    executor_kind: str = "serial"
    schedule_mode: str = DEFAULT_SCHEDULE_MODE

    @property
    def elapsed_seconds(self) -> float:
        return self.ended_at - self.started_at


@dataclass(frozen=True)
class MatchProgressSnapshot:
    match_index: int
    match_round_number: int
    alpha: int
    player_count: int
    phase: str
    current_actor: str | None
    active_label: str | None
    active_detail: str | None
    active_percent: float | None
    auction_actions_completed: int
    played_cards_completed: int
    total_card_plays: int
    elapsed_seconds: float


@dataclass
class ProgressDisplay:
    is_tty: bool = field(default_factory=lambda: sys.stdout.isatty())
    rendered_lines: int = 0
    last_rendered_at: float = 0.0
    last_lines: tuple[str, ...] = field(default_factory=tuple)

    def render(self, lines: Sequence[str]) -> None:
        now = time.monotonic()
        rendered_lines = tuple(lines)
        minimum_interval = 0.1 if self.is_tty else 2.0
        if now - self.last_rendered_at < minimum_interval:
            return
        if rendered_lines == self.last_lines and now - self.last_rendered_at < 5.0:
            return

        if self.is_tty and self.rendered_lines > 0:
            self._rewind()
            self._clear_block()
            self._rewind()

        output = "\n".join(rendered_lines)
        if self.is_tty:
            sys.stdout.write(output)
        else:
            sys.stdout.write(output + "\n")
        sys.stdout.flush()
        self.rendered_lines = len(rendered_lines)
        self.last_rendered_at = now
        self.last_lines = rendered_lines

    def clear(self) -> None:
        if self.is_tty and self.rendered_lines > 0:
            self._rewind()
            self._clear_block()
            self._rewind()
            sys.stdout.flush()
        self.rendered_lines = 0
        self.last_lines = ()

    def _rewind(self) -> None:
        if self.rendered_lines > 1:
            sys.stdout.write(f"\x1b[{self.rendered_lines - 1}F")
        else:
            sys.stdout.write("\r")

    def _clear_block(self) -> None:
        for index in range(self.rendered_lines):
            sys.stdout.write("\x1b[2K")
            if index < self.rendered_lines - 1:
                sys.stdout.write("\n")


def available_bot_ids(include_hidden: bool = False) -> list[str]:
    ready_ids = [spec.id for spec in READY_BOTS]
    if not include_hidden:
        return ready_ids
    return ready_ids + [spec.id for spec in HIDDEN_BOTS]


def resolve_requested_bot_ids(
    *,
    requested_bot_ids: Sequence[str] | None,
    include_hidden: bool = False,
) -> list[str]:
    if requested_bot_ids is None or len(requested_bot_ids) == 0:
        return available_bot_ids(include_hidden=include_hidden)

    resolved_ids: list[str] = []
    seen: set[str] = set()
    for bot_id in requested_bot_ids:
        spec = get_ready_bot_spec(bot_id)
        if spec.id in seen:
            raise ValueError(f"duplicate bot id requested: {spec.id}")
        seen.add(spec.id)
        resolved_ids.append(spec.id)
    return resolved_ids


def players_per_game() -> int:
    return DEFAULT_PLAYERS_PER_GAME


def validate_bot_pool_for_matches(bot_ids: Sequence[str]) -> None:
    if len(bot_ids) < MIN_RATED_BOTS:
        raise ValueError("continuous-sim requires at least two bots")


def build_random_match(
    bot_ids: Sequence[str],
    *,
    rng: random.Random,
) -> tuple[SeatAssignment, ...]:
    validate_bot_pool_for_matches(bot_ids)

    if len(bot_ids) == MIN_RATED_BOTS:
        participants = [
            SeatAssignment(seat_key=bot_id, bot_id=bot_id)
            for bot_id in bot_ids
        ]
        participants.append(
            SeatAssignment(
                seat_key=FILLER_SEAT_KEY,
                bot_id=FILLER_BOT_ID,
                is_rated=False,
            )
        )
        rng.shuffle(participants)
        return tuple(participants)

    selected_bot_ids = rng.sample(list(bot_ids), players_per_game())
    rng.shuffle(selected_bot_ids)
    return tuple(
        SeatAssignment(seat_key=bot_id, bot_id=bot_id)
        for bot_id in selected_bot_ids
    )


def _build_balanced_cycle_entries(
    bot_ids: Sequence[str],
) -> list[tuple[SeatAssignment, ...]]:
    validate_bot_pool_for_matches(bot_ids)
    normalized_bot_ids = tuple(sorted(bot_ids))
    if len(normalized_bot_ids) == MIN_RATED_BOTS:
        base_trios = [tuple(
            list(
                SeatAssignment(seat_key=bot_id, bot_id=bot_id)
                for bot_id in normalized_bot_ids
            )
            + [
                SeatAssignment(
                    seat_key=FILLER_SEAT_KEY,
                    bot_id=FILLER_BOT_ID,
                    is_rated=False,
                )
            ]
        )]
    else:
        base_trios = [
            tuple(SeatAssignment(seat_key=bot_id, bot_id=bot_id) for bot_id in trio_bot_ids)
            for trio_bot_ids in itertools.combinations(
                normalized_bot_ids,
                players_per_game(),
            )
        ]

    cycle_entries: list[tuple[SeatAssignment, ...]] = []
    for trio in base_trios:
        cycle_entries.extend(
            tuple(permutation)
            for permutation in itertools.permutations(trio)
        )
    return cycle_entries


def schedule_cycle_size(
    bot_ids: Sequence[str],
    *,
    schedule_mode: str,
) -> int | None:
    if schedule_mode != "balanced":
        return None
    return len(_build_balanced_cycle_entries(bot_ids))


def iter_match_tasks(
    bot_ids: Sequence[str],
    *,
    alpha: int,
    seed: int | None,
    schedule_mode: str = DEFAULT_SCHEDULE_MODE,
) -> Iterable[MatchTask]:
    schedule_rng = random.Random(seed)
    match_index = 1
    while True:
        if schedule_mode == "random":
            cycle_entries = [
                build_random_match(
                    bot_ids,
                    rng=schedule_rng,
                )
            ]
        elif schedule_mode == "balanced":
            cycle_entries = _build_balanced_cycle_entries(bot_ids)
            schedule_rng.shuffle(cycle_entries)
        else:
            raise ValueError(f"unsupported schedule mode: {schedule_mode}")

        for participants in cycle_entries:
            yield MatchTask(
                match_index=match_index,
                participants=participants,
                alpha=alpha,
                seed=schedule_rng.randrange(2**63) if seed is not None else None,
            )
            match_index += 1


def _participant_label(participant: SeatAssignment) -> str:
    if participant.is_rated:
        return participant.bot_id
    return f"{participant.bot_id} (filler)"


def _format_matchup(participants: Sequence[SeatAssignment]) -> str:
    return " vs ".join(_participant_label(participant) for participant in participants)


def _compact_bot_name(bot_id: str) -> str:
    compact_names = {
        "optimal-bot": "opt",
        "random": "rnd",
        "greedy": "grd",
        "stupid": "stp",
        "o-one-trick-minmax": "o1",
    }
    if bot_id in compact_names:
        return compact_names[bot_id]

    omniscient_match = re.fullmatch(r"o-(\d+)-trick-minmax", bot_id)
    if omniscient_match:
        return f"o{omniscient_match.group(1)}"

    hidden_match = re.fullmatch(r"(\d+)-trick-minmax", bot_id)
    if hidden_match:
        return f"l{hidden_match.group(1)}"

    return bot_id[:6]


def _compact_participant_name(participant: SeatAssignment) -> str:
    suffix = "" if participant.is_rated else "*"
    return f"{_compact_bot_name(participant.bot_id)}{suffix}"


def _format_matchup_compact(participants: Sequence[SeatAssignment]) -> str:
    return " vs ".join(
        _compact_participant_name(participant)
        for participant in participants
    )


def _current_actor_name(controller: MatchController) -> str | None:
    if controller.session.phase == "auction":
        return controller.session.auction.current_bidder_name
    if controller.session.phase == "play":
        return controller.session.game.curr_player.name
    return None


def _build_match_progress_snapshot(
    controller: MatchController,
    task: MatchTask,
    *,
    started_at: float,
) -> MatchProgressSnapshot:
    phase = controller.session.phase
    bot_progress = controller.get_bot_progress()
    active_percent = None
    active_label = None
    active_detail = None
    if bot_progress.get("active"):
        active_percent = float(bot_progress.get("percent_complete") or 0.0)
        active_label = bot_progress.get("label")
        active_detail = bot_progress.get("detail")

    played_cards_completed = 0
    total_card_plays = len(task.participants) * 6
    if phase in {"play", "round_complete", "match_complete"}:
        round_state = controller.session.game.round_state
        played_cards_completed = sum(
            len(trick.plays) for trick in round_state.trick_history
        ) + len(round_state.current_trick.plays)

    return MatchProgressSnapshot(
        match_index=task.match_index,
        match_round_number=controller.session.round_number,
        alpha=task.alpha,
        player_count=len(task.participants),
        phase=phase,
        current_actor=_current_actor_name(controller),
        active_label=active_label,
        active_detail=active_detail,
        active_percent=active_percent,
        auction_actions_completed=len(controller.session.auction.state.bid_history),
        played_cards_completed=played_cards_completed,
        total_card_plays=total_card_plays,
        elapsed_seconds=max(0.0, time.perf_counter() - started_at),
    )


def _run_match_task_with_progress(
    task: MatchTask,
    progress_queue,
) -> MatchOutcome:
    if task.seed is not None:
        random.seed(task.seed)

    player_names = [f"Player {index}" for index in range(1, len(task.participants) + 1)]
    bots = {
        player_name: build_ready_bot(participant.bot_id, player_name)
        for player_name, participant in zip(player_names, task.participants)
    }
    controller = MatchController.create(
        num_players=len(task.participants),
        player_names=player_names,
        teams=[(player_name,) for player_name in player_names],
        bots=bots,
        target_score=TARGET_SCORE,
    )

    started_at = time.perf_counter()
    stop_event = threading.Event()

    def publish_progress_loop() -> None:
        while not stop_event.is_set():
            progress_queue.put(
                _build_match_progress_snapshot(
                    controller,
                    task,
                    started_at=started_at,
                )
            )
            stop_event.wait(0.1)

    publisher = threading.Thread(target=publish_progress_loop, daemon=True)
    publisher.start()
    try:
        match_result = Simulator(controller).run_match(task.alpha)
    finally:
        stop_event.set()
        publisher.join(timeout=0.2)
        progress_queue.put(
            _build_match_progress_snapshot(
                controller,
                task,
                started_at=started_at,
            )
        )

    elapsed_seconds = time.perf_counter() - started_at
    participant_by_player_name = dict(zip(player_names, task.participants))
    winner_seat_keys = tuple(
        sorted(
            participant_by_player_name[winner_name].seat_key
            for winner_name in match_result.winner_names
        )
    )
    scores_by_seat = tuple(
        (
            participant.seat_key,
            match_result.final_scores[player_name],
        )
        for player_name, participant in zip(player_names, task.participants)
    )
    scores_by_bot = tuple(
        (
            participant.bot_id,
            match_result.final_scores[player_name],
        )
        for player_name, participant in zip(player_names, task.participants)
        if participant.is_rated
    )
    return MatchOutcome(
        match_index=task.match_index,
        participants=task.participants,
        rounds_played=match_result.rounds_played,
        is_draw=match_result.is_draw,
        winner_seat_keys=winner_seat_keys,
        scores_by_seat=scores_by_seat,
        scores_by_bot=scores_by_bot,
        elapsed_seconds=elapsed_seconds,
        seed=task.seed,
    )


def compute_multiplayer_elo_deltas(
    ratings: dict[str, EloEntry],
    score_by_bot: dict[str, int],
    *,
    k_factor: float,
) -> dict[str, float]:
    if len(score_by_bot) < 2:
        return {bot_id: 0.0 for bot_id in score_by_bot}

    deltas = {bot_id: 0.0 for bot_id in score_by_bot}
    bot_ids = list(score_by_bot)
    comparison_scale = k_factor / (len(bot_ids) - 1)

    for bot_id, opponent_id, actual_score, expected_score in _iter_pairwise_results(
        ratings,
        score_by_bot,
    ):
        adjustment = comparison_scale * (actual_score - expected_score)
        deltas[bot_id] += adjustment
        deltas[opponent_id] -= adjustment

    return deltas


def _top_scoring_participant_keys(
    scores_by_seat: Sequence[tuple[str, int]],
) -> tuple[str, ...]:
    top_score = max(score for _, score in scores_by_seat)
    return tuple(
        seat_key
        for seat_key, score in scores_by_seat
        if score == top_score
    )


def _top_scoring_bot_ids(scores_by_bot: Sequence[tuple[str, int]]) -> tuple[str, ...]:
    top_score = max(score for _, score in scores_by_bot)
    return tuple(
        sorted(bot_id for bot_id, score in scores_by_bot if score == top_score)
    )


def _expected_pairwise_score(
    bot_rating: float,
    opponent_rating: float,
) -> float:
    return 1.0 / (
        1.0 + 10.0 ** ((opponent_rating - bot_rating) / 400.0)
    )


def _pairwise_information(expected_score: float) -> float:
    return (PAIRWISE_ELO_Q**2) * expected_score * (1.0 - expected_score)


def _iter_pairwise_results(
    ratings: dict[str, EloEntry],
    score_by_bot: dict[str, int],
) -> Iterable[tuple[str, str, float, float]]:
    bot_ids = list(score_by_bot)
    for index, bot_id in enumerate(bot_ids):
        for opponent_id in bot_ids[index + 1:]:
            bot_score = score_by_bot[bot_id]
            opponent_score = score_by_bot[opponent_id]
            if bot_score > opponent_score:
                actual_score = 1.0
            elif bot_score < opponent_score:
                actual_score = 0.0
            else:
                actual_score = 0.5

            expected_score = _expected_pairwise_score(
                ratings[bot_id].rating,
                ratings[opponent_id].rating,
            )
            yield bot_id, opponent_id, actual_score, expected_score


def _rating_confidence_half_width(entry: EloEntry) -> float | None:
    if entry.information <= 0:
        return None
    return PAIRWISE_CI_Z / math.sqrt(entry.information)


def _format_confidence_half_width(entry: EloEntry) -> str:
    confidence_half_width = _rating_confidence_half_width(entry)
    if confidence_half_width is None:
        return "--"
    return f"±{confidence_half_width:.1f}"


def apply_match_outcome_to_elo(
    ratings: dict[str, EloEntry],
    outcome: MatchOutcome,
    *,
    k_factor: float,
) -> dict[str, float]:
    score_by_bot = dict(outcome.scores_by_bot)
    pairwise_results = tuple(_iter_pairwise_results(ratings, score_by_bot))
    deltas = compute_multiplayer_elo_deltas(
        ratings,
        score_by_bot,
        k_factor=k_factor,
    )

    for bot_id, delta in deltas.items():
        ratings[bot_id].rating += delta
        ratings[bot_id].games_played += 1

    for bot_id, opponent_id, _actual_score, expected_score in pairwise_results:
        information = _pairwise_information(expected_score)
        ratings[bot_id].pairwise_games += 1
        ratings[opponent_id].pairwise_games += 1
        ratings[bot_id].information += information
        ratings[opponent_id].information += information

    top_bots = set(_top_scoring_bot_ids(outcome.scores_by_bot))
    for bot_id, _ in outcome.scores_by_bot:
        if bot_id in top_bots:
            if len(top_bots) == 1:
                ratings[bot_id].wins += 1
            else:
                ratings[bot_id].draws += 1
        else:
            ratings[bot_id].losses += 1

    return deltas


def _format_scores_by_seat(
    participants: Sequence[SeatAssignment],
    scores_by_seat: Sequence[tuple[str, int]],
) -> str:
    participant_by_seat_key = {
        participant.seat_key: participant
        for participant in participants
    }
    ordered_scores = sorted(scores_by_seat, key=lambda item: (-item[1], item[0]))
    return " | ".join(
        f"{_participant_label(participant_by_seat_key[seat_key])}:{score}"
        for seat_key, score in ordered_scores
    )


def _format_bot_deltas(
    deltas: dict[str, float],
    ratings: dict[str, EloEntry],
) -> str:
    ordered = sorted(deltas.items(), key=lambda item: (-item[1], item[0]))
    return " | ".join(
        f"{bot_id} {delta:+.1f} -> {ratings[bot_id].rating:.1f}"
        for bot_id, delta in ordered
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_progress_bar(
    progress: float,
    *,
    width: int = 24,
) -> str:
    normalized = _clamp(progress, 0.0, 1.0)
    filled = min(width, int(normalized * width))
    if 0.0 < normalized < 1.0 and filled >= width:
        filled = width - 1
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _truncate(value: str | None, max_length: int) -> str:
    if not value:
        return ""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def _snapshot_progress(snapshot: MatchProgressSnapshot) -> float:
    if snapshot.active_percent is not None:
        return _clamp(snapshot.active_percent / 100.0, 0.0, 0.999)
    if snapshot.phase in {"round_complete", "match_complete"}:
        return 1.0
    if snapshot.phase == "play" and snapshot.total_card_plays > 0:
        return _clamp(
            snapshot.played_cards_completed / snapshot.total_card_plays,
            0.0,
            0.999,
        )
    if snapshot.phase == "auction":
        return _clamp(
            snapshot.auction_actions_completed / max(1, snapshot.player_count),
            0.0,
            0.999,
        )
    return 0.0


def _snapshot_status_text(snapshot: MatchProgressSnapshot) -> str:
    if snapshot.active_percent is not None:
        return snapshot.active_label or "thinking"
    return snapshot.phase


def _snapshot_detail_text(snapshot: MatchProgressSnapshot) -> str:
    if snapshot.active_percent is not None:
        return _truncate(snapshot.active_detail, 36)
    if snapshot.phase == "play":
        return f"cards {snapshot.played_cards_completed}/{snapshot.total_card_plays}"
    if snapshot.phase == "auction":
        return f"bids {snapshot.auction_actions_completed}"
    return ""


def _compact_status_text(snapshot: MatchProgressSnapshot | None) -> str:
    if snapshot is None:
        return "queue"
    if snapshot.active_label:
        depth_match = re.search(r"Searching (\d+) tricks ahead", snapshot.active_label)
        if depth_match:
            return f"s{depth_match.group(1)}"
        return _truncate(snapshot.active_label.lower(), 6)

    phase_names = {
        "auction": "bid",
        "play": "play",
        "round_complete": "round",
        "match_complete": "match",
    }
    return phase_names.get(snapshot.phase, _truncate(snapshot.phase, 6))


def _compact_actor_name(actor_name: str | None) -> str:
    if not actor_name:
        return "--"
    player_match = re.fullmatch(r"Player (\d+)", actor_name)
    if player_match:
        return f"P{player_match.group(1)}"
    return _truncate(actor_name, 4)


def _compact_detail_text(snapshot: MatchProgressSnapshot | None) -> str:
    if snapshot is None:
        return ""

    detail = _snapshot_detail_text(snapshot)
    world_match = re.fullmatch(r"([A-Z0-9]+) in world (\d+)/(\d+)", detail)
    if world_match:
        return f"{world_match.group(1)} w{world_match.group(2)}/{world_match.group(3)}"

    cards_match = re.fullmatch(r"cards (\d+)/(\d+)", detail)
    if cards_match:
        return f"c{cards_match.group(1)}/{cards_match.group(2)}"

    bids_match = re.fullmatch(r"bids (\d+)", detail)
    if bids_match:
        return f"b{bids_match.group(1)}"

    return _truncate(detail, 12)


def _format_schedule_summary(
    *,
    schedule_mode: str,
    completed_games: int,
    cycle_size: int | None,
) -> str:
    if schedule_mode != "balanced" or cycle_size in {None, 0}:
        return f"Schedule | {schedule_mode}"

    cycle_number = (completed_games // cycle_size) + 1
    cycle_progress = completed_games % cycle_size
    return (
        f"Schedule | balanced | cycle {cycle_number} | "
        f"{cycle_progress}/{cycle_size} in current cycle"
    )


def _render_match_progress_line(
    *,
    slot_index: int,
    task: MatchTask | None,
    progress_snapshot: MatchProgressSnapshot | None,
) -> str:
    if task is None:
        return f"  s{slot_index:02d} idle"

    if progress_snapshot is not None:
        progress = _snapshot_progress(progress_snapshot)
        elapsed_text = f"{int(progress_snapshot.elapsed_seconds):>3d}s"
        round_text = f"r{progress_snapshot.match_round_number}/{task.alpha}"
    else:
        progress = 0.0
        elapsed_text = "  --s"
        round_text = f"r1/{task.alpha}"

    status = _compact_status_text(progress_snapshot)
    actor_text = _compact_actor_name(
        None if progress_snapshot is None else progress_snapshot.current_actor
    )
    detail_text = _compact_detail_text(progress_snapshot)
    matchup_text = _truncate(_format_matchup_compact(task.participants), 34)
    detail_suffix = f" {detail_text}" if detail_text else ""
    return (
        f"  s{slot_index:02d} g{task.match_index:03d} "
        f"{progress * 100:>5.1f}% {elapsed_text} "
        f"{round_text:<6} {actor_text:<3} {status:<5} "
        f"{matchup_text:<34}{detail_suffix}"
    )


def _render_live_leaderboard_lines(
    ratings: dict[str, EloEntry],
    *,
    recent_deltas: dict[str, float] | None,
) -> list[str]:
    ranked_rows = sorted(
        ratings.items(),
        key=lambda item: (-item[1].rating, item[0]),
    )
    lines = [
        "Elo",
        "  rk bot   elo   c95    g   w   d   l     dE",
    ]
    for rank, (bot_id, entry) in enumerate(ranked_rows, start=1):
        delta = 0.0 if recent_deltas is None else recent_deltas.get(bot_id, 0.0)
        delta_text = "" if recent_deltas is None or abs(delta) < 0.05 else f"{delta:+.1f}"
        confidence_text = _format_confidence_half_width(entry)
        lines.append(
            f"  {rank:>2} {_compact_bot_name(bot_id):<4} "
            f"{entry.rating:>6.1f} {confidence_text:>5} {entry.games_played:>5} "
            f"{entry.wins:>3} {entry.draws:>3} {entry.losses:>3} {delta_text:>6}"
        )
    return lines


def _render_live_progress_lines(
    *,
    slot_count: int,
    active_tasks_by_slot: dict[int, MatchTask],
    progress_by_match_id: dict[int, MatchProgressSnapshot],
    completed_games: int,
    max_games: int | None,
    ratings: dict[str, EloEntry],
    recent_deltas: dict[str, float] | None,
    last_result_text: str | None,
    schedule_mode: str,
    cycle_size: int | None,
) -> list[str]:
    if max_games is None:
        overall_summary = f"{completed_games} games complete"
    else:
        overall_summary = f"{completed_games}/{max_games} games complete"

    lines = [
        f"Continuous sim | {overall_summary} | {len(active_tasks_by_slot)}/{slot_count} running",
        f"Last result | {last_result_text or 'none yet'}",
        _format_schedule_summary(
            schedule_mode=schedule_mode,
            completed_games=completed_games,
            cycle_size=cycle_size,
        ),
        "",
    ]
    lines.extend(_render_live_leaderboard_lines(ratings, recent_deltas=recent_deltas))
    lines.append("")
    lines.append("Matches")
    for slot_index in range(1, slot_count + 1):
        task = active_tasks_by_slot.get(slot_index)
        progress_snapshot = (
            None if task is None else progress_by_match_id.get(task.match_index)
        )
        lines.append(
            _render_match_progress_line(
                slot_index=slot_index,
                task=task,
                progress_snapshot=progress_snapshot,
            )
        )
    return lines


def _write_progress_snapshot(
    display: ProgressDisplay,
    *,
    slot_count: int,
    active_tasks_by_slot: dict[int, MatchTask],
    progress_by_match_id: dict[int, MatchProgressSnapshot],
    completed_games: int,
    max_games: int | None,
    ratings: dict[str, EloEntry],
    recent_deltas: dict[str, float] | None,
    last_result_text: str | None,
    schedule_mode: str,
    cycle_size: int | None,
) -> None:
    display.render(
        _render_live_progress_lines(
            slot_count=slot_count,
            active_tasks_by_slot=active_tasks_by_slot,
            progress_by_match_id=progress_by_match_id,
            completed_games=completed_games,
            max_games=max_games,
            ratings=ratings,
            recent_deltas=recent_deltas,
            last_result_text=last_result_text,
            schedule_mode=schedule_mode,
            cycle_size=cycle_size,
        )
    )


def _drain_progress_queue(
    progress_queue,
    progress_by_match_id: dict[int, MatchProgressSnapshot],
) -> None:
    while True:
        try:
            snapshot = progress_queue.get_nowait()
        except queue.Empty:
            return
        progress_by_match_id[snapshot.match_index] = snapshot


def render_match_summary(
    outcome: MatchOutcome,
    *,
    deltas: dict[str, float],
    ratings: dict[str, EloEntry],
) -> str:
    participant_by_seat_key = {
        participant.seat_key: participant
        for participant in outcome.participants
    }
    top_participants = _top_scoring_participant_keys(outcome.scores_by_seat)
    if outcome.is_draw:
        if len(top_participants) == 1:
            winner_text = (
                f"{_participant_label(participant_by_seat_key[top_participants[0]])} "
                "(alpha draw)"
            )
        else:
            winner_text = (
                f"{' | '.join(_participant_label(participant_by_seat_key[seat_key]) for seat_key in top_participants)} "
                "(tied alpha draw)"
            )
    else:
        winner_text = (
            " | ".join(
                _participant_label(participant_by_seat_key[seat_key])
                for seat_key in outcome.winner_seat_keys
            )
            if outcome.winner_seat_keys
            else " | ".join(
                _participant_label(participant_by_seat_key[seat_key])
                for seat_key in top_participants
            )
        )

    lines = [
        (
            f"[game {outcome.match_index}] "
            f"players={len(outcome.participants)} rounds={outcome.rounds_played} "
            f"elapsed={outcome.elapsed_seconds:.2f}s winner={winner_text}"
        ),
        f"  matchup: {_format_matchup(outcome.participants)}",
        f"  scores: {_format_scores_by_seat(outcome.participants, outcome.scores_by_seat)}",
        f"  elo: {_format_bot_deltas(deltas, ratings)}",
    ]
    return "\n".join(lines)


def _format_recent_result(outcome: MatchOutcome) -> str:
    participant_by_seat_key = {
        participant.seat_key: participant
        for participant in outcome.participants
    }
    winning_seat_keys = outcome.winner_seat_keys or _top_scoring_participant_keys(
        outcome.scores_by_seat
    )
    winner_text = " | ".join(
        _truncate(
            _compact_participant_name(participant_by_seat_key[seat_key]),
            10,
        )
        for seat_key in winning_seat_keys
    )
    losing_participants = tuple(
        participant_by_seat_key[seat_key]
        for seat_key, _ in outcome.scores_by_seat
        if seat_key not in winning_seat_keys
    )
    if outcome.is_draw and len(winning_seat_keys) > 1:
        verb = "tied at alpha with"
    elif outcome.is_draw:
        verb = "led at alpha over"
    else:
        verb = "beat"
    return (
        f"g{outcome.match_index:03d} "
        f"{winner_text} "
        f"{verb} "
        f"{_truncate(_format_matchup_compact(losing_participants), 18) if losing_participants else 'field'} "
        f"({outcome.elapsed_seconds:.1f}s)"
    )


def render_leaderboard(
    ratings: dict[str, EloEntry],
    *,
    recent_deltas: dict[str, float] | None = None,
) -> str:
    ranked_rows = sorted(
        ratings.items(),
        key=lambda item: (-item[1].rating, item[0]),
    )
    id_width = max(len("bot"), *(len(bot_id) for bot_id in ratings))
    lines = [
        (
            f"{'#':>2}  {'bot':<{id_width}}  {'elo':>8}  {'ci95':>6}  {'games':>5}  "
            f"{'W':>3}  {'D':>3}  {'L':>3}  {'Δ':>6}"
        )
    ]
    for rank, (bot_id, entry) in enumerate(ranked_rows, start=1):
        delta = 0.0 if recent_deltas is None else recent_deltas.get(bot_id, 0.0)
        delta_text = "" if recent_deltas is None or abs(delta) < 0.05 else f"{delta:+.1f}"
        confidence_text = _format_confidence_half_width(entry)
        lines.append(
            f"{rank:>2}  {bot_id:<{id_width}}  {entry.rating:>8.1f}  {confidence_text:>6}  "
            f"{entry.games_played:>5}  {entry.wins:>3}  {entry.draws:>3}  "
            f"{entry.losses:>3}  {delta_text:>6}"
        )
    return "\n".join(lines)


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def _build_initial_ratings(
    bot_ids: Sequence[str],
    *,
    initial_rating: float,
    persisted_ratings: dict[str, EloEntry] | None = None,
    persisted_fingerprints: dict[str, str] | None = None,
) -> dict[str, EloEntry]:
    ratings: dict[str, EloEntry] = {}
    for bot_id in bot_ids:
        current_fingerprint = get_ready_bot_rating_fingerprint(bot_id)
        persisted_fingerprint = (
            bot_id
            if persisted_fingerprints is None
            else persisted_fingerprints.get(bot_id, bot_id)
        )
        entry = None if persisted_ratings is None else persisted_ratings.get(bot_id)
        if entry is not None and persisted_fingerprint != current_fingerprint:
            entry = None
        if entry is None:
            entry = EloEntry(rating=initial_rating)
            if persisted_ratings is not None:
                persisted_ratings[bot_id] = entry
        if persisted_fingerprints is not None:
            persisted_fingerprints[bot_id] = current_fingerprint
        ratings[bot_id] = entry
    return ratings


def _serialize_elo_entry(entry: EloEntry) -> dict[str, float | int]:
    return {
        "rating": round(entry.rating, 6),
        "games_played": entry.games_played,
        "wins": entry.wins,
        "draws": entry.draws,
        "losses": entry.losses,
        "pairwise_games": entry.pairwise_games,
        "information": round(entry.information, 12),
    }


def _deserialize_elo_entry(
    bot_id: str,
    payload: object,
) -> EloEntry:
    if not isinstance(payload, dict):
        raise ValueError(f"invalid Elo entry for {bot_id}: expected object")

    try:
        rating = float(payload["rating"])
        games_played = int(payload.get("games_played", 0))
        wins = int(payload.get("wins", 0))
        draws = int(payload.get("draws", 0))
        losses = int(payload.get("losses", 0))
        pairwise_games = int(payload.get("pairwise_games", games_played * 2))
        information = float(
            payload.get(
                "information",
                pairwise_games * _pairwise_information(0.5),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Elo entry for {bot_id}") from exc

    if rating <= 0:
        raise ValueError(f"invalid Elo entry for {bot_id}: rating must be positive")
    if min(games_played, wins, draws, losses, pairwise_games) < 0:
        raise ValueError(
            f"invalid Elo entry for {bot_id}: counters must be non-negative"
        )
    if wins + draws + losses > games_played:
        raise ValueError(
            f"invalid Elo entry for {bot_id}: result counters exceed games played"
        )
    if information < 0:
        raise ValueError(
            f"invalid Elo entry for {bot_id}: information must be non-negative"
        )

    return EloEntry(
        rating=rating,
        games_played=games_played,
        wins=wins,
        draws=draws,
        losses=losses,
        pairwise_games=pairwise_games,
        information=information,
    )


def load_persisted_rating_state(
    elo_file: Path,
) -> tuple[dict[str, EloEntry], dict[str, str]]:
    if not elo_file.exists():
        return {}, {}

    try:
        payload = json.loads(elo_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Elo JSON in {elo_file}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"invalid Elo JSON in {elo_file}: expected top-level object")

    ratings_payload = payload.get("ratings", payload)
    version = payload.get("version")
    if version is not None and version != ELO_FILE_VERSION:
        raise ValueError(
            f"unsupported Elo file version in {elo_file}: {version}"
        )
    if not isinstance(ratings_payload, dict):
        raise ValueError(f"invalid Elo JSON in {elo_file}: expected ratings object")
    fingerprints_payload = payload.get("fingerprints", {})
    if not isinstance(fingerprints_payload, dict):
        raise ValueError(
            f"invalid Elo JSON in {elo_file}: expected fingerprints object"
        )

    ratings: dict[str, EloEntry] = {}
    fingerprints: dict[str, str] = {}
    for bot_id, entry_payload in ratings_payload.items():
        if not isinstance(bot_id, str) or not bot_id:
            raise ValueError(f"invalid Elo JSON in {elo_file}: bad bot id")
        ratings[bot_id] = _deserialize_elo_entry(bot_id, entry_payload)
        fingerprint = fingerprints_payload.get(bot_id, bot_id)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"invalid Elo JSON in {elo_file}: bad fingerprint for {bot_id}"
            )
        fingerprints[bot_id] = fingerprint
    return ratings, fingerprints


def load_persisted_ratings(elo_file: Path) -> dict[str, EloEntry]:
    ratings, _ = load_persisted_rating_state(elo_file)
    return ratings


def save_persisted_ratings(
    elo_file: Path,
    ratings: dict[str, EloEntry],
    *,
    bot_fingerprints: dict[str, str] | None = None,
) -> None:
    elo_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ELO_FILE_VERSION,
        "ratings": {
            bot_id: _serialize_elo_entry(entry)
            for bot_id, entry in sorted(ratings.items())
        },
    }
    if bot_fingerprints is not None:
        payload["fingerprints"] = {
            bot_id: fingerprint
            for bot_id, fingerprint in sorted(bot_fingerprints.items())
        }
    temp_file = elo_file.with_name(f".{elo_file.name}.tmp")
    temp_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_file, elo_file)


def _build_parallel_executor(
    workers: int,
) -> tuple[ProcessPoolExecutor | ThreadPoolExecutor, str]:
    try:
        mp_context = mp.get_context("spawn")
        return (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
            ),
            "process",
        )
    except (OSError, PermissionError, ValueError):
        return (ThreadPoolExecutor(max_workers=workers), "thread")


def run_continuous_sim(
    *,
    bot_ids: Sequence[str],
    alpha: int,
    k_factor: float,
    initial_rating: float,
    max_games: int | None,
    duration_seconds: float | None,
    workers: int,
    seed: int | None,
    leaderboard_every: int,
    elo_file: Path | None = None,
    load_from_elo_file: bool = True,
    schedule_mode: str = DEFAULT_SCHEDULE_MODE,
) -> ContinuousSimResult:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if k_factor <= 0:
        raise ValueError("k_factor must be positive")
    if initial_rating <= 0:
        raise ValueError("initial_rating must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if leaderboard_every <= 0:
        raise ValueError("leaderboard_every must be positive")
    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be positive when provided")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive when provided")

    resolved_bot_ids = list(bot_ids)
    validate_bot_pool_for_matches(resolved_bot_ids)
    if elo_file is None:
        persisted_ratings = None
        persisted_fingerprints = None
    elif load_from_elo_file:
        persisted_ratings, persisted_fingerprints = load_persisted_rating_state(
            elo_file
        )
    else:
        persisted_ratings = {}
        persisted_fingerprints = {}
    ratings = _build_initial_ratings(
        resolved_bot_ids,
        initial_rating=initial_rating,
        persisted_ratings=persisted_ratings,
        persisted_fingerprints=persisted_fingerprints,
    )
    if elo_file is not None and persisted_ratings is not None:
        save_persisted_ratings(
            elo_file,
            persisted_ratings,
            bot_fingerprints=persisted_fingerprints,
        )
    started_at = time.perf_counter()
    deadline = (
        time.monotonic() + duration_seconds
        if duration_seconds is not None
        else None
    )
    task_iter = iter_match_tasks(
        resolved_bot_ids,
        alpha=alpha,
        seed=seed,
        schedule_mode=schedule_mode,
    )
    completed_games = 0
    scheduled_games = 0
    progress_display = ProgressDisplay()
    slot_count = workers if max_games is None else min(workers, max_games)
    cycle_size = schedule_cycle_size(
        resolved_bot_ids,
        schedule_mode=schedule_mode,
    )
    recent_deltas: dict[str, float] | None = None
    last_result_text: str | None = None

    def should_schedule_more() -> bool:
        if max_games is not None and scheduled_games >= max_games:
            return False
        if deadline is not None and time.monotonic() >= deadline:
            return False
        return True

    def handle_outcome(outcome: MatchOutcome) -> None:
        nonlocal completed_games, recent_deltas, last_result_text
        completed_games += 1
        recent_deltas = apply_match_outcome_to_elo(
            ratings,
            outcome,
            k_factor=k_factor,
        )
        last_result_text = _format_recent_result(outcome)
        if elo_file is not None and persisted_ratings is not None:
            save_persisted_ratings(
                elo_file,
                persisted_ratings,
                bot_fingerprints=persisted_fingerprints,
            )

    def next_task() -> MatchTask:
        nonlocal scheduled_games
        task = next(task_iter)
        scheduled_games += 1
        return task

    def render_progress(
        *,
        active_tasks_by_slot: dict[int, MatchTask],
        progress_by_match_id: dict[int, MatchProgressSnapshot],
    ) -> None:
        _write_progress_snapshot(
            progress_display,
            slot_count=slot_count,
            active_tasks_by_slot=active_tasks_by_slot,
            progress_by_match_id=progress_by_match_id,
            completed_games=completed_games,
            max_games=max_games,
            ratings=ratings,
            recent_deltas=recent_deltas,
            last_result_text=last_result_text,
            schedule_mode=schedule_mode,
            cycle_size=cycle_size,
        )

    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
    executor_kind = "serial"
    progress_manager = None
    try:
        if workers == 1:
            executor = ThreadPoolExecutor(max_workers=1)
            executor_kind = "serial"
            progress_queue = queue.Queue()
        else:
            executor, executor_kind = _build_parallel_executor(workers)
            if executor_kind == "process":
                progress_manager = mp.Manager()
                progress_queue = progress_manager.Queue()
            else:
                progress_queue = queue.Queue()

        active_tasks_by_slot: dict[int, MatchTask] = {}
        progress_by_match_id: dict[int, MatchProgressSnapshot] = {}
        futures_by_slot: dict[int, object] = {}
        slots_by_future: dict[object, int] = {}
        disable_parallel_thread_seeds = executor_kind == "thread" and workers > 1

        def submit_task(task: MatchTask):
            submitted_task = (
                replace(task, seed=None)
                if disable_parallel_thread_seeds and task.seed is not None
                else task
            )
            return executor.submit(
                _run_match_task_with_progress,
                submitted_task,
                progress_queue,
            )

        for slot_index in range(1, slot_count + 1):
            if not should_schedule_more():
                break
            task = next_task()
            active_tasks_by_slot[slot_index] = task
            future = submit_task(task)
            futures_by_slot[slot_index] = future
            slots_by_future[future] = slot_index

        render_progress(
            active_tasks_by_slot=active_tasks_by_slot,
            progress_by_match_id=progress_by_match_id,
        )

        while slots_by_future:
            completed_futures, _ = wait(
                set(slots_by_future),
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            _drain_progress_queue(progress_queue, progress_by_match_id)

            if not completed_futures:
                render_progress(
                    active_tasks_by_slot=active_tasks_by_slot,
                    progress_by_match_id=progress_by_match_id,
                )
                continue

            for future in completed_futures:
                slot_index = slots_by_future.pop(future)
                futures_by_slot.pop(slot_index, None)
                task = active_tasks_by_slot.pop(slot_index)
                outcome = future.result()
                progress_by_match_id.pop(outcome.match_index, None)
                handle_outcome(outcome)

                if should_schedule_more():
                    next_match = next_task()
                    active_tasks_by_slot[slot_index] = next_match
                    next_future = submit_task(next_match)
                    futures_by_slot[slot_index] = next_future
                    slots_by_future[next_future] = slot_index

            _drain_progress_queue(progress_queue, progress_by_match_id)
            if active_tasks_by_slot:
                render_progress(
                    active_tasks_by_slot=active_tasks_by_slot,
                    progress_by_match_id=progress_by_match_id,
                )

        progress_display.clear()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if progress_manager is not None:
            progress_manager.shutdown()
        progress_display.clear()

    if elo_file is not None and persisted_ratings is not None:
        save_persisted_ratings(
            elo_file,
            persisted_ratings,
            bot_fingerprints=persisted_fingerprints,
        )

    ended_at = time.perf_counter()
    return ContinuousSimResult(
        games_completed=completed_games,
        started_at=started_at,
        ended_at=ended_at,
        ratings=ratings,
        executor_kind=executor_kind,
        schedule_mode=schedule_mode,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously simulate console-only Smear three-player matches across the ready "
            "bot pool and update live Elo ratings after every completed game."
        ),
    )
    parser.add_argument(
        "--bots",
        nargs="*",
        default=None,
        help=(
            "optional subset of ready bot ids to include; "
            "defaults to every visible ready bot"
        ),
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="include hidden and legacy bot ids in the default bot pool",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="stop after this many completed matches; default is to run until interrupted",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="optional wall-clock stop limit in hours",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=DEFAULT_ALPHA,
        help="maximum rounds per match before declaring a draw",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_worker_count(),
        help="number of worker processes to run in parallel",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional seed for deterministic bot sampling, filler seating, and per-match RNG",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=DEFAULT_K_FACTOR,
        help="Elo K-factor applied to the multiplayer pairwise update",
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=DEFAULT_INITIAL_ELO,
        help="starting Elo rating for every bot",
    )
    parser.add_argument(
        "--leaderboard-every",
        type=int,
        default=1,
        help="print the full leaderboard after every N completed matches",
    )
    parser.add_argument(
        "--elo-file",
        type=Path,
        default=DEFAULT_ELO_FILE,
        help="JSON file used to load and persist Elo across runs",
    )
    parser.add_argument(
        "--fresh-ratings",
        action="store_true",
        help=(
            "ignore any saved Elo JSON at startup and begin this run from "
            "--initial-rating instead"
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=("balanced", "random"),
        default=DEFAULT_SCHEDULE_MODE,
        help=(
            "match scheduling strategy; balanced cycles every trio and seat order "
            "before repeating"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    duration_seconds = (
        None if args.duration_hours is None else args.duration_hours * 3600.0
    )
    bot_ids = resolve_requested_bot_ids(
        requested_bot_ids=args.bots,
        include_hidden=args.include_hidden,
    )
    player_count = players_per_game()
    uses_filler = len(bot_ids) == MIN_RATED_BOTS

    print(
        "Starting continuous simulation "
        f"with {len(bot_ids)} bots, format=3-player free-for-all "
        f"({player_count} seats{' with random filler' if uses_filler else ''}), "
        f"workers={args.workers}, alpha={args.alpha}, schedule={args.schedule}, "
        f"k_factor={args.k_factor}, seed={args.seed}",
        flush=True,
    )
    print(f"Bot pool: {', '.join(bot_ids)}", flush=True)
    print(f"Elo file: {args.elo_file}", flush=True)
    print(
        "Elo startup: "
        + (
            "fresh from --initial-rating"
            if args.fresh_ratings
            else "load from JSON if present"
        ),
        flush=True,
    )
    if args.games is None and duration_seconds is None:
        print("Run limit: until interrupted", flush=True)
    else:
        limit_parts: list[str] = []
        if args.games is not None:
            limit_parts.append(f"{args.games} games")
        if duration_seconds is not None:
            limit_parts.append(f"{args.duration_hours} hours")
        print(f"Run limit: {' or '.join(limit_parts)}", flush=True)

    try:
        result = run_continuous_sim(
            bot_ids=bot_ids,
            alpha=args.alpha,
            k_factor=args.k_factor,
            initial_rating=args.initial_rating,
            max_games=args.games,
            duration_seconds=duration_seconds,
            workers=args.workers,
            seed=args.seed,
            leaderboard_every=args.leaderboard_every,
            elo_file=args.elo_file,
            load_from_elo_file=not args.fresh_ratings,
            schedule_mode=args.schedule,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2

    if result.executor_kind != "serial":
        print(f"Parallel executor: {result.executor_kind}", flush=True)

    print(
        "\nFinal leaderboard "
        f"after {result.games_completed} game"
        f"{'' if result.games_completed == 1 else 's'} "
        f"({result.elapsed_seconds:.2f}s):",
        flush=True,
    )
    print(render_leaderboard(result.ratings), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
