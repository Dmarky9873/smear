from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import random
from pathlib import Path
import sys
import time
import threading
from typing import Callable

try:
    from .bots.human_information_minimax_n_trick_bot import HumanInformationMinimaxNTrickPlayer
    from .bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from .bots.neural_3p_bot import NeuralThreePlayerBot, NeuralThreePlayerV1Bot
    from .bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from .bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from .bots.optimal_bot import OptimalBotPlayer
    from .bots.optimal_bot_tuning import OptimalBotCandidate
    from .bots.registry import build_ready_bot, get_ready_bot_spec
    from .bots.search_eval import evaluate_terminal_round_utility
    from .engine import apply_auction_action_for_search, apply_trick_action_to_state
    from .gameplay import MatchController
    from .models import AuctionEvent, Card, Play
except ImportError:
    from bots.human_information_minimax_n_trick_bot import HumanInformationMinimaxNTrickPlayer
    from bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from bots.neural_3p_bot import NeuralThreePlayerBot, NeuralThreePlayerV1Bot
    from bots.neural_3p_features import (
        encode_auction_candidate,
        encode_auction_state,
        encode_play_candidate,
        encode_play_state,
        ordered_legal_auction_actions,
        ordered_legal_cards,
    )
    from bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
    from bots.optimal_bot import OptimalBotPlayer
    from bots.optimal_bot_tuning import OptimalBotCandidate
    from bots.registry import build_ready_bot, get_ready_bot_spec
    from bots.search_eval import evaluate_terminal_round_utility
    from engine import apply_auction_action_for_search, apply_trick_action_to_state
    from gameplay import MatchController
    from models import AuctionEvent, Card, Play


DEFAULT_OUTPUT = NeuralThreePlayerBot.MODEL_FILE
DEFAULT_PLAY_HIDDEN_DIM = 48
DEFAULT_AUCTION_HIDDEN_DIM = 32
DEFAULT_PLAY_VALUE_HIDDEN_DIM = 48
DEFAULT_AUCTION_VALUE_HIDDEN_DIM = 32
DEFAULT_TEACHER_POOL = "optimal-bot:4,1-trick-minmax:2,greedy:1"
DEFAULT_PLAY_VALUE_WEIGHT = 0.8
DEFAULT_AUCTION_VALUE_WEIGHT = 0.55
DEFAULT_PLAY_ROLLOUT_DEPTH = 3
DEFAULT_AUCTION_ROLLOUT_DEPTH = 1
DEFAULT_WARM_START_LR_SCALE = 0.35
DEFAULT_WARM_START_EPOCH_SCALE = 0.6
DEFAULT_GRADIENT_CLIP = 3.0
DEFAULT_STUDENT_AGREEMENT_KEEP_PROB = 0.2
DEFAULT_TEACHER_SAMPLE_SCALE = 0.6
DEFAULT_TEACHER_TARGET_TEMPERATURE = 0.35
THREE_PLAYER_NAMES = ["Player 1", "Player 2", "Player 3"]


@dataclass(frozen=True)
class TeacherSpec:
    bot_id: str
    weight: float = 1.0


@dataclass(frozen=True)
class PendingValueExample:
    player_name: str
    features: list[float]
    weight: float = 1.0


@dataclass(frozen=True)
class TeacherDecision:
    action: Card | AuctionEvent
    bot_id: str
    candidate_actions: list[Card | AuctionEvent] | None = None
    target_distribution: list[float] | None = None


@dataclass
class TrainingDataset:
    play_policy_examples: list[ChoiceExample] = field(default_factory=list)
    auction_policy_examples: list[ChoiceExample] = field(default_factory=list)
    play_value_examples: list[RegressionExample] = field(default_factory=list)
    auction_value_examples: list[RegressionExample] = field(default_factory=list)

    def extend(self, other: "TrainingDataset") -> None:
        self.play_policy_examples.extend(other.play_policy_examples)
        self.auction_policy_examples.extend(other.auction_policy_examples)
        self.play_value_examples.extend(other.play_value_examples)
        self.auction_value_examples.extend(other.auction_value_examples)


def _scaled_count(count: int, scale: float, *, minimum: int = 1) -> int:
    if count <= 0:
        return 0
    if scale <= 0:
        raise ValueError("scale must be positive")
    return max(minimum, int(round(count * scale)))


def _scaled_optimal_candidate(
    candidate: OptimalBotCandidate,
    *,
    sample_scale: float,
) -> OptimalBotCandidate:
    return OptimalBotCandidate(
        id=f"{candidate.id}-train-x{sample_scale:.2f}",
        depth=candidate.depth,
        play_determinization_samples=_scaled_count(
            candidate.play_determinization_samples,
            sample_scale,
        ),
        min_play_determinization_samples=min(
            candidate.min_play_determinization_samples,
            _scaled_count(
                candidate.play_determinization_samples,
                sample_scale,
            ),
        ),
        auction_determinization_samples=_scaled_count(
            candidate.auction_determinization_samples,
            sample_scale,
        ),
        three_player_auction_determinization_samples=_scaled_count(
            candidate.three_player_auction_determinization_samples,
            sample_scale,
        ),
    )


def _configure_teacher_bot_for_training(
    bot,
    *,
    sample_scale: float,
):
    if sample_scale >= 0.999:
        return bot

    if isinstance(bot, OptimalBotPlayer):
        bot.THREE_PLAYER_PROFILE = _scaled_optimal_candidate(
            bot.THREE_PLAYER_PROFILE,
            sample_scale=sample_scale,
        )
        bot.MULTIPLAYER_PROFILE = _scaled_optimal_candidate(
            bot.MULTIPLAYER_PROFILE,
            sample_scale=sample_scale,
        )
        bot.PROFILE = bot.MULTIPLAYER_PROFILE
        return bot

    if isinstance(bot, HumanInformationMinimaxNTrickPlayer):
        original_samples = int(bot.DETERMINIZATION_SAMPLES)
        bot.DETERMINIZATION_SAMPLES = _scaled_count(
            original_samples,
            sample_scale,
        )
        bot.MIN_DETERMINIZATION_SAMPLES = min(
            int(bot.MIN_DETERMINIZATION_SAMPLES),
            bot.DETERMINIZATION_SAMPLES,
        )
        bot.AUCTION_DETERMINIZATION_SAMPLES = _scaled_count(
            int(bot.AUCTION_DETERMINIZATION_SAMPLES),
            sample_scale,
        )
        bot.THREE_PLAYER_AUCTION_DETERMINIZATION_SAMPLES = _scaled_count(
            int(bot.THREE_PLAYER_AUCTION_DETERMINIZATION_SAMPLES),
            sample_scale,
        )
        return bot

    if isinstance(bot, OmniscientMinimaxNTrickPlayer):
        bot.AUCTION_DETERMINIZATION_SAMPLES = _scaled_count(
            int(bot.AUCTION_DETERMINIZATION_SAMPLES),
            sample_scale,
        )
        return bot

    return bot


class TrainingBotProvider:
    def __init__(
        self,
        *,
        teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    ):
        self._teacher_sample_scale = teacher_sample_scale
        self._bots: dict[tuple[str, str], object] = {}

    def get_bot(self, *, bot_id: str, player_name: str):
        cache_key = (bot_id, player_name)
        bot = self._bots.get(cache_key)
        if bot is None:
            bot = build_ready_bot(bot_id, player_name)
            bot = _configure_teacher_bot_for_training(
                bot,
                sample_scale=self._teacher_sample_scale,
            )
            self._bots[cache_key] = bot
        return bot


class TeacherOracle:
    def __init__(
        self,
        teacher_specs: list[TeacherSpec],
        seed: int,
        *,
        bot_provider: TrainingBotProvider,
        teacher_target_temperature: float,
    ):
        if not teacher_specs:
            raise ValueError("teacher_specs must not be empty")
        self._teacher_specs = list(teacher_specs)
        self._rng = random.Random(seed)
        self._bot_provider = bot_provider
        self._teacher_target_temperature = teacher_target_temperature

    def sample_teacher_spec(self) -> TeacherSpec:
        total_weight = sum(spec.weight for spec in self._teacher_specs)
        threshold = self._rng.random() * total_weight
        running_total = 0.0
        for spec in self._teacher_specs:
            running_total += spec.weight
            if threshold <= running_total:
                return spec
        return self._teacher_specs[-1]

    def choose_auction_action(
        self,
        *,
        teacher_spec: TeacherSpec | None = None,
        auction_state,
        player_name: str,
        hand: set[Card],
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
    ) -> TeacherDecision:
        selected_teacher = teacher_spec or self.sample_teacher_spec()
        bot = self._bot_provider.get_bot(
            bot_id=selected_teacher.bot_id,
            player_name=player_name,
        )
        bot._cards = set(hand)
        if hasattr(bot, "set_match_context"):
            bot.set_match_context(
                player_names=player_names,
                teams=teams,
                match_scores=match_scores,
                target_score=target_score,
                auction_state=auction_state,
                round_state=None,
            )
        return _build_teacher_decision_for_auction(
            bot=bot,
            auction_state=auction_state,
            bot_id=selected_teacher.bot_id,
            temperature=self._teacher_target_temperature,
        )

    def choose_card(
        self,
        *,
        teacher_spec: TeacherSpec | None = None,
        round_state,
        player_name: str,
        auction_state,
        player_names: list[str],
        teams: list[tuple[str, ...]],
        match_scores: dict[str, int],
        target_score: int,
    ) -> TeacherDecision:
        selected_teacher = teacher_spec or self.sample_teacher_spec()
        bot = self._bot_provider.get_bot(
            bot_id=selected_teacher.bot_id,
            player_name=player_name,
        )
        bot._cards = set(round_state.current_player.cards)
        if hasattr(bot, "set_match_context"):
            bot.set_match_context(
                player_names=player_names,
                teams=teams,
                match_scores=match_scores,
                target_score=target_score,
                auction_state=auction_state,
                round_state=round_state,
            )
        return _build_teacher_decision_for_card(
            bot=bot,
            round_state=round_state,
            bot_id=selected_teacher.bot_id,
            temperature=self._teacher_target_temperature,
        )


def _softmax_distribution(
    scores: list[float],
    *,
    temperature: float,
) -> list[float]:
    if not scores:
        raise ValueError("scores must not be empty")
    clamped_temperature = max(temperature, 1e-3)
    max_score = max(scores)
    exp_scores = [
        math.exp((score - max_score) / clamped_temperature)
        for score in scores
    ]
    total = sum(exp_scores)
    if total <= 0.0:
        raise ValueError("teacher scores produced a non-positive softmax total")
    return [value / total for value in exp_scores]


def _extract_scored_candidates(raw_candidates) -> list[tuple[object, float]] | None:
    if raw_candidates is None:
        return None
    parsed_candidates: list[tuple[object, float]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
            return None
        action = candidate[0]
        try:
            score = float(candidate[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        parsed_candidates.append((action, score))
    if not parsed_candidates:
        return None
    return parsed_candidates


def _call_optional_scoring_method(
    bot,
    method_name: str,
    state,
) -> list[tuple[object, float]] | None:
    scoring_method = getattr(bot, method_name, None)
    if scoring_method is None:
        return None
    try:
        raw_candidates = scoring_method(state, show_progress=False)
    except TypeError:
        raw_candidates = scoring_method(state)
    return _extract_scored_candidates(raw_candidates)


def _build_soft_teacher_decision(
    *,
    action: Card | AuctionEvent,
    bot_id: str,
    scored_candidates: list[tuple[object, float]] | None,
    temperature: float,
) -> TeacherDecision:
    if not scored_candidates:
        return TeacherDecision(action=action, bot_id=bot_id)

    candidate_actions = [candidate_action for candidate_action, _score in scored_candidates]
    if action not in candidate_actions:
        return TeacherDecision(action=action, bot_id=bot_id)

    return TeacherDecision(
        action=action,
        bot_id=bot_id,
        candidate_actions=list(candidate_actions),
        target_distribution=_softmax_distribution(
            [score for _candidate_action, score in scored_candidates],
            temperature=temperature,
        ),
    )


def _build_teacher_decision_for_auction(
    *,
    bot,
    auction_state,
    bot_id: str,
    temperature: float,
    selected_action: AuctionEvent | None = None,
) -> TeacherDecision:
    scored_candidates = _call_optional_scoring_method(
        bot,
        "score_auction_candidates",
        auction_state,
    )
    if selected_action is None:
        if scored_candidates is not None and hasattr(bot, "select_best_scored_auction_action"):
            selected_action = bot.select_best_scored_auction_action(scored_candidates)
        else:
            selected_action = bot.choose_auction_action(auction_state)
    return _build_soft_teacher_decision(
        action=selected_action,
        bot_id=bot_id,
        scored_candidates=scored_candidates,
        temperature=temperature,
    )


def _build_teacher_decision_for_card(
    *,
    bot,
    round_state,
    bot_id: str,
    temperature: float,
    selected_card: Card | None = None,
) -> TeacherDecision:
    scored_candidates = _call_optional_scoring_method(
        bot,
        "score_card_candidates",
        round_state,
    )
    if selected_card is None:
        if scored_candidates is not None and hasattr(bot, "select_best_scored_card"):
            selected_card = bot.select_best_scored_card(scored_candidates)
        else:
            selected_card = bot.choose_card(round_state)
    return _build_soft_teacher_decision(
        action=selected_card,
        bot_id=bot_id,
        scored_candidates=scored_candidates,
        temperature=temperature,
    )


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _format_teacher_pool(teacher_specs: list[TeacherSpec]) -> str:
    return ", ".join(
        f"{teacher_spec.bot_id}x{teacher_spec.weight:g}"
        for teacher_spec in teacher_specs
    )


def _format_dataset_counts(dataset: TrainingDataset) -> str:
    return (
        f"play_policy={len(dataset.play_policy_examples)} "
        f"auction_policy={len(dataset.auction_policy_examples)} "
        f"play_value={len(dataset.play_value_examples)} "
        f"auction_value={len(dataset.auction_value_examples)}"
    )


def _format_training_metrics(training_report: dict) -> str:
    return (
        f"play_acc={training_report['play_history'][-1]['accuracy']:.3f} "
        f"auction_acc={training_report['auction_history'][-1]['accuracy']:.3f} "
        f"play_value_mse={training_report['play_value_history'][-1]['mse']:.4f} "
        f"auction_value_mse={training_report['auction_value_history'][-1]['mse']:.4f}"
    )


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = max(0.0, min(1.0, completed / total))
    filled = min(width, int(round(width * ratio)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class LiveProgressDisplay:
    def __init__(
        self,
        *,
        verbose: bool,
        label: str,
        total: int,
        started_at: float,
        refresh_interval: float = 1.0,
    ):
        self._verbose = verbose
        self._label = label
        self._total = max(total, 0)
        self._started_at = started_at
        self._refresh_interval = refresh_interval
        self._completed = 0
        self._detail = ""
        self._estimated_total_seconds: float | None = None
        self._last_line_length = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, detail: str = "") -> None:
        if not self._verbose:
            return
        with self._lock:
            self._detail = detail
        self._render()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(
        self,
        *,
        completed: int | None = None,
        detail: str | None = None,
    ) -> None:
        if not self._verbose:
            return
        with self._lock:
            if completed is not None:
                self._completed = max(0, min(completed, self._total))
                elapsed = max(0.0, time.perf_counter() - self._started_at)
                if self._completed > 0:
                    self._estimated_total_seconds = (
                        elapsed / self._completed
                    ) * self._total
            if detail is not None:
                self._detail = detail
        self._render()

    def stop(
        self,
        *,
        completed: int | None = None,
        detail: str | None = None,
    ) -> None:
        if not self._verbose:
            return
        with self._lock:
            if completed is not None:
                self._completed = max(0, min(completed, self._total))
                elapsed = max(0.0, time.perf_counter() - self._started_at)
                if self._completed > 0:
                    self._estimated_total_seconds = (
                        elapsed / self._completed
                    ) * self._total
            if detail is not None:
                self._detail = detail
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._refresh_interval + 0.1)
        self._render(final=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self._refresh_interval):
            self._render()

    def _render(self, *, final: bool = False) -> None:
        if not self._verbose:
            return
        with self._lock:
            completed = self._completed
            detail = self._detail
            estimated_total_seconds = self._estimated_total_seconds

        elapsed = max(0.0, time.perf_counter() - self._started_at)
        percent = (100.0 * completed / self._total) if self._total > 0 else 0.0
        message = (
            f"[{self._label}] {_progress_bar(completed, self._total)} "
            f"{completed}/{self._total} ({percent:5.1f}%) "
            f"elapsed={_format_seconds(elapsed)}"
        )
        if completed < self._total and estimated_total_seconds is not None:
            eta = max(0.0, estimated_total_seconds - elapsed)
            message += f" eta={_format_seconds(eta)}"
        if detail:
            message += f" {detail}"

        padded_message = message.ljust(self._last_line_length)
        sys.stderr.write("\r" + padded_message)
        if final:
            sys.stderr.write("\n")
        sys.stderr.flush()
        self._last_line_length = len(padded_message)


def _log_progress(
    *,
    verbose: bool,
    label: str,
    completed: int,
    total: int,
    started_at: float,
    detail: str = "",
) -> None:
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = (completed / elapsed) if elapsed > 0 and completed > 0 else 0.0
    remaining = max(total - completed, 0)
    eta = (remaining / rate) if rate > 0 else 0.0
    percent = (100.0 * completed / total) if total > 0 else 0.0
    message = (
        f"[{label}] {_progress_bar(completed, total)} "
        f"{completed}/{total} ({percent:5.1f}%) "
        f"elapsed={_format_seconds(elapsed)}"
    )
    if completed < total and rate > 0:
        message += f" eta={_format_seconds(eta)}"
    if detail:
        message += f" {detail}"
    _log(verbose, message)


def _make_choice_epoch_logger(
    *,
    progress: LiveProgressDisplay,
) -> Callable[[dict[str, float]], None]:
    def _callback(metrics: dict[str, float]) -> None:
        progress.update(
            completed=int(metrics["epoch"]),
            detail=f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f}",
        )

    return _callback


def _make_regression_epoch_logger(
    *,
    progress: LiveProgressDisplay,
) -> Callable[[dict[str, float]], None]:
    def _callback(metrics: dict[str, float]) -> None:
        progress.update(
            completed=int(metrics["epoch"]),
            detail=f"mse={metrics['mse']:.4f}",
        )

    return _callback


def _assert_finite_training_report(training_report: dict) -> None:
    final_metrics = [
        training_report["play_history"][-1]["loss"],
        training_report["play_history"][-1]["accuracy"],
        training_report["auction_history"][-1]["loss"],
        training_report["auction_history"][-1]["accuracy"],
        training_report["play_value_history"][-1]["mse"],
        training_report["auction_value_history"][-1]["mse"],
    ]
    if not all(math.isfinite(value) for value in final_metrics):
        raise FloatingPointError("training produced non-finite metrics")


def _scaled_epoch_count(base_epochs: int, scale: float) -> int:
    if base_epochs <= 0:
        raise ValueError("base_epochs must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    return max(1, int(round(base_epochs * scale)))


def _teacher_weight_by_bot_id(teacher_specs: list[TeacherSpec]) -> dict[str, float]:
    return {
        teacher_spec.bot_id: teacher_spec.weight
        for teacher_spec in teacher_specs
    }


def _policy_example_weight(
    *,
    actor_mode: str,
    teacher_weight: float,
    teacher_matches_environment: bool,
) -> float:
    weight = max(teacher_weight, 1.0)
    if actor_mode == "student" and not teacher_matches_environment:
        weight *= 2.5
    return weight


def normalize_utility(utility: float, target_score: int) -> float:
    scale = max(float(target_score) * 2.0, 1.0)
    return utility / scale


def parse_teacher_specs(
    teacher_pool: str | None = None,
    *,
    fallback_teacher_bot_id: str | None = None,
) -> list[TeacherSpec]:
    raw_spec = teacher_pool.strip() if teacher_pool is not None else ""
    if not raw_spec:
        if fallback_teacher_bot_id is None:
            raise ValueError("teacher_pool cannot be empty")
        raw_spec = fallback_teacher_bot_id

    specs: list[TeacherSpec] = []
    for item in raw_spec.split(","):
        token = item.strip()
        if not token:
            continue
        bot_id, separator, raw_weight = token.partition(":")
        cleaned_bot_id = bot_id.strip()
        if not cleaned_bot_id:
            raise ValueError(f"invalid teacher token: {token!r}")
        get_ready_bot_spec(cleaned_bot_id)
        if separator:
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise ValueError(f"invalid teacher weight in {token!r}") from exc
        else:
            weight = 1.0
        if weight <= 0:
            raise ValueError(f"teacher weights must be positive: {token!r}")
        specs.append(TeacherSpec(cleaned_bot_id, weight))

    if not specs:
        raise ValueError("teacher_pool cannot be empty")
    return specs


def _serialize_teacher_specs(teacher_specs: list[TeacherSpec]) -> list[dict]:
    return [
        {"bot_id": teacher_spec.bot_id, "weight": teacher_spec.weight}
        for teacher_spec in teacher_specs
    ]


def load_model_bundle(model_path: str | Path) -> dict:
    return json.loads(Path(model_path).read_text(encoding="utf-8"))


def _build_student_bots(model_bundle: dict) -> dict[str, NeuralThreePlayerBot]:
    return {
        player_name: NeuralThreePlayerBot(player_name, model_bundle=model_bundle)
        for player_name in THREE_PLAYER_NAMES
    }


def _build_match_controller_for_actor(
    *,
    actor_mode: str,
    teacher_specs: list[TeacherSpec],
    student_bundle: dict | None,
    seat_rng: random.Random,
    bot_provider: TrainingBotProvider,
) -> MatchController:
    if actor_mode == "teacher":
        sampled_bot_ids = {
            player_name: seat_rng.choices(
                [spec.bot_id for spec in teacher_specs],
                weights=[spec.weight for spec in teacher_specs],
                k=1,
            )[0]
            for player_name in THREE_PLAYER_NAMES
        }
        controller = MatchController.create(
            num_players=3,
            player_names=THREE_PLAYER_NAMES,
            teams=None,
            player_bot_ids={player_name: None for player_name in THREE_PLAYER_NAMES},
            bots={
                player_name: bot_provider.get_bot(
                    bot_id=bot_id,
                    player_name=player_name,
                )
                for player_name, bot_id in sampled_bot_ids.items()
            },
            auto_run_bots=False,
        )
        controller.session.player_bot_ids = sampled_bot_ids
        return controller

    if student_bundle is None:
        raise ValueError("student_bundle is required for student rollouts")
    return MatchController.create(
        num_players=3,
        player_names=THREE_PLAYER_NAMES,
        teams=None,
        bots=_build_student_bots(student_bundle),
        auto_run_bots=False,
    )


def _resolve_round_value_targets(
    *,
    controller: MatchController,
    round_start_match_scores: dict[str, int],
) -> dict[str, float]:
    round_state = controller.session.game.round_state
    auction_state = controller.session.auction.state
    teams = controller.session.teams
    target_score = controller.session.target_score
    return {
        player_name: normalize_utility(
            evaluate_terminal_round_utility(
                round_state=round_state,
                auction_state=auction_state,
                match_scores=round_start_match_scores,
                teams=teams,
                target_score=target_score,
                player_name=player_name,
            ),
            target_score,
        )
        for player_name in controller.session.player_names
    }


def _collect_rollout_examples(
    *,
    actor_mode: str,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    student_bundle: dict | None = None,
    student_agreement_keep_prob: float = DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    if actor_mode not in {"teacher", "student"}:
        raise ValueError("actor_mode must be 'teacher' or 'student'")
    if match_count <= 0:
        raise ValueError("match_count must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    dataset = TrainingDataset()
    bot_provider = TrainingBotProvider(
        teacher_sample_scale=teacher_sample_scale,
    )
    teacher_oracle = TeacherOracle(
        teacher_specs,
        seed + 10_000,
        bot_provider=bot_provider,
        teacher_target_temperature=teacher_target_temperature,
    )
    seat_rng = random.Random(seed)
    example_rng = random.Random(seed + 20_000)
    teacher_label_counts: dict[str, int] = {}
    acting_bot_counts: dict[str, int] = {}
    forced_auction_actions = 0
    forced_play_actions = 0
    teacher_queries = 0
    teacher_weight_map = _teacher_weight_by_bot_id(teacher_specs)
    started_at = time.perf_counter()
    progress = LiveProgressDisplay(
        verbose=verbose,
        label=f"collect:{actor_mode}",
        total=match_count,
        started_at=started_at,
    )

    _log(
        verbose,
        (
            f"[collect:{actor_mode}] start matches={match_count} alpha={alpha} "
            f"seed={seed} teacher_pool={_format_teacher_pool(teacher_specs)}"
        ),
    )
    progress.start(detail=_format_dataset_counts(dataset))

    for match_index in range(match_count):
        random.seed(seed + match_index)
        controller = _build_match_controller_for_actor(
            actor_mode=actor_mode,
            teacher_specs=teacher_specs,
            student_bundle=student_bundle,
            seat_rng=seat_rng,
            bot_provider=bot_provider,
        )
        round_start_match_scores = dict(controller.session.match_scores)
        pending_play_values: list[PendingValueExample] = []
        pending_auction_values: list[PendingValueExample] = []

        for round_index in range(1, alpha + 1):
            while controller.session.phase in {"auction", "play"}:
                player_name = controller._current_bot_name()
                if player_name is None:
                    raise ValueError("match controller had no active bot turn")

                controller._sync_bot_hand(player_name)
                bot = controller.session.bots[player_name]

                if controller.session.phase == "auction":
                    auction_state = controller.session.auction.state
                    legal_actions = ordered_legal_auction_actions(auction_state)
                    acting_hand = set(bot.cards)
                    forced_action = legal_actions[0] if len(legal_actions) == 1 else None
                    if forced_action is not None:
                        forced_auction_actions += 1
                        environment_action = forced_action
                    else:
                        environment_action = bot.choose_auction_action(auction_state)
                        if environment_action not in legal_actions:
                            raise ValueError(
                                f"actor selected illegal auction action {environment_action}"
                            )

                    if actor_mode == "teacher":
                        teacher_bot_id = controller.session.player_bot_ids[player_name]
                        if teacher_bot_id is None:
                            raise ValueError("teacher rollout was missing a configured bot id")
                        teacher_decision = _build_teacher_decision_for_auction(
                            bot=bot,
                            auction_state=auction_state,
                            bot_id=teacher_bot_id,
                            temperature=teacher_target_temperature,
                            selected_action=environment_action,
                        )
                        teacher_action = teacher_decision.action
                        keep_policy_example = forced_action is None
                    else:
                        if forced_action is not None:
                            sampled_teacher_spec = teacher_oracle.sample_teacher_spec()
                            teacher_decision = TeacherDecision(
                                action=forced_action,
                                bot_id=sampled_teacher_spec.bot_id,
                            )
                            teacher_action = teacher_decision.action
                            teacher_bot_id = teacher_decision.bot_id
                            keep_policy_example = False
                        else:
                            teacher_queries += 1
                            teacher_decision = teacher_oracle.choose_auction_action(
                                auction_state=auction_state,
                                player_name=player_name,
                                hand=acting_hand,
                                player_names=controller.session.player_names,
                                teams=controller.session.teams,
                                match_scores=controller.session.match_scores,
                                target_score=controller.session.target_score,
                            )
                            teacher_action = teacher_decision.action
                            teacher_bot_id = teacher_decision.bot_id
                            if teacher_action not in legal_actions:
                                raise ValueError(
                                    f"teacher oracle selected illegal auction action {teacher_action}"
                                )
                            keep_policy_example = True
                    example_weight = _policy_example_weight(
                        actor_mode=actor_mode,
                        teacher_weight=teacher_weight_map.get(teacher_bot_id, 1.0),
                        teacher_matches_environment=(teacher_action == environment_action),
                    )
                    if (
                        keep_policy_example
                        and actor_mode == "student"
                        and teacher_action == environment_action
                        and example_rng.random() > student_agreement_keep_prob
                    ):
                        keep_policy_example = False

                    if keep_policy_example:
                        policy_actions = (
                            teacher_decision.candidate_actions
                            if teacher_decision.candidate_actions is not None
                            else legal_actions
                        )
                        dataset.auction_policy_examples.append(
                            ChoiceExample(
                                candidate_features=[
                                    encode_auction_candidate(
                                        auction_state=auction_state,
                                        acting_player_name=player_name,
                                        hand=acting_hand,
                                        candidate_action=legal_action,
                                        match_scores=controller.session.match_scores,
                                        target_score=controller.session.target_score,
                                    )
                                    for legal_action in policy_actions
                                ],
                                chosen_index=policy_actions.index(teacher_action),
                                weight=example_weight,
                                target_distribution=teacher_decision.target_distribution,
                            )
                        )
                    teacher_label_counts[teacher_bot_id] = (
                        teacher_label_counts.get(teacher_bot_id, 0) + 1
                    )
                    acting_key = (
                        controller.session.player_bot_ids.get(player_name)
                        if actor_mode == "teacher"
                        else controller.session.bots[player_name]._bundle_bot_id
                    )
                    acting_bot_counts[acting_key] = acting_bot_counts.get(acting_key, 0) + 1

                    successor_auction = deepcopy(auction_state)
                    apply_auction_action_for_search(successor_auction, environment_action)
                    pending_auction_values.append(
                        PendingValueExample(
                            player_name=player_name,
                            features=encode_auction_state(
                                auction_state=successor_auction,
                                perspective_player_name=player_name,
                                hand=acting_hand,
                                match_scores=round_start_match_scores,
                                target_score=controller.session.target_score,
                            ),
                            weight=example_weight,
                        )
                    )

                    controller.session.auction.apply_event(environment_action)
                    controller._advance_or_finalize_auction()
                    continue

                round_state = controller.session.game.round_state
                legal_cards = ordered_legal_cards(round_state)
                forced_card = legal_cards[0] if len(legal_cards) == 1 else None
                if forced_card is not None:
                    forced_play_actions += 1
                    environment_card = forced_card
                else:
                    environment_card = bot.choose_card(round_state)
                    if environment_card not in legal_cards:
                        raise ValueError(f"actor selected illegal card {environment_card.code}")

                if actor_mode == "teacher":
                    teacher_bot_id = controller.session.player_bot_ids[player_name]
                    if teacher_bot_id is None:
                        raise ValueError("teacher rollout was missing a configured bot id")
                    teacher_decision = _build_teacher_decision_for_card(
                        bot=bot,
                        round_state=round_state,
                        bot_id=teacher_bot_id,
                        temperature=teacher_target_temperature,
                        selected_card=environment_card,
                    )
                    teacher_card = teacher_decision.action
                    keep_policy_example = forced_card is None
                else:
                    if forced_card is not None:
                        sampled_teacher_spec = teacher_oracle.sample_teacher_spec()
                        teacher_decision = TeacherDecision(
                            action=forced_card,
                            bot_id=sampled_teacher_spec.bot_id,
                        )
                        teacher_card = teacher_decision.action
                        teacher_bot_id = teacher_decision.bot_id
                        keep_policy_example = False
                    else:
                        teacher_queries += 1
                        teacher_decision = teacher_oracle.choose_card(
                            round_state=round_state,
                            player_name=player_name,
                            auction_state=controller.session.auction.state,
                            player_names=controller.session.player_names,
                            teams=controller.session.teams,
                            match_scores=controller.session.match_scores,
                            target_score=controller.session.target_score,
                        )
                        teacher_card = teacher_decision.action
                        teacher_bot_id = teacher_decision.bot_id
                        if teacher_card not in legal_cards:
                            raise ValueError(
                                f"teacher oracle selected illegal card {teacher_card.code}"
                            )
                        keep_policy_example = True
                example_weight = _policy_example_weight(
                    actor_mode=actor_mode,
                    teacher_weight=teacher_weight_map.get(teacher_bot_id, 1.0),
                    teacher_matches_environment=(teacher_card == environment_card),
                )
                if (
                    keep_policy_example
                    and actor_mode == "student"
                    and teacher_card == environment_card
                    and example_rng.random() > student_agreement_keep_prob
                ):
                    keep_policy_example = False

                if keep_policy_example:
                    policy_cards = (
                        teacher_decision.candidate_actions
                        if teacher_decision.candidate_actions is not None
                        else legal_cards
                    )
                    dataset.play_policy_examples.append(
                        ChoiceExample(
                            candidate_features=[
                                encode_play_candidate(
                                    round_state=round_state,
                                    acting_player_name=player_name,
                                    candidate_card=legal_card,
                                    match_scores=controller.session.match_scores,
                                    target_score=controller.session.target_score,
                                    auction_state=controller.session.auction.state,
                                )
                                for legal_card in policy_cards
                            ],
                            chosen_index=policy_cards.index(teacher_card),
                            weight=example_weight,
                            target_distribution=teacher_decision.target_distribution,
                        )
                    )
                teacher_label_counts[teacher_bot_id] = (
                    teacher_label_counts.get(teacher_bot_id, 0) + 1
                )
                acting_key = (
                    controller.session.player_bot_ids.get(player_name)
                    if actor_mode == "teacher"
                    else controller.session.bots[player_name]._bundle_bot_id
                )
                acting_bot_counts[acting_key] = acting_bot_counts.get(acting_key, 0) + 1

                successor_round = apply_trick_action_to_state(
                    round_state,
                    Play(round_state.current_player, environment_card),
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
                            weight=example_weight,
                        )
                    )

                controller.session.game.apply_trick_action(
                    Play(controller.session.game.curr_player, environment_card)
                )
                controller._score_terminal_round_if_needed()

            if controller.session.game.round_state.is_terminal:
                round_targets = _resolve_round_value_targets(
                    controller=controller,
                    round_start_match_scores=round_start_match_scores,
                )
                dataset.play_value_examples.extend(
                    RegressionExample(
                        example.features,
                        round_targets[example.player_name],
                        weight=example.weight,
                    )
                    for example in pending_play_values
                )
                dataset.auction_value_examples.extend(
                    RegressionExample(
                        example.features,
                        round_targets[example.player_name],
                        weight=example.weight,
                    )
                    for example in pending_auction_values
                )
                pending_play_values = []
                pending_auction_values = []

            if controller.session.is_match_complete:
                break

            if round_index < alpha:
                controller.next_round(auto_run_bots=False)
                round_start_match_scores = dict(controller.session.match_scores)

        progress.update(
            completed=match_index + 1,
            detail=_format_dataset_counts(dataset),
        )

    report = {
        "actor_mode": actor_mode,
        "matches": match_count,
        "alpha": alpha,
        "teacher_label_counts": teacher_label_counts,
        "acting_bot_counts": acting_bot_counts,
        "play_policy_examples": len(dataset.play_policy_examples),
        "auction_policy_examples": len(dataset.auction_policy_examples),
        "play_value_examples": len(dataset.play_value_examples),
        "auction_value_examples": len(dataset.auction_value_examples),
        "forced_auction_actions": forced_auction_actions,
        "forced_play_actions": forced_play_actions,
        "teacher_queries": teacher_queries,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    progress.stop(
        completed=match_count,
        detail=_format_dataset_counts(dataset),
    )
    _log(
        verbose,
        (
            f"[collect:{actor_mode}] done "
            f"{_format_dataset_counts(dataset)} "
            f"elapsed={_format_seconds(report['elapsed_seconds'])} "
            f"forced_auction={forced_auction_actions} forced_play={forced_play_actions} "
            f"teacher_queries={teacher_queries} "
            f"labels={report['teacher_label_counts']}"
        ),
    )
    return dataset, report


def collect_teacher_examples(
    *,
    teacher_bot_id: str,
    match_count: int,
    alpha: int,
    seed: int,
) -> tuple[list[ChoiceExample], list[ChoiceExample]]:
    dataset, _ = _collect_rollout_examples(
        actor_mode="teacher",
        teacher_specs=parse_teacher_specs(fallback_teacher_bot_id=teacher_bot_id),
        match_count=match_count,
        alpha=alpha,
        seed=seed,
    )
    return dataset.play_policy_examples, dataset.auction_policy_examples


def collect_teacher_training_dataset(
    *,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    student_agreement_keep_prob: float = DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    return _collect_rollout_examples(
        actor_mode="teacher",
        teacher_specs=teacher_specs,
        match_count=match_count,
        alpha=alpha,
        seed=seed,
        student_agreement_keep_prob=student_agreement_keep_prob,
        teacher_sample_scale=teacher_sample_scale,
        teacher_target_temperature=teacher_target_temperature,
        verbose=verbose,
    )


def collect_dagger_training_dataset(
    *,
    student_bundle: dict,
    teacher_specs: list[TeacherSpec],
    match_count: int,
    alpha: int,
    seed: int,
    student_agreement_keep_prob: float = DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
    teacher_sample_scale: float = DEFAULT_TEACHER_SAMPLE_SCALE,
    teacher_target_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    verbose: bool = False,
) -> tuple[TrainingDataset, dict]:
    return _collect_rollout_examples(
        actor_mode="student",
        teacher_specs=teacher_specs,
        match_count=match_count,
        alpha=alpha,
        seed=seed,
        student_bundle=student_bundle,
        student_agreement_keep_prob=student_agreement_keep_prob,
        teacher_sample_scale=teacher_sample_scale,
        teacher_target_temperature=teacher_target_temperature,
        verbose=verbose,
    )


def _initialize_model(
    *,
    initial_bundle: dict | None,
    bundle_key: str,
    input_dim: int,
    hidden_dim: int,
    seed: int,
) -> ScalarMLP:
    if initial_bundle is not None and bundle_key in initial_bundle:
        model = ScalarMLP.from_dict(initial_bundle[bundle_key]).copy()
        if model.input_dim != input_dim:
            raise ValueError(
                f"warm-started model {bundle_key} expected input_dim={model.input_dim}, got {input_dim}"
            )
        return model
    return ScalarMLP.initialize(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )


def train_neural_3p_bundle(
    *,
    play_examples: list[ChoiceExample],
    auction_examples: list[ChoiceExample],
    play_value_examples: list[RegressionExample],
    auction_value_examples: list[RegressionExample],
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
    seed: int,
    teacher_specs: list[TeacherSpec],
    initial_bundle: dict | None = None,
    play_value_weight: float = DEFAULT_PLAY_VALUE_WEIGHT,
    auction_value_weight: float = DEFAULT_AUCTION_VALUE_WEIGHT,
    play_rollout_depth: int = DEFAULT_PLAY_ROLLOUT_DEPTH,
    auction_rollout_depth: int = DEFAULT_AUCTION_ROLLOUT_DEPTH,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
    bot_id: str = "neural-3p-v2",
    bundle_version: int = 2,
    extra_metadata: dict | None = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    if not play_examples:
        raise ValueError("play_examples must not be empty")
    if not auction_examples:
        raise ValueError("auction_examples must not be empty")
    if not play_value_examples:
        raise ValueError("play_value_examples must not be empty")
    if not auction_value_examples:
        raise ValueError("auction_value_examples must not be empty")

    play_input_dim = len(play_examples[0].candidate_features[0])
    auction_input_dim = len(auction_examples[0].candidate_features[0])
    play_value_input_dim = len(play_value_examples[0].features)
    auction_value_input_dim = len(auction_value_examples[0].features)

    started_at = time.perf_counter()
    _log(
        verbose,
        (
            "[train] start "
            f"play_examples={len(play_examples)} auction_examples={len(auction_examples)} "
            f"play_value_examples={len(play_value_examples)} "
            f"auction_value_examples={len(auction_value_examples)} "
            f"seed={seed} "
            f"hidden_dims=({play_hidden_dim},{auction_hidden_dim},{play_value_hidden_dim},{auction_value_hidden_dim}) "
            f"epochs=({play_epochs},{auction_epochs},{play_value_epochs},{auction_value_epochs}) "
            f"lrs=({play_learning_rate:.4f},{auction_learning_rate:.4f},{play_value_learning_rate:.4f},{auction_value_learning_rate:.4f})"
        ),
    )
    _log_progress(
        verbose=verbose,
        label="train:models",
        completed=0,
        total=4,
        started_at=started_at,
        detail="initializing models",
    )

    play_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="play_model",
        input_dim=play_input_dim,
        hidden_dim=play_hidden_dim,
        seed=seed,
    )
    auction_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="auction_model",
        input_dim=auction_input_dim,
        hidden_dim=auction_hidden_dim,
        seed=seed + 1,
    )
    play_value_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="play_value_model",
        input_dim=play_value_input_dim,
        hidden_dim=play_value_hidden_dim,
        seed=seed + 2,
    )
    auction_value_model = _initialize_model(
        initial_bundle=initial_bundle,
        bundle_key="auction_value_model",
        input_dim=auction_value_input_dim,
        hidden_dim=auction_value_hidden_dim,
        seed=seed + 3,
    )

    play_progress = LiveProgressDisplay(
        verbose=verbose,
        label="train:play-policy",
        total=play_epochs,
        started_at=time.perf_counter(),
    )
    play_progress.start(detail=f"examples={len(play_examples)}")
    play_history = play_model.train_choice_examples(
        play_examples,
        epochs=play_epochs,
        learning_rate=play_learning_rate,
        l2=l2,
        seed=seed,
        gradient_clip=gradient_clip,
        progress_callback=_make_choice_epoch_logger(
            progress=play_progress,
        ),
    )
    play_progress.stop(
        completed=play_epochs,
        detail=(
            f"loss={play_history[-1]['loss']:.4f} "
            f"acc={play_history[-1]['accuracy']:.3f}"
        ),
    )
    _log_progress(
        verbose=verbose,
        label="train:models",
        completed=1,
        total=4,
        started_at=started_at,
        detail=f"play_policy acc={play_history[-1]['accuracy']:.3f}",
    )
    auction_progress = LiveProgressDisplay(
        verbose=verbose,
        label="train:auction-policy",
        total=auction_epochs,
        started_at=time.perf_counter(),
    )
    auction_progress.start(detail=f"examples={len(auction_examples)}")
    auction_history = auction_model.train_choice_examples(
        auction_examples,
        epochs=auction_epochs,
        learning_rate=auction_learning_rate,
        l2=l2,
        seed=seed + 1,
        gradient_clip=gradient_clip,
        progress_callback=_make_choice_epoch_logger(
            progress=auction_progress,
        ),
    )
    auction_progress.stop(
        completed=auction_epochs,
        detail=(
            f"loss={auction_history[-1]['loss']:.4f} "
            f"acc={auction_history[-1]['accuracy']:.3f}"
        ),
    )
    _log_progress(
        verbose=verbose,
        label="train:models",
        completed=2,
        total=4,
        started_at=started_at,
        detail=f"auction_policy acc={auction_history[-1]['accuracy']:.3f}",
    )
    play_value_progress = LiveProgressDisplay(
        verbose=verbose,
        label="train:play-value",
        total=play_value_epochs,
        started_at=time.perf_counter(),
    )
    play_value_progress.start(detail=f"examples={len(play_value_examples)}")
    play_value_history = play_value_model.train_regression_examples(
        play_value_examples,
        epochs=play_value_epochs,
        learning_rate=play_value_learning_rate,
        l2=l2,
        seed=seed + 2,
        gradient_clip=gradient_clip,
        progress_callback=_make_regression_epoch_logger(
            progress=play_value_progress,
        ),
    )
    play_value_progress.stop(
        completed=play_value_epochs,
        detail=f"mse={play_value_history[-1]['mse']:.4f}",
    )
    _log_progress(
        verbose=verbose,
        label="train:models",
        completed=3,
        total=4,
        started_at=started_at,
        detail=f"play_value mse={play_value_history[-1]['mse']:.4f}",
    )
    auction_value_progress = LiveProgressDisplay(
        verbose=verbose,
        label="train:auction-value",
        total=auction_value_epochs,
        started_at=time.perf_counter(),
    )
    auction_value_progress.start(detail=f"examples={len(auction_value_examples)}")
    auction_value_history = auction_value_model.train_regression_examples(
        auction_value_examples,
        epochs=auction_value_epochs,
        learning_rate=auction_value_learning_rate,
        l2=l2,
        seed=seed + 3,
        gradient_clip=gradient_clip,
        progress_callback=_make_regression_epoch_logger(
            progress=auction_value_progress,
        ),
    )
    auction_value_progress.stop(
        completed=auction_value_epochs,
        detail=f"mse={auction_value_history[-1]['mse']:.4f}",
    )
    _log_progress(
        verbose=verbose,
        label="train:models",
        completed=4,
        total=4,
        started_at=started_at,
        detail=f"auction_value mse={auction_value_history[-1]['mse']:.4f}",
    )

    bundle = {
        "version": bundle_version,
        "bot_id": bot_id,
        "teacher_pool": _serialize_teacher_specs(teacher_specs),
        "play_model": play_model.to_dict(),
        "auction_model": auction_model.to_dict(),
        "play_value_model": play_value_model.to_dict(),
        "auction_value_model": auction_value_model.to_dict(),
        "inference": {
            "play_policy_weight": 1.0,
            "play_value_weight": play_value_weight,
            "auction_policy_weight": 1.0,
            "auction_value_weight": auction_value_weight,
            "play_rollout_depth": play_rollout_depth,
            "auction_rollout_depth": auction_rollout_depth,
        },
    }
    if extra_metadata:
        bundle.update(extra_metadata)
    training_report = {
        "play_history": play_history,
        "auction_history": auction_history,
        "play_value_history": play_value_history,
        "auction_value_history": auction_value_history,
        "play_examples": len(play_examples),
        "auction_examples": len(auction_examples),
        "play_value_examples": len(play_value_examples),
        "auction_value_examples": len(auction_value_examples),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    _assert_finite_training_report(training_report)
    _log(
        verbose,
        f"[train] done {_format_training_metrics(training_report)} "
        f"elapsed={_format_seconds(training_report['elapsed_seconds'])}",
    )
    return bundle, training_report


def train_with_dagger(
    *,
    teacher_specs: list[TeacherSpec],
    bootstrap_matches: int,
    dagger_matches: int,
    alpha: int,
    dagger_iterations: int,
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
    initial_bundle: dict | None = None,
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
    bot_id: str = "neural-3p-v2",
    verbose: bool = False,
) -> tuple[dict, dict]:
    _log(
        verbose,
        (
            f"[dagger] bootstrap_matches={bootstrap_matches} dagger_matches={dagger_matches} "
            f"dagger_iterations={dagger_iterations} teacher_pool={_format_teacher_pool(teacher_specs)} "
            f"warm_start_lr_scale={warm_start_lr_scale:.2f} "
            f"warm_start_epoch_scale={warm_start_epoch_scale:.2f} "
            f"teacher_sample_scale={teacher_sample_scale:.2f} "
            f"teacher_target_temperature={teacher_target_temperature:.2f}"
        ),
    )
    aggregate_dataset, bootstrap_report = collect_teacher_training_dataset(
        teacher_specs=teacher_specs,
        match_count=bootstrap_matches,
        alpha=alpha,
        seed=seed,
        student_agreement_keep_prob=student_agreement_keep_prob,
        teacher_sample_scale=teacher_sample_scale,
        teacher_target_temperature=teacher_target_temperature,
        verbose=verbose,
    )
    _log(verbose, f"[dagger] bootstrap dataset {_format_dataset_counts(aggregate_dataset)}")

    bundle, training_report = train_neural_3p_bundle(
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
        seed=seed,
        teacher_specs=teacher_specs,
        initial_bundle=initial_bundle,
        play_value_weight=play_value_weight,
        auction_value_weight=auction_value_weight,
        play_rollout_depth=play_rollout_depth,
        auction_rollout_depth=auction_rollout_depth,
        gradient_clip=gradient_clip,
        bot_id=bot_id,
        extra_metadata={
            "training_mode": "search_distillation",
            "teacher_target_temperature": teacher_target_temperature,
        },
        verbose=verbose,
    )
    _log(verbose, f"[dagger] post-bootstrap {_format_training_metrics(training_report)}")

    dagger_reports: list[dict] = []
    for dagger_iteration in range(dagger_iterations):
        _log(verbose, f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} collecting student rollouts")
        dagger_dataset, dagger_report = collect_dagger_training_dataset(
            student_bundle=bundle,
            teacher_specs=teacher_specs,
            match_count=dagger_matches,
            alpha=alpha,
            seed=seed + 1_000 + dagger_iteration,
            student_agreement_keep_prob=student_agreement_keep_prob,
            teacher_sample_scale=teacher_sample_scale,
            teacher_target_temperature=teacher_target_temperature,
            verbose=verbose,
        )
        aggregate_dataset.extend(dagger_dataset)
        _log(
            verbose,
            (
                f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} "
                f"aggregate {_format_dataset_counts(aggregate_dataset)}"
            ),
        )
        bundle, training_report = train_neural_3p_bundle(
            play_examples=aggregate_dataset.play_policy_examples,
            auction_examples=aggregate_dataset.auction_policy_examples,
            play_value_examples=aggregate_dataset.play_value_examples,
            auction_value_examples=aggregate_dataset.auction_value_examples,
            play_hidden_dim=play_hidden_dim,
            auction_hidden_dim=auction_hidden_dim,
            play_value_hidden_dim=play_value_hidden_dim,
            auction_value_hidden_dim=auction_value_hidden_dim,
            play_epochs=_scaled_epoch_count(play_epochs, warm_start_epoch_scale),
            auction_epochs=_scaled_epoch_count(auction_epochs, warm_start_epoch_scale),
            play_value_epochs=_scaled_epoch_count(
                play_value_epochs, warm_start_epoch_scale
            ),
            auction_value_epochs=_scaled_epoch_count(
                auction_value_epochs, warm_start_epoch_scale
            ),
            play_learning_rate=play_learning_rate * warm_start_lr_scale,
            auction_learning_rate=auction_learning_rate * warm_start_lr_scale,
            play_value_learning_rate=play_value_learning_rate * warm_start_lr_scale,
            auction_value_learning_rate=auction_value_learning_rate * warm_start_lr_scale,
            l2=l2,
            seed=seed + (dagger_iteration + 1) * 100,
            teacher_specs=teacher_specs,
            initial_bundle=bundle,
            play_value_weight=play_value_weight,
            auction_value_weight=auction_value_weight,
            play_rollout_depth=play_rollout_depth,
            auction_rollout_depth=auction_rollout_depth,
            gradient_clip=gradient_clip,
            bot_id=bot_id,
            extra_metadata={
                "training_mode": "search_distillation",
                "teacher_target_temperature": teacher_target_temperature,
            },
            verbose=verbose,
        )
        _log(
            verbose,
            f"[dagger] iteration {dagger_iteration + 1}/{dagger_iterations} {_format_training_metrics(training_report)}",
        )
        dagger_reports.append(
            {
                "iteration": dagger_iteration + 1,
                **dagger_report,
                "aggregate_play_examples": len(aggregate_dataset.play_policy_examples),
                "aggregate_auction_examples": len(aggregate_dataset.auction_policy_examples),
                "aggregate_play_value_examples": len(aggregate_dataset.play_value_examples),
                "aggregate_auction_value_examples": len(aggregate_dataset.auction_value_examples),
            }
        )

    report = {
        "bootstrap": bootstrap_report,
        "dagger_iterations": dagger_reports,
        "training": training_report,
        "teacher_pool": _serialize_teacher_specs(teacher_specs),
        "teacher_target_temperature": teacher_target_temperature,
    }
    return bundle, report


def save_model_bundle(bundle: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(bundle), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the dependency-free 3-player singleton neural smear bot."
    )
    parser.add_argument(
        "--teacher",
        default=None,
        help="legacy single-teacher bot id; used when --teacher-pool is omitted",
    )
    parser.add_argument(
        "--teacher-pool",
        default=DEFAULT_TEACHER_POOL,
        help=(
            "comma-separated teacher mixture using bot_id[:weight], for example "
            "'optimal-bot:4,1-trick-minmax:2,greedy:1'"
        ),
    )
    parser.add_argument(
        "--bootstrap-matches",
        type=int,
        default=36,
        help="number of mixed-teacher rollout matches for the initial supervised pass",
    )
    parser.add_argument(
        "--dagger-matches",
        type=int,
        default=18,
        help="number of student rollout matches per DAgger iteration",
    )
    parser.add_argument(
        "--dagger-iterations",
        type=int,
        default=2,
        help="number of DAgger relabel-and-retrain iterations",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=12,
        help="maximum rounds per collected match",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for data collection and initialization",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="optional starting checkpoint to warm-start from",
    )
    parser.add_argument("--play-hidden-dim", type=int, default=DEFAULT_PLAY_HIDDEN_DIM)
    parser.add_argument("--auction-hidden-dim", type=int, default=DEFAULT_AUCTION_HIDDEN_DIM)
    parser.add_argument(
        "--play-value-hidden-dim",
        type=int,
        default=DEFAULT_PLAY_VALUE_HIDDEN_DIM,
    )
    parser.add_argument(
        "--auction-value-hidden-dim",
        type=int,
        default=DEFAULT_AUCTION_VALUE_HIDDEN_DIM,
    )
    parser.add_argument("--play-epochs", type=int, default=16)
    parser.add_argument("--auction-epochs", type=int, default=16)
    parser.add_argument("--play-value-epochs", type=int, default=18)
    parser.add_argument("--auction-value-epochs", type=int, default=18)
    parser.add_argument("--play-learning-rate", type=float, default=0.028)
    parser.add_argument("--auction-learning-rate", type=float, default=0.034)
    parser.add_argument("--play-value-learning-rate", type=float, default=0.024)
    parser.add_argument("--auction-value-learning-rate", type=float, default=0.028)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--warm-start-lr-scale",
        type=float,
        default=DEFAULT_WARM_START_LR_SCALE,
        help="multiplier applied to learning rates for DAgger retraining after bootstrap",
    )
    parser.add_argument(
        "--warm-start-epoch-scale",
        type=float,
        default=DEFAULT_WARM_START_EPOCH_SCALE,
        help="multiplier applied to epochs for DAgger retraining after bootstrap",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=DEFAULT_GRADIENT_CLIP,
        help="absolute gradient clip applied inside the dependency-free optimizer",
    )
    parser.add_argument(
        "--student-agreement-keep-prob",
        type=float,
        default=DEFAULT_STUDENT_AGREEMENT_KEEP_PROB,
        help="when student and teacher agree during DAgger, keep this fraction of policy examples",
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
        "--play-value-weight",
        type=float,
        default=DEFAULT_PLAY_VALUE_WEIGHT,
        help="runtime weight for the play value head during one-ply lookahead",
    )
    parser.add_argument(
        "--auction-value-weight",
        type=float,
        default=DEFAULT_AUCTION_VALUE_WEIGHT,
        help="runtime weight for the auction value head during one-ply lookahead",
    )
    parser.add_argument(
        "--play-rollout-depth",
        type=int,
        default=DEFAULT_PLAY_ROLLOUT_DEPTH,
        help="number of learned play plies to roll forward when scoring a candidate move",
    )
    parser.add_argument(
        "--auction-rollout-depth",
        type=int,
        default=DEFAULT_AUCTION_ROLLOUT_DEPTH,
        help="number of learned auction plies to roll forward when scoring a candidate action",
    )
    parser.add_argument(
        "--bot-id",
        default="neural-3p-v2",
        help="bot id to store in the checkpoint metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the trained model bundle",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress live training progress logs and only print the final JSON summary",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    teacher_specs = parse_teacher_specs(
        args.teacher_pool,
        fallback_teacher_bot_id=args.teacher,
    )
    initial_bundle = (
        load_model_bundle(args.initial_model)
        if args.initial_model is not None
        else None
    )
    bundle, report = train_with_dagger(
        teacher_specs=teacher_specs,
        bootstrap_matches=args.bootstrap_matches,
        dagger_matches=args.dagger_matches,
        alpha=args.alpha,
        dagger_iterations=args.dagger_iterations,
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
        initial_bundle=initial_bundle,
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
        bot_id=args.bot_id,
        verbose=not args.quiet,
    )
    save_model_bundle(bundle, args.output)
    _log(
        not args.quiet,
        (
            f"[train] wrote {args.output} "
            f"{_format_training_metrics(report['training'])}"
        ),
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "teacher_pool": _serialize_teacher_specs(teacher_specs),
                "final_play_accuracy": report["training"]["play_history"][-1]["accuracy"],
                "final_auction_accuracy": report["training"]["auction_history"][-1]["accuracy"],
                "final_play_value_mse": report["training"]["play_value_history"][-1]["mse"],
                "final_auction_value_mse": report["training"]["auction_value_history"][-1]["mse"],
                **report,
            }
        )
    )


if __name__ == "__main__":
    main()
