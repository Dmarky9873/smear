from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from ..constants import HAND_SIZE
    from .base import BotPlayer
    from .greedy_bot import GreedyPlayer
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
    )
    from .legacy_human_information_minimax_n_trick_bot import (
        LegacyHumanInformationMinimaxNTrickPlayer,
    )
    from .legacy_human_information_minimax_one_trick_bot import (
        LegacyHumanInformationMinimaxOneTrickPlayer,
    )
    from .legacy_omniscient_minimax_n_trick_bot import (
        LegacyOmniscientMinimaxNTrickPlayer,
    )
    from .legacy_omniscient_minimax_one_trick_bot import (
        LegacyOmniscientMinimaxOneTrickPlayer,
    )
    from .neural_3p_bot import (
        NeuralThreePlayerBot,
        NeuralThreePlayerV1Bot,
        NeuralThreePlayerV3Bot,
        NeuralThreePlayerV4Bot,
    )
    from .omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from .omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
    from .optimal_bot import OptimalBotPlayer
    from .random_bot import RandomPlayer
    from .stupid_bot import StupidBot
except ImportError:
    from constants import HAND_SIZE
    from bots.base import BotPlayer
    from bots.greedy_bot import GreedyPlayer
    from bots.human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from bots.human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
    )
    from bots.legacy_human_information_minimax_n_trick_bot import (
        LegacyHumanInformationMinimaxNTrickPlayer,
    )
    from bots.legacy_human_information_minimax_one_trick_bot import (
        LegacyHumanInformationMinimaxOneTrickPlayer,
    )
    from bots.legacy_omniscient_minimax_n_trick_bot import (
        LegacyOmniscientMinimaxNTrickPlayer,
    )
    from bots.legacy_omniscient_minimax_one_trick_bot import (
        LegacyOmniscientMinimaxOneTrickPlayer,
    )
    from bots.neural_3p_bot import (
        NeuralThreePlayerBot,
        NeuralThreePlayerV1Bot,
        NeuralThreePlayerV3Bot,
        NeuralThreePlayerV4Bot,
    )
    from bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from bots.omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
    from bots.optimal_bot import OptimalBotPlayer
    from bots.random_bot import RandomPlayer
    from bots.stupid_bot import StupidBot


@dataclass(frozen=True)
class ReadyBotSpec:
    id: str
    label: str
    description: str
    factory: Callable[[str], BotPlayer]
    rating_fingerprint: str | None = None


MAX_VISIBLE_PRESET_DEPTH = 3


def _build_n_trick_specs(
    *,
    min_depth: int = 2,
    max_depth: int = HAND_SIZE,
) -> tuple[ReadyBotSpec, ...]:
    specs: list[ReadyBotSpec] = []
    for depth in range(min_depth, max_depth + 1):
        specs.append(
            ReadyBotSpec(
                id=f"{depth}-trick-minmax",
                label=f"L-{depth} Minmax",
                description=(
                    f"Searches the next {depth} completed tricks over sampled "
                    "hidden-information worlds."
                ),
                factory=lambda player_name, depth=depth: HumanInformationMinimaxNTrickPlayer(
                    player_name,
                    depth=depth,
                ),
            )
        )
        specs.append(
            ReadyBotSpec(
                id=f"o-{depth}-trick-minmax",
                label=f"Omniscient L-{depth} Minmax",
                description=(
                    f"Chooses the mathematically perfect card over the next {depth} "
                    "completed tricks."
                ),
                factory=lambda player_name, depth=depth: OmniscientMinimaxNTrickPlayer(
                    player_name,
                    depth=depth,
                ),
            )
        )
    return tuple(specs)


def _build_legacy_n_trick_specs(
    *,
    min_depth: int = 2,
    max_depth: int = HAND_SIZE,
) -> tuple[ReadyBotSpec, ...]:
    specs: list[ReadyBotSpec] = []
    for depth in range(min_depth, max_depth + 1):
        specs.append(
            ReadyBotSpec(
                id=f"legacy-{depth}-trick-minmax",
                label=f"Legacy L-{depth} Minmax",
                description=(
                    f"Legacy sampled hidden-information minimax over the next "
                    f"{depth} completed tricks."
                ),
                factory=lambda player_name, depth=depth: LegacyHumanInformationMinimaxNTrickPlayer(
                    player_name,
                    depth=depth,
                ),
            )
        )
        specs.append(
            ReadyBotSpec(
                id=f"legacy-o-{depth}-trick-minmax",
                label=f"Legacy Omniscient L-{depth} Minmax",
                description=(
                    f"Legacy perfect-information minimax over the next {depth} "
                    "completed tricks."
                ),
                factory=lambda player_name, depth=depth: LegacyOmniscientMinimaxNTrickPlayer(
                    player_name,
                    depth=depth,
                ),
            )
        )
    return tuple(specs)


READY_BOTS: tuple[ReadyBotSpec, ...] = (
    ReadyBotSpec(
        id="random",
        label="Random",
        description="Chooses uniformly from the legal auction and card actions returned by the backend.",
        factory=lambda player_name: RandomPlayer(player_name),
    ),
    ReadyBotSpec(
        id="greedy",
        label="Greedy",
        description="Uses a basic algorithm to choose auction and trick actions.",
        factory=lambda player_name: GreedyPlayer(player_name),
    ),
    ReadyBotSpec(
        id="stupid",
        label="Stupid",
        description="Auctions very stupidly; always bids one higher than the previous highest bid. Hard when you're about to go out.",
        factory=lambda player_name: StupidBot(player_name),
    ),
    ReadyBotSpec(
        id="optimal-bot",
        label="Optimal Bot",
        description=(
            "Adaptive sampled hidden-information minimax that uses the stronger "
            "depth-2 profile in three-player games and the depth-3 baseline in "
            "larger games."
        ),
        factory=lambda player_name: OptimalBotPlayer(player_name),
        rating_fingerprint="optimal-bot:v2",
    ),
    ReadyBotSpec(
        id="neural-3p-v1",
        label="Neural 3P v1",
        description=(
            "A tiny dependency-free neural policy for three-player singleton "
            "smear, with greedy fallback outside that format."
        ),
        factory=lambda player_name: NeuralThreePlayerV1Bot(
            player_name,
            model_path=NeuralThreePlayerBot.MODEL_FILE_V1,
        ),
        rating_fingerprint="neural-3p-v1:v1",
    ),
    ReadyBotSpec(
        id="neural-3p-v2",
        label="Neural 3P v2",
        description=(
            "A dependency-free 3-player singleton neural bot with mixed-teacher "
            "distillation, DAgger relabeling, and a value-guided one-ply lookahead."
        ),
        factory=lambda player_name: NeuralThreePlayerBot(
            player_name,
            model_path=NeuralThreePlayerBot.MODEL_FILE_V2,
        ),
        rating_fingerprint="neural-3p-v2:v1",
    ),
    ReadyBotSpec(
        id="neural-3p-v3",
        label="Neural 3P v3",
        description=(
            "A dependency-free 3-player singleton neural bot trained from v2 by "
            "advantage-weighted self-play and promoted head-to-head."
        ),
        factory=lambda player_name: NeuralThreePlayerV3Bot(
            player_name,
            model_path=NeuralThreePlayerBot.MODEL_FILE_V3,
        ),
        rating_fingerprint="neural-3p-v3:v1",
    ),
    ReadyBotSpec(
        id="neural-3p-v4",
        label="Neural 3P v4",
        description=(
            "A dependency-free 3-player singleton neural bot that alternates "
            "teacher-guided imitation refreshes with promoted self-play."
        ),
        factory=lambda player_name: NeuralThreePlayerV4Bot(
            player_name,
            model_path=NeuralThreePlayerBot.MODEL_FILE_V4,
        ),
        rating_fingerprint="neural-3p-v4:v1",
    ),
    ReadyBotSpec(
        id="1-trick-minmax",
        label="L-1 Minmax",
        description="Searches the current trick over sampled hidden-information worlds.",
        factory=lambda player_name: HumanInformationMinimaxOneTrickPlayer(
            player_name
        ),
    ),
    ReadyBotSpec(
        id="o-one-trick-minmax",
        label="Omniscient L-1 Minmax",
        description="Chooses the mathematically perfect card for this current trick.",
        factory=lambda player_name: OmniscientMinimaxOneTrickPlayer(
            player_name),
    ),
    *_build_n_trick_specs(max_depth=MAX_VISIBLE_PRESET_DEPTH),
)

HIDDEN_BOTS: tuple[ReadyBotSpec, ...] = (
    ReadyBotSpec(
        id="one-trick-minmax",
        label="L-1 Minmax",
        description="Alias for the default one-trick sampled hidden-information minimax bot.",
        factory=lambda player_name: HumanInformationMinimaxOneTrickPlayer(
            player_name
        ),
    ),
    ReadyBotSpec(
        id="legacy-one-trick-minmax",
        label="Legacy L-1 Minmax",
        description="Legacy one-trick sampled hidden-information minimax.",
        factory=lambda player_name: LegacyHumanInformationMinimaxOneTrickPlayer(
            player_name
        ),
    ),
    ReadyBotSpec(
        id="legacy-o-one-trick-minmax",
        label="Legacy Omniscient L-1 Minmax",
        description="Legacy one-trick perfect-information minimax.",
        factory=lambda player_name: LegacyOmniscientMinimaxOneTrickPlayer(
            player_name
        ),
    ),
    *_build_n_trick_specs(min_depth=MAX_VISIBLE_PRESET_DEPTH + 1),
    *_build_legacy_n_trick_specs(),
)

READY_BOT_MAP = {bot.id: bot for bot in (*READY_BOTS, *HIDDEN_BOTS)}


def get_ready_bot_spec(bot_id: str) -> ReadyBotSpec:
    try:
        return READY_BOT_MAP[bot_id]
    except KeyError as exc:
        raise ValueError(f"unknown bot id: {bot_id}") from exc


def get_ready_bot_rating_fingerprint(bot_id: str) -> str:
    spec = get_ready_bot_spec(bot_id)
    return spec.rating_fingerprint or spec.id


def build_ready_bot(bot_id: str, player_name: str) -> BotPlayer:
    spec = get_ready_bot_spec(bot_id)
    return spec.factory(player_name)


def list_ready_bot_metadata() -> list[dict[str, str]]:
    return [
        {
            "id": bot.id,
            "label": bot.label,
            "description": bot.description,
        }
        for bot in READY_BOTS
    ]
