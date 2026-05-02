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
    from .omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from .omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
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
    from bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
    from bots.omniscient_minimax_one_trick_bot import OmniscientMinimaxOneTrickPlayer
    from bots.random_bot import RandomPlayer
    from bots.stupid_bot import StupidBot


@dataclass(frozen=True)
class ReadyBotSpec:
    id: str
    label: str
    description: str
    factory: Callable[[str], BotPlayer]


def _build_n_trick_specs() -> tuple[ReadyBotSpec, ...]:
    specs: list[ReadyBotSpec] = []
    for depth in range(2, HAND_SIZE + 1):
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
        id="one-trick-minmax",
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
        factory=lambda player_name: OmniscientMinimaxOneTrickPlayer(player_name),
    ),
    *_build_n_trick_specs(),
)

READY_BOT_MAP = {bot.id: bot for bot in READY_BOTS}


def get_ready_bot_spec(bot_id: str) -> ReadyBotSpec:
    try:
        return READY_BOT_MAP[bot_id]
    except KeyError as exc:
        raise ValueError(f"unknown bot id: {bot_id}") from exc


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
