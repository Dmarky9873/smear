from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from .base import BotPlayer
    from .greedy_bot import GreedyPlayer
    from .minimax_one_trick_bot import MinimaxOneTrickPlayer
    from .random_bot import RandomPlayer
    from .stupid_bot import StupidBot
    from .o_minimax_one_trick_bot import OMNISCIENT_MinimaxOneTrickPlayer
except ImportError:
    from bots.base import BotPlayer
    from bots.greedy_bot import GreedyPlayer
    from bots.minimax_one_trick_bot import MinimaxOneTrickPlayer
    from bots.random_bot import RandomPlayer
    from bots.stupid_bot import StupidBot
    from bots.o_minimax_one_trick_bot import OMNISCIENT_MinimaxOneTrickPlayer


@dataclass(frozen=True)
class ReadyBotSpec:
    id: str
    label: str
    description: str
    factory: Callable[[str], BotPlayer]


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
        factory=lambda player_name: MinimaxOneTrickPlayer(player_name),
    ),
    ReadyBotSpec(
        id="o-one-trick-minmax",
        label="Omniscient L-1 Minmax",
        description="Chooses the mathematically perfect card for this current trick.",
        factory=lambda player_name: OMNISCIENT_MinimaxOneTrickPlayer(
            player_name)
    ),
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
