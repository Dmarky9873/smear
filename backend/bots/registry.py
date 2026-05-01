from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from .base import BotPlayer
    from .random_bot import RandomPlayer
except ImportError:
    from base import BotPlayer
    from random_bot import RandomPlayer


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
