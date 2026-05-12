from __future__ import annotations

try:
    from backend.models import Card
    from .omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
except ImportError:
    from models import Card
    from .omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer


class OmniscientMinimaxOneTrickPlayer(OmniscientMinimaxNTrickPlayer):
    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
    ):
        super().__init__(name, cards, depth=1)


OMNISCIENT_MinimaxOneTrickPlayer = OmniscientMinimaxOneTrickPlayer
