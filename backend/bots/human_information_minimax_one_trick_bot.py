from __future__ import annotations

try:
    from backend.models import Card
    from .human_information_minimax_n_trick_bot import HumanInformationMinimaxNTrickPlayer
except ImportError:
    from models import Card
    from .human_information_minimax_n_trick_bot import HumanInformationMinimaxNTrickPlayer


class HumanInformationMinimaxOneTrickPlayer(HumanInformationMinimaxNTrickPlayer):
    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
    ):
        super().__init__(name, cards, depth=1)


MinimaxOneTrickPlayer = HumanInformationMinimaxOneTrickPlayer
