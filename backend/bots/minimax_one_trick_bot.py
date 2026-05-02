from __future__ import annotations

try:
    from .human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
        MinimaxOneTrickPlayer,
    )
except ImportError:
    from bots.human_information_minimax_one_trick_bot import (
        HumanInformationMinimaxOneTrickPlayer,
        MinimaxOneTrickPlayer,
    )
