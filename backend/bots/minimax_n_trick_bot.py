from __future__ import annotations

try:
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
        MinimaxNTrickPlayer,
    )
except ImportError:
    from bots.human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
        MinimaxNTrickPlayer,
    )
