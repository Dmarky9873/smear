from __future__ import annotations

try:
    from .omniscient_minimax_one_trick_bot import (
        OMNISCIENT_MinimaxOneTrickPlayer,
        OmniscientMinimaxOneTrickPlayer,
    )
except ImportError:
    from bots.omniscient_minimax_one_trick_bot import (
        OMNISCIENT_MinimaxOneTrickPlayer,
        OmniscientMinimaxOneTrickPlayer,
    )


MinimaxOneTrickPlayer = OMNISCIENT_MinimaxOneTrickPlayer
