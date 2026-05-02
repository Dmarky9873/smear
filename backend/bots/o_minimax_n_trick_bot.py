from __future__ import annotations

try:
    from .omniscient_minimax_n_trick_bot import (
        OMNISCIENT_MinimaxNTrickPlayer,
        OmniscientMinimaxNTrickPlayer,
    )
except ImportError:
    from bots.omniscient_minimax_n_trick_bot import (
        OMNISCIENT_MinimaxNTrickPlayer,
        OmniscientMinimaxNTrickPlayer,
    )
