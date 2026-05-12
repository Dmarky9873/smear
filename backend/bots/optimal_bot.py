from __future__ import annotations

try:
    from backend.models import Card
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .optimal_bot_tuning import OPTIMAL_BOT_CANDIDATE
except ImportError:
    from models import Card
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .optimal_bot_tuning import OPTIMAL_BOT_CANDIDATE


class OptimalBotPlayer(HumanInformationMinimaxNTrickPlayer):
    """Depth-3 sampled minimax tuned to keep most of the search strength."""

    PROFILE = OPTIMAL_BOT_CANDIDATE
    DETERMINIZATION_SAMPLES = PROFILE.play_determinization_samples
    MIN_DETERMINIZATION_SAMPLES = PROFILE.min_play_determinization_samples
    AUCTION_DETERMINIZATION_SAMPLES = PROFILE.auction_determinization_samples
    THREE_PLAYER_AUCTION_DETERMINIZATION_SAMPLES = (
        PROFILE.three_player_auction_determinization_samples
    )

    def __init__(
        self,
        name: str,
        cards: set[Card] | None = None,
    ):
        super().__init__(name, cards, depth=self.PROFILE.depth)
