from __future__ import annotations

try:
    from backend.models import Card
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .optimal_bot_tuning import (
        OPTIMAL_BOT_MULTIPLAYER_CANDIDATE,
        OPTIMAL_BOT_THREE_PLAYER_CANDIDATE,
        OptimalBotCandidate,
    )
except ImportError:
    from models import Card
    from .human_information_minimax_n_trick_bot import (
        HumanInformationMinimaxNTrickPlayer,
    )
    from .optimal_bot_tuning import (
        OPTIMAL_BOT_MULTIPLAYER_CANDIDATE,
        OPTIMAL_BOT_THREE_PLAYER_CANDIDATE,
        OptimalBotCandidate,
    )


class OptimalBotPlayer(HumanInformationMinimaxNTrickPlayer):
    """Adaptive sampled minimax tuned for strong three-player ladder play."""

    THREE_PLAYER_PROFILE = OPTIMAL_BOT_THREE_PLAYER_CANDIDATE
    MULTIPLAYER_PROFILE = OPTIMAL_BOT_MULTIPLAYER_CANDIDATE
    PROFILE = MULTIPLAYER_PROFILE
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

    def _profile_for_player_count(
        self,
        player_count: int | None,
    ) -> OptimalBotCandidate:
        if player_count == 3:
            return self.THREE_PLAYER_PROFILE
        return self.MULTIPLAYER_PROFILE

    def _profile_determinization_sample_count(
        self,
        profile: OptimalBotCandidate,
        *,
        player_count: int | None,
    ) -> int:
        if player_count == 3:
            if profile.depth <= 2:
                return profile.play_determinization_samples
            if profile.depth == 3:
                return max(
                    profile.min_play_determinization_samples,
                    (profile.play_determinization_samples * 7) // 12,
                )
            if profile.depth == 4:
                return max(
                    profile.min_play_determinization_samples,
                    profile.play_determinization_samples // 3,
                )
            return max(
                profile.min_play_determinization_samples,
                profile.play_determinization_samples // 5,
            )

        if profile.depth <= 2:
            return profile.play_determinization_samples
        if profile.depth == 3:
            return max(
                profile.min_play_determinization_samples,
                profile.play_determinization_samples // 2,
            )
        if profile.depth == 4:
            return max(
                profile.min_play_determinization_samples,
                profile.play_determinization_samples // 4,
            )
        return profile.min_play_determinization_samples

    def _determinization_sample_count(self, *, player_count: int | None = None) -> int:
        profile = self._profile_for_player_count(player_count)
        return self._profile_determinization_sample_count(
            profile,
            player_count=player_count,
        )

    def _auction_determinization_sample_count(self, auction_state) -> int:
        profile = self._profile_for_player_count(len(auction_state.player_names))
        if len(auction_state.player_names) == 3:
            return profile.three_player_auction_determinization_samples
        return profile.auction_determinization_samples

    def _run_with_profile_depth(
        self,
        profile: OptimalBotCandidate,
        fn,
        state,
    ):
        original_depth = self.depth
        self.depth = profile.depth
        try:
            return fn(state)
        finally:
            self.depth = original_depth

    def choose_card(self, round_state):
        profile = self._profile_for_player_count(len(round_state.players))
        return self._run_with_profile_depth(profile, super().choose_card, round_state)

    def choose_auction_action(self, auction_state):
        profile = self._profile_for_player_count(len(auction_state.player_names))
        return self._run_with_profile_depth(
            profile,
            super().choose_auction_action,
            auction_state,
        )
