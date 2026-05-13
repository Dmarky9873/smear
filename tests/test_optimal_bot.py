import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.bots.optimal_bot import OptimalBotPlayer
from backend.bots.optimal_bot_tuning import (
    DEPTH_2_SPEED_CANDIDATE,
    DEPTH_3_BASELINE_CANDIDATE,
    DEPTH_3_FAST_CANDIDATE,
    DEPTH_4_SEARCH_CANDIDATE,
    LEGACY_OPTIMAL_BOT_CANDIDATE,
    OPTIMAL_BOT_MULTIPLAYER_CANDIDATE,
    OPTIMAL_BOT_THREE_PLAYER_CANDIDATE,
    OPTIMAL_BOT_TUNING_CANDIDATES,
    OptimalBotCandidateBenchmark,
    balance_score,
    select_optimal_bot_candidate,
)
from backend.bots.registry import build_ready_bot, list_ready_bot_metadata
from backend.models import AuctionState


class OptimalBotTests(unittest.TestCase):
    def test_ready_bot_registry_builds_optimal_bot(self):
        bot = build_ready_bot("optimal-bot", "A")

        self.assertIsInstance(bot, OptimalBotPlayer)
        self.assertEqual(bot.depth, OPTIMAL_BOT_MULTIPLAYER_CANDIDATE.depth)

    def test_ready_bot_metadata_lists_optimal_bot(self):
        bot_ids = {bot["id"] for bot in list_ready_bot_metadata()}

        self.assertIn("optimal-bot", bot_ids)

    def test_optimal_bot_uses_adaptive_profiles(self):
        bot = OptimalBotPlayer("A")

        self.assertEqual(
            bot._profile_for_player_count(3),
            OPTIMAL_BOT_THREE_PLAYER_CANDIDATE,
        )
        self.assertEqual(
            bot._profile_for_player_count(4),
            OPTIMAL_BOT_MULTIPLAYER_CANDIDATE,
        )
        self.assertEqual(
            bot._determinization_sample_count(player_count=4),
            6,
        )
        self.assertEqual(
            bot._determinization_sample_count(player_count=3),
            12,
        )
        self.assertEqual(
            bot._auction_determinization_sample_count(
                AuctionState(
                    dealer_index=0,
                    current_bidder_index=1,
                    player_names=["A", "B", "C"],
                )
            ),
            OPTIMAL_BOT_THREE_PLAYER_CANDIDATE.three_player_auction_determinization_samples,
        )
        self.assertEqual(
            bot._auction_determinization_sample_count(
                AuctionState(
                    dealer_index=0,
                    current_bidder_index=1,
                    player_names=["A", "B", "C", "D"],
                )
            ),
            OPTIMAL_BOT_MULTIPLAYER_CANDIDATE.auction_determinization_samples,
        )

    def test_balance_score_requires_positive_runtime(self):
        with self.assertRaisesRegex(ValueError, "seconds_per_game must be positive"):
            balance_score(
                OptimalBotCandidateBenchmark(
                    candidate=LEGACY_OPTIMAL_BOT_CANDIDATE,
                    win_rate=1.0,
                    seconds_per_game=0.0,
                ),
                fastest_seconds_per_game=1.0,
            )

    def test_optimal_bot_uses_three_player_depth_during_search(self):
        bot = OptimalBotPlayer("A")

        with patch(
            "backend.bots.human_information_minimax_n_trick_bot."
            "HumanInformationMinimaxNTrickPlayer.choose_card",
            autospec=True,
            side_effect=lambda player, round_state: player.depth,
        ) as choose_card_mock:
            selected_depth = bot.choose_card(
                SimpleNamespace(players=["A", "B", "C"])
            )

        self.assertEqual(selected_depth, OPTIMAL_BOT_THREE_PLAYER_CANDIDATE.depth)
        self.assertEqual(bot.depth, OPTIMAL_BOT_MULTIPLAYER_CANDIDATE.depth)
        choose_card_mock.assert_called_once()

    def test_select_optimal_bot_candidate_prefers_best_balance(self):
        self.assertIn(LEGACY_OPTIMAL_BOT_CANDIDATE, OPTIMAL_BOT_TUNING_CANDIDATES)

        selected = select_optimal_bot_candidate(
            [
                OptimalBotCandidateBenchmark(
                    candidate=DEPTH_2_SPEED_CANDIDATE,
                    win_rate=0.50,
                    seconds_per_game=4.15,
                ),
                OptimalBotCandidateBenchmark(
                    candidate=DEPTH_3_FAST_CANDIDATE,
                    win_rate=0.75,
                    seconds_per_game=7.35,
                ),
                OptimalBotCandidateBenchmark(
                    candidate=LEGACY_OPTIMAL_BOT_CANDIDATE,
                    win_rate=1.00,
                    seconds_per_game=8.17,
                ),
                OptimalBotCandidateBenchmark(
                    candidate=DEPTH_3_BASELINE_CANDIDATE,
                    win_rate=0.50,
                    seconds_per_game=12.22,
                ),
                OptimalBotCandidateBenchmark(
                    candidate=DEPTH_4_SEARCH_CANDIDATE,
                    win_rate=0.25,
                    seconds_per_game=18.05,
                ),
            ]
        )

        self.assertEqual(selected.candidate, LEGACY_OPTIMAL_BOT_CANDIDATE)

    def test_select_optimal_bot_candidate_rejects_empty_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "at least one benchmark result is required",
        ):
            select_optimal_bot_candidate([])


if __name__ == "__main__":
    unittest.main()
