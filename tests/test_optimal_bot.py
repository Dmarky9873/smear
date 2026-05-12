import unittest

from backend.bots.optimal_bot import OptimalBotPlayer
from backend.bots.optimal_bot_tuning import (
    DEPTH_2_SPEED_CANDIDATE,
    DEPTH_3_BASELINE_CANDIDATE,
    DEPTH_3_FAST_CANDIDATE,
    DEPTH_4_SEARCH_CANDIDATE,
    OPTIMAL_BOT_CANDIDATE,
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
        self.assertEqual(bot.depth, OPTIMAL_BOT_CANDIDATE.depth)

    def test_ready_bot_metadata_lists_optimal_bot(self):
        bot_ids = {bot["id"] for bot in list_ready_bot_metadata()}

        self.assertIn("optimal-bot", bot_ids)

    def test_optimal_bot_uses_tuned_determinization_schedule(self):
        bot = OptimalBotPlayer("A")

        self.assertEqual(
            bot._determinization_sample_count(player_count=4),
            5,
        )
        self.assertEqual(
            bot._determinization_sample_count(player_count=3),
            5,
        )
        self.assertEqual(
            bot._auction_determinization_sample_count(
                AuctionState(
                    dealer_index=0,
                    current_bidder_index=1,
                    player_names=["A", "B", "C"],
                )
            ),
            OPTIMAL_BOT_CANDIDATE.three_player_auction_determinization_samples,
        )

    def test_balance_score_requires_positive_runtime(self):
        with self.assertRaisesRegex(ValueError, "seconds_per_game must be positive"):
            balance_score(
                OptimalBotCandidateBenchmark(
                    candidate=OPTIMAL_BOT_CANDIDATE,
                    win_rate=1.0,
                    seconds_per_game=0.0,
                ),
                fastest_seconds_per_game=1.0,
            )

    def test_select_optimal_bot_candidate_prefers_best_balance(self):
        self.assertIn(OPTIMAL_BOT_CANDIDATE, OPTIMAL_BOT_TUNING_CANDIDATES)

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
                    candidate=OPTIMAL_BOT_CANDIDATE,
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

        self.assertEqual(selected.candidate, OPTIMAL_BOT_CANDIDATE)

    def test_select_optimal_bot_candidate_rejects_empty_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "at least one benchmark result is required",
        ):
            select_optimal_bot_candidate([])


if __name__ == "__main__":
    unittest.main()
