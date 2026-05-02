import unittest

from backend.gameplay import MatchController


class MatchControllerBotStepTests(unittest.TestCase):
    def test_manual_mode_keeps_initial_bot_turn_until_advanced(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=["stupid", None, None],
            auto_run_bots=False,
        )

        self.assertEqual(controller.session.phase, "auction")
        self.assertEqual(controller.session.auction.current_bidder_name, "A")
        self.assertIsNone(controller.session.auction.state.highest_bid)

        controller.advance_bot_turn()

        self.assertEqual(controller.session.auction.current_bidder_name, "B")
        self.assertEqual(controller.session.auction.state.highest_bid, 1)
        self.assertEqual(controller.session.auction.state.highest_bidder_name, "A")

    def test_auto_run_bots_still_advances_to_next_human_turn(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=["stupid", None, None],
            auto_run_bots=True,
        )

        self.assertEqual(controller.session.phase, "auction")
        self.assertEqual(controller.session.auction.current_bidder_name, "B")
        self.assertEqual(controller.session.auction.state.highest_bid, 1)
        self.assertEqual(controller.session.auction.state.highest_bidder_name, "A")

    def test_advancing_bot_turn_on_human_turn_raises(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=["stupid", None, None],
            auto_run_bots=False,
        )

        controller.advance_bot_turn()

        with self.assertRaisesRegex(ValueError, "human-controlled player 'B'"):
            controller.advance_bot_turn()


if __name__ == "__main__":
    unittest.main()
