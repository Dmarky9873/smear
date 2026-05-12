import unittest

from backend.bots.greedy_bot import GreedyPlayer
from backend.bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
from backend.models import AuctionState, Card


class AuctionBiddingHeuristicsTests(unittest.TestCase):
    def _opening_auction_state(self) -> AuctionState:
        return AuctionState(
            dealer_index=2,
            current_bidder_index=0,
            player_names=["A", "B", "C"],
        )

    def test_greedy_threshold_opens_with_guaranteed_high(self):
        bot = GreedyPlayer(
            "A",
            {
                Card("AH"),
                Card("KS"),
                Card("QD"),
                Card("JC"),
                Card("10C"),
                Card("AS"),
            },
            use_rollout_auction=False,
        )

        action = bot.choose_auction_action(self._opening_auction_state())

        self.assertEqual(action.action, "bid")
        self.assertEqual(action.amount, 1)

    def test_greedy_threshold_opens_two_with_guaranteed_high_and_low(self):
        bot = GreedyPlayer(
            "A",
            {
                Card("AH"),
                Card("10H"),
                Card("KS"),
                Card("QD"),
                Card("JC"),
                Card("AS"),
            },
            use_rollout_auction=False,
        )

        action = bot.choose_auction_action(self._opening_auction_state())

        self.assertEqual(action.action, "bid")
        self.assertEqual(action.amount, 2)
        self.assertEqual(bot._opening_suit_override, "H")

    def test_second_joker_does_not_increase_bid_strength_again(self):
        one_joker_bot = GreedyPlayer(
            "A",
            {
                Card("AS"),
                Card("J1"),
                Card("10H"),
                Card("10D"),
                Card("10C"),
                Card("9C"),
            },
            use_rollout_auction=False,
        )
        two_joker_bot = GreedyPlayer(
            "A",
            {
                Card("AS"),
                Card("J1"),
                Card("J2"),
                Card("10H"),
                Card("10D"),
                Card("10C"),
            },
            use_rollout_auction=False,
        )

        self.assertEqual(
            one_joker_bot._estimate_hand_strength("S"),
            two_joker_bot._estimate_hand_strength("S"),
        )

    def test_greedy_rollout_opening_bid_stays_at_guaranteed_floor(self):
        bot = GreedyPlayer(
            "A",
            {
                Card("AS"),
                Card("J1"),
                Card("J2"),
                Card("10H"),
                Card("10D"),
                Card("10C"),
            },
            use_rollout_auction=True,
        )

        action = bot.choose_auction_action(self._opening_auction_state())

        self.assertEqual(action.action, "bid")
        self.assertEqual(action.amount, 1)
        self.assertEqual(bot._opening_suit_override, "S")

    def test_omniscient_opening_bid_uses_guaranteed_points_rule(self):
        bot = OmniscientMinimaxNTrickPlayer(
            "A",
            {
                Card("AH"),
                Card("10H"),
                Card("KS"),
                Card("QD"),
                Card("JC"),
                Card("AS"),
            },
            depth=2,
        )

        action = bot.choose_auction_action(self._opening_auction_state())

        self.assertEqual(action.action, "bid")
        self.assertEqual(action.amount, 2)
        self.assertEqual(bot._opening_suit_override, "H")


if __name__ == "__main__":
    unittest.main()
