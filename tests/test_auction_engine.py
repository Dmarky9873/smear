import unittest

from backend.engine import Auction, get_legal_auction_actions
from backend.models import AuctionEvent


class AuctionRulesTests(unittest.TestCase):
    def test_auction_completes_after_each_player_acts_once(self):
        auction = Auction(["A", "B", "C", "D"], dealer="D")

        auction.apply_event(AuctionEvent("A", "bid", 1))
        auction.apply_event(AuctionEvent("B", "bid", 2))
        auction.apply_event(AuctionEvent("C", "pass"))
        state = auction.apply_event(AuctionEvent("D", "bid", 3))

        self.assertTrue(state.is_complete)
        self.assertEqual(state.highest_bid, 3)
        self.assertEqual(state.highest_bidder_name, "D")
        self.assertEqual(state.current_bidder_name, "D")
        self.assertEqual(len(state.bid_history), 4)

    def test_last_player_must_open_if_nobody_has_bid(self):
        auction = Auction(["A", "B", "C"], dealer="C")

        auction.apply_event(AuctionEvent("A", "pass"))
        auction.apply_event(AuctionEvent("B", "pass"))
        legal_actions = get_legal_auction_actions(auction.state)

        self.assertEqual(
            legal_actions,
            [
                AuctionEvent("C", "bid", 1),
                AuctionEvent("C", "bid", 2),
                AuctionEvent("C", "bid", 3),
                AuctionEvent("C", "bid", 4),
                AuctionEvent("C", "bid", 5),
                AuctionEvent("C", "bid", 6),
            ],
        )

    def test_bid_of_six_does_not_end_auction_early(self):
        auction = Auction(["A", "B", "C"], dealer="C")

        state = auction.apply_event(AuctionEvent("A", "bid", 6))
        self.assertFalse(state.is_complete)
        self.assertEqual(state.current_bidder_name, "B")
        self.assertEqual(
            get_legal_auction_actions(state),
            [AuctionEvent("B", "pass")],
        )

        state = auction.apply_event(AuctionEvent("B", "pass"))
        self.assertFalse(state.is_complete)
        self.assertEqual(state.current_bidder_name, "C")

        state = auction.apply_event(AuctionEvent("C", "pass"))
        self.assertTrue(state.is_complete)
        self.assertEqual(state.highest_bidder_name, "A")
        self.assertEqual(state.current_bidder_name, "A")


if __name__ == "__main__":
    unittest.main()
