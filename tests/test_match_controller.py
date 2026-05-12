import unittest

from backend.gameplay import MatchController
from backend.constants import HAND_SIZE
from backend.models import Deck, Player, RoundState, Team, TrickState


class MatchControllerBotStepTests(unittest.TestCase):
    def _build_terminal_round_state(self) -> RoundState:
        player_a = Player("A", set())
        player_b = Player("B", set())
        player_c = Player("C", set())
        players = [player_a, player_b, player_c]
        deck = Deck("10")
        terminal_trick = TrickState(player_a, [], players, "H")
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=terminal_trick,
            hidden_cards=set(deck.get_copy()),
            trick_history=[terminal_trick for _ in range(HAND_SIZE)],
            teams=[Team([player], set()) for player in players],
            deck=deck,
        )

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

    def test_greedy_bot_can_take_an_auction_turn(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=["greedy", None, None],
            auto_run_bots=False,
        )

        controller.advance_bot_turn()

        self.assertEqual(controller.session.auction.current_bidder_name, "B")
        self.assertEqual(len(controller.session.auction.state.bid_history), 1)
        self.assertEqual(
            controller.session.auction.state.bid_history[0].bidder_name,
            "A",
        )

    def test_next_round_preserves_last_round_score_for_ui(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=[None, None, None],
            auto_run_bots=False,
        )
        controller.session.last_round_score = {"summary": "previous round"}
        controller.session.last_scored_round_number = 1
        controller.session.game._round_state = self._build_terminal_round_state()

        controller.next_round(auto_run_bots=False)

        self.assertEqual(
            controller.session.last_round_score,
            {"summary": "previous round"},
        )
        self.assertEqual(controller.session.last_scored_round_number, 1)
        self.assertEqual(controller.session.round_number, 2)
        self.assertEqual(controller.session.phase, "auction")

    def test_get_score_returns_most_recent_round_during_next_round(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=[None, None, None],
            auto_run_bots=False,
        )
        controller.session.last_round_score = {"summary": "previous round"}
        controller.session.last_scored_round_number = 1

        self.assertEqual(
            controller.get_score(),
            {"summary": "previous round"},
        )

    def test_omniscient_bot_can_take_an_auction_turn(self):
        controller = MatchController.create(
            num_players=3,
            player_names=["A", "B", "C"],
            teams=None,
            player_bot_ids=["o-one-trick-minmax", None, None],
            auto_run_bots=False,
        )

        controller.advance_bot_turn()

        self.assertEqual(controller.session.auction.current_bidder_name, "B")
        self.assertEqual(len(controller.session.auction.state.bid_history), 1)
        self.assertEqual(
            controller.session.auction.state.bid_history[0].bidder_name,
            "A",
        )


if __name__ == "__main__":
    unittest.main()
