import unittest

from backend.engine import (
    Game,
    apply_trick_action_for_search,
    apply_trick_action_to_state,
    undo_trick_action_for_search,
)
from backend.models import Card, Deck, Play, Player, RoundState, Team, TrickState


class RoundStateTransitionTests(unittest.TestCase):
    def _state_snapshot(self, state: RoundState) -> dict:
        return {
            "current_player": state.current_player.name,
            "trump": state.trump,
            "current_trick_leader": state.current_trick.leader.name,
            "current_trick_plays": [
                (play.player.name, play.card.code) for play in state.current_trick.plays
            ],
            "trick_history": [
                [(play.player.name, play.card.code) for play in trick.plays]
                for trick in state.trick_history
            ],
            "hands": {
                player.name: sorted(card.code for card in player.cards)
                for player in state.players
            },
            "captures": {
                player.name: sorted(card.code for card in player.captured_cards)
                for player in state.players
            },
        }

    def _build_opening_state(self) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("KD")})
        player_b = Player("B", {Card("QC")})
        player_c = Player("C", {Card("JS")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        return RoundState(
            players=players,
            current_player=player_a,
            trump=None,
            current_trick=TrickState(player_a, [], players, None),
            hidden_cards=set(),
            trick_history=[],
            teams=teams,
            deck=Deck("10"),
        )

    def _build_terminal_trick_state(self) -> RoundState:
        player_a = Player("A", {Card("2D")})
        player_b = Player("B", {Card("3S")})
        player_c = Player("C", {Card("AH")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        current_trick = TrickState(
            leader=player_a,
            plays=[Play(player_a, Card("10C")), Play(player_b, Card("J1"))],
            players=players,
            trump="H",
        )
        return RoundState(
            players=players,
            current_player=player_c,
            trump="H",
            current_trick=current_trick,
            hidden_cards=set(),
            trick_history=[],
            teams=teams,
            deck=Deck("10"),
        )

    def test_apply_trick_action_to_state_does_not_mutate_original(self):
        round_state = self._build_opening_state()
        original_snapshot = self._state_snapshot(round_state)

        next_state = apply_trick_action_to_state(
            round_state,
            Play(round_state.current_player, Card("AH")),
        )

        self.assertEqual(self._state_snapshot(round_state), original_snapshot)
        self.assertIsNot(next_state, round_state)
        self.assertEqual(next_state.current_player.name, "B")
        self.assertEqual(next_state.trump, "H")
        self.assertEqual(
            [(play.player.name, play.card.code) for play in next_state.current_trick.plays],
            [("A", "AH")],
        )
        self.assertNotIn(Card("AH"), next_state.players[0].cards)

    def test_search_apply_and_undo_restore_non_terminal_state(self):
        round_state = self._build_opening_state()
        original_snapshot = self._state_snapshot(round_state)

        undo = apply_trick_action_for_search(
            round_state,
            Play(round_state.current_player, Card("AH")),
        )

        self.assertNotEqual(self._state_snapshot(round_state), original_snapshot)

        undo_trick_action_for_search(round_state, undo)

        self.assertEqual(self._state_snapshot(round_state), original_snapshot)

    def test_search_apply_and_undo_restore_terminal_trick_state(self):
        round_state = self._build_terminal_trick_state()
        original_snapshot = self._state_snapshot(round_state)

        undo = apply_trick_action_for_search(
            round_state,
            Play(round_state.current_player, Card("AH")),
        )

        self.assertNotEqual(self._state_snapshot(round_state), original_snapshot)

        undo_trick_action_for_search(round_state, undo)

        self.assertEqual(self._state_snapshot(round_state), original_snapshot)

    def test_game_apply_trick_action_matches_pure_helper(self):
        round_state = self._build_terminal_trick_state()
        pure_result = apply_trick_action_to_state(
            round_state,
            Play(round_state.current_player, Card("AH")),
        )

        game = Game(3, ["A", "B", "C"], [("A",), ("B",), ("C",)])
        game._round_state = self._build_terminal_trick_state()
        trick_completed = game.apply_trick_action(Play(game.curr_player, Card("AH")))

        self.assertTrue(trick_completed)
        self.assertEqual(self._state_snapshot(game.round_state), self._state_snapshot(pure_result))
        self.assertEqual(game.round_state.current_player.name, "C")
        self.assertEqual(len(game.round_state.trick_history), 1)
        self.assertEqual(sorted(card.code for card in game.round_state.current_player.captured_cards), ["10C", "AH", "J1"])


if __name__ == "__main__":
    unittest.main()
