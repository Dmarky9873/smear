import unittest

from backend.engine import get_trick_winner, score_round_details
from backend.models import Card, Deck, Play, Player, RoundState, Team, TrickState


class RoundScoringTests(unittest.TestCase):
    def _build_low_owner_not_capturer_state(self) -> RoundState:
        player_a = Player("A", set())
        player_b = Player("B", set())
        player_c = Player("C", set())
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        trick_history = [
            TrickState(
                player_a,
                [
                    Play(player_a, Card("10H")),
                    Play(player_b, Card("AH")),
                    Play(player_c, Card("10S")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("JH")),
                    Play(player_c, Card("KD")),
                    Play(player_a, Card("JD")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("KH")),
                    Play(player_c, Card("QD")),
                    Play(player_a, Card("KC")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("QH")),
                    Play(player_c, Card("AD")),
                    Play(player_a, Card("QC")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("AC")),
                    Play(player_c, Card("KS")),
                    Play(player_a, Card("QS")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("JC")),
                    Play(player_c, Card("J1")),
                    Play(player_a, Card("10D")),
                ],
                players,
                "H",
            ),
        ]

        for trick in trick_history:
            winner = get_trick_winner(trick)
            for play in trick.plays:
                winner.capture(play)

        deck = Deck("10")
        played_cards = {
            play.card
            for trick in trick_history
            for play in trick.plays
        }
        hidden_cards = set(deck.get_copy()) - played_cards

        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=hidden_cards,
            trick_history=trick_history,
            teams=teams,
            deck=deck,
        )

    def test_low_award_does_not_change_game_card_ownership(self):
        round_state = self._build_low_owner_not_capturer_state()

        details = score_round_details(round_state)
        results_by_name = {
            result["name"]: result
            for result in details["results"]
        }

        self.assertEqual(details["awards"]["low"]["unit_name"], "A")
        self.assertEqual(details["awards"]["high"]["unit_name"], "B")
        self.assertEqual(results_by_name["A"]["breakdown"]["low"], 1)
        self.assertEqual(results_by_name["A"]["game_total"], 0)
        self.assertNotIn(
            Card("10H"),
            results_by_name["A"]["captured_cards"],
        )
        self.assertIn(
            Card("10H"),
            results_by_name["B"]["captured_cards"],
        )
        self.assertEqual(results_by_name["B"]["game_total"], 54)
        self.assertEqual(details["hidden_cards"], round_state.hidden_cards)


if __name__ == "__main__":
    unittest.main()
