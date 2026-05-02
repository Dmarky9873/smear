import unittest

from backend.bots.minimax_one_trick_bot import MinimaxOneTrickPlayer
from backend.engine import get_legal_actions
from backend.models import Card, Deck, Player, RoundState, Team, TrickState


class MinimaxOneTrickBotTests(unittest.TestCase):
    def test_choose_card_uses_team_membership_by_name(self):
        player_a = Player("A", {Card("AH"), Card("10C")})
        player_b = Player("B", {Card("KC")})
        player_c = Player("C", {Card("2C")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        round_state = RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=set(),
            trick_history=[],
            teams=teams,
            deck=Deck("10"),
        )

        bot = MinimaxOneTrickPlayer("A")
        chosen_card = bot.choose_card(round_state)

        self.assertIn(chosen_card, get_legal_actions(round_state))
        self.assertEqual(chosen_card.code, "AH")


if __name__ == "__main__":
    unittest.main()
