import unittest

from backend.bots.human_information_minimax_one_trick_bot import (
    HumanInformationMinimaxOneTrickPlayer,
)
from backend.bots.legacy_human_information_minimax_n_trick_bot import (
    LegacyHumanInformationMinimaxNTrickPlayer,
)
from backend.bots.legacy_omniscient_minimax_n_trick_bot import (
    LegacyOmniscientMinimaxNTrickPlayer,
)
from backend.bots.minimax_one_trick_bot import MinimaxOneTrickPlayer
from backend.bots.o_minimax_one_trick_bot import OMNISCIENT_MinimaxOneTrickPlayer
from backend.bots.omniscient_minimax_one_trick_bot import (
    OmniscientMinimaxOneTrickPlayer,
)
from backend.bots.registry import build_ready_bot
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

        bot = OmniscientMinimaxOneTrickPlayer("A")
        chosen_card = bot.choose_card(round_state)

        self.assertIn(chosen_card, get_legal_actions(round_state))
        self.assertEqual(chosen_card.code, "AH")

    def _build_hidden_information_state(
        self,
        *,
        player_b_card: Card,
        player_c_card: Card,
    ) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("10C")})
        player_b = Player("B", {player_b_card})
        player_c = Player("C", {player_c_card})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        deck = Deck("10")
        hidden_cards = (
            set(deck.get_copy())
            - player_a.cards
            - player_b.cards
            - player_c.cards
        )
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=hidden_cards,
            trick_history=[],
            teams=teams,
            deck=deck,
        )

    def test_hidden_information_bot_ignores_actual_opponent_hands(self):
        public_state_a = self._build_hidden_information_state(
            player_b_card=Card("KC"),
            player_c_card=Card("QD"),
        )
        public_state_b = self._build_hidden_information_state(
            player_b_card=Card("QS"),
            player_c_card=Card("KD"),
        )

        bot = HumanInformationMinimaxOneTrickPlayer(
            "A",
            cards=set(public_state_a.current_player.cards),
        )
        choice_a = bot.choose_card(public_state_a)
        choice_b = bot.choose_card(public_state_b)

        self.assertIn(choice_a, get_legal_actions(public_state_a))
        self.assertEqual(choice_a, choice_b)

    def test_registry_builds_hidden_information_bot_for_default_minimax_id(self):
        bot = build_ready_bot("one-trick-minmax", "A")
        self.assertIsInstance(bot, HumanInformationMinimaxOneTrickPlayer)

    def test_registry_builds_omniscient_bot_for_omniscient_minimax_id(self):
        bot = build_ready_bot("o-one-trick-minmax", "A")
        self.assertIsInstance(bot, OmniscientMinimaxOneTrickPlayer)

    def test_registry_builds_hidden_legacy_minimax_ids(self):
        self.assertIsInstance(
            build_ready_bot("legacy-2-trick-minmax", "A"),
            LegacyHumanInformationMinimaxNTrickPlayer,
        )
        self.assertIsInstance(
            build_ready_bot("legacy-o-2-trick-minmax", "A"),
            LegacyOmniscientMinimaxNTrickPlayer,
        )

    def test_legacy_import_paths_remain_compatible(self):
        self.assertIs(MinimaxOneTrickPlayer, HumanInformationMinimaxOneTrickPlayer)
        self.assertIs(
            OMNISCIENT_MinimaxOneTrickPlayer,
            OmniscientMinimaxOneTrickPlayer,
        )


if __name__ == "__main__":
    unittest.main()
