from backend.engine import get_legal_actions
from random import choice
from base import BotPlayer


class RandomPlayer(BotPlayer):
    def choose_card(self, round_state):
        return choice(get_legal_actions(round_state))

    # def auction
