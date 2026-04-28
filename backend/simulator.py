from engine import Game, get_legal_actions
from models import Play


class Simulator:
    game: Game

    def __init__(self, game: Game):
        self._game = game

    def run_trick(self, trump):
        trick = self.game._round_state.current_trick
        round = self.game._round_state
        trick.trump = trump
        round.trump = trump
        is_terminal = False
        while not is_terminal:
            curr_player = self.game.curr_player
            legal_actions = get_legal_actions(round)
            print(f"Current player: {curr_player}")
            card = None
            while card not in legal_actions:
                print(
                    f"please select a card to play from the legal cards: {legal_actions}")
                card = input().upper().strip()
            play = Play(curr_player, card)

            is_terminal = self.game.apply_trick_action(play)
