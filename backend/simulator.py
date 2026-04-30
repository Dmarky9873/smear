from engine import Game, get_legal_actions, score_round
from models import Play, Card, Player


class Simulator:
    game: Game

    def __init__(self, game: Game):
        self._game = game

    def _run_trick(self):
        is_terminal = False
        while not is_terminal:
            curr_player = self._game.curr_player
            legal_actions = get_legal_actions(self._game._round_state)
            print(f"Current player: {curr_player}")
            card = None
            while card not in legal_actions:
                print(
                    f"please select a card to play from the legal cards: {legal_actions}")
                card = Card(input().upper().strip())
            play = Play(curr_player, card)

            is_terminal = self._game.apply_trick_action(play)

    def _run_round(self, auction_winner: Player) -> dict:
        self._game.set_starting_player(auction_winner)

        while not self._game._round_state.is_terminal:
            self._run_trick()

        return score_round(self._game._round_state)
