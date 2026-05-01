try:
    from .engine import Game, get_legal_actions, score_round
    from .models import Play, Card, Player
except ImportError:
    from engine import Game, get_legal_actions, score_round
    from models import Play, Card, Player


class Simulator:
    game: Game

    def __init__(self, game: Game):
        self._game = game
