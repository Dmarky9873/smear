from backend.models import Player, RoundState


class BotPlayer(Player):
    """Abstract base bot class"""

    def choose_card(self, round_state: RoundState):
        raise NotImplementedError
