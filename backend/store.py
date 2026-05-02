from __future__ import annotations

try:
    from .gameplay import GameSession, MatchController, RoundNotTerminalError
except ImportError:
    from gameplay import GameSession, MatchController, RoundNotTerminalError


class GameNotInitializedError(RuntimeError):
    """Raised when a request needs a game but none exists."""


class GameStore:
    def __init__(self):
        self._controller: MatchController | None = None

    def require_controller(self) -> MatchController:
        if self._controller is None:
            raise GameNotInitializedError("No game exists yet. Create one first.")
        return self._controller

    def require_session(self) -> GameSession:
        return self.require_controller().session

    def create_game(
        self,
        num_players: int,
        player_names: list[str],
        teams: list[list[str]] | None,
        player_bots: list[str | None] | None = None,
    ) -> GameSession:
        self._controller = MatchController.create(
            num_players=num_players,
            player_names=player_names,
            teams=teams,
            player_bot_ids=player_bots,
            auto_run_bots=True,
        )
        return self._controller.session

    def reset_round(self) -> GameSession:
        return self.require_controller().reset_round(auto_run_bots=True)

    def next_round(self) -> GameSession:
        return self.require_controller().next_round(auto_run_bots=True)

    def get_state(self) -> GameSession:
        return self.require_controller().get_state()

    def get_legal_actions(self) -> list[dict]:
        return self.require_controller().get_legal_actions()

    def place_bid(self, amount: int) -> GameSession:
        return self.require_controller().place_bid(amount, auto_run_bots=True)

    def pass_auction(self) -> GameSession:
        return self.require_controller().pass_auction(auto_run_bots=True)

    def play_card(self, card_code: str) -> GameSession:
        return self.require_controller().play_card(card_code, auto_run_bots=True)

    def get_score(self) -> dict:
        return self.require_controller().get_score()


game_store = GameStore()
