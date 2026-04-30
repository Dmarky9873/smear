from __future__ import annotations

try:
    from .engine import Game, get_legal_actions, score_round
    from .models import Card, Play
except ImportError:
    from engine import Game, get_legal_actions, score_round
    from models import Card, Play


class GameNotInitializedError(RuntimeError):
    """Raised when a request needs a game but none exists."""


class RoundNotTerminalError(RuntimeError):
    """Raised when score is requested before the round is complete."""


def _normalize_teams(
    player_names: list[str],
    teams: list[list[str]] | None,
) -> list[tuple[str, ...]]:
    if teams is None:
        return [(name,) for name in player_names]

    seen_players: set[str] = set()
    normalized: list[tuple[str, ...]] = []
    expected_players = set(player_names)

    for raw_team in teams:
        cleaned_team = tuple(name.strip() for name in raw_team if name.strip())
        if not cleaned_team:
            raise ValueError("teams cannot contain empty groups")

        for name in cleaned_team:
            if name not in expected_players:
                raise ValueError(f"unknown player in teams: {name}")
            if name in seen_players:
                raise ValueError(f"player listed multiple times in teams: {name}")
            seen_players.add(name)

        normalized.append(cleaned_team)

    if seen_players != expected_players:
        missing = sorted(expected_players - seen_players)
        raise ValueError(
            f"every player must appear in exactly one team; missing: {', '.join(missing)}"
        )

    return normalized


class GameStore:
    def __init__(self):
        self._game: Game | None = None

    def require_game(self) -> Game:
        if self._game is None:
            raise GameNotInitializedError("No game exists yet. Create one first.")
        return self._game

    def create_game(
        self,
        num_players: int,
        player_names: list[str],
        teams: list[list[str]] | None,
    ) -> Game:
        normalized_names = [name.strip() for name in player_names]
        if len(normalized_names) != num_players:
            raise ValueError("num_players must match the number of player_names")
        if any(not name for name in normalized_names):
            raise ValueError("player_names cannot contain blank values")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("player_names must be unique")

        normalized_teams = _normalize_teams(normalized_names, teams)
        self._game = Game(num_players, normalized_names, normalized_teams)
        return self._game

    def reset_round(self) -> Game:
        game = self.require_game()
        game.reset_round()
        return game

    def get_state(self) -> Game:
        return self.require_game()

    def get_legal_actions(self) -> list[Card]:
        game = self.require_game()
        return list(get_legal_actions(game.round_state))

    def play_card(self, card_code: str) -> Game:
        game = self.require_game()
        card = Card(card_code.upper().strip())
        play = Play(game.curr_player, card)
        game.apply_trick_action(play)
        return game

    def get_score(self) -> dict[str, int]:
        game = self.require_game()
        if not game.round_state.is_terminal:
            raise RoundNotTerminalError("Round is not terminal yet.")
        return score_round(game.round_state)


game_store = GameStore()
