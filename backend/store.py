from __future__ import annotations

from dataclasses import dataclass, field
import os
from threading import RLock
import time

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
        auto_run_bots: bool = True,
    ) -> GameSession:
        self._controller = MatchController.create(
            num_players=num_players,
            player_names=player_names,
            teams=teams,
            player_bot_ids=player_bots,
            auto_run_bots=auto_run_bots,
        )
        return self._controller.session

    def reset_round(self, auto_run_bots: bool = True) -> GameSession:
        return self.require_controller().reset_round(auto_run_bots=auto_run_bots)

    def next_round(self, auto_run_bots: bool = True) -> GameSession:
        return self.require_controller().next_round(auto_run_bots=auto_run_bots)

    def get_state(self) -> GameSession:
        return self.require_controller().get_state()

    def get_legal_actions(self) -> list[dict]:
        return self.require_controller().get_legal_actions()

    def place_bid(self, amount: int, auto_run_bots: bool = True) -> GameSession:
        return self.require_controller().place_bid(
            amount,
            auto_run_bots=auto_run_bots,
        )

    def pass_auction(self, auto_run_bots: bool = True) -> GameSession:
        return self.require_controller().pass_auction(auto_run_bots=auto_run_bots)

    def play_card(self, card_code: str, auto_run_bots: bool = True) -> GameSession:
        return self.require_controller().play_card(
            card_code,
            auto_run_bots=auto_run_bots,
        )

    def advance_bot_turn(self) -> GameSession:
        return self.require_controller().advance_bot_turn()

    def get_score(self) -> dict:
        return self.require_controller().get_score()

    def get_bot_progress(self) -> dict:
        return self.require_controller().get_bot_progress()


def _load_session_ttl_seconds() -> int:
    raw_value = os.getenv("SMEAR_SESSION_TTL_HOURS", "12").strip()
    try:
        hours = float(raw_value)
    except ValueError:
        hours = 12.0

    return max(1, int(hours * 60 * 60))


@dataclass
class SessionEntry:
    store: GameStore = field(default_factory=GameStore)
    lock: RLock = field(default_factory=RLock)
    last_accessed_at: float = field(default_factory=time.monotonic)


class SessionGameStore:
    def __init__(self, session_ttl_seconds: int):
        self._session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = RLock()

    def _prune_expired_sessions_locked(self) -> None:
        now = time.monotonic()
        expired_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - entry.last_accessed_at > self._session_ttl_seconds
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    def _get_entry(self, session_id: str) -> SessionEntry:
        with self._lock:
            self._prune_expired_sessions_locked()
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = SessionEntry()
                self._sessions[session_id] = entry
            entry.last_accessed_at = time.monotonic()
            return entry

    def _run(self, session_id: str, operation):
        entry = self._get_entry(session_id)
        with entry.lock:
            entry.last_accessed_at = time.monotonic()
            return operation(entry.store)

    def create_game(
        self,
        session_id: str,
        num_players: int,
        player_names: list[str],
        teams: list[list[str]] | None,
        player_bots: list[str | None] | None = None,
        auto_run_bots: bool = True,
    ) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.create_game(
                num_players=num_players,
                player_names=player_names,
                teams=teams,
                player_bots=player_bots,
                auto_run_bots=auto_run_bots,
            ),
        )

    def reset_round(self, session_id: str, auto_run_bots: bool = True) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.reset_round(auto_run_bots=auto_run_bots),
        )

    def next_round(self, session_id: str, auto_run_bots: bool = True) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.next_round(auto_run_bots=auto_run_bots),
        )

    def get_state(self, session_id: str) -> GameSession:
        return self._run(session_id, lambda store: store.get_state())

    def get_legal_actions(self, session_id: str) -> list[dict]:
        return self._run(session_id, lambda store: store.get_legal_actions())

    def place_bid(
        self,
        session_id: str,
        amount: int,
        auto_run_bots: bool = True,
    ) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.place_bid(
                amount,
                auto_run_bots=auto_run_bots,
            ),
        )

    def pass_auction(self, session_id: str, auto_run_bots: bool = True) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.pass_auction(auto_run_bots=auto_run_bots),
        )

    def play_card(
        self,
        session_id: str,
        card_code: str,
        auto_run_bots: bool = True,
    ) -> GameSession:
        return self._run(
            session_id,
            lambda store: store.play_card(
                card_code,
                auto_run_bots=auto_run_bots,
            ),
        )

    def advance_bot_turn(self, session_id: str) -> GameSession:
        return self._run(session_id, lambda store: store.advance_bot_turn())

    def get_score(self, session_id: str) -> dict:
        return self._run(session_id, lambda store: store.get_score())

    def get_bot_progress(self, session_id: str) -> dict:
        return self._run(session_id, lambda store: store.get_bot_progress())


game_store = SessionGameStore(session_ttl_seconds=_load_session_ttl_seconds())
