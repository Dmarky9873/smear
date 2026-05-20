from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import string
from threading import RLock
import time


LOBBY_CODE_ALPHABET = string.ascii_uppercase + string.digits
LOBBY_CODE_LENGTH = 5
DEFAULT_LOBBY_TTL_SECONDS = 12 * 60 * 60


class LobbyNotFoundError(RuntimeError):
    """Raised when a lobby code does not map to an active lobby."""


class LobbyPermissionError(RuntimeError):
    """Raised when a lobby action is attempted by the wrong player."""


@dataclass
class LobbySeat:
    index: int
    player_name: str | None = None
    player_token: str | None = None
    is_host: bool = False

    @property
    def is_occupied(self) -> bool:
        return self.player_name is not None and self.player_token is not None


@dataclass
class Lobby:
    code: str
    session_id: str
    num_players: int
    teams: list[list[int]] | None
    seats: list[LobbySeat]
    host_token: str
    status: str = "waiting"
    revision: int = 1
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def is_full(self) -> bool:
        return all(seat.is_occupied for seat in self.seats)

    @property
    def player_names(self) -> list[str]:
        return [
            seat.player_name if seat.player_name is not None else f"Seat {seat.index + 1}"
            for seat in self.seats
        ]

    @property
    def named_teams(self) -> list[list[str]] | None:
        if self.teams is None:
            return None
        player_names = self.player_names
        return [
            [player_names[seat_index] for seat_index in team]
            for team in self.teams
        ]

    def find_seat_by_token(self, player_token: str) -> LobbySeat | None:
        return next(
            (seat for seat in self.seats if seat.player_token == player_token),
            None,
        )


def _normalize_lobby_code(code: str) -> str:
    normalized = code.strip().upper()
    if not normalized:
        raise ValueError("lobby code is required")
    if len(normalized) > 16:
        raise ValueError("lobby code is too long")
    if any(character not in LOBBY_CODE_ALPHABET for character in normalized):
        raise ValueError("lobby codes can only contain letters and numbers")
    return normalized


def _generate_lobby_code(existing_codes: set[str]) -> str:
    for _ in range(100):
        code = "".join(
            secrets.choice(LOBBY_CODE_ALPHABET)
            for _ in range(LOBBY_CODE_LENGTH)
        )
        if code not in existing_codes:
            return code
    raise RuntimeError("could not generate a unique lobby code")


def _generate_player_token() -> str:
    return secrets.token_urlsafe(24)


def _normalize_player_name(player_name: str) -> str:
    normalized = player_name.strip()
    if not normalized:
        raise ValueError("player name is required")
    if len(normalized) > 32:
        raise ValueError("player names must be 32 characters or fewer")
    return normalized


def _normalize_seat_index(seat_index: int, num_players: int) -> int:
    if seat_index < 0 or seat_index >= num_players:
        raise ValueError("seat index is outside this lobby")
    return seat_index


def _normalize_teams(
    teams: list[list[int]] | None,
    num_players: int,
) -> list[list[int]] | None:
    if teams is None:
        return None

    expected_indexes = set(range(num_players))
    seen_indexes: set[int] = set()
    normalized: list[list[int]] = []

    for team in teams:
        if not team:
            raise ValueError("teams cannot contain empty groups")
        normalized_team: list[int] = []
        for seat_index in team:
            normalized_index = _normalize_seat_index(seat_index, num_players)
            if normalized_index in seen_indexes:
                raise ValueError("each seat can only appear on one team")
            seen_indexes.add(normalized_index)
            normalized_team.append(normalized_index)
        normalized.append(normalized_team)

    if seen_indexes != expected_indexes:
        missing_indexes = sorted(expected_indexes - seen_indexes)
        missing_seats = ", ".join(str(index + 1) for index in missing_indexes)
        raise ValueError(
            f"every seat must appear in exactly one team; missing seats: {missing_seats}"
        )

    return normalized


class LobbyStore:
    def __init__(self, lobby_ttl_seconds: int = DEFAULT_LOBBY_TTL_SECONDS):
        self._lobby_ttl_seconds = lobby_ttl_seconds
        self._lobbies: dict[str, Lobby] = {}
        self._lock = RLock()

    def _prune_expired_lobbies_locked(self) -> None:
        now = time.monotonic()
        expired_codes = [
            code
            for code, lobby in self._lobbies.items()
            if now - lobby.updated_at > self._lobby_ttl_seconds
        ]
        for code in expired_codes:
            self._lobbies.pop(code, None)

    def _touch_locked(self, lobby: Lobby, *, increment_revision: bool = False) -> None:
        lobby.updated_at = time.monotonic()
        if increment_revision:
            lobby.revision += 1

    def create_lobby(
        self,
        *,
        host_name: str,
        num_players: int,
        teams: list[list[int]] | None,
        host_seat_index: int = 0,
    ) -> tuple[Lobby, str]:
        if num_players < 3 or num_players > 8:
            raise ValueError("num_players must be between 3 and 8")

        normalized_host_name = _normalize_player_name(host_name)
        normalized_host_seat_index = _normalize_seat_index(
            host_seat_index,
            num_players,
        )
        normalized_teams = _normalize_teams(teams, num_players)

        with self._lock:
            self._prune_expired_lobbies_locked()
            code = _generate_lobby_code(set(self._lobbies))
            host_token = _generate_player_token()
            seats = [LobbySeat(index=index) for index in range(num_players)]
            seats[normalized_host_seat_index] = LobbySeat(
                index=normalized_host_seat_index,
                player_name=normalized_host_name,
                player_token=host_token,
                is_host=True,
            )
            lobby = Lobby(
                code=code,
                session_id=f"lobby:{code}",
                num_players=num_players,
                teams=normalized_teams,
                seats=seats,
                host_token=host_token,
            )
            self._lobbies[code] = lobby
            return lobby, host_token

    def get_lobby(self, code: str) -> Lobby:
        normalized_code = _normalize_lobby_code(code)
        with self._lock:
            self._prune_expired_lobbies_locked()
            lobby = self._lobbies.get(normalized_code)
            if lobby is None:
                raise LobbyNotFoundError("Lobby not found.")
            self._touch_locked(lobby)
            return lobby

    def join_lobby(
        self,
        *,
        code: str,
        player_name: str,
        seat_index: int | None,
    ) -> tuple[Lobby, str]:
        normalized_name = _normalize_player_name(player_name)
        normalized_code = _normalize_lobby_code(code)

        with self._lock:
            self._prune_expired_lobbies_locked()
            lobby = self._lobbies.get(normalized_code)
            if lobby is None:
                raise LobbyNotFoundError("Lobby not found.")
            if lobby.status != "waiting":
                raise ValueError("This lobby has already started.")

            occupied_names = {
                seat.player_name.lower()
                for seat in lobby.seats
                if seat.player_name is not None
            }
            if normalized_name.lower() in occupied_names:
                raise ValueError("That player name is already in this lobby.")

            if seat_index is None:
                seat = next(
                    (candidate for candidate in lobby.seats if not candidate.is_occupied),
                    None,
                )
                if seat is None:
                    raise ValueError("This lobby is full.")
            else:
                seat = lobby.seats[
                    _normalize_seat_index(seat_index, lobby.num_players)
                ]
                if seat.is_occupied:
                    raise ValueError("That seat is already taken.")

            player_token = _generate_player_token()
            seat.player_name = normalized_name
            seat.player_token = player_token
            self._touch_locked(lobby, increment_revision=True)
            return lobby, player_token

    def require_player(self, code: str, player_token: str | None) -> tuple[Lobby, LobbySeat]:
        if not player_token:
            raise LobbyPermissionError("A player token is required for this lobby.")

        lobby = self.get_lobby(code)
        seat = lobby.find_seat_by_token(player_token)
        if seat is None:
            raise LobbyPermissionError("This player is not part of the lobby.")
        return lobby, seat

    def require_host(self, code: str, player_token: str | None) -> tuple[Lobby, LobbySeat]:
        lobby, seat = self.require_player(code, player_token)
        if player_token != lobby.host_token:
            raise LobbyPermissionError("Only the lobby host can do that.")
        return lobby, seat

    def mark_started(self, code: str) -> Lobby:
        normalized_code = _normalize_lobby_code(code)
        with self._lock:
            lobby = self._lobbies.get(normalized_code)
            if lobby is None:
                raise LobbyNotFoundError("Lobby not found.")
            lobby.status = "active"
            self._touch_locked(lobby, increment_revision=True)
            return lobby

    def touch_lobby(self, code: str) -> Lobby:
        normalized_code = _normalize_lobby_code(code)
        with self._lock:
            lobby = self._lobbies.get(normalized_code)
            if lobby is None:
                raise LobbyNotFoundError("Lobby not found.")
            self._touch_locked(lobby, increment_revision=True)
            return lobby


lobby_store = LobbyStore()
