from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

try:
    from .engine import Auction, Game, get_trick_winner, score_round_details
    from .gameplay import GameSession, MAX_BID, MatchController
    from .models import (
        AuctionEvent,
        AuctionState,
        Card,
        Deck,
        Play,
        Player,
        RoundState,
        Team,
        TrickState,
    )
except ImportError:
    from engine import Auction, Game, get_trick_winner, score_round_details
    from gameplay import GameSession, MAX_BID, MatchController
    from models import (
        AuctionEvent,
        AuctionState,
        Card,
        Deck,
        Play,
        Player,
        RoundState,
        Team,
        TrickState,
    )


STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistedSession:
    controller: MatchController
    revision: int


class SQLiteSessionRepository:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        if self._initialized:
            return

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS game_sessions (
                    session_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
        self._initialized = True

    def load(self, session_id: str) -> PersistedSession | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT revision, snapshot_json
                FROM game_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        snapshot = json.loads(row["snapshot_json"])
        return PersistedSession(
            controller=restore_match_controller(snapshot),
            revision=int(row["revision"]),
        )

    def save(
        self,
        session_id: str,
        controller: MatchController,
        revision: int,
    ) -> None:
        self._ensure_schema()
        snapshot_json = json.dumps(
            dump_match_controller(controller),
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO game_sessions (session_id, revision, snapshot_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    revision = excluded.revision,
                    snapshot_json = excluded.snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, revision, snapshot_json, time.time()),
            )


def _card_codes(cards: set[Card] | list[Card]) -> list[str]:
    return sorted(card.code for card in cards)


def _dump_play(play: Play) -> dict:
    return {
        "player_name": play.player.name,
        "card_code": play.card.code,
    }


def _dump_trick(trick: TrickState) -> dict:
    return {
        "leader_name": trick.leader.name,
        "plays": [_dump_play(play) for play in trick.plays],
        "trump": trick.trump,
    }


def _dump_auction_event(event: AuctionEvent) -> dict:
    return {
        "bidder_name": event.bidder_name,
        "action": event.action,
        "amount": event.amount,
    }


def _dump_score_value(value: Any) -> Any:
    if isinstance(value, Card):
        return {"__type": "card", "code": value.code}
    if isinstance(value, set):
        return {
            "__type": "set",
            "items": [_dump_score_value(item) for item in sorted(value, key=repr)],
        }
    if isinstance(value, tuple):
        return [_dump_score_value(item) for item in value]
    if isinstance(value, list):
        return [_dump_score_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump_score_value(item) for key, item in value.items()}
    return value


def _restore_score_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_score_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    value_type = value.get("__type")
    if value_type == "card":
        return Card(value["code"])
    if value_type == "set":
        return {_restore_score_value(item) for item in value["items"]}
    return {key: _restore_score_value(item) for key, item in value.items()}


def dump_match_controller(controller: MatchController) -> dict:
    session = controller.session
    round_state = session.game.round_state
    auction_state = session.auction.state
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "player_names": session.player_names,
        "teams": [list(team) for team in session.teams],
        "dealer_index": session.dealer_index,
        "player_bot_ids": session.player_bot_ids,
        "match_scores": session.match_scores,
        "target_score": session.target_score,
        "round_number": session.round_number,
        "last_scored_round_number": session.last_scored_round_number,
        "last_round_score": _dump_score_value(session.last_round_score),
        "auction": {
            "dealer_index": auction_state.dealer_index,
            "current_bidder_index": auction_state.current_bidder_index,
            "player_names": auction_state.player_names,
            "highest_bid": auction_state.highest_bid,
            "highest_bidder_name": auction_state.highest_bidder_name,
            "passed_player_names": sorted(auction_state.passed_player_names),
            "bid_history": [
                _dump_auction_event(event) for event in auction_state.bid_history
            ],
            "is_complete": auction_state.is_complete,
        },
        "round": {
            "low": session.game.low,
            "players": [
                {
                    "name": player.name,
                    "cards": _card_codes(player.cards),
                }
                for player in round_state.players
            ],
            "current_player_name": round_state.current_player.name,
            "trump": round_state.trump,
            "hidden_cards": _card_codes(round_state.hidden_cards),
            "current_trick": _dump_trick(round_state.current_trick),
            "trick_history": [
                _dump_trick(trick) for trick in round_state.trick_history
            ],
        },
    }


def _restore_play(payload: dict, player_by_name: dict[str, Player]) -> Play:
    return Play(player_by_name[payload["player_name"]], Card(payload["card_code"]))


def _restore_trick(
    payload: dict,
    player_by_name: dict[str, Player],
    players: list[Player],
) -> TrickState:
    return TrickState(
        leader=player_by_name[payload["leader_name"]],
        plays=[
            _restore_play(play_payload, player_by_name)
            for play_payload in payload["plays"]
        ],
        players=players,
        trump=payload["trump"],
    )


def _restore_auction(payload: dict) -> Auction:
    auction_state = AuctionState(
        dealer_index=payload["dealer_index"],
        current_bidder_index=payload["current_bidder_index"],
        player_names=list(payload["player_names"]),
        highest_bid=payload["highest_bid"],
        highest_bidder_name=payload["highest_bidder_name"],
        passed_player_names=set(payload["passed_player_names"]),
        bid_history=[
            AuctionEvent(
                bidder_name=event["bidder_name"],
                action=event["action"],
                amount=event["amount"],
            )
            for event in payload["bid_history"]
        ],
        is_complete=payload["is_complete"],
    )
    return Auction.from_state(auction_state, max_bid=MAX_BID)


def restore_match_controller(snapshot: dict) -> MatchController:
    schema_version = snapshot.get("schema_version")
    if schema_version != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported game session schema: {schema_version}")

    player_names = list(snapshot["player_names"])
    teams = [tuple(team) for team in snapshot["teams"]]
    round_payload = snapshot["round"]
    players = [
        Player(
            player_payload["name"],
            {Card(card_code) for card_code in player_payload["cards"]},
        )
        for player_payload in round_payload["players"]
    ]
    player_by_name = {player.name: player for player in players}
    team_states = [
        Team([player_by_name[player_name] for player_name in team], set())
        for team in teams
    ]
    current_trick = _restore_trick(
        round_payload["current_trick"],
        player_by_name,
        players,
    )
    trick_history = [
        _restore_trick(trick_payload, player_by_name, players)
        for trick_payload in round_payload["trick_history"]
    ]
    round_state = RoundState(
        players=players,
        current_player=player_by_name[round_payload["current_player_name"]],
        trump=round_payload["trump"],
        current_trick=current_trick,
        hidden_cards={Card(card_code) for card_code in round_payload["hidden_cards"]},
        trick_history=trick_history,
        teams=team_states,
        deck=Deck(round_payload["low"]),
    )

    for trick in trick_history:
        if trick.is_terminal:
            winner = get_trick_winner(trick)
            for play in trick.plays:
                winner.capture(play)

    game = object.__new__(Game)
    game._low, game._num_hiding, game._num_dealt = Game._calculate_low(
        game,
        len(player_names),
    )
    game._low = round_payload["low"]
    game._round_state = round_state

    session = GameSession(
        game=game,
        player_names=player_names,
        teams=teams,
        dealer_index=snapshot["dealer_index"],
        auction=_restore_auction(snapshot["auction"]),
        player_bot_ids=dict(snapshot["player_bot_ids"]),
        match_scores=dict(snapshot["match_scores"]),
        bots={},
        target_score=snapshot["target_score"],
        round_number=snapshot["round_number"],
        last_round_score=_restore_score_value(snapshot["last_round_score"]),
        last_scored_round_number=snapshot["last_scored_round_number"],
    )
    session.bots = MatchController._build_ready_bot_controllers(
        session.player_bot_ids,
    )
    controller = MatchController(session)
    controller._sync_all_bot_hands()

    if (
        session.last_round_score is None
        and session.last_scored_round_number == session.round_number
        and session.game.round_state.is_terminal
    ):
        session.last_round_score = controller._apply_bid_penalty(
            score_round_details(session.game.round_state),
        )

    return controller
