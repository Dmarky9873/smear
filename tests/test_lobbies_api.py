import tempfile
import unittest

from fastapi import HTTPException

import backend.api as api_module
from backend.lobbies import LobbyStore
from backend.persistence import SQLiteSessionRepository
from backend.realtime import LobbyEventHub
from backend.schemas import (
    AddLobbyBotRequest,
    CreateLobbyRequest,
    JoinLobbyRequest,
    LobbyBidRequest,
    StartLobbyRequest,
)
from backend.store import SessionGameStore


class LobbyApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_game_store = api_module.game_store
        self.original_lobby_store = api_module.lobby_store
        self.original_lobby_events = api_module.lobby_events
        api_module.game_store = SessionGameStore(
            session_ttl_seconds=3600,
            repository=SQLiteSessionRepository(f"{self.temp_dir.name}/sessions.sqlite3"),
        )
        api_module.lobby_store = LobbyStore()
        api_module.lobby_events = LobbyEventHub()

    async def asyncTearDown(self):
        api_module.game_store = self.original_game_store
        api_module.lobby_store = self.original_lobby_store
        api_module.lobby_events = self.original_lobby_events
        self.temp_dir.cleanup()

    async def _started_three_player_lobby(self):
        host_lobby = await api_module.create_lobby(
            CreateLobbyRequest(
                host_name="A",
                num_players=3,
                teams=None,
            )
        )
        code = host_lobby["code"]
        host_token = host_lobby["you"]["player_token"]
        player_b = await api_module.join_lobby(
            code,
            JoinLobbyRequest(player_name="B", seat_index=1),
        )
        player_c = await api_module.join_lobby(
            code,
            JoinLobbyRequest(player_name="C", seat_index=2),
        )

        started = await api_module.start_lobby(
            code,
            StartLobbyRequest(player_token=host_token),
        )
        return started, host_token, player_b["you"]["player_token"], player_c

    async def test_lobby_start_returns_player_scoped_game_state(self):
        started, host_token, player_b_token, _player_c = (
            await self._started_three_player_lobby()
        )

        self.assertEqual(started["status"], "active")
        self.assertEqual(started["game_state"]["phase"], "auction")
        self.assertGreater(len(started["legal_actions"]), 0)

        host_players = started["game_state"]["round"]["players"]
        self.assertEqual(len(host_players[0]["cards"]), 6)
        self.assertEqual(len(host_players[1]["cards"]), 0)
        self.assertEqual(host_players[1]["card_count"], 6)
        self.assertEqual(started["game_state"]["round"]["hidden_cards"], [])

        lobby = api_module.lobby_store.get_lobby(started["code"])
        player_b_view = api_module._serialize_lobby(lobby, player_b_token)
        player_b_players = player_b_view["game_state"]["round"]["players"]

        self.assertEqual(len(player_b_players[0]["cards"]), 0)
        self.assertEqual(len(player_b_players[1]["cards"]), 6)
        self.assertEqual(player_b_view["legal_actions"], [])

    async def test_lobby_actions_require_current_player(self):
        started, _host_token, player_b_token, _player_c = (
            await self._started_three_player_lobby()
        )

        with self.assertRaises(HTTPException) as exc:
            await api_module.place_lobby_bid(
                started["code"],
                LobbyBidRequest(player_token=player_b_token, amount=1),
            )

        self.assertEqual(exc.exception.status_code, 403)

    async def test_host_can_fill_lobby_seat_with_bot(self):
        host_lobby = await api_module.create_lobby(
            CreateLobbyRequest(
                host_name="A",
                num_players=3,
                teams=None,
            )
        )
        code = host_lobby["code"]
        host_token = host_lobby["you"]["player_token"]

        player_c = await api_module.join_lobby(
            code,
            JoinLobbyRequest(player_name="C", seat_index=2),
        )
        with_bot = await api_module.add_lobby_bot(
            code,
            AddLobbyBotRequest(
                player_token=host_token,
                seat_index=1,
                bot_id="stupid",
                player_name="Bot",
            ),
        )

        self.assertTrue(with_bot["is_full"])
        self.assertTrue(with_bot["seats"][1]["is_bot"])
        self.assertEqual(with_bot["seats"][1]["bot_id"], "stupid")

        started = await api_module.start_lobby(
            code,
            StartLobbyRequest(player_token=host_token),
        )
        self.assertEqual(started["status"], "active")
        self.assertEqual(
            started["game_state"]["round"]["players"][1]["bot_id"],
            "stupid",
        )

        after_host_bid = await api_module.place_lobby_bid(
            code,
            LobbyBidRequest(player_token=host_token, amount=1),
        )

        self.assertEqual(
            after_host_bid["game_state"]["auction"]["current_bidder_name"],
            player_c["you"]["player_name"],
        )
        self.assertEqual(after_host_bid["legal_actions"], [])


if __name__ == "__main__":
    unittest.main()
