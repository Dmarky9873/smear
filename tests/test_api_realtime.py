import tempfile
import unittest

import backend.api as api_module
from backend.persistence import SQLiteSessionRepository
from backend.realtime import GameEventHub
from backend.serializers import serialize_game
from backend.store import SessionGameStore


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.messages: list[dict] = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload: dict):
        self.messages.append(payload)


class ApiRealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_game_state_reaches_session_socket(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_store = api_module.game_store
            original_events = api_module.game_events
            api_module.game_store = SessionGameStore(
                session_ttl_seconds=3600,
                repository=SQLiteSessionRepository(f"{temp_dir}/sessions.sqlite3"),
            )
            api_module.game_events = GameEventHub()
            websocket = FakeWebSocket()

            try:
                await api_module.game_events.connect("socket-test", websocket)
                session = api_module.game_store.create_game(
                    "socket-test",
                    num_players=3,
                    player_names=["A", "B", "C"],
                    teams=None,
                    player_bots=[None, None, None],
                    auto_run_bots=False,
                )

                await api_module._broadcast_game_state(
                    "socket-test",
                    serialize_game(session),
                )

                self.assertTrue(websocket.accepted)
                self.assertEqual(len(websocket.messages), 1)
                self.assertEqual(websocket.messages[0]["type"], "game_state")
                self.assertEqual(websocket.messages[0]["revision"], 1)
                self.assertEqual(websocket.messages[0]["state"]["phase"], "auction")
                self.assertEqual(
                    websocket.messages[0]["state"]["auction"]["current_bidder_name"],
                    "A",
                )
            finally:
                api_module.game_store = original_store
                api_module.game_events = original_events


if __name__ == "__main__":
    unittest.main()
