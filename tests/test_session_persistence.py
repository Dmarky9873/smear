import unittest

from backend.persistence import SQLiteSessionRepository
from backend.store import SessionGameStore


class SessionPersistenceTests(unittest.TestCase):
    def test_game_state_survives_store_recreation(self):
        with self.subTest("persist and reload from sqlite"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = f"{temp_dir}/sessions.sqlite3"
                first_store = SessionGameStore(
                    session_ttl_seconds=3600,
                    repository=SQLiteSessionRepository(db_path),
                )
                first_store.create_game(
                    "session-a",
                    num_players=3,
                    player_names=["A", "B", "C"],
                    teams=None,
                    player_bots=[None, None, None],
                    auto_run_bots=False,
                )
                first_store.place_bid(
                    "session-a",
                    amount=1,
                    auto_run_bots=False,
                )
                first_revision = first_store.get_revision("session-a")

                second_store = SessionGameStore(
                    session_ttl_seconds=3600,
                    repository=SQLiteSessionRepository(db_path),
                )
                restored = second_store.get_state("session-a")

                self.assertEqual(second_store.get_revision("session-a"), first_revision)
                self.assertEqual(restored.auction.state.highest_bid, 1)
                self.assertEqual(restored.auction.state.highest_bidder_name, "A")
                self.assertEqual(restored.auction.current_bidder_name, "B")
                self.assertEqual(restored.player_names, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
