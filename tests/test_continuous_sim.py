import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.continuous_sim import (
    ContinuousSimResult,
    EloEntry,
    _compact_bot_name,
    _build_initial_ratings,
    _build_parallel_executor,
    compute_multiplayer_elo_deltas,
    iter_match_tasks,
    load_persisted_rating_state,
    load_persisted_ratings,
    save_persisted_ratings,
)


class ContinuousSimHelperTests(unittest.TestCase):
    def test_continuous_sim_result_exposes_games_per_hour(self):
        result = ContinuousSimResult(
            games_completed=120,
            started_at=100.0,
            ended_at=3700.0,
            ratings={},
        )

        self.assertAlmostEqual(result.elapsed_seconds, 3600.0)
        self.assertAlmostEqual(result.games_per_hour, 120.0)

    def test_compact_bot_name_shows_neural_versions_explicitly(self):
        self.assertEqual(_compact_bot_name("neural-3p-v1"), "n3v1")
        self.assertEqual(_compact_bot_name("neural-3p-v2"), "n3v2")
        self.assertEqual(_compact_bot_name("neural-3p-v3"), "n3v3")

    def test_iter_match_tasks_samples_unique_three_player_matchups(self):
        bot_ids = [f"bot-{index}" for index in range(10)]

        task = next(
            iter_match_tasks(
                bot_ids,
                alpha=50,
                seed=0,
            )
        )

        self.assertEqual(len(task.participants), 3)
        rated_bot_ids = [participant.bot_id for participant in task.participants]
        self.assertEqual(len(rated_bot_ids), 3)
        self.assertEqual(len(set(rated_bot_ids)), 3)
        self.assertEqual(
            sorted(rated_bot_ids),
            sorted(participant.seat_key for participant in task.participants),
        )
        self.assertTrue(set(rated_bot_ids).issubset(set(bot_ids)))

    def test_balanced_schedule_covers_every_trio_and_seat_order_before_repeating(self):
        bot_ids = ["alpha", "beta", "gamma", "delta"]
        task_iter = iter_match_tasks(
            bot_ids,
            alpha=50,
            seed=0,
            schedule_mode="balanced",
        )

        first_cycle = [next(task_iter) for _ in range(24)]
        seen_seat_orders = {
            tuple(participant.bot_id for participant in task.participants)
            for task in first_cycle
        }
        trio_counts: dict[tuple[str, ...], int] = {}
        for task in first_cycle:
            trio_key = tuple(
                sorted(participant.bot_id for participant in task.participants)
            )
            trio_counts[trio_key] = trio_counts.get(trio_key, 0) + 1

        self.assertEqual(len(seen_seat_orders), 24)
        self.assertEqual(len(trio_counts), 4)
        self.assertTrue(all(count == 6 for count in trio_counts.values()))

    def test_iter_match_tasks_uses_random_filler_with_two_bots(self):
        task = next(
            iter_match_tasks(
                ["greedy", "stupid"],
                alpha=50,
                seed=0,
            )
        )

        self.assertEqual(len(task.participants), 3)
        rated_participants = [
            participant for participant in task.participants if participant.is_rated
        ]
        filler_participants = [
            participant for participant in task.participants if not participant.is_rated
        ]
        self.assertEqual(sorted(participant.bot_id for participant in rated_participants), ["greedy", "stupid"])
        self.assertEqual(len(filler_participants), 1)
        self.assertEqual(filler_participants[0].bot_id, "random")

    def test_balanced_schedule_rotates_two_bot_filler_seating(self):
        task_iter = iter_match_tasks(
            ["greedy", "stupid"],
            alpha=50,
            seed=0,
            schedule_mode="balanced",
        )

        first_cycle = [next(task_iter) for _ in range(6)]
        seat_orders = {
            tuple(participant.bot_id for participant in task.participants)
            for task in first_cycle
        }

        self.assertEqual(len(seat_orders), 6)

    def test_multiplayer_elo_averages_pairwise_results(self):
        ratings = {
            "alpha": EloEntry(rating=1500.0),
            "beta": EloEntry(rating=1500.0),
            "gamma": EloEntry(rating=1500.0),
        }
        deltas = compute_multiplayer_elo_deltas(
            ratings,
            {"alpha": 21, "beta": 10, "gamma": 0},
            k_factor=32.0,
        )

        self.assertAlmostEqual(deltas["alpha"], 16.0)
        self.assertAlmostEqual(deltas["beta"], 0.0)
        self.assertAlmostEqual(deltas["gamma"], -16.0)

    def test_persisted_elo_round_trips_through_json_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_file = Path(temp_dir) / "elo.json"
            original_ratings = {
                "alpha": EloEntry(
                    rating=1512.5,
                    games_played=4,
                    wins=2,
                    draws=1,
                    losses=1,
                )
            }

            save_persisted_ratings(
                elo_file,
                original_ratings,
                bot_fingerprints={"alpha": "alpha:v1"},
            )
            loaded_ratings = load_persisted_ratings(elo_file)
            loaded_rating_state, loaded_fingerprints = load_persisted_rating_state(
                elo_file
            )

        self.assertEqual(set(loaded_ratings), {"alpha"})
        self.assertEqual(set(loaded_rating_state), {"alpha"})
        self.assertAlmostEqual(loaded_ratings["alpha"].rating, 1512.5)
        self.assertEqual(loaded_ratings["alpha"].games_played, 4)
        self.assertEqual(loaded_ratings["alpha"].wins, 2)
        self.assertEqual(loaded_ratings["alpha"].draws, 1)
        self.assertEqual(loaded_ratings["alpha"].losses, 1)
        self.assertEqual(loaded_ratings["alpha"].pairwise_games, 0)
        self.assertEqual(loaded_ratings["alpha"].information, 0.0)
        self.assertEqual(loaded_fingerprints, {"alpha": "alpha:v1"})

    def test_build_initial_ratings_resets_changed_bot_fingerprint(self):
        persisted_ratings = {
            "optimal-bot": EloEntry(
                rating=1725.0,
                games_played=40,
                wins=24,
                losses=16,
            )
        }
        persisted_fingerprints = {
            "optimal-bot": "optimal-bot",
        }

        ratings = _build_initial_ratings(
            ["optimal-bot"],
            initial_rating=1500.0,
            persisted_ratings=persisted_ratings,
            persisted_fingerprints=persisted_fingerprints,
        )

        self.assertEqual(ratings["optimal-bot"].rating, 1500.0)
        self.assertEqual(ratings["optimal-bot"].games_played, 0)
        self.assertEqual(
            persisted_fingerprints["optimal-bot"],
            "optimal-bot:v2",
        )

    @patch("backend.continuous_sim.ProcessPoolExecutor", side_effect=PermissionError)
    def test_parallel_executor_falls_back_to_threads_when_processes_unavailable(
        self,
        _process_pool_mock,
    ):
        executor, executor_kind = _build_parallel_executor(2)
        try:
            self.assertEqual(executor_kind, "thread")
        finally:
            executor.shutdown(wait=True)


class ContinuousSimEntryPointTests(unittest.TestCase):
    def test_console_entrypoint_runs_single_serial_three_player_game(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_file = Path(temp_dir) / "elo.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "continuous-sim",
                    "--games",
                    "1",
                    "--workers",
                    "1",
                    "--alpha",
                    "1",
                    "--seed",
                    "0",
                    "--elo-file",
                    str(elo_file),
                    "--bots",
                    "greedy",
                    "stupid",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("Starting continuous simulation", completed.stdout)
        self.assertIn("Continuous sim |", completed.stdout)
        self.assertIn("Elo", completed.stdout)
        self.assertIn("Matches", completed.stdout)
        self.assertIn("Elo file:", completed.stdout)
        self.assertIn("Elo startup: load from JSON if present", completed.stdout)
        self.assertIn("schedule=balanced", completed.stdout)
        self.assertIn("Schedule | balanced", completed.stdout)
        self.assertIn("games/hour", completed.stdout)
        self.assertIn("random filler", completed.stdout)
        self.assertIn("Final leaderboard after 1 game", completed.stdout)

    def test_console_entrypoint_reuses_saved_elo_file_on_second_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_file = Path(temp_dir) / "elo.json"
            command = [
                sys.executable,
                "continuous-sim",
                "--games",
                "1",
                "--workers",
                "1",
                "--alpha",
                "1",
                "--seed",
                "0",
                "--elo-file",
                str(elo_file),
                "--bots",
                "greedy",
                "stupid",
            ]

            first_run = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                first_run.returncode,
                0,
                msg=f"stdout:\n{first_run.stdout}\nstderr:\n{first_run.stderr}",
            )

            first_payload = json.loads(elo_file.read_text(encoding="utf-8"))
            self.assertEqual(
                first_payload["ratings"]["greedy"]["games_played"],
                1,
            )
            self.assertEqual(
                first_payload["ratings"]["stupid"]["games_played"],
                1,
            )
            self.assertNotIn("random", first_payload["ratings"])

            second_run = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                second_run.returncode,
                0,
                msg=f"stdout:\n{second_run.stdout}\nstderr:\n{second_run.stderr}",
            )

            second_payload = json.loads(elo_file.read_text(encoding="utf-8"))
            self.assertEqual(
                second_payload["ratings"]["greedy"]["games_played"],
                2,
            )
            self.assertEqual(
                second_payload["ratings"]["stupid"]["games_played"],
                2,
            )

    def test_console_entrypoint_fresh_ratings_ignores_saved_elo_on_startup(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_file = Path(temp_dir) / "elo.json"
            seed_payload = {
                "version": 1,
                "ratings": {
                    "greedy": {
                        "rating": 1750.0,
                        "games_played": 10,
                        "wins": 8,
                        "draws": 0,
                        "losses": 2,
                    },
                    "stupid": {
                        "rating": 1250.0,
                        "games_played": 10,
                        "wins": 2,
                        "draws": 0,
                        "losses": 8,
                    },
                },
            }
            elo_file.write_text(json.dumps(seed_payload), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "continuous-sim",
                    "--games",
                    "1",
                    "--workers",
                    "1",
                    "--alpha",
                    "1",
                    "--seed",
                    "0",
                    "--elo-file",
                    str(elo_file),
                    "--fresh-ratings",
                    "--bots",
                    "greedy",
                    "stupid",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn(
                "Elo startup: fresh from --initial-rating",
                completed.stdout,
            )

            payload = json.loads(elo_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["ratings"]["greedy"]["games_played"], 1)
            self.assertEqual(payload["ratings"]["stupid"]["games_played"], 1)
            self.assertNotEqual(payload["ratings"]["greedy"]["rating"], 1750.0)


if __name__ == "__main__":
    unittest.main()
