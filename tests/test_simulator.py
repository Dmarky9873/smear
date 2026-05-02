import unittest
from unittest.mock import patch

from backend.gameplay import MatchResult
from backend.simulator import benchmark_models


class SimulatorTeamSizeTests(unittest.TestCase):
    @patch("backend.simulator.time.perf_counter", side_effect=[10.0, 12.5])
    @patch("backend.simulator.Simulator.run_match")
    @patch("backend.simulator.MatchController.create")
    def test_team_size_builds_same_model_pairs(
        self,
        create_mock,
        run_match_mock,
        perf_counter_mock,
    ):
        captured_create_kwargs = {}

        def fake_create(**kwargs):
            captured_create_kwargs.update(kwargs)
            return object()

        create_mock.side_effect = fake_create
        run_match_mock.return_value = MatchResult(
            rounds_played=3,
            is_draw=False,
            winner_names=["Player 1 / Player 2"],
            final_scores={
                "Player 1 / Player 2": 21,
                "Player 3 / Player 4": 10,
            },
        )

        result = benchmark_models(
            1,
            10,
            "random",
            "greedy",
            show_progress=False,
            team_size=2,
        )

        self.assertEqual(captured_create_kwargs["num_players"], 4)
        self.assertEqual(
            captured_create_kwargs["player_names"],
            ["Player 1", "Player 2", "Player 3", "Player 4"],
        )
        self.assertEqual(
            captured_create_kwargs["teams"],
            [
                ("Player 1", "Player 2"),
                ("Player 3", "Player 4"),
            ],
        )
        self.assertEqual(result["team_size"], 2)
        self.assertEqual(result["players_per_game"], 4)
        self.assertEqual(result["elapsed_seconds"], 2.5)
        self.assertEqual(result["average_seconds_per_game"], 2.5)
        self.assertEqual(result["average_seconds_per_round"], 2.5 / 3)
        self.assertEqual(result["games_per_second"], 0.4)
        self.assertEqual(result["rounds_per_second"], 1.2)
        self.assertEqual(result["models"]["random"]["games_won"], 1.0)
        self.assertEqual(result["models"]["greedy"]["games_won"], 0.0)
        self.assertEqual(
            result["teams"],
            [
                {
                    "team_name": "Player 1 / Player 2",
                    "player_names": ["Player 1", "Player 2"],
                    "model_keys": ["random", "random"],
                    "model_labels": ["Random", "Random"],
                },
                {
                    "team_name": "Player 3 / Player 4",
                    "player_names": ["Player 3", "Player 4"],
                    "model_keys": ["greedy", "greedy"],
                    "model_labels": ["Greedy", "Greedy"],
                },
            ],
        )
        self.assertEqual(perf_counter_mock.call_count, 2)

    def test_team_size_rejects_more_than_eight_total_players(self):
        with self.assertRaisesRegex(ValueError, "exceeds the 8-player limit"):
            benchmark_models(
                1,
                10,
                "random",
                "greedy",
                "stupid",
                "one-trick-minmax",
                "random",
                show_progress=False,
                team_size=2,
            )


if __name__ == "__main__":
    unittest.main()
