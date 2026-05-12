import unittest
from unittest.mock import call
from unittest.mock import patch

from backend.bots.human_information_minimax_n_trick_bot import (
    HumanInformationMinimaxNTrickPlayer,
)
from backend.bots.omniscient_minimax_n_trick_bot import OmniscientMinimaxNTrickPlayer
from backend.gameplay import MatchResult
from backend.simulator import benchmark_models, compare_models_objectively


class SimulatorTeamSizeTests(unittest.TestCase):
    @patch("backend.simulator.time.perf_counter", side_effect=[3.0, 4.0])
    @patch("backend.simulator.Simulator.run_match")
    @patch("backend.simulator.MatchController.create")
    def test_three_player_mode_fills_missing_third_seat_with_random(
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
            rounds_played=1,
            is_draw=False,
            winner_names=["Player 1"],
            final_scores={
                "Player 1": 21,
                "Player 2": 0,
                "Player 3": 0,
            },
        )

        result = benchmark_models(
            1,
            10,
            "greedy",
            "stupid",
            show_progress=False,
            three_player=True,
        )

        self.assertEqual(captured_create_kwargs["num_players"], 3)
        self.assertEqual(
            captured_create_kwargs["player_names"],
            ["Player 1", "Player 2", "Player 3"],
        )
        self.assertTrue(result["three_player"])
        self.assertEqual(result["players_per_game"], 3)
        self.assertIn("random", result["models"])
        self.assertEqual(perf_counter_mock.call_count, 2)

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

    def test_three_player_mode_rejects_more_than_three_models(self):
        with self.assertRaisesRegex(
            ValueError,
            "three-player mode accepts at most three models",
        ):
            benchmark_models(
                1,
                10,
                "random",
                "greedy",
                "stupid",
                "one-trick-minmax",
                show_progress=False,
                three_player=True,
            )

    def test_three_player_mode_requires_single_seat_teams(self):
        with self.assertRaisesRegex(
            ValueError,
            "three-player mode requires team_size 1",
        ):
            benchmark_models(
                1,
                10,
                "random",
                "greedy",
                show_progress=False,
                team_size=2,
                three_player=True,
            )

    @patch("backend.simulator.time.perf_counter", side_effect=[1.0, 2.0])
    @patch("backend.simulator.Simulator.run_match")
    @patch("backend.simulator.MatchController.create")
    def test_depth_overrides_minimax_ready_bot_ids(
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
            rounds_played=1,
            is_draw=False,
            winner_names=["Player 1"],
            final_scores={
                "Player 1": 21,
                "Player 2": 0,
                "Player 3": 0,
            },
        )

        result = benchmark_models(
            1,
            10,
            "one-trick-minmax",
            "o-one-trick-minmax",
            show_progress=False,
            depth=3,
        )

        controllers = captured_create_kwargs["bots"]
        self.assertIsInstance(
            controllers["Player 1"],
            HumanInformationMinimaxNTrickPlayer,
        )
        self.assertEqual(controllers["Player 1"].depth, 3)
        self.assertIsInstance(
            controllers["Player 2"],
            OmniscientMinimaxNTrickPlayer,
        )
        self.assertEqual(controllers["Player 2"].depth, 3)
        self.assertEqual(result["minimax_depth"], 3)
        self.assertIn("3-trick-minmax", result["models"])
        self.assertIn("o-3-trick-minmax", result["models"])
        self.assertEqual(perf_counter_mock.call_count, 2)

    def test_depth_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "depth must be positive"):
            benchmark_models(
                1,
                10,
                "one-trick-minmax",
                "random",
                show_progress=False,
                depth=0,
            )

    @patch("backend.simulator.time.perf_counter", side_effect=[5.0, 6.0])
    @patch("backend.simulator.random.seed")
    @patch("backend.simulator.Simulator.run_match")
    @patch("backend.simulator.MatchController.create")
    def test_fair_mode_rotates_seat_assignments_with_paired_seed(
        self,
        create_mock,
        run_match_mock,
        seed_mock,
        perf_counter_mock,
    ):
        created_bots = []

        def fake_create(**kwargs):
            created_bots.append(kwargs["bots"])
            return object()

        create_mock.side_effect = fake_create
        run_match_mock.return_value = MatchResult(
            rounds_played=2,
            is_draw=True,
            winner_names=[],
            final_scores={
                "Player 1": 5,
                "Player 2": 5,
                "Player 3": 5,
            },
        )

        result = benchmark_models(
            6,
            10,
            "greedy",
            "stupid",
            show_progress=False,
            fair=True,
            seed=123,
        )

        self.assertEqual(result["comparison_mode"], "fair")
        self.assertEqual(result["seed"], 123)
        self.assertEqual(result["fair_schedule"]["assignment_count"], 6)
        self.assertTrue(result["fair_schedule"]["fully_balanced"])
        self.assertEqual(
            [assignment["games_scheduled"] for assignment in result["fair_schedule"]["assignments"]],
            [1, 1, 1, 1, 1, 1],
        )

        signatures = {
            tuple(type(bots[f"Player {index}"]).__name__ for index in range(1, 4))
            for bots in created_bots
        }
        self.assertEqual(len(signatures), 6)
        self.assertEqual(seed_mock.call_args_list, [call(123)] * 6)
        self.assertEqual(perf_counter_mock.call_count, 2)

    @patch("backend.simulator.time.perf_counter", side_effect=[2.0, 3.0])
    @patch("backend.simulator.random.seed")
    @patch("backend.simulator.Simulator.run_match")
    @patch("backend.simulator.MatchController.create")
    def test_fair_mode_defaults_seed_and_distributes_extra_games(
        self,
        create_mock,
        run_match_mock,
        seed_mock,
        perf_counter_mock,
    ):
        create_mock.return_value = object()
        run_match_mock.return_value = MatchResult(
            rounds_played=1,
            is_draw=True,
            winner_names=[],
            final_scores={
                "Player 1": 0,
                "Player 2": 0,
                "Player 3": 0,
            },
        )

        result = compare_models_objectively(
            7,
            10,
            "greedy",
            "stupid",
            show_progress=False,
        )

        self.assertEqual(result["seed"], 0)
        self.assertEqual(
            [assignment["games_scheduled"] for assignment in result["fair_schedule"]["assignments"]],
            [2, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            seed_mock.call_args_list,
            [call(0), call(0), call(0), call(0), call(0), call(0), call(1)],
        )
        self.assertEqual(perf_counter_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
