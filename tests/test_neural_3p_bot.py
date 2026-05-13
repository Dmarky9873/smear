import json
import math
import tempfile
import unittest
from pathlib import Path

from backend.bots.neural_3p_bot import (
    NeuralThreePlayerBot,
    NeuralThreePlayerV1Bot,
    NeuralThreePlayerV3Bot,
)
from backend.bots.neural_3p_features import (
    encode_auction_candidate,
    encode_auction_state,
    encode_play_candidate,
    encode_play_state,
    ordered_legal_auction_actions,
    ordered_legal_cards,
)
from backend.bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
from backend.bots.neural_model import (
    DEFAULT_TRAINING_BACKEND,
    is_torch_available,
    resolve_training_backend,
)
from backend.bots.registry import build_ready_bot, list_ready_bot_metadata
from backend.models import AuctionState, Card, Play, Player, RoundState, Team, TrickState
from backend.self_play_neural_3p_v3 import (
    SelfPlayDatasetHistory,
    load_persisted_replay_history,
    prune_replay_shards,
    reset_replay_state_after_promotion,
    resolve_self_play_exploration,
    save_replay_shard,
    train_with_self_play,
)
from backend.self_train_neural_3p import (
    _build_incumbent_baseline_cache_from_evaluation,
    candidate_meets_promotion_criteria,
)
from backend.train_neural_3p_bot import (
    ComparisonMetric,
    _render_iteration_comparison_table,
    collect_teacher_training_dataset,
    load_model_bundle,
    parse_teacher_specs,
    save_model_bundle,
    train_neural_3p_bundle,
    train_with_dagger,
)


class NeuralThreePlayerBotTests(unittest.TestCase):
    def _build_supported_round_state(self) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("10H"), Card("KC")})
        player_b = Player("B", {Card("QD")})
        player_c = Player("C", {Card("KD")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        player_b.capture(Play(player_b, Card("J1")))
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(
                leader=player_b,
                plays=[
                    Play(player_b, Card("QD")),
                    Play(player_c, Card("KD")),
                ],
                players=players,
                trump="H",
            ),
            hidden_cards={Card("JC"), Card("10S")},
            trick_history=[],
            teams=teams,
            deck=type("DeckStub", (), {"low": "10"})(),
        )

    def _build_supported_auction_state(self) -> AuctionState:
        return AuctionState(
            dealer_index=2,
            current_bidder_index=0,
            player_names=["A", "B", "C"],
        )

    def test_feature_encoders_return_stable_non_empty_vectors(self):
        round_state = self._build_supported_round_state()
        auction_state = self._build_supported_auction_state()

        play_candidate_features = encode_play_candidate(
            round_state=round_state,
            acting_player_name="A",
            candidate_card=Card("AH"),
            match_scores={"A": 4, "B": 2, "C": 1},
            target_score=21,
            auction_state=auction_state,
        )
        play_state_features = encode_play_state(
            round_state=round_state,
            perspective_player_name="A",
            match_scores={"A": 4, "B": 2, "C": 1},
            target_score=21,
            auction_state=auction_state,
        )
        auction_candidate_features = encode_auction_candidate(
            auction_state=auction_state,
            acting_player_name="A",
            hand={
                Card("AH"),
                Card("10H"),
                Card("KS"),
                Card("QD"),
                Card("JC"),
                Card("AS"),
            },
            candidate_action=ordered_legal_auction_actions(auction_state)[0],
            match_scores={"A": 4, "B": 2, "C": 1},
            target_score=21,
        )
        auction_state_features = encode_auction_state(
            auction_state=auction_state,
            perspective_player_name="A",
            hand={
                Card("AH"),
                Card("10H"),
                Card("KS"),
                Card("QD"),
                Card("JC"),
                Card("AS"),
            },
            match_scores={"A": 4, "B": 2, "C": 1},
            target_score=21,
        )

        self.assertGreater(len(play_candidate_features), 32)
        self.assertGreater(len(play_state_features), 32)
        self.assertGreater(len(auction_candidate_features), 32)
        self.assertGreater(len(auction_state_features), 32)
        self.assertEqual(
            len(play_candidate_features),
            len(
                encode_play_candidate(
                    round_state=round_state,
                    acting_player_name="A",
                    candidate_card=Card("10H"),
                    match_scores={"A": 4, "B": 2, "C": 1},
                    target_score=21,
                    auction_state=auction_state,
                )
            ),
        )
        self.assertEqual(
            len(play_state_features),
            len(
                encode_play_state(
                    round_state=round_state,
                    perspective_player_name="A",
                    match_scores={"A": 0, "B": 0, "C": 0},
                    target_score=21,
                    auction_state=auction_state,
                )
            ),
        )
        self.assertEqual(
            len(auction_candidate_features),
            len(
                encode_auction_candidate(
                    auction_state=auction_state,
                    acting_player_name="A",
                    hand={
                        Card("AH"),
                        Card("10H"),
                        Card("KS"),
                        Card("QD"),
                        Card("JC"),
                        Card("AS"),
                    },
                    candidate_action=ordered_legal_auction_actions(auction_state)[-1],
                    match_scores={"A": 4, "B": 2, "C": 1},
                    target_score=21,
                )
            ),
        )
        self.assertEqual(
            len(auction_state_features),
            len(
                encode_auction_state(
                    auction_state=auction_state,
                    perspective_player_name="A",
                    hand={
                        Card("AH"),
                        Card("10H"),
                        Card("KS"),
                        Card("QD"),
                        Card("JC"),
                        Card("AS"),
                    },
                    match_scores={"A": 0, "B": 0, "C": 0},
                    target_score=21,
                )
            ),
        )

    def test_scalar_mlp_can_fit_tiny_choice_dataset(self):
        model = ScalarMLP.initialize(input_dim=2, hidden_dim=6, seed=0)
        examples = [
            ChoiceExample(
                candidate_features=[[1.0, 0.0], [0.0, 1.0]],
                chosen_index=0,
            ),
            ChoiceExample(
                candidate_features=[[0.0, 1.0], [1.0, 0.0]],
                chosen_index=1,
            ),
        ]

        history = model.train_choice_examples(
            examples,
            epochs=80,
            learning_rate=0.1,
            seed=0,
            backend="python",
        )

        self.assertGreater(history[-1]["accuracy"], 0.99)
        self.assertEqual(model.predict_index(examples[0].candidate_features), 0)
        self.assertEqual(model.predict_index(examples[1].candidate_features), 1)

    def test_scalar_mlp_can_fit_tiny_soft_target_dataset(self):
        model = ScalarMLP.initialize(input_dim=2, hidden_dim=6, seed=0)
        examples = [
            ChoiceExample(
                candidate_features=[[1.0, 0.0], [0.0, 1.0]],
                chosen_index=0,
                target_distribution=[0.85, 0.15],
            ),
            ChoiceExample(
                candidate_features=[[0.0, 1.0], [1.0, 0.0]],
                chosen_index=1,
                target_distribution=[0.10, 0.90],
            ),
        ]

        history = model.train_choice_examples(
            examples,
            epochs=80,
            learning_rate=0.1,
            seed=0,
            backend="python",
        )

        self.assertGreater(history[-1]["accuracy"], 0.99)
        scores = model.score_many(examples[0].candidate_features)
        max_score = max(scores)
        probabilities = [math.exp(score - max_score) for score in scores]
        total = sum(probabilities)
        normalized = [value / total for value in probabilities]
        self.assertGreater(normalized[0], normalized[1])

    def test_training_backend_resolution_matches_available_runtime(self):
        expected_backend = "torch" if is_torch_available() else "python"
        self.assertEqual(
            resolve_training_backend(DEFAULT_TRAINING_BACKEND),
            expected_backend,
        )
        self.assertEqual(resolve_training_backend("python"), "python")

    def test_scalar_mlp_explicit_torch_backend_requires_dependency(self):
        if is_torch_available():
            self.skipTest("torch is installed in this environment")

        model = ScalarMLP.initialize(input_dim=2, hidden_dim=6, seed=0)
        examples = [
            ChoiceExample(
                candidate_features=[[1.0, 0.0], [0.0, 1.0]],
                chosen_index=0,
            ),
        ]

        with self.assertRaises(ImportError):
            model.train_choice_examples(
                examples,
                epochs=1,
                learning_rate=0.1,
                backend="torch",
            )

    def test_scalar_mlp_can_fit_tiny_choice_dataset_with_torch_backend(self):
        if not is_torch_available():
            self.skipTest("torch is not installed in this environment")

        model = ScalarMLP.initialize(input_dim=2, hidden_dim=6, seed=0)
        examples = [
            ChoiceExample(
                candidate_features=[[1.0, 0.0], [0.0, 1.0]],
                chosen_index=0,
            ),
            ChoiceExample(
                candidate_features=[[0.0, 1.0], [1.0, 0.0]],
                chosen_index=1,
            ),
        ]

        history = model.train_choice_examples(
            examples,
            epochs=80,
            learning_rate=0.1,
            seed=0,
            backend="torch",
            batch_size=2,
        )

        self.assertGreater(history[-1]["accuracy"], 0.99)

    def test_scalar_mlp_can_fit_tiny_regression_dataset(self):
        model = ScalarMLP.initialize(input_dim=2, hidden_dim=6, seed=0)
        examples = [
            RegressionExample(features=[1.0, 0.0], target=1.0),
            RegressionExample(features=[0.0, 1.0], target=-1.0),
        ]

        history = model.train_regression_examples(
            examples,
            epochs=120,
            learning_rate=0.08,
            seed=0,
            backend="python",
        )

        self.assertLess(history[-1]["mse"], 0.02)
        self.assertGreater(model.score([1.0, 0.0]), 0.7)
        self.assertLess(model.score([0.0, 1.0]), -0.7)

    def test_ready_bot_registry_builds_neural_bots(self):
        legacy_bot = build_ready_bot("neural-3p-v1", "A")
        upgraded_bot = build_ready_bot("neural-3p-v2", "A")
        self_play_bot = build_ready_bot("neural-3p-v3", "A")

        self.assertIsInstance(legacy_bot, NeuralThreePlayerV1Bot)
        self.assertIsInstance(upgraded_bot, NeuralThreePlayerBot)
        self.assertIsInstance(self_play_bot, NeuralThreePlayerBot)
        bot_ids = {bot["id"] for bot in list_ready_bot_metadata()}
        self.assertIn("neural-3p-v1", bot_ids)
        self.assertIn("neural-3p-v2", bot_ids)
        self.assertIn("neural-3p-v3", bot_ids)

    def test_neural_bot_chooses_legal_card_in_supported_context(self):
        round_state = self._build_supported_round_state()
        bot = build_ready_bot("neural-3p-v2", "A")
        bot._cards = set(round_state.current_player.cards)
        bot.set_match_context(
            player_names=["A", "B", "C"],
            teams=[("A",), ("B",), ("C",)],
            match_scores={"A": 0, "B": 0, "C": 0},
            target_score=21,
            auction_state=self._build_supported_auction_state(),
            round_state=round_state,
        )

        chosen_card = bot.choose_card(round_state)

        self.assertIn(chosen_card, ordered_legal_cards(round_state))

    def test_neural_bot_falls_back_gracefully_in_four_player_auction(self):
        auction_state = AuctionState(
            dealer_index=3,
            current_bidder_index=0,
            player_names=["A", "B", "C", "D"],
        )
        bot = build_ready_bot("neural-3p-v2", "A")
        bot._cards = {
            Card("AH"),
            Card("KH"),
            Card("QH"),
            Card("AD"),
            Card("KD"),
            Card("QD"),
        }

        action = bot.choose_auction_action(auction_state)

        self.assertIn(action, ordered_legal_auction_actions(auction_state))

    def test_training_helpers_collect_and_save_v2_model_bundle(self):
        teacher_specs = parse_teacher_specs("greedy")
        dataset, collection_report = collect_teacher_training_dataset(
            teacher_specs=teacher_specs,
            match_count=1,
            alpha=1,
            seed=0,
        )
        self.assertGreater(collection_report["play_policy_examples"], 0)
        self.assertGreater(collection_report["auction_policy_examples"], 0)
        self.assertGreater(collection_report["play_value_examples"], 0)
        self.assertGreater(collection_report["auction_value_examples"], 0)

        bundle, report = train_neural_3p_bundle(
            play_examples=dataset.play_policy_examples,
            auction_examples=dataset.auction_policy_examples,
            play_value_examples=dataset.play_value_examples,
            auction_value_examples=dataset.auction_value_examples,
            play_hidden_dim=8,
            auction_hidden_dim=8,
            play_value_hidden_dim=8,
            auction_value_hidden_dim=8,
            play_epochs=2,
            auction_epochs=2,
            play_value_epochs=2,
            auction_value_epochs=2,
            play_learning_rate=0.05,
            auction_learning_rate=0.05,
            play_value_learning_rate=0.05,
            auction_value_learning_rate=0.05,
            l2=0.0,
            seed=0,
            teacher_specs=teacher_specs,
            bot_id="neural-3p-v2",
        )

        self.assertEqual(bundle["version"], 2)
        self.assertIn("play_history", report)
        self.assertIn("auction_history", report)
        self.assertIn("play_value_history", report)
        self.assertIn("auction_value_history", report)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.json"
            save_model_bundle(bundle, output_path)
            trained_bot = NeuralThreePlayerBot("A", model_path=output_path)
            round_state = self._build_supported_round_state()
            trained_bot._cards = set(round_state.current_player.cards)
            trained_bot.set_match_context(
                player_names=["A", "B", "C"],
                teams=[("A",), ("B",), ("C",)],
                match_scores={"A": 0, "B": 0, "C": 0},
                target_score=21,
                auction_state=self._build_supported_auction_state(),
                round_state=round_state,
            )

            chosen_card = trained_bot.choose_card(round_state)

        self.assertIn(chosen_card, ordered_legal_cards(round_state))

    def test_search_teacher_collection_emits_soft_policy_targets(self):
        teacher_specs = parse_teacher_specs("1-trick-minmax")
        dataset, _collection_report = collect_teacher_training_dataset(
            teacher_specs=teacher_specs,
            match_count=1,
            alpha=1,
            seed=0,
            teacher_target_temperature=0.35,
        )

        play_soft_examples = [
            example
            for example in dataset.play_policy_examples
            if example.target_distribution is not None
        ]
        auction_soft_examples = [
            example
            for example in dataset.auction_policy_examples
            if example.target_distribution is not None
        ]

        self.assertGreater(len(play_soft_examples), 0)
        self.assertGreater(len(auction_soft_examples), 0)
        self.assertAlmostEqual(sum(play_soft_examples[0].target_distribution), 1.0)
        self.assertAlmostEqual(sum(auction_soft_examples[0].target_distribution), 1.0)

    def test_iteration_comparison_table_lists_changed_and_same_metrics(self):
        table = _render_iteration_comparison_table(
            prefix="[test]",
            title="comparison",
            current_snapshot={
                "play_accuracy": 0.82,
                "decision": "promoted",
            },
            previous_snapshot={
                "play_accuracy": 0.80,
                "decision": "promoted",
            },
            metrics=(
                ComparisonMetric(
                    key="play_accuracy",
                    label="Play accuracy",
                    kind="float",
                    digits=3,
                    higher_is_better=True,
                    tolerance=1e-6,
                ),
                ComparisonMetric(
                    key="decision",
                    label="Decision",
                    kind="text",
                ),
            ),
        )

        self.assertIn("[test] comparison", table)
        self.assertIn("metric", table)
        self.assertIn("Play accuracy", table)
        self.assertIn("Decision", table)
        self.assertIn("summary  changed 1 | same 1 | new 0", table)
        self.assertIn("same     Decision", table)

    def test_train_with_dagger_runs_small_end_to_end_cycle(self):
        bundle, report = train_with_dagger(
            teacher_specs=parse_teacher_specs("greedy"),
            bootstrap_matches=2,
            dagger_matches=2,
            alpha=1,
            dagger_iterations=1,
            seed=0,
            play_hidden_dim=8,
            auction_hidden_dim=8,
            play_value_hidden_dim=8,
            auction_value_hidden_dim=8,
            play_epochs=1,
            auction_epochs=1,
            play_value_epochs=1,
            auction_value_epochs=1,
            play_learning_rate=0.03,
            auction_learning_rate=0.03,
            play_value_learning_rate=0.03,
            auction_value_learning_rate=0.03,
            l2=0.0,
            workers=2,
            bot_id="neural-3p-v2",
        )

        self.assertEqual(bundle["version"], 2)
        self.assertEqual(len(report["dagger_iterations"]), 1)
        self.assertGreater(report["bootstrap"]["play_policy_examples"], 0)
        self.assertEqual(report["bootstrap"]["workers"], 2)

    def test_train_with_self_play_runs_small_end_to_end_cycle(self):
        bundle, report = train_with_self_play(
            initial_bundle=load_model_bundle(NeuralThreePlayerBot.MODEL_FILE_V2),
            self_play_matches=2,
            alpha=1,
            self_play_iterations=1,
            replay_window=1,
            seed=0,
            play_hidden_dim=8,
            auction_hidden_dim=8,
            play_value_hidden_dim=8,
            auction_value_hidden_dim=8,
            play_epochs=1,
            auction_epochs=1,
            play_value_epochs=1,
            auction_value_epochs=1,
            play_learning_rate=0.02,
            auction_learning_rate=0.02,
            play_value_learning_rate=0.02,
            auction_value_learning_rate=0.02,
            l2=0.0,
            play_value_weight=0.5,
            auction_value_weight=0.3,
            play_rollout_depth=1,
            auction_rollout_depth=1,
            play_temperature=0.8,
            auction_temperature=0.7,
            epsilon=0.05,
            temperature_decay=1.0,
            epsilon_decay=1.0,
            policy_advantage_threshold=-0.25,
            policy_advantage_scale=2.0,
            value_weight_scale=1.0,
            winner_policy_weight=0.5,
            gradient_clip=1.0,
            workers=2,
        )

        self.assertEqual(bundle["version"], 3)
        self.assertEqual(bundle["bot_id"], "neural-3p-v3")
        self.assertEqual(bundle["training_mode"], "self_play")
        self.assertEqual(len(report["self_play_iterations"]), 1)
        collection_report = report["self_play_iterations"][0]["collection"]
        training_report = report["self_play_iterations"][0]["training"]
        self.assertGreater(collection_report["play_policy_examples"], 0)
        self.assertGreater(collection_report["auction_policy_examples"], 0)
        self.assertEqual(collection_report["workers"], 2)
        self.assertEqual(
            training_report["trainer_backend"],
            resolve_training_backend(DEFAULT_TRAINING_BACKEND),
        )

    def test_train_with_self_play_persists_replay_history_across_calls(self):
        replay_history = SelfPlayDatasetHistory()
        initial_bundle = load_model_bundle(NeuralThreePlayerBot.MODEL_FILE_V2)
        common_kwargs = {
            "self_play_matches": 1,
            "alpha": 1,
            "self_play_iterations": 1,
            "replay_window": 2,
            "play_hidden_dim": 8,
            "auction_hidden_dim": 8,
            "play_value_hidden_dim": 8,
            "auction_value_hidden_dim": 8,
            "play_epochs": 1,
            "auction_epochs": 1,
            "play_value_epochs": 1,
            "auction_value_epochs": 1,
            "play_learning_rate": 0.02,
            "auction_learning_rate": 0.02,
            "play_value_learning_rate": 0.02,
            "auction_value_learning_rate": 0.02,
            "l2": 0.0,
            "play_value_weight": 0.5,
            "auction_value_weight": 0.3,
            "play_rollout_depth": 1,
            "auction_rollout_depth": 1,
            "play_temperature": 0.8,
            "auction_temperature": 0.7,
            "epsilon": 0.05,
            "temperature_decay": 1.0,
            "epsilon_decay": 1.0,
            "policy_advantage_threshold": -0.25,
            "policy_advantage_scale": 2.0,
            "value_weight_scale": 1.0,
            "winner_policy_weight": 0.5,
            "gradient_clip": 1.0,
            "workers": 1,
            "replay_history": replay_history,
        }

        _, first_report = train_with_self_play(
            initial_bundle=initial_bundle,
            seed=0,
            **common_kwargs,
        )
        _, second_report = train_with_self_play(
            initial_bundle=initial_bundle,
            seed=1,
            **common_kwargs,
        )

        first_iteration = first_report["self_play_iterations"][0]
        second_iteration = second_report["self_play_iterations"][0]

        first_collection_play = first_iteration["collection"]["play_policy_examples"]
        second_collection_play = second_iteration["collection"]["play_policy_examples"]
        second_aggregate_play = second_iteration["training"]["play_examples"]

        self.assertEqual(first_iteration["replay_history_iterations"], 1)
        self.assertEqual(second_iteration["replay_history_iterations"], 2)
        self.assertEqual(len(replay_history.iterations), 2)
        self.assertGreater(first_collection_play, 0)
        self.assertGreater(second_collection_play, 0)
        self.assertEqual(
            second_aggregate_play,
            first_collection_play + second_collection_play,
        )

    def test_self_play_replay_shards_persist_and_reload_recent_window(self):
        replay_history = SelfPlayDatasetHistory()
        initial_bundle = load_model_bundle(NeuralThreePlayerBot.MODEL_FILE_V2)
        common_kwargs = {
            "initial_bundle": initial_bundle,
            "self_play_matches": 1,
            "alpha": 1,
            "self_play_iterations": 1,
            "replay_window": 2,
            "play_hidden_dim": 8,
            "auction_hidden_dim": 8,
            "play_value_hidden_dim": 8,
            "auction_value_hidden_dim": 8,
            "play_epochs": 1,
            "auction_epochs": 1,
            "play_value_epochs": 1,
            "auction_value_epochs": 1,
            "play_learning_rate": 0.02,
            "auction_learning_rate": 0.02,
            "play_value_learning_rate": 0.02,
            "auction_value_learning_rate": 0.02,
            "l2": 0.0,
            "play_value_weight": 0.5,
            "auction_value_weight": 0.3,
            "play_rollout_depth": 1,
            "auction_rollout_depth": 1,
            "play_temperature": 0.8,
            "auction_temperature": 0.7,
            "epsilon": 0.05,
            "temperature_decay": 1.0,
            "epsilon_decay": 1.0,
            "policy_advantage_threshold": -0.25,
            "policy_advantage_scale": 2.0,
            "value_weight_scale": 1.0,
            "winner_policy_weight": 0.5,
            "gradient_clip": 1.0,
            "workers": 1,
            "replay_history": replay_history,
            "verbose": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_store_dir = Path(tmpdir) / "replay-store"

            train_with_self_play(seed=0, **common_kwargs)
            first_dataset = replay_history.iterations[-1]
            save_replay_shard(
                replay_store_dir=replay_store_dir,
                dataset=first_dataset,
                run_id="run-a",
                iteration=1,
                source_bot_id="neural-3p-v3",
            )

            train_with_self_play(seed=1, **common_kwargs)
            second_dataset = replay_history.iterations[-1]
            save_replay_shard(
                replay_store_dir=replay_store_dir,
                dataset=second_dataset,
                run_id="run-b",
                iteration=1,
                source_bot_id="neural-3p-v3",
            )

            train_with_self_play(seed=2, **common_kwargs)
            third_dataset = replay_history.iterations[-1]
            save_replay_shard(
                replay_store_dir=replay_store_dir,
                dataset=third_dataset,
                run_id="run-c",
                iteration=1,
                source_bot_id="neural-3p-v3",
            )

            loaded_history = load_persisted_replay_history(
                replay_store_dir=replay_store_dir,
                replay_window=2,
                persisted_replay_limit=3,
                verbose=False,
            )

            self.assertEqual(len(loaded_history.iterations), 2)
            self.assertEqual(
                len(loaded_history.aggregate().play_policy_examples),
                len(second_dataset.play_policy_examples) + len(third_dataset.play_policy_examples),
            )

            prune_replay_shards(
                replay_store_dir=replay_store_dir,
                replay_window=2,
                persisted_replay_limit=2,
                verbose=False,
            )
            self.assertEqual(len(list(replay_store_dir.glob("*.json"))), 2)

    def test_replay_pruning_ignores_unrelated_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_store_dir = Path(tmpdir) / "replay-store"
            replay_store_dir.mkdir(parents=True, exist_ok=True)
            unrelated_path = replay_store_dir / "aaa-config.json"
            unrelated_path.write_text(json.dumps({"keep": True}), encoding="utf-8")

            for iteration in range(1, 3):
                save_replay_shard(
                    replay_store_dir=replay_store_dir,
                    dataset=SelfPlayDatasetHistory().aggregate(),
                    run_id="run-a",
                    iteration=iteration,
                    source_bot_id="neural-3p-v3",
                )

            prune_replay_shards(
                replay_store_dir=replay_store_dir,
                replay_window=1,
                persisted_replay_limit=1,
                verbose=False,
            )

            self.assertTrue(unrelated_path.exists())
            self.assertEqual(
                sorted(path.name for path in replay_store_dir.glob("*.json")),
                ["aaa-config.json", "run-a-iteration-002.json"],
            )

    def test_reset_replay_state_after_promotion_clears_managed_shards_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_store_dir = Path(tmpdir) / "replay-store"
            replay_store_dir.mkdir(parents=True, exist_ok=True)
            unrelated_path = replay_store_dir / "notes.json"
            unrelated_path.write_text(json.dumps({"keep": True}), encoding="utf-8")

            for iteration in range(1, 3):
                save_replay_shard(
                    replay_store_dir=replay_store_dir,
                    dataset=SelfPlayDatasetHistory().aggregate(),
                    run_id="run-a",
                    iteration=iteration,
                    source_bot_id="neural-3p-v3",
                )

            replay_history = reset_replay_state_after_promotion(
                replay_store_dir=replay_store_dir,
                verbose=False,
            )

            self.assertEqual(len(replay_history.iterations), 0)
            self.assertTrue(unrelated_path.exists())
            self.assertEqual(
                sorted(path.name for path in replay_store_dir.glob("*.json")),
                ["notes.json"],
            )

    def test_load_persisted_replay_history_skips_invalid_shards(self):
        dataset = SelfPlayDatasetHistory().aggregate()
        dataset.play_policy_examples.append(
            ChoiceExample(
                candidate_features=[[0.1, 0.2], [0.3, 0.4]],
                chosen_index=1,
                weight=1.0,
                target_distribution=[0.25, 0.75],
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_store_dir = Path(tmpdir) / "replay-store"
            save_replay_shard(
                replay_store_dir=replay_store_dir,
                dataset=dataset,
                run_id="run-a",
                iteration=1,
                source_bot_id="neural-3p-v3",
            )
            (replay_store_dir / "run-b-iteration-002.json").write_text(
                '{"schema_version": 1,',
                encoding="utf-8",
            )

            loaded_history = load_persisted_replay_history(
                replay_store_dir=replay_store_dir,
                replay_window=4,
                persisted_replay_limit=4,
                verbose=False,
            )

            self.assertEqual(len(loaded_history.iterations), 1)
            self.assertEqual(
                len(loaded_history.aggregate().play_policy_examples),
                len(dataset.play_policy_examples),
            )

    def test_named_bot_versions_default_to_expected_model_files(self):
        v1_bot = NeuralThreePlayerV1Bot("A")
        v3_bot = NeuralThreePlayerV3Bot("B")

        self.assertEqual(v1_bot._model_path.name, NeuralThreePlayerBot.MODEL_FILE_V1.name)
        self.assertEqual(v3_bot._model_path.name, NeuralThreePlayerBot.MODEL_FILE_V3.name)

    def test_resolve_self_play_exploration_anneals_and_respects_floors(self):
        self.assertEqual(
            resolve_self_play_exploration(
                play_temperature=0.9,
                auction_temperature=0.75,
                epsilon=0.08,
                temperature_decay=0.96,
                epsilon_decay=0.94,
                iteration_index=0,
            ),
            (0.9, 0.75, 0.08),
        )

        play_temp, auction_temp, epsilon = resolve_self_play_exploration(
            play_temperature=0.9,
            auction_temperature=0.75,
            epsilon=0.08,
            temperature_decay=0.5,
            epsilon_decay=0.5,
            iteration_index=12,
        )

        self.assertEqual(play_temp, 0.2)
        self.assertEqual(auction_temp, 0.15)
        self.assertEqual(epsilon, 0.01)

    def test_promoted_candidate_baseline_cache_renames_candidate_to_incumbent(self):
        candidate_vs_greedy = {
            "models": {
                "candidate": {"label": "candidate", "win_percentage": 22.2},
                "greedy": {"label": "Greedy", "win_percentage": 33.3},
                "random": {"label": "Random", "win_percentage": 0.0},
            }
        }
        candidate_vs_optimal = {
            "models": {
                "candidate": {"label": "candidate", "win_percentage": 5.6},
                "optimal-bot": {"label": "Optimal Bot", "win_percentage": 44.4},
                "random": {"label": "Random", "win_percentage": 0.0},
            }
        }
        evaluation_report = {
            "vs_greedy": candidate_vs_greedy,
            "vs_optimal_bot": candidate_vs_optimal,
            "incumbent_vs_greedy": None,
            "incumbent_vs_optimal": None,
        }

        cache = _build_incumbent_baseline_cache_from_evaluation(
            evaluation_report,
            promoted_candidate=True,
        )

        self.assertEqual(
            cache["vs_greedy"]["models"]["incumbent"]["win_percentage"],
            22.2,
        )
        self.assertEqual(
            cache["vs_optimal"]["models"]["incumbent"]["win_percentage"],
            5.6,
        )
        self.assertNotIn("candidate", cache["vs_greedy"]["models"])
        self.assertNotIn("candidate", cache["vs_optimal"]["models"])
        self.assertIn("candidate", candidate_vs_greedy["models"])
        self.assertIn("candidate", candidate_vs_optimal["models"])

    def test_candidate_promotion_requires_no_baseline_regression(self):
        evaluation_report = {
            "vs_incumbent": {
                "models": {
                    "candidate": {"win_percentage": 35.0},
                    "incumbent": {"win_percentage": 20.0},
                }
            },
            "vs_greedy": {
                "models": {
                    "candidate": {"win_percentage": 29.0},
                }
            },
            "vs_optimal_bot": {
                "models": {
                    "candidate": {"win_percentage": 8.0},
                }
            },
            "incumbent_vs_greedy": {
                "models": {
                    "incumbent": {"win_percentage": 30.0},
                }
            },
            "incumbent_vs_optimal": {
                "models": {
                    "incumbent": {"win_percentage": 8.0},
                }
            },
        }

        self.assertFalse(candidate_meets_promotion_criteria(evaluation_report))

    def test_candidate_promotion_accepts_clear_head_to_head_upgrade(self):
        evaluation_report = {
            "vs_incumbent": {
                "models": {
                    "candidate": {"win_percentage": 35.0},
                    "incumbent": {"win_percentage": 20.0},
                }
            },
            "vs_greedy": {
                "models": {
                    "candidate": {"win_percentage": 30.0},
                }
            },
            "vs_optimal_bot": {
                "models": {
                    "candidate": {"win_percentage": 8.0},
                }
            },
            "incumbent_vs_greedy": {
                "models": {
                    "incumbent": {"win_percentage": 30.0},
                }
            },
            "incumbent_vs_optimal": {
                "models": {
                    "incumbent": {"win_percentage": 8.0},
                }
            },
        }

        self.assertTrue(candidate_meets_promotion_criteria(evaluation_report))


if __name__ == "__main__":
    unittest.main()
