import tempfile
import unittest
from pathlib import Path

from backend.bots.neural_3p_bot import NeuralThreePlayerBot, NeuralThreePlayerV1Bot
from backend.bots.neural_3p_features import (
    encode_auction_candidate,
    encode_auction_state,
    encode_play_candidate,
    encode_play_state,
    ordered_legal_auction_actions,
    ordered_legal_cards,
)
from backend.bots.neural_model import ChoiceExample, RegressionExample, ScalarMLP
from backend.bots.registry import build_ready_bot, list_ready_bot_metadata
from backend.models import AuctionState, Card, Play, Player, RoundState, Team, TrickState
from backend.train_neural_3p_bot import (
    collect_teacher_training_dataset,
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
        )

        self.assertGreater(history[-1]["accuracy"], 0.99)
        self.assertEqual(model.predict_index(examples[0].candidate_features), 0)
        self.assertEqual(model.predict_index(examples[1].candidate_features), 1)

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
        )

        self.assertLess(history[-1]["mse"], 0.02)
        self.assertGreater(model.score([1.0, 0.0]), 0.7)
        self.assertLess(model.score([0.0, 1.0]), -0.7)

    def test_ready_bot_registry_builds_neural_bots(self):
        legacy_bot = build_ready_bot("neural-3p-v1", "A")
        upgraded_bot = build_ready_bot("neural-3p-v2", "A")

        self.assertIsInstance(legacy_bot, NeuralThreePlayerV1Bot)
        self.assertIsInstance(upgraded_bot, NeuralThreePlayerBot)
        bot_ids = {bot["id"] for bot in list_ready_bot_metadata()}
        self.assertIn("neural-3p-v1", bot_ids)
        self.assertIn("neural-3p-v2", bot_ids)

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

    def test_train_with_dagger_runs_small_end_to_end_cycle(self):
        bundle, report = train_with_dagger(
            teacher_specs=parse_teacher_specs("greedy"),
            bootstrap_matches=1,
            dagger_matches=1,
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
            bot_id="neural-3p-v2",
        )

        self.assertEqual(bundle["version"], 2)
        self.assertEqual(len(report["dagger_iterations"]), 1)
        self.assertGreater(report["bootstrap"]["play_policy_examples"], 0)


if __name__ == "__main__":
    unittest.main()
