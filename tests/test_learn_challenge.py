from __future__ import annotations

import unittest

from backend.learn import generate_learn_challenge
from backend.schemas import LearnChallengeResponse


def _action_key(action: dict) -> tuple:
    return (
        action["type"],
        action.get("amount"),
        action.get("card_code"),
    )


class LearnChallengeTests(unittest.TestCase):
    def test_auction_challenge_returns_legal_optimal_option(self):
        challenge = generate_learn_challenge(
            seed=1,
            preferred_phase="auction",
        )

        option_keys = {_action_key(option) for option in challenge["options"]}

        LearnChallengeResponse.model_validate(challenge)
        self.assertEqual(challenge["phase"], "auction")
        self.assertEqual(challenge["actor_name"], "You")
        self.assertGreaterEqual(len(challenge["options"]), 2)
        self.assertIn(_action_key(challenge["best_action"]), option_keys)
        self.assertTrue(challenge["best_action_explanation"])

    def test_play_challenge_returns_legal_optimal_option(self):
        challenge = generate_learn_challenge(
            seed=2,
            preferred_phase="play",
        )

        option_keys = {_action_key(option) for option in challenge["options"]}

        LearnChallengeResponse.model_validate(challenge)
        self.assertEqual(challenge["phase"], "play")
        self.assertEqual(challenge["state"]["round"]["current_player_name"], "You")
        self.assertGreaterEqual(len(challenge["options"]), 2)
        self.assertIn(_action_key(challenge["best_action"]), option_keys)
        self.assertTrue(challenge["best_action_explanation"])

    def test_challenge_can_use_selected_bot(self):
        challenge = generate_learn_challenge(
            seed=3,
            preferred_phase="auction",
            bot_id="greedy",
        )

        LearnChallengeResponse.model_validate(challenge)
        self.assertEqual(challenge["best_bot_id"], "greedy")
        self.assertEqual(challenge["best_bot_label"], "Greedy")

    def test_auction_challenge_rejects_six_bid_without_ace(self):
        for seed in range(40):
            challenge = generate_learn_challenge(
                seed=seed,
                preferred_phase="auction",
            )
            best_action = challenge["best_action"]
            if best_action["type"] != "bid" or best_action.get("amount") != 6:
                continue

            hand = next(
                player["cards"]
                for player in challenge["state"]["round"]["players"]
                if player["name"] == "You"
            )
            self.assertTrue(any(card["rank"] == "A" for card in hand))


if __name__ == "__main__":
    unittest.main()
