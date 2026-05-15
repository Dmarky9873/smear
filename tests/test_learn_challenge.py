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


if __name__ == "__main__":
    unittest.main()
