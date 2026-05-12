import unittest
from unittest.mock import patch

from backend.bots.human_information_minimax_n_trick_bot import (
    HumanInformationMinimaxNTrickPlayer,
)
from backend.bots.human_information_minimax_one_trick_bot import (
    HumanInformationMinimaxOneTrickPlayer,
)
from backend.bots.minimax_n_trick_bot import MinimaxNTrickPlayer
from backend.bots.o_minimax_n_trick_bot import OMNISCIENT_MinimaxNTrickPlayer
from backend.bots.omniscient_minimax_n_trick_bot import (
    OmniscientMinimaxNTrickPlayer,
)
from backend.bots.omniscient_minimax_one_trick_bot import (
    OmniscientMinimaxOneTrickPlayer,
)
from backend.bots.registry import build_ready_bot, list_ready_bot_metadata
from backend.bots.search_eval import (
    _should_rollout_cutoff_exactly,
    apply_bid_and_match_rules,
    ensure_captured_plays_synchronized,
    rollout_round_to_utility,
    score_unit_name,
)
from backend.engine import (
    apply_trick_action_for_search,
    get_legal_actions,
    score_round_details,
    undo_trick_action_for_search,
)
from backend.models import AuctionState, Card, Deck, Play, Player, RoundState, Team, TrickState


class ExactOmniscientMinimaxNTrickPlayer(OmniscientMinimaxNTrickPlayer):
    def _search_exact(
        self,
        round_state: RoundState,
        remaining_tricks: int,
        team_member_names: set[str],
    ) -> int:
        if remaining_tricks <= 0 or round_state.is_terminal:
            return 0

        legal_actions = list(get_legal_actions(round_state))
        if not legal_actions:
            return 0

        maximizing = round_state.current_player.name in team_member_names
        trick_count_before = len(round_state.trick_history)
        if maximizing:
            value = float("-inf")
            for card in legal_actions:
                play = Play(round_state.current_player, card)
                undo = apply_trick_action_for_search(round_state, play)
                try:
                    trick_completed = len(round_state.trick_history) > trick_count_before
                    immediate_value = (
                        self._evaluate_trick(
                            round_state.trick_history[-1],
                            team_member_names,
                        )
                        if trick_completed
                        else 0
                    )
                    value = max(
                        value,
                        immediate_value
                        + self._search_exact(
                            round_state,
                            remaining_tricks - 1 if trick_completed else remaining_tricks,
                            team_member_names,
                        ),
                    )
                finally:
                    undo_trick_action_for_search(round_state, undo)
            return int(value)

        value = float("inf")
        for card in legal_actions:
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                immediate_value = (
                    self._evaluate_trick(
                        round_state.trick_history[-1],
                        team_member_names,
                    )
                    if trick_completed
                    else 0
                )
                value = min(
                    value,
                    immediate_value
                    + self._search_exact(
                        round_state,
                        remaining_tricks - 1 if trick_completed else remaining_tricks,
                        team_member_names,
                    ),
                )
            finally:
                undo_trick_action_for_search(round_state, undo)
        return int(value)

    def choose_card_exact(self, round_state: RoundState) -> Card:
        team_member_names = self._team_member_names(round_state)
        best_card = None
        best_score = float("-inf")

        for card in list(get_legal_actions(round_state)):
            trick_count_before = len(round_state.trick_history)
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                immediate_value = (
                    self._evaluate_trick(
                        round_state.trick_history[-1],
                        team_member_names,
                    )
                    if trick_completed
                    else 0
                )
                score = immediate_value + self._search_exact(
                    round_state,
                    self.depth - 1 if trick_completed else self.depth,
                    team_member_names,
                )
            finally:
                undo_trick_action_for_search(round_state, undo)

            if score > best_score:
                best_score = score
                best_card = card

        if best_card is None:
            raise ValueError("exact minimax bot could not find a legal card")
        return best_card

    def root_scores_exact(self, round_state: RoundState) -> dict[str, int]:
        team_member_names = self._team_member_names(round_state)
        scores: dict[str, int] = {}

        for card in list(get_legal_actions(round_state)):
            trick_count_before = len(round_state.trick_history)
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                immediate_value = (
                    self._evaluate_trick(
                        round_state.trick_history[-1],
                        team_member_names,
                    )
                    if trick_completed
                    else 0
                )
                scores[card.code] = immediate_value + self._search_exact(
                    round_state,
                    self.depth - 1 if trick_completed else self.depth,
                    team_member_names,
                )
            finally:
                undo_trick_action_for_search(round_state, undo)

        return scores

    def _search_exact_utility(
        self,
        round_state: RoundState,
        remaining_tricks: int,
        team_member_names: set[str],
    ) -> float:
        if round_state.is_terminal or remaining_tricks <= 0:
            return self._leaf_state_utility(round_state, None)

        legal_actions = list(get_legal_actions(round_state))
        if not legal_actions:
            return self._leaf_state_utility(round_state, None)

        maximizing = round_state.current_player.name in team_member_names
        trick_count_before = len(round_state.trick_history)
        value = float("-inf") if maximizing else float("inf")

        for card in legal_actions:
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                child_value = self._search_exact_utility(
                    round_state,
                    remaining_tricks - 1 if trick_completed else remaining_tricks,
                    team_member_names,
                )
                if maximizing:
                    value = max(value, child_value)
                else:
                    value = min(value, child_value)
            finally:
                undo_trick_action_for_search(round_state, undo)

        return value

    def root_scores_exact_utility(self, round_state: RoundState) -> dict[str, float]:
        team_member_names = self._team_member_names(round_state)
        scores: dict[str, float] = {}

        for card in list(get_legal_actions(round_state)):
            trick_count_before = len(round_state.trick_history)
            play = Play(round_state.current_player, card)
            undo = apply_trick_action_for_search(round_state, play)
            try:
                trick_completed = len(round_state.trick_history) > trick_count_before
                scores[card.code] = self._search_exact_utility(
                    round_state,
                    self.depth - 1 if trick_completed else self.depth,
                    team_member_names,
                )
            finally:
                undo_trick_action_for_search(round_state, undo)

        return scores


class MinimaxNTrickBotTests(unittest.TestCase):
    def _build_single_trick_state(self) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("10C")})
        player_b = Player("B", {Card("KC")})
        player_c = Player("C", {Card("2C")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=set(),
            trick_history=[],
            teams=teams,
            deck=Deck("10"),
        )

    def _build_hidden_information_state(
        self,
        *,
        player_b_cards: set[Card],
        player_c_cards: set[Card],
    ) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("10C")})
        player_b = Player("B", set(player_b_cards))
        player_c = Player("C", set(player_c_cards))
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        deck = Deck("10")
        hidden_cards = (
            set(deck.get_copy())
            - player_a.cards
            - player_b.cards
            - player_c.cards
        )
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=hidden_cards,
            trick_history=[],
            teams=teams,
            deck=deck,
        )

    def _build_two_trick_state(self) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("10C")})
        player_b = Player("B", {Card("KH"), Card("QC")})
        player_c = Player("C", {Card("2H"), Card("AD")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        deck = Deck("10")
        hidden_cards = (
            set(deck.get_copy())
            - player_a.cards
            - player_b.cards
            - player_c.cards
        )
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=hidden_cards,
            trick_history=[],
            teams=teams,
            deck=deck,
        )

    def _build_ordering_state(self) -> RoundState:
        player_a = Player("A", {Card("AH"), Card("AD")})
        player_b = Player("B", {Card("KC")})
        player_c = Player("C", {Card("2S")})
        players = [player_b, player_a, player_c]
        teams = [Team([player], set()) for player in players]
        deck = Deck("10")
        hidden_cards = (
            set(deck.get_copy())
            - player_a.cards
            - player_b.cards
            - player_c.cards
        )
        player_b.play_card(Card("KC"))
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(
                player_b,
                [Play(player_b, Card("KC"))],
                players,
                "H",
            ),
            hidden_cards=hidden_cards,
            trick_history=[],
            teams=teams,
            deck=deck,
        )

    def _build_transposition_regression_state(self) -> RoundState:
        player_a = Player("A", {Card("10C"), Card("AD"), Card("JC"), Card("JD"), Card("KC")})
        player_b = Player("B", {Card("10H"), Card("J1"), Card("JH"), Card("KH"), Card("KS")})
        player_c = Player("C", {Card("AH"), Card("AS"), Card("KD"), Card("QC"), Card("QS")})
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        completed_trick = TrickState(
            player_a,
            [
                Play(player_a, Card("10D")),
                Play(player_b, Card("QD")),
                Play(player_c, Card("JD")),
            ],
            players,
            "D",
        )
        deck = Deck("10")
        hidden_cards = (
            set(deck.get_copy())
            - player_a.cards
            - player_b.cards
            - player_c.cards
            - {play.card for play in completed_trick.plays}
        )
        return RoundState(
            players=players,
            current_player=player_b,
            trump="D",
            current_trick=TrickState(player_b, [], players, "D"),
            hidden_cards=hidden_cards,
            trick_history=[completed_trick],
            teams=teams,
            deck=deck,
        )

    def _build_terminal_scoring_state(self) -> RoundState:
        player_a = Player("A", set())
        player_b = Player("B", set())
        player_c = Player("C", set())
        players = [player_a, player_b, player_c]
        teams = [Team([player], set()) for player in players]
        trick_history = [
            TrickState(
                player_a,
                [
                    Play(player_a, Card("JH")),
                    Play(player_b, Card("QS")),
                    Play(player_c, Card("JC")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_a,
                [
                    Play(player_a, Card("AH")),
                    Play(player_b, Card("KD")),
                    Play(player_c, Card("J1")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_a,
                [
                    Play(player_a, Card("KH")),
                    Play(player_b, Card("AD")),
                    Play(player_c, Card("J2")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_a,
                [
                    Play(player_a, Card("QH")),
                    Play(player_b, Card("KS")),
                    Play(player_c, Card("QC")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_a,
                [
                    Play(player_a, Card("AS")),
                    Play(player_b, Card("AC")),
                    Play(player_c, Card("JS")),
                ],
                players,
                "H",
            ),
            TrickState(
                player_b,
                [
                    Play(player_b, Card("JD")),
                    Play(player_c, Card("QD")),
                    Play(player_a, Card("KC")),
                ],
                players,
                "H",
            ),
        ]
        deck = Deck("J")
        hidden_cards = set(deck.get_copy()) - {
            play.card
            for trick in trick_history
            for play in trick.plays
        }
        return RoundState(
            players=players,
            current_player=player_a,
            trump="H",
            current_trick=TrickState(player_a, [], players, "H"),
            hidden_cards=hidden_cards,
            trick_history=trick_history,
            teams=teams,
            deck=deck,
        )

    def test_omniscient_depth_one_matches_one_trick_bot(self):
        round_state = self._build_single_trick_state()
        depth_one_bot = OmniscientMinimaxNTrickPlayer("A", depth=1)
        baseline_bot = OmniscientMinimaxOneTrickPlayer("A")

        self.assertEqual(
            depth_one_bot.choose_card(round_state),
            baseline_bot.choose_card(round_state),
        )

    def test_human_information_depth_one_matches_one_trick_bot(self):
        round_state = self._build_hidden_information_state(
            player_b_cards={Card("KC")},
            player_c_cards={Card("QD")},
        )
        depth_one_bot = HumanInformationMinimaxNTrickPlayer(
            "A",
            cards=set(round_state.current_player.cards),
            depth=1,
        )
        baseline_bot = HumanInformationMinimaxOneTrickPlayer(
            "A",
            cards=set(round_state.current_player.cards),
        )

        self.assertEqual(
            depth_one_bot.choose_card(round_state),
            baseline_bot.choose_card(round_state),
        )

    def test_human_information_depth_one_ignores_actual_opponent_hands(self):
        public_state_a = self._build_hidden_information_state(
            player_b_cards={Card("KC"), Card("QS")},
            player_c_cards={Card("QD"), Card("KD")},
        )
        public_state_b = self._build_hidden_information_state(
            player_b_cards={Card("JD"), Card("AS")},
            player_c_cards={Card("10D"), Card("AC")},
        )

        bot = HumanInformationMinimaxNTrickPlayer(
            "A",
            cards=set(public_state_a.current_player.cards),
            depth=1,
        )
        choice_a = bot.choose_card(public_state_a)
        choice_b = bot.choose_card(public_state_b)

        self.assertIn(choice_a, get_legal_actions(public_state_a))
        self.assertEqual(choice_a, choice_b)

    def test_n_trick_bots_can_search_past_the_current_trick(self):
        round_state = self._build_two_trick_state()

        omniscient_bot = OmniscientMinimaxNTrickPlayer("A", depth=2)
        human_bot = HumanInformationMinimaxNTrickPlayer(
            "A",
            cards=set(round_state.current_player.cards),
            depth=2,
        )

        omniscient_choice = omniscient_bot.choose_card(round_state)
        human_choice = human_bot.choose_card(round_state)

        legal_actions = get_legal_actions(round_state)
        self.assertIn(omniscient_choice, legal_actions)
        self.assertIn(human_choice, legal_actions)

    def test_move_ordering_prefers_immediate_trick_winners(self):
        round_state = self._build_ordering_state()
        bot = OmniscientMinimaxNTrickPlayer("A", depth=2)

        ordered_actions = bot._ordered_legal_actions(round_state)

        self.assertEqual(ordered_actions[0], Card("AH"))

    def test_leaf_state_utility_enables_hybrid_cutoff_for_deeper_search(self):
        round_state = self._build_two_trick_state()
        bot = OmniscientMinimaxNTrickPlayer("A", depth=3)

        with patch(
            "backend.bots.omniscient_minimax_n_trick_bot.rollout_round_to_utility",
            return_value=7.0,
        ) as rollout_mock:
            value = bot._leaf_state_utility(round_state, None)

        self.assertEqual(value, 7.0)
        rollout_mock.assert_called_once()
        self.assertEqual(rollout_mock.call_args.kwargs["hybrid_cutoff"], True)

    def test_leaf_state_utility_keeps_exact_rollout_for_shallower_search(self):
        round_state = self._build_two_trick_state()
        bot = OmniscientMinimaxNTrickPlayer("A", depth=2)

        with patch(
            "backend.bots.omniscient_minimax_n_trick_bot.rollout_round_to_utility",
            return_value=5.0,
        ) as rollout_mock:
            value = bot._leaf_state_utility(round_state, None)

        self.assertEqual(value, 5.0)
        rollout_mock.assert_called_once()
        self.assertEqual(rollout_mock.call_args.kwargs["hybrid_cutoff"], False)

    def test_hybrid_cutoff_uses_partial_evaluator_when_trump_is_set(self):
        round_state = self._build_two_trick_state()

        with (
            patch(
                "backend.bots.search_eval.evaluate_terminal_round_utility",
                return_value=7.0,
            ) as terminal_mock,
            patch(
                "backend.bots.search_eval.estimate_partial_match_utility",
                side_effect=[3.0],
            ) as estimate_mock,
        ):
            value = rollout_round_to_utility(
                round_state=round_state,
                auction_state=None,
                match_scores=None,
                teams=[("A",), ("B",), ("C",)],
                target_score=21,
                player_name="A",
                hybrid_cutoff=True,
            )

        self.assertEqual(value, 3.0)
        terminal_mock.assert_not_called()
        estimate_mock.assert_called_once()

    def test_hybrid_cutoff_keeps_exact_rollout_for_near_terminal_states(self):
        round_state = self._build_ordering_state()
        self.assertTrue(_should_rollout_cutoff_exactly(round_state))

    def test_determinization_samples_stay_higher_for_three_player_games(self):
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=2)
            ._determinization_sample_count(player_count=4),
            12,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=3)
            ._determinization_sample_count(player_count=4),
            6,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=4)
            ._determinization_sample_count(player_count=4),
            3,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=5)
            ._determinization_sample_count(player_count=4),
            2,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=3)
            ._determinization_sample_count(player_count=3),
            8,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=4)
            ._determinization_sample_count(player_count=3),
            4,
        )
        self.assertEqual(
            HumanInformationMinimaxNTrickPlayer("A", depth=3)
            ._auction_determinization_sample_count(
                AuctionState(
                    dealer_index=0,
                    current_bidder_index=1,
                    player_names=["A", "B", "C"],
                )
            ),
            8,
        )

    def test_transposition_table_preserves_exact_best_action(self):
        round_state = self._build_transposition_regression_state()
        cached_bot = OmniscientMinimaxNTrickPlayer("B", depth=3)
        exact_bot = ExactOmniscientMinimaxNTrickPlayer("B", depth=3)
        exact_scores = exact_bot.root_scores_exact_utility(round_state)
        chosen_card = cached_bot.choose_card(round_state)

        self.assertEqual(
            exact_scores[chosen_card.code],
            max(exact_scores.values()),
        )

    def test_terminal_search_scoring_matches_round_scoring_rules(self):
        round_state = self._build_terminal_scoring_state()
        ensure_captured_plays_synchronized(round_state)
        auction_state = AuctionState(
            dealer_index=0,
            current_bidder_index=1,
            player_names=["A", "B", "C"],
            highest_bid=5,
            highest_bidder_name="B",
            passed_player_names={"A", "C"},
            is_complete=True,
        )
        teams = [("A",), ("B",), ("C",)]
        match_scores = {"A": 20, "B": 19, "C": 18}

        _, projected_scores = apply_bid_and_match_rules(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=match_scores,
            teams=teams,
            target_score=21,
        )

        details = score_round_details(round_state)
        manual_scores = {
            score_unit_name(team): float(match_scores.get(score_unit_name(team), 0))
            for team in teams
        }
        round_points = {
            result["name"]: float(result["total_points"])
            for result in details["results"]
        }
        for team in teams:
            unit_name = score_unit_name(team)
            delta = round_points[unit_name]
            if unit_name == "B":
                delta = float(-auction_state.highest_bid)
            next_score = manual_scores[unit_name] + delta
            if unit_name != "B" and next_score > 20:
                next_score = 20.0
            manual_scores[unit_name] = next_score

        self.assertEqual(projected_scores, manual_scores)

    def test_depth_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "depth must be at least 1"):
            OmniscientMinimaxNTrickPlayer("A", depth=0)

        with self.assertRaisesRegex(ValueError, "depth must be at least 1"):
            HumanInformationMinimaxNTrickPlayer("A", depth=0)

    def test_legacy_import_paths_remain_compatible(self):
        self.assertIs(MinimaxNTrickPlayer, HumanInformationMinimaxNTrickPlayer)
        self.assertIs(
            OMNISCIENT_MinimaxNTrickPlayer,
            OmniscientMinimaxNTrickPlayer,
        )

    def test_registry_builds_human_information_depth_presets(self):
        bot = build_ready_bot("2-trick-minmax", "A")
        self.assertIsInstance(bot, HumanInformationMinimaxNTrickPlayer)
        self.assertEqual(bot.depth, 2)

    def test_registry_builds_omniscient_depth_presets(self):
        bot = build_ready_bot("o-3-trick-minmax", "A")
        self.assertIsInstance(bot, OmniscientMinimaxNTrickPlayer)
        self.assertEqual(bot.depth, 3)

    def test_registry_metadata_lists_n_trick_presets(self):
        bot_ids = {bot["id"] for bot in list_ready_bot_metadata()}
        self.assertIn("2-trick-minmax", bot_ids)
        self.assertIn("o-2-trick-minmax", bot_ids)
        self.assertIn("6-trick-minmax", bot_ids)
        self.assertIn("o-6-trick-minmax", bot_ids)


if __name__ == "__main__":
    unittest.main()
