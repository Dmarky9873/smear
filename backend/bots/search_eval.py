from __future__ import annotations

try:
    from backend.constants import GAME_VALUES, HAND_SIZE, RANKS, RANK_ORDER
    from backend.engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_trick_winner,
        undo_trick_action_for_search,
    )
    from backend.models import (
        AuctionState,
        Card,
        Deck,
        Play,
        Player,
        RoundState,
        Team,
        TrickState,
        get_cards_value,
        would_win,
    )
except ImportError:
    from constants import GAME_VALUES, HAND_SIZE, RANKS, RANK_ORDER
    from engine import (
        apply_trick_action_for_search,
        get_legal_actions,
        get_trick_winner,
        undo_trick_action_for_search,
    )
    from models import (
        AuctionState,
        Card,
        Deck,
        Play,
        Player,
        RoundState,
        Team,
        TrickState,
        get_cards_value,
        would_win,
    )


def score_unit_name(member_names: tuple[str, ...] | list[str]) -> str:
    return " / ".join(member_names)


_SUIT_ORDER = ("H", "D", "C", "S")


def score_unit_name_for_player(
    player_name: str,
    teams: list[tuple[str, ...]],
) -> str:
    for team in teams:
        if player_name in team:
            return score_unit_name(team)
    return player_name


def default_teams(player_names: list[str]) -> list[tuple[str, ...]]:
    return [(player_name,) for player_name in player_names]


def calculate_functional_deck_low(num_players: int) -> str:
    dealt = HAND_SIZE * num_players
    best_low = None
    best_diff = float("inf")

    for index, rank in enumerate(RANKS):
        remaining_ranks = RANKS[index:]
        deck_size = 4 * len(remaining_ranks) + 2
        hiding = deck_size - dealt

        if hiding <= 0:
            continue

        diff = abs(hiding - 2)
        if diff < best_diff:
            best_diff = diff
            best_low = rank

    if best_low is None:
        raise ValueError(f"could not determine functional deck low for {num_players} players")
    return best_low


def build_round_state_for_world(
    *,
    player_names: list[str],
    teams: list[tuple[str, ...]],
    hands: dict[str, set[Card]],
    hidden_cards: set[Card],
    low: str,
    starting_player_name: str,
) -> RoundState:
    players = [Player(player_name, set(hands[player_name])) for player_name in player_names]
    players_by_name = {player.name: player for player in players}
    starting_player = players_by_name[starting_player_name]

    return RoundState(
        players=players,
        current_player=starting_player,
        trump=None,
        current_trick=TrickState(starting_player, [], players, None),
        hidden_cards=set(hidden_cards),
        trick_history=[],
        teams=[
            Team([players_by_name[player_name] for player_name in team], set())
            for team in teams
        ],
        deck=Deck(low),
    )


def synchronize_captured_plays(round_state: RoundState) -> None:
    for player in round_state.players:
        player._captured_plays = set()

    for trick in round_state.trick_history:
        winner = get_trick_winner(trick)
        for play in trick.plays:
            winner.capture(play)


def ensure_captured_plays_synchronized(round_state: RoundState) -> None:
    expected_captured_count = sum(len(trick.plays) for trick in round_state.trick_history)
    actual_captured_count = sum(
        len(player.captured_plays)
        for player in round_state.players
    )
    if actual_captured_count != expected_captured_count:
        synchronize_captured_plays(round_state)


def _score_terminal_unit_points(
    round_state: RoundState,
    teams: list[tuple[str, ...]],
) -> dict[str, int]:
    if not round_state.is_terminal:
        raise ValueError("round is not in terminal state")

    if round_state.trump is None:
        raise ValueError("round trump has not been set")

    player_to_unit: dict[str, str] = {}
    captured_cards_by_unit: dict[str, set[Card]] = {}
    point_totals = {score_unit_name(team): 0 for team in teams}

    for team in round_state.teams:
        member_names = tuple(player.name for player in team.constituents)
        unit_name = score_unit_name(member_names)
        captured_cards = {
            play.card
            for player in team.constituents
            for play in player.captured_plays
        }
        captured_cards_by_unit[unit_name] = captured_cards
        for member_name in member_names:
            player_to_unit[member_name] = unit_name

    visible_trump_cards = [
        card
        for card in round_state.deck.get_copy()
        if (
            not card.is_joker
            and card.suit == round_state.trump
            and card not in round_state.hidden_cards
        )
    ]
    if not visible_trump_cards:
        raise ValueError("no non-hidden trump cards are available to score high/low")

    low_card = min(visible_trump_cards, key=lambda card: RANK_ORDER[card.rank])
    high_card = max(visible_trump_cards, key=lambda card: RANK_ORDER[card.rank])
    jack_card = next(
        (card for card in visible_trump_cards if card.rank == "J"),
        None,
    )

    low_owner_name = next(
        (
            play.player.name
            for trick in round_state.trick_history
            for play in trick.plays
            if play.card == low_card
        ),
        None,
    )
    if low_owner_name is None:
        raise ValueError("could not determine owner of low card from completed plays")
    point_totals[player_to_unit[low_owner_name]] += 1

    high_unit_name = next(
        (
            unit_name
            for unit_name, captured_cards in captured_cards_by_unit.items()
            if high_card in captured_cards
        ),
        None,
    )
    if high_unit_name is None:
        raise ValueError("could not determine owner of high card from scoring units")
    point_totals[high_unit_name] += 1

    if jack_card is not None:
        jack_unit_name = next(
            (
                unit_name
                for unit_name, captured_cards in captured_cards_by_unit.items()
                if jack_card in captured_cards
            ),
            None,
        )
        if jack_unit_name is None:
            raise ValueError("could not determine owner of jack card from scoring units")
        point_totals[jack_unit_name] += 1

    game_totals: dict[str, int] = {}
    for unit_name, captured_cards in captured_cards_by_unit.items():
        point_totals[unit_name] += sum(1 for card in captured_cards if card.is_joker)
        game_totals[unit_name] = sum(
            GAME_VALUES.get(card.rank, 0)
            for card in captured_cards
            if not card.is_joker
        )

    max_game_total = max(game_totals.values(), default=0)
    game_winners = [
        unit_name
        for unit_name, game_total in game_totals.items()
        if game_total == max_game_total
    ]
    if len(game_winners) == 1 and max_game_total > 0:
        point_totals[game_winners[0]] += 1

    return point_totals


def apply_bid_and_match_rules(
    *,
    round_state: RoundState,
    auction_state: AuctionState | None,
    match_scores: dict[str, int] | None,
    teams: list[tuple[str, ...]],
    target_score: int,
) -> tuple[dict, dict[str, float]]:
    round_totals = _score_terminal_unit_points(round_state, teams)
    round_score = {
        "results": [
            {
                "name": score_unit_name(team),
                "member_names": list(team),
                "total_points": round_totals.get(score_unit_name(team), 0),
                "match_delta": float(round_totals.get(score_unit_name(team), 0)),
                "bid_amount": None,
                "made_bid": None,
            }
            for team in teams
        ],
    }
    projected_scores = {
        score_unit_name(team): float((match_scores or {}).get(score_unit_name(team), 0))
        for team in teams
    }

    bid_summary = {
        "bidder_name": None,
        "unit_name": None,
        "amount": None,
        "points_won": None,
        "made_bid": None,
        "match_delta": None,
    }

    if auction_state is not None:
        bidder_name = auction_state.highest_bidder_name
        bid_amount = auction_state.highest_bid
        bid_summary["bidder_name"] = bidder_name
        bid_summary["amount"] = bid_amount

        if bidder_name is not None and bid_amount is not None:
            bidder_result = next(
                (
                    result
                    for result in round_score["results"]
                    if bidder_name in result["member_names"]
                ),
                None,
            )
            if bidder_result is None:
                raise ValueError("could not map highest bidder to a scoring unit")

            made_bid = bidder_result["total_points"] >= bid_amount
            bidder_result["bid_amount"] = bid_amount
            bidder_result["made_bid"] = made_bid
            bidder_result["match_delta"] = (
                float(bidder_result["total_points"]) if made_bid else float(-bid_amount)
            )

            bid_summary.update(
                {
                    "unit_name": bidder_result["name"],
                    "points_won": bidder_result["total_points"],
                    "made_bid": made_bid,
                    "match_delta": bidder_result["match_delta"],
                }
            )

    round_score["bid_summary"] = bid_summary

    auction_winning_unit_name = bid_summary["unit_name"]
    non_bidder_win_cap = target_score - 1
    for result in round_score["results"]:
        current_score = projected_scores.get(result["name"], 0.0)
        next_score = current_score + float(result["match_delta"])
        if (
            auction_winning_unit_name is not None
            and result["name"] != auction_winning_unit_name
            and next_score > non_bidder_win_cap
        ):
            next_score = float(non_bidder_win_cap)
            result["match_delta"] = next_score - current_score

        projected_scores[result["name"]] = next_score

    return round_score, projected_scores


def match_utility_for_player(
    *,
    player_name: str,
    teams: list[tuple[str, ...]],
    projected_scores: dict[str, float],
    target_score: int,
) -> float:
    my_unit_name = score_unit_name_for_player(player_name, teams)
    my_score = projected_scores.get(my_unit_name, 0.0)
    opponent_scores = [
        score
        for unit_name, score in projected_scores.items()
        if unit_name != my_unit_name
    ]
    utility = my_score - max(opponent_scores, default=0.0)

    if projected_scores:
        max_score = max(projected_scores.values())
        winners = [
            unit_name
            for unit_name, score in projected_scores.items()
            if score >= target_score and score == max_score
        ]
        if my_unit_name in winners:
            utility += target_score
        elif winners:
            utility -= target_score

    return float(utility)


def evaluate_terminal_round_utility(
    *,
    round_state: RoundState,
    auction_state: AuctionState | None,
    match_scores: dict[str, int] | None,
    teams: list[tuple[str, ...]],
    target_score: int,
    player_name: str,
) -> float:
    _, projected_scores = apply_bid_and_match_rules(
        round_state=round_state,
        auction_state=auction_state,
        match_scores=match_scores,
        teams=teams,
        target_score=target_score,
    )
    return match_utility_for_player(
        player_name=player_name,
        teams=teams,
        projected_scores=projected_scores,
        target_score=target_score,
    )


def estimate_partial_match_utility(
    *,
    round_state: RoundState,
    auction_state: AuctionState | None,
    match_scores: dict[str, int] | None,
    teams: list[tuple[str, ...]],
    target_score: int,
    player_name: str,
) -> float:
    unit_names = {team: score_unit_name(team) for team in teams}
    unit_scores = {unit_name: 0.0 for unit_name in unit_names.values()}
    player_to_unit = {
        member_name: unit_name
        for team, unit_name in unit_names.items()
        for member_name in team
    }

    for team in round_state.teams:
        unit_name = player_to_unit[team.constituents[0].name]
        captured_score = 0.0
        hand_score = 0.0
        for player in team.constituents:
            for play in player.captured_plays:
                card = play.card
                if card.is_joker:
                    captured_score += 1.0
                else:
                    captured_score += GAME_VALUES.get(card.rank, 0)
            for card in player.cards:
                if card.is_joker:
                    hand_score += 1.0
                else:
                    hand_score += GAME_VALUES.get(card.rank, 0)
        unit_scores[unit_name] += captured_score + (0.5 * hand_score)

    if round_state.trump is not None:
        visible_trump_cards = [
            card for card in round_state.deck.get_copy()
            if (
                not card.is_joker
                and card.suit == round_state.trump
                and card not in round_state.hidden_cards
            )
        ]
        if visible_trump_cards:
            high_card = max(visible_trump_cards, key=lambda card: RANK_ORDER[card.rank])
            low_card = min(visible_trump_cards, key=lambda card: RANK_ORDER[card.rank])
            jack_card = next(
                (card for card in visible_trump_cards if card.rank == "J"),
                None,
            )

            card_unit_map: dict[Card, tuple[str, bool]] = {}
            for trick in round_state.trick_history:
                for play in trick.plays:
                    card_unit_map[play.card] = (
                        player_to_unit[play.player.name],
                        True,
                    )
            for play in round_state.current_trick.plays:
                card_unit_map.setdefault(
                    play.card,
                    (player_to_unit[play.player.name], False),
                )
            for player in round_state.players:
                unit_name = player_to_unit[player.name]
                for card in player.cards:
                    card_unit_map.setdefault(card, (unit_name, False))

            low_unit_name, _ = card_unit_map.get(low_card, (None, False))
            if low_unit_name is not None:
                unit_scores[low_unit_name] += 1.0

            for card in [high_card, jack_card]:
                if card is None:
                    continue
                unit_name, is_certain = card_unit_map.get(card, (None, False))
                if unit_name is not None:
                    unit_scores[unit_name] += 1.0 if is_certain else 0.5

    projected_scores = {
        unit_name: float((match_scores or {}).get(unit_name, 0))
        for unit_name in unit_names.values()
    }
    bid_unit_name = None
    bid_amount = None
    if auction_state is not None and auction_state.highest_bidder_name is not None:
        bid_unit_name = score_unit_name_for_player(
            auction_state.highest_bidder_name,
            teams,
        )
        bid_amount = auction_state.highest_bid

    for unit_name, estimated_points in unit_scores.items():
        delta = estimated_points
        if bid_unit_name is not None and bid_amount is not None and unit_name == bid_unit_name:
            delta = estimated_points if estimated_points >= bid_amount else float(-bid_amount)
        next_score = projected_scores[unit_name] + delta
        if (
            bid_unit_name is not None
            and unit_name != bid_unit_name
            and next_score > target_score - 1
        ):
            next_score = float(target_score - 1)
        projected_scores[unit_name] = next_score

    return match_utility_for_player(
        player_name=player_name,
        teams=teams,
        projected_scores=projected_scores,
        target_score=target_score,
    )


def _trump_cards(cards: set[Card], trump_suit: str) -> list[Card]:
    return [
        card for card in cards
        if not card.is_joker and card.suit == trump_suit
    ]


def _count_jokers(cards: set[Card]) -> int:
    return sum(1 for card in cards if card.is_joker)


def _has_high_trump(cards: set[Card], trump_suit: str) -> bool:
    return any(card.rank == "A" for card in _trump_cards(cards, trump_suit))


def _has_low_trump_candidate(cards: set[Card], trump_suit: str) -> bool:
    trump_cards = _trump_cards(cards, trump_suit)
    if not trump_cards:
        return False
    lowest_trump = min(trump_cards, key=lambda card: RANK_ORDER[card.rank])
    return RANK_ORDER[lowest_trump.rank] <= RANK_ORDER["10"]


def _has_game_strength(cards: set[Card]) -> bool:
    game_total = sum(
        GAME_VALUES.get(card.rank, 0)
        for card in cards
        if not card.is_joker
    )
    valuable_card_count = sum(
        1
        for card in cards
        if not card.is_joker and GAME_VALUES.get(card.rank, 0) > 0
    )
    return game_total >= 16 or valuable_card_count >= 4


def _has_trump_control(cards: set[Card], trump_suit: str) -> bool:
    trump_cards = _trump_cards(cards, trump_suit)
    if len(trump_cards) >= 2:
        return True
    if not trump_cards:
        return False
    if _count_jokers(cards) > 0:
        return True
    return any(card.rank in {"A", "K", "Q", "J"} for card in trump_cards)


def _estimate_hand_strength(cards: set[Card], trump_suit: str) -> int:
    estimate = 0
    if _has_high_trump(cards, trump_suit):
        estimate += 1
    if _has_low_trump_candidate(cards, trump_suit):
        estimate += 1
    estimate += _count_jokers(cards)
    if _has_game_strength(cards):
        estimate += 1
    if _has_trump_control(cards, trump_suit):
        estimate += 1
    return max(0, min(6, estimate))


def _preferred_suit_key(cards: set[Card], trump_suit: str) -> tuple[int, int, int, int]:
    trump_cards = _trump_cards(cards, trump_suit)
    highest_rank = max(
        (RANK_ORDER[card.rank] for card in trump_cards),
        default=-1,
    )
    total_rank = sum(RANK_ORDER[card.rank] for card in trump_cards)
    return (
        _estimate_hand_strength(cards, trump_suit),
        highest_rank,
        len(trump_cards),
        total_rank,
    )


def select_preferred_suit(cards: set[Card]) -> str:
    candidate_suits = {
        card.suit
        for card in cards
        if not card.is_joker and card.suit is not None
    }
    ordered_suits = (
        tuple(suit for suit in _SUIT_ORDER if suit in candidate_suits)
        if candidate_suits
        else _SUIT_ORDER
    )
    return max(
        ordered_suits,
        key=lambda suit: _preferred_suit_key(cards, suit),
    )


def choose_rollout_card(
    round_state: RoundState,
    legal_cards: list[Card] | None = None,
) -> Card:
    if legal_cards is None:
        legal_cards = list(get_legal_actions(round_state))
    if not legal_cards:
        raise ValueError("rollout policy could not find a legal card")
    preferred_suit = (
        select_preferred_suit(round_state.current_player.cards)
        if round_state.trump is None
        else round_state.trump
    )

    value_by_card: dict[Card, int] = {}
    trick_cards = {play.card for play in round_state.current_trick.plays}
    for card in legal_cards:
        if would_win(card, round_state.current_trick):
            value_by_card[card] = get_cards_value(trick_cards.union({card}))
        else:
            value_by_card[card] = -get_cards_value({card})

    max_value = max(value_by_card.values())
    best_cards = [
        card
        for card, value in value_by_card.items()
        if value == max_value
    ]
    return max(
        best_cards,
        key=lambda card: (
            card.suit == preferred_suit,
            len(RANK_ORDER) + 1 if card.is_joker else RANK_ORDER[card.rank],
            card.code,
        ),
    )


def _should_rollout_cutoff_exactly(
    round_state: RoundState,
    *,
    exact_rollout_action_threshold: int | None = None,
) -> bool:
    if round_state.is_terminal or round_state.trump is None:
        return True

    threshold = (
        len(round_state.players)
        if exact_rollout_action_threshold is None
        else exact_rollout_action_threshold
    )
    remaining_actions = sum(len(player.cards) for player in round_state.players)
    return remaining_actions <= threshold


def rollout_round_to_utility(
    *,
    round_state: RoundState,
    auction_state: AuctionState | None,
    match_scores: dict[str, int] | None,
    teams: list[tuple[str, ...]],
    target_score: int,
    player_name: str,
    hybrid_cutoff: bool = False,
    exact_rollout_action_threshold: int | None = None,
) -> float:
    prefer_exact_cutoff = hybrid_cutoff and _should_rollout_cutoff_exactly(
        round_state,
        exact_rollout_action_threshold=exact_rollout_action_threshold,
    )

    if round_state.is_terminal:
        return evaluate_terminal_round_utility(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=match_scores,
            teams=teams,
            target_score=target_score,
            player_name=player_name,
        )

    if hybrid_cutoff and not prefer_exact_cutoff:
        return estimate_partial_match_utility(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=match_scores,
            teams=teams,
            target_score=target_score,
            player_name=player_name,
        )

    undos = []
    try:
        while not round_state.is_terminal:
            legal_cards = list(get_legal_actions(round_state))
            if not legal_cards:
                if prefer_exact_cutoff:
                    try:
                        return evaluate_terminal_round_utility(
                            round_state=round_state,
                            auction_state=auction_state,
                            match_scores=match_scores,
                            teams=teams,
                            target_score=target_score,
                            player_name=player_name,
                        )
                    except ValueError:
                        pass
                return estimate_partial_match_utility(
                    round_state=round_state,
                    auction_state=auction_state,
                    match_scores=match_scores,
                    teams=teams,
                    target_score=target_score,
                    player_name=player_name,
                )
            card = choose_rollout_card(round_state, legal_cards)
            undos.append(
                apply_trick_action_for_search(
                    round_state,
                    Play(round_state.current_player, card),
                    validate_legal=False,
                )
            )

        return evaluate_terminal_round_utility(
            round_state=round_state,
            auction_state=auction_state,
            match_scores=match_scores,
            teams=teams,
            target_score=target_score,
            player_name=player_name,
        )
    finally:
        for undo in reversed(undos):
            undo_trick_action_for_search(round_state, undo)
