from __future__ import annotations

from dataclasses import dataclass

try:
    from backend.constants import GAME_VALUES, HAND_SIZE, RANK_ORDER
    from backend.engine import get_legal_actions, get_legal_auction_actions
    from backend.models import AuctionEvent, AuctionState, Card, RoundState, would_win
except ImportError:
    from constants import GAME_VALUES, HAND_SIZE, RANK_ORDER
    from engine import get_legal_actions, get_legal_auction_actions
    from models import AuctionEvent, AuctionState, Card, RoundState, would_win


THREE_PLAYER_LOW_RANK = "10"
THREE_PLAYER_RANKS = ("10", "J", "Q", "K", "A")
SUIT_ORDER = ("H", "D", "C", "S")
THREE_PLAYER_CARD_CODES = tuple(
    f"{rank}{suit}"
    for suit in SUIT_ORDER
    for rank in THREE_PLAYER_RANKS
) + ("J1", "J2")
CARD_INDEX = {
    card_code: index
    for index, card_code in enumerate(THREE_PLAYER_CARD_CODES)
}


@dataclass(frozen=True)
class RelativeSeatContext:
    ordered_names: tuple[str, str, str]
    self_name: str
    left_name: str
    right_name: str


def _one_hot(index: int | None, size: int) -> list[float]:
    vector = [0.0 for _ in range(size)]
    if index is not None and 0 <= index < size:
        vector[index] = 1.0
    return vector


def _zero_vector(size: int) -> list[float]:
    return [0.0 for _ in range(size)]


def _normalized_score(score: int | float, target_score: int) -> float:
    if target_score <= 0:
        raise ValueError("target_score must be positive")
    clipped_score = max(-target_score, min(target_score, float(score)))
    return clipped_score / target_score


def _encode_card(card: Card | None) -> list[float]:
    if card is None:
        return _zero_vector(len(THREE_PLAYER_CARD_CODES))
    vector = _zero_vector(len(THREE_PLAYER_CARD_CODES))
    try:
        vector[CARD_INDEX[card.code]] = 1.0
    except KeyError as exc:
        raise ValueError(
            f"card {card.code} is not part of the supported 3-player deck"
        ) from exc
    return vector


def _encode_cards(cards: set[Card]) -> list[float]:
    vector = _zero_vector(len(THREE_PLAYER_CARD_CODES))
    for card in cards:
        try:
            vector[CARD_INDEX[card.code]] = 1.0
        except KeyError as exc:
            raise ValueError(
                f"card {card.code} is not part of the supported 3-player deck"
            ) from exc
    return vector


def _trump_one_hot(trump_suit: str | None) -> list[float]:
    if trump_suit is None:
        return _one_hot(4, 5)
    return _one_hot(SUIT_ORDER.index(trump_suit), 5)


def _suit_one_hot(card: Card) -> list[float]:
    if card.is_joker:
        return _one_hot(4, 5)
    return _one_hot(SUIT_ORDER.index(card.suit), 5)


def _rank_one_hot(card: Card) -> list[float]:
    if card.is_joker:
        return _one_hot(5, 6)
    return _one_hot(THREE_PLAYER_RANKS.index(card.rank), 6)


def _relative_seat_context(
    player_names: list[str],
    acting_player_name: str,
) -> RelativeSeatContext:
    if len(player_names) != 3:
        raise ValueError("the neural 3-player bot only supports exactly three players")
    try:
        acting_index = player_names.index(acting_player_name)
    except ValueError as exc:
        raise ValueError(
            f"acting player {acting_player_name} is not seated in this game"
        ) from exc
    ordered_names = tuple(
        player_names[(acting_index + offset) % len(player_names)]
        for offset in range(len(player_names))
    )
    return RelativeSeatContext(
        ordered_names=ordered_names,
        self_name=ordered_names[0],
        left_name=ordered_names[1],
        right_name=ordered_names[2],
    )


def _relative_seat_index(
    seat_context: RelativeSeatContext,
    player_name: str,
) -> int:
    return seat_context.ordered_names.index(player_name)


def is_supported_three_player_singleton_context(
    player_names: list[str],
    teams: list[tuple[str, ...]] | None,
) -> bool:
    if len(player_names) != 3:
        return False
    if teams is None:
        return True
    if len(teams) != 3:
        return False
    return all(len(team) == 1 for team in teams)


def ordered_legal_cards(round_state: RoundState) -> list[Card]:
    return sorted(get_legal_actions(round_state), key=lambda card: card.code)


def ordered_legal_auction_actions(auction_state: AuctionState) -> list[AuctionEvent]:
    actions = list(get_legal_auction_actions(auction_state))
    return sorted(
        actions,
        key=lambda action: (
            action.action == "pass",
            action.amount or 0,
        ),
    )


def _public_void_constraints(
    round_state: RoundState,
    acting_player_name: str,
) -> tuple[dict[str, set[str]], set[str]]:
    suit_voids = {
        player.name: set()
        for player in round_state.players
        if player.name != acting_player_name
    }
    players_without_joker: set[str] = set()

    for trick in [*round_state.trick_history, round_state.current_trick]:
        if len(trick.plays) < 2:
            continue

        lead_card = trick.plays[0].card
        if lead_card.is_joker:
            continue

        for play in trick.plays[1:]:
            player_name = play.player.name
            if player_name == acting_player_name or play.card.is_joker:
                continue
            if play.card.suit == lead_card.suit or play.card.suit == trick.trump:
                continue

            suit_voids[player_name].add(lead_card.suit)
            if lead_card.suit == trick.trump:
                players_without_joker.add(player_name)

    return suit_voids, players_without_joker


def _count_suit(cards: set[Card], suit: str) -> int:
    return sum(1 for card in cards if not card.is_joker and card.suit == suit)


def _suit_game_value(cards: set[Card], suit: str) -> int:
    return sum(
        GAME_VALUES.get(card.rank, 0)
        for card in cards
        if not card.is_joker and card.suit == suit
    )


def _count_jokers(cards: set[Card]) -> int:
    return sum(1 for card in cards if card.is_joker)


def _suit_strength(cards: set[Card], suit: str) -> float:
    suit_cards = [
        card
        for card in cards
        if not card.is_joker and card.suit == suit
    ]
    if not suit_cards:
        return 0.0

    strength = 0.0
    if any(card.rank == "A" for card in suit_cards):
        strength += 1.0
    if any(card.rank == THREE_PLAYER_LOW_RANK for card in suit_cards):
        strength += 1.0
    if any(card.rank == "J" for card in suit_cards):
        strength += 1.0
    if len(suit_cards) >= 2:
        strength += 1.0
    if _suit_game_value(cards, suit) >= 12:
        strength += 1.0
    if _count_jokers(cards) > 0:
        strength += 0.5
    return strength


def _preferred_suit(cards: set[Card]) -> str:
    candidate_suits = [
        suit for suit in SUIT_ORDER
        if _count_suit(cards, suit) > 0
    ]
    if not candidate_suits:
        return SUIT_ORDER[0]
    return max(
        candidate_suits,
        key=lambda suit: (
            _suit_strength(cards, suit),
            _count_suit(cards, suit),
            _suit_game_value(cards, suit),
            suit,
        ),
    )


def _match_score_vector(
    seat_context: RelativeSeatContext,
    match_scores: dict[str, int] | None,
    target_score: int,
) -> list[float]:
    if match_scores is None:
        match_scores = {
            player_name: 0
            for player_name in seat_context.ordered_names
        }
    return [
        _normalized_score(match_scores.get(player_name, 0), target_score)
        for player_name in seat_context.ordered_names
    ]


def _highest_bid_one_hot(auction_state: AuctionState | None) -> list[float]:
    if auction_state is None or auction_state.highest_bid is None:
        return _one_hot(0, 7)
    return _one_hot(auction_state.highest_bid, 7)


def _highest_bidder_one_hot(
    seat_context: RelativeSeatContext,
    auction_state: AuctionState | None,
) -> list[float]:
    if auction_state is None or auction_state.highest_bidder_name is None:
        return _one_hot(3, 4)
    return _one_hot(
        _relative_seat_index(seat_context, auction_state.highest_bidder_name),
        4,
    )


def _bid_history_features(
    seat_context: RelativeSeatContext,
    auction_state: AuctionState,
) -> list[float]:
    bid_amount_by_player = {
        player_name: 0.0
        for player_name in auction_state.player_names
    }
    bid_made_by_player = {
        player_name: 0.0
        for player_name in auction_state.player_names
    }
    pass_by_player = {
        player_name: 0.0
        for player_name in auction_state.player_names
    }

    for event in auction_state.bid_history:
        if event.action == "pass":
            pass_by_player[event.bidder_name] = 1.0
        elif event.amount is not None:
            bid_amount_by_player[event.bidder_name] = event.amount / 6.0
            bid_made_by_player[event.bidder_name] = 1.0

    features: list[float] = []
    for player_name in seat_context.ordered_names:
        features.append(pass_by_player[player_name])
        features.append(bid_made_by_player[player_name])
        features.append(bid_amount_by_player[player_name])
    return features


def _current_trick_features(
    *,
    seat_context: RelativeSeatContext,
    round_state: RoundState,
) -> list[float]:
    features: list[float] = []
    current_trick = round_state.current_trick
    for slot_index in range(2):
        if slot_index < len(current_trick.plays):
            play = current_trick.plays[slot_index]
            features.extend(
                _one_hot(_relative_seat_index(seat_context, play.player.name), 3)
            )
            features.extend(_encode_card(play.card))
        else:
            features.extend(_zero_vector(3))
            features.extend(_zero_vector(len(THREE_PLAYER_CARD_CODES)))
    return features


def _captured_card_count(player) -> float:
    return len(player.captured_plays) / float(HAND_SIZE * 3)


def _captured_card_value(player) -> float:
    total_value = sum(
        1 if play.card.is_joker else GAME_VALUES.get(play.card.rank, 0)
        for play in player.captured_plays
    )
    return total_value / 30.0


def encode_play_state(
    *,
    round_state: RoundState,
    perspective_player_name: str,
    match_scores: dict[str, int] | None,
    target_score: int,
    auction_state: AuctionState | None,
) -> list[float]:
    player_names = [player.name for player in round_state.players]
    seat_context = _relative_seat_context(player_names, perspective_player_name)
    player_by_name = {
        player.name: player
        for player in round_state.players
    }
    perspective_player = player_by_name[perspective_player_name]
    public_seen_cards = {
        play.card
        for trick in round_state.trick_history
        for play in trick.plays
    } | {play.card for play in round_state.current_trick.plays}
    voids, no_joker_players = _public_void_constraints(
        round_state,
        perspective_player_name,
    )
    current_trick = round_state.current_trick
    trick_points = sum(
        (1 if play.card.is_joker else GAME_VALUES.get(play.card.rank, 0))
        for play in current_trick.plays
    )
    preferred_suit = _preferred_suit(perspective_player.cards)
    lead_card = current_trick.plays[0].card if current_trick.plays else None

    features: list[float] = [1.0]
    features.extend(_encode_cards(set(perspective_player.cards)))
    features.extend(_encode_cards(public_seen_cards - set(perspective_player.cards)))
    features.extend(_current_trick_features(seat_context=seat_context, round_state=round_state))
    features.extend(_trump_one_hot(round_state.trump))
    features.extend(_match_score_vector(seat_context, match_scores, target_score))
    features.extend(
        [
            len(player_by_name[player_name].cards) / HAND_SIZE
            for player_name in seat_context.ordered_names
        ]
    )
    features.extend(
        [
            _captured_card_count(player_by_name[player_name])
            for player_name in seat_context.ordered_names
        ]
    )
    features.extend(
        [
            _captured_card_value(player_by_name[player_name])
            for player_name in seat_context.ordered_names
        ]
    )
    features.extend(_one_hot(len(current_trick.plays), 3))
    features.extend(_one_hot(len(round_state.trick_history), HAND_SIZE))
    features.extend(_highest_bid_one_hot(auction_state))
    features.extend(_highest_bidder_one_hot(seat_context, auction_state))
    features.extend(
        _one_hot(_relative_seat_index(seat_context, round_state.current_player.name), 3)
    )

    for opponent_name in (seat_context.left_name, seat_context.right_name):
        features.extend(
            [
                1.0 if suit in voids.get(opponent_name, set()) else 0.0
                for suit in SUIT_ORDER
            ]
        )
    features.extend(
        [
            1.0 if opponent_name in no_joker_players else 0.0
            for opponent_name in (seat_context.left_name, seat_context.right_name)
        ]
    )
    features.extend(
        [
            1.0 if round_state.trump is None else 0.0,
            1.0 if not current_trick.plays else 0.0,
            1.0 if lead_card is not None and lead_card.is_joker else 0.0,
            trick_points / 16.0,
            max(_suit_strength(perspective_player.cards, suit) for suit in SUIT_ORDER) / 5.5,
            _suit_strength(perspective_player.cards, preferred_suit) / 5.5,
            1.0 if preferred_suit == "H" else 0.0,
            1.0 if preferred_suit == "D" else 0.0,
            1.0 if preferred_suit == "C" else 0.0,
            1.0 if preferred_suit == "S" else 0.0,
        ]
    )

    return features


def encode_auction_state(
    *,
    auction_state: AuctionState,
    perspective_player_name: str,
    hand: set[Card],
    match_scores: dict[str, int] | None,
    target_score: int,
) -> list[float]:
    seat_context = _relative_seat_context(
        auction_state.player_names,
        perspective_player_name,
    )
    preferred_suit = _preferred_suit(hand)
    suit_strengths = {
        suit: _suit_strength(hand, suit)
        for suit in SUIT_ORDER
    }
    sorted_strengths = sorted(suit_strengths.values(), reverse=True)
    legal_actions = ordered_legal_auction_actions(auction_state)
    can_pass = any(action.action == "pass" for action in legal_actions)
    joker_count = _count_jokers(hand)
    total_game_value = sum(
        1 if card.is_joker else GAME_VALUES.get(card.rank, 0)
        for card in hand
    )

    features: list[float] = [1.0]
    features.extend(_encode_cards(set(hand)))

    for suit in SUIT_ORDER:
        features.append(_count_suit(hand, suit) / HAND_SIZE)
    for suit in SUIT_ORDER:
        features.append(_suit_game_value(hand, suit) / 20.0)
    for suit in SUIT_ORDER:
        features.append(suit_strengths[suit] / 5.5)
    for suit in SUIT_ORDER:
        features.append(
            1.0
            if any(
                not card.is_joker and card.suit == suit and card.rank == "A"
                for card in hand
            )
            else 0.0
        )
    for suit in SUIT_ORDER:
        features.append(
            1.0
            if any(
                not card.is_joker
                and card.suit == suit
                and card.rank == THREE_PLAYER_LOW_RANK
                for card in hand
            )
            else 0.0
        )

    features.extend(_one_hot(joker_count, 3))
    features.extend(
        [
            total_game_value / 30.0,
            max(sorted_strengths[0], 0.0) / 5.5,
            max(sorted_strengths[1], 0.0) / 5.5,
            1.0 if preferred_suit == "H" else 0.0,
            1.0 if preferred_suit == "D" else 0.0,
            1.0 if preferred_suit == "C" else 0.0,
            1.0 if preferred_suit == "S" else 0.0,
        ]
    )

    features.extend(_match_score_vector(seat_context, match_scores, target_score))
    features.extend(_highest_bid_one_hot(auction_state))
    features.extend(_highest_bidder_one_hot(seat_context, auction_state))
    features.extend(
        _one_hot(
            _relative_seat_index(seat_context, auction_state.current_bidder_name),
            3,
        )
    )
    features.extend(
        _one_hot(_relative_seat_index(seat_context, auction_state.dealer_name), 3)
    )
    features.extend(_bid_history_features(seat_context, auction_state))
    features.extend(
        [
            len(auction_state.passed_player_names) / 3.0,
            len(auction_state.bid_history) / 9.0,
            1.0 if auction_state.is_complete else 0.0,
            1.0 if can_pass else 0.0,
            1.0 if auction_state.highest_bid is None else 0.0,
        ]
    )

    return features


def encode_play_candidate(
    *,
    round_state: RoundState,
    acting_player_name: str,
    candidate_card: Card,
    match_scores: dict[str, int] | None,
    target_score: int,
    auction_state: AuctionState | None,
) -> list[float]:
    player_names = [player.name for player in round_state.players]
    seat_context = _relative_seat_context(player_names, acting_player_name)
    player_by_name = {
        player.name: player
        for player in round_state.players
    }
    acting_player = player_by_name[acting_player_name]
    public_seen_cards = {
        play.card
        for trick in round_state.trick_history
        for play in trick.plays
    } | {play.card for play in round_state.current_trick.plays}
    voids, no_joker_players = _public_void_constraints(round_state, acting_player_name)
    current_trick = round_state.current_trick
    trick_points = sum(
        (1 if play.card.is_joker else GAME_VALUES.get(play.card.rank, 0))
        for play in current_trick.plays
    )
    preferred_suit = _preferred_suit(acting_player.cards)
    lead_card = current_trick.plays[0].card if current_trick.plays else None

    features: list[float] = [1.0]
    features.extend(_encode_cards(set(acting_player.cards)))
    features.extend(_encode_cards(public_seen_cards - set(acting_player.cards)))

    for slot_index in range(2):
        if slot_index < len(current_trick.plays):
            play = current_trick.plays[slot_index]
            features.extend(
                _one_hot(_relative_seat_index(seat_context, play.player.name), 3)
            )
            features.extend(_encode_card(play.card))
        else:
            features.extend(_zero_vector(3))
            features.extend(_zero_vector(len(THREE_PLAYER_CARD_CODES)))

    features.extend(_trump_one_hot(round_state.trump))
    features.extend(_match_score_vector(seat_context, match_scores, target_score))
    features.extend(
        [
            len(player_by_name[player_name].cards) / HAND_SIZE
            for player_name in seat_context.ordered_names
        ]
    )
    features.extend(_one_hot(len(current_trick.plays), 3))
    features.extend(_one_hot(len(round_state.trick_history), HAND_SIZE))
    features.extend(_highest_bid_one_hot(auction_state))
    features.extend(_highest_bidder_one_hot(seat_context, auction_state))

    for opponent_name in (seat_context.left_name, seat_context.right_name):
        features.extend(
            [
                1.0 if suit in voids.get(opponent_name, set()) else 0.0
                for suit in SUIT_ORDER
            ]
        )
    features.extend(
        [
            1.0 if opponent_name in no_joker_players else 0.0
            for opponent_name in (seat_context.left_name, seat_context.right_name)
        ]
    )

    action_suit = candidate_card.suit
    if candidate_card.is_joker and round_state.trump is not None:
        action_suit = round_state.trump

    same_suit_cards_before = [
        card
        for card in acting_player.cards
        if (
            not card.is_joker
            and action_suit is not None
            and card.suit == action_suit
        )
    ]
    higher_same_suit = 0
    lower_same_suit = 0
    if action_suit is not None and not candidate_card.is_joker:
        candidate_rank_order = RANK_ORDER[candidate_card.rank]
        higher_same_suit = sum(
            1
            for card in same_suit_cards_before
            if card.rank is not None and RANK_ORDER[card.rank] > candidate_rank_order
        )
        lower_same_suit = sum(
            1
            for card in same_suit_cards_before
            if card.rank is not None and RANK_ORDER[card.rank] < candidate_rank_order
        )

    features.extend(_encode_card(candidate_card))
    features.extend(_suit_one_hot(candidate_card))
    features.extend(_rank_one_hot(candidate_card))
    features.extend(
        [
            (1.0 if candidate_card.is_joker else GAME_VALUES.get(candidate_card.rank, 0) / 10.0),
            1.0 if candidate_card.is_joker else 0.0,
            1.0
            if (
                round_state.trump is not None
                and (
                    candidate_card.is_joker
                    or candidate_card.suit == round_state.trump
                )
            )
            else 0.0,
            1.0 if would_win(candidate_card, current_trick) else 0.0,
            1.0 if not current_trick.plays else 0.0,
            1.0 if round_state.trump is None else 0.0,
            1.0
            if (
                lead_card is not None
                and not candidate_card.is_joker
                and not lead_card.is_joker
                and candidate_card.suit == lead_card.suit
            )
            else 0.0,
            1.0
            if (
                round_state.trump is None
                and not candidate_card.is_joker
                and candidate_card.suit == preferred_suit
            )
            else 0.0,
            len(same_suit_cards_before) / HAND_SIZE,
            max(len(same_suit_cards_before) - (0 if candidate_card.is_joker else 1), 0)
            / max(HAND_SIZE - 1, 1),
            higher_same_suit / 4.0,
            lower_same_suit / 4.0,
            (_suit_strength(acting_player.cards, action_suit) / 5.5)
            if action_suit is not None
            else 0.0,
            1.0
            if (
                not candidate_card.is_joker
                and candidate_card.rank == "A"
            )
            else 0.0,
            1.0
            if (
                not candidate_card.is_joker
                and candidate_card.rank == "J"
            )
            else 0.0,
            1.0
            if (
                not candidate_card.is_joker
                and candidate_card.rank == THREE_PLAYER_LOW_RANK
            )
            else 0.0,
            trick_points / 16.0,
        ]
    )

    return features


def encode_auction_candidate(
    *,
    auction_state: AuctionState,
    acting_player_name: str,
    hand: set[Card],
    candidate_action: AuctionEvent,
    match_scores: dict[str, int] | None,
    target_score: int,
) -> list[float]:
    seat_context = _relative_seat_context(auction_state.player_names, acting_player_name)
    preferred_suit = _preferred_suit(hand)
    suit_strengths = {
        suit: _suit_strength(hand, suit)
        for suit in SUIT_ORDER
    }
    sorted_strengths = sorted(suit_strengths.values(), reverse=True)
    legal_actions = ordered_legal_auction_actions(auction_state)
    can_pass = any(action.action == "pass" for action in legal_actions)
    next_bid = (
        1 if auction_state.highest_bid is None
        else auction_state.highest_bid + 1
    )
    joker_count = _count_jokers(hand)
    total_game_value = sum(
        1 if card.is_joker else GAME_VALUES.get(card.rank, 0)
        for card in hand
    )

    features: list[float] = [1.0]
    features.extend(_encode_cards(set(hand)))

    for suit in SUIT_ORDER:
        features.append(_count_suit(hand, suit) / HAND_SIZE)
    for suit in SUIT_ORDER:
        features.append(_suit_game_value(hand, suit) / 20.0)
    for suit in SUIT_ORDER:
        features.append(suit_strengths[suit] / 5.5)
    for suit in SUIT_ORDER:
        features.append(
            1.0
            if any(
                not card.is_joker and card.suit == suit and card.rank == "A"
                for card in hand
            )
            else 0.0
        )
    for suit in SUIT_ORDER:
        features.append(
            1.0
            if any(
                not card.is_joker and card.suit == suit and card.rank == THREE_PLAYER_LOW_RANK
                for card in hand
            )
            else 0.0
        )

    features.extend(_one_hot(joker_count, 3))
    features.extend(
        [
            total_game_value / 30.0,
            max(sorted_strengths[0], 0.0) / 5.5,
            max(sorted_strengths[1], 0.0) / 5.5,
            1.0 if preferred_suit == "H" else 0.0,
            1.0 if preferred_suit == "D" else 0.0,
            1.0 if preferred_suit == "C" else 0.0,
            1.0 if preferred_suit == "S" else 0.0,
        ]
    )

    features.extend(_match_score_vector(seat_context, match_scores, target_score))
    features.extend(_highest_bid_one_hot(auction_state))
    features.extend(_highest_bidder_one_hot(seat_context, auction_state))
    features.extend(_one_hot(_relative_seat_index(seat_context, auction_state.current_bidder_name), 3))
    features.extend(_one_hot(_relative_seat_index(seat_context, auction_state.dealer_name), 3))
    features.extend(_bid_history_features(seat_context, auction_state))

    features.extend(
        [
            1.0 if candidate_action.action == "pass" else 0.0,
            1.0 if candidate_action.action == "bid" else 0.0,
            1.0 if can_pass else 0.0,
            1.0 if auction_state.highest_bid is None else 0.0,
            1.0 if not can_pass else 0.0,
            1.0 if candidate_action.amount == next_bid else 0.0,
            ((candidate_action.amount or 0) / 6.0),
            ((candidate_action.amount or 0) - (auction_state.highest_bid or 0)) / 6.0,
        ]
    )
    if candidate_action.action == "bid" and candidate_action.amount is not None:
        features.extend(_one_hot(candidate_action.amount - 1, 6))
    else:
        features.extend(_zero_vector(6))

    return features
