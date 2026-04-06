from __future__ import annotations

import random

from app.models import (
    AuctionHistoryEntry,
    Card,
    GameActionRequest,
    GameState,
    LegalAction,
    NewGameRequest,
    PlayedCard,
    Player,
)

SUITS = ("clubs", "diamonds", "hearts", "spades")
RANKS = ("9", "10", "J", "Q", "K", "A")
RANK_VALUES = {rank: index for index, rank in enumerate(RANKS)}
HAND_SIZE = 5


def create_mock_game_state(game_id: str, request: NewGameRequest) -> GameState:
    """Create a predictable mock game state for local UI and API testing."""
    # TODO: Replace this with true Smear round setup, seating, dealing, and metadata.
    player_count = max(2, request.player_count)
    rng = random.Random(request.seed)
    deck = _build_mock_deck(rng)
    players = _build_players(player_count, request.player_names, deck)
    team_ids = sorted({player.team_id if player.team_id is not None else player.id for player in players})

    state = GameState(
        game_id=game_id,
        phase="auction",
        players=players,
        current_player_id=0,
        dealer_id=player_count - 1,
        scores={f"team_{team_id}": 0 for team_id in team_ids},
        round_points={f"team_{team_id}": 0 for team_id in team_ids},
        logs=[f"Created mock game {game_id}."],
        debug={
            "seed": request.seed,
            "debug_mode": request.debug,
            "turn_index": 0,
            "auction_turns": 0,
            "resets": 0,
            "notes": [
                "Placeholder Smear scaffold. Replace mock_logic.py with real rules.",
                "Hands are visible for debugging by design.",
            ],
            "todo": {
                "state_initialization": "Implement real Smear shuffling, dealing, and starting player selection.",
                "move_generation": "Replace placeholder legal action generation with rule-aware bidding and trick play.",
                "bidding_logic": "Implement real bidding constraints and auction termination.",
                "trick_resolution": "Implement actual trick winner evaluation and follow-suit enforcement.",
                "scoring": "Implement real round scoring and game-over conditions.",
                "ai": "Replace placeholder AI action selection with minimax / determinization / imperfect-information search.",
            },
        },
    )
    state.legal_actions = compute_legal_actions(state)
    return state


def compute_legal_actions(state: GameState) -> list[LegalAction]:
    """Return the currently legal mock actions."""
    # TODO: Replace this with real Smear legal move generation.
    if state.phase == "auction":
        next_bid = 2 if state.current_bid is None else state.current_bid + 1
        return [
            LegalAction(type="bid", value=next_bid),
            LegalAction(type="bid", value=next_bid + 1),
            LegalAction(type="pass"),
        ]

    if state.phase == "choose_trump":
        return [LegalAction(type="choose_trump", suit=suit) for suit in SUITS]

    if state.phase == "play":
        player = _get_player(state, state.current_player_id)
        return [LegalAction(type="play_card", card_id=card.id) for card in player.hand]

    return []


def apply_action_to_state(state: GameState, action: GameActionRequest) -> GameState:
    """Apply a mock action to the state in-place and recompute legal actions."""
    # TODO: Replace this dispatcher with real state transitions and validation.
    matched_action = _find_matching_action(state.legal_actions, action)
    if matched_action is None:
        raise ValueError("Action is not legal in the current placeholder state.")

    if state.phase == "auction":
        _apply_auction_action(state, matched_action)
    elif state.phase == "choose_trump":
        _apply_choose_trump_action(state, matched_action)
    elif state.phase == "play":
        _apply_play_action(state, matched_action)
    else:
        raise ValueError(f"Cannot apply actions during phase '{state.phase}'.")

    state.debug["turn_index"] = int(state.debug.get("turn_index", 0)) + 1
    state.debug["last_action"] = matched_action.model_dump()
    state.legal_actions = compute_legal_actions(state)
    return state


def advance_turn(state: GameState) -> None:
    """Move to the next player in seat order."""
    player_ids = [player.id for player in state.players]
    current_index = player_ids.index(state.current_player_id)
    next_index = (current_index + 1) % len(player_ids)
    state.current_player_id = player_ids[next_index]


def choose_placeholder_ai_action(state: GameState) -> LegalAction | None:
    """Choose a simple deterministic action for the placeholder AI."""
    # TODO: Replace this with minimax / determinization / imperfect-information search.
    legal_actions = compute_legal_actions(state)
    if not legal_actions:
        return None
    return legal_actions[0]


def _apply_auction_action(state: GameState, action: LegalAction) -> None:
    player = _get_player(state, state.current_player_id)
    auction_history = state.auction_history

    if action.type == "pass":
        player.has_passed = True
        auction_history.append(AuctionHistoryEntry(player_id=player.id, action="pass"))
        state.logs.append(f"{player.name} passed.")
    else:
        player.bid = action.value
        player.has_passed = False
        state.current_bid = action.value
        state.winning_bidder_id = player.id
        auction_history.append(AuctionHistoryEntry(player_id=player.id, action="bid", value=action.value))
        state.logs.append(f"{player.name} bid {action.value}.")

    state.debug["auction_turns"] = int(state.debug.get("auction_turns", 0)) + 1

    if _auction_is_complete(state):
        _finish_mock_auction(state)
        return

    advance_turn(state)


def _apply_choose_trump_action(state: GameState, action: LegalAction) -> None:
    state.trump_suit = action.suit
    state.phase = "play"
    state.leading_player_id = state.winning_bidder_id if state.winning_bidder_id is not None else 0
    state.current_player_id = state.leading_player_id
    state.logs.append(f"Trump selected: {action.suit}.")


def _apply_play_action(state: GameState, action: LegalAction) -> None:
    player = _get_player(state, state.current_player_id)
    card = _remove_card_from_hand(player, action.card_id)
    played_card = PlayedCard(player_id=player.id, card=card)
    state.current_trick.append(played_card)
    state.logs.append(f"{player.name} played {card.rank} of {card.suit or 'none'}.")

    if len(state.current_trick) < len(state.players):
        advance_turn(state)
        return

    _resolve_mock_trick(state)


def _auction_is_complete(state: GameState) -> bool:
    turns_taken = int(state.debug.get("auction_turns", 0))
    players_still_active = [player for player in state.players if not player.has_passed]
    return turns_taken >= len(state.players) or (
        state.winning_bidder_id is not None and len(players_still_active) <= 1
    )


def _finish_mock_auction(state: GameState) -> None:
    # TODO: Replace this placeholder with actual Smear auction resolution.
    if state.winning_bidder_id is None:
        fallback_bidder = state.dealer_id if state.dealer_id is not None else 0
        state.winning_bidder_id = fallback_bidder
        state.current_bid = 2
        state.logs.append("No bids were made. Dealer assigned winning bid of 2 in mock mode.")

    state.phase = "choose_trump"
    state.current_player_id = state.winning_bidder_id
    state.logs.append("Auction complete. Waiting for mock trump selection.")


def _resolve_mock_trick(state: GameState) -> None:
    # TODO: Replace this with real follow-suit enforcement and trick winner evaluation.
    trick = list(state.current_trick)
    winner = max(trick, key=lambda played: _mock_card_strength(played.card, state.trump_suit))
    winning_player = _get_player(state, winner.player_id)
    winning_player.captured_cards.extend([played.card for played in trick])
    state.completed_tricks.append(trick)
    state.current_trick = []
    state.leading_player_id = winner.player_id
    state.current_player_id = winner.player_id

    team_key = f"team_{winning_player.team_id if winning_player.team_id is not None else winning_player.id}"
    state.round_points[team_key] = state.round_points.get(team_key, 0) + 1
    state.logs.append(f"{winning_player.name} won the mock trick.")

    if any(player.hand for player in state.players):
        return

    state.phase = "round_over"
    for team_key_name, points in state.round_points.items():
        state.scores[team_key_name] = state.scores.get(team_key_name, 0) + points
    state.logs.append("Mock round complete. Use reset or new game to continue testing.")


def _build_mock_deck(rng: random.Random) -> list[Card]:
    deck = [
        Card(id=f"{rank}-{suit}", rank=rank, suit=suit)
        for suit in SUITS
        for rank in RANKS
    ]
    deck.extend(
        [
            Card(id="joker-red", rank="JOKER", suit=None, is_joker=True),
            Card(id="joker-black", rank="JOKER", suit=None, is_joker=True),
        ]
    )
    rng.shuffle(deck)
    return deck


def _build_players(player_count: int, player_names: list[str] | None, deck: list[Card]) -> list[Player]:
    players: list[Player] = []
    resolved_names = player_names or []

    for player_id in range(player_count):
        hand = [deck.pop() for _ in range(HAND_SIZE) if deck]
        team_id = player_id % 2 if player_count % 2 == 0 else None
        players.append(
            Player(
                id=player_id,
                name=resolved_names[player_id] if player_id < len(resolved_names) else f"Player {player_id + 1}",
                team_id=team_id,
                hand=hand,
            )
        )

    return players


def _find_matching_action(legal_actions: list[LegalAction], action: GameActionRequest) -> LegalAction | None:
    for legal_action in legal_actions:
        if (
            legal_action.type == action.type
            and legal_action.value == action.value
            and legal_action.card_id == action.card_id
            and legal_action.suit == action.suit
        ):
            return legal_action
    return None


def _get_player(state: GameState, player_id: int) -> Player:
    for player in state.players:
        if player.id == player_id:
            return player
    raise ValueError(f"Player {player_id} not found.")


def _remove_card_from_hand(player: Player, card_id: str | None) -> Card:
    if card_id is None:
        raise ValueError("card_id is required for play_card actions.")

    for index, card in enumerate(player.hand):
        if card.id == card_id:
            return player.hand.pop(index)

    raise ValueError(f"Card '{card_id}' is not in {player.name}'s hand.")


def _mock_card_strength(card: Card, trump_suit: str | None) -> int:
    if card.is_joker:
        return 100

    base_value = RANK_VALUES.get(card.rank, 0)
    if trump_suit and card.suit == trump_suit:
        return base_value + 50
    return base_value
