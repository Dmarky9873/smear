from __future__ import annotations

try:
    from .constants import RANK_ORDER
    from .engine import Game, get_trick_winner
    from .models import Card, Play, Player, RoundState, Team, TrickState
except ImportError:
    from constants import RANK_ORDER
    from engine import Game, get_trick_winner
    from models import Card, Play, Player, RoundState, Team, TrickState


SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3}


def sort_cards(cards: list[Card] | set[Card]) -> list[Card]:
    return sorted(
        cards,
        key=lambda card: (
            1 if card.is_joker else 0,
            SUIT_ORDER.get(card.suit or "", 99),
            RANK_ORDER.get(card.rank or "", 99),
            card.code,
        ),
    )


def serialize_card(card: Card) -> dict:
    return {
        "code": card.code,
        "rank": card.rank,
        "suit": card.suit,
        "is_joker": card.is_joker,
    }


def serialize_player(player: Player) -> dict:
    captured_cards = sort_cards(player.captured_cards)
    return {
        "name": player.name,
        "cards": [serialize_card(card) for card in sort_cards(player.cards)],
        "captured_cards": [serialize_card(card) for card in captured_cards],
        "captured_count": len(captured_cards),
    }


def serialize_play(play: Play) -> dict:
    return {
        "player_name": play.player.name,
        "card": serialize_card(play.card),
    }


def serialize_trick(trick: TrickState) -> dict:
    winner_name = None
    if trick.is_terminal and trick.trump is not None:
        winner_name = get_trick_winner(trick).name

    return {
        "leader_name": trick.leader.name,
        "plays": [serialize_play(play) for play in trick.plays],
        "trump": trick.trump,
        "is_terminal": trick.is_terminal,
        "winner_name": winner_name,
    }


def serialize_team(team: Team) -> dict:
    captured_cards = sort_cards(
        {
            card
            for player in team.constituents
            for card in player.captured_cards
        }
    )
    return {
        "constituents": [player.name for player in team.constituents],
        "captured_cards": [serialize_card(card) for card in captured_cards],
        "captured_count": len(captured_cards),
    }


def serialize_round_state(round_state: RoundState) -> dict:
    hidden_cards = sort_cards(round_state.hidden_cards)
    return {
        "players": [serialize_player(player) for player in round_state.players],
        "current_player_name": round_state.current_player.name,
        "trump": round_state.trump,
        "current_trick": serialize_trick(round_state.current_trick),
        "hidden_cards_count": len(hidden_cards),
        "hidden_cards": [serialize_card(card) for card in hidden_cards],
        "trick_history": [
            serialize_trick(trick) for trick in round_state.trick_history
        ],
        "teams": [serialize_team(team) for team in round_state.teams],
        "is_terminal": round_state.is_terminal,
    }


def serialize_game(game: Game) -> dict:
    return {
        "num_players": game.num_players,
        "low": game.low,
        "round": serialize_round_state(game.round_state),
    }
