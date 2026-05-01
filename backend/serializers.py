from __future__ import annotations

try:
    from .bots.registry import get_ready_bot_spec
    from .constants import RANK_ORDER
    from .engine import Game, get_trick_winner
    from .models import Card, Play, Player, RoundState, Team, TrickState
    from .store import GameSession
except ImportError:
    from bots.registry import get_ready_bot_spec
    from constants import RANK_ORDER
    from engine import Game, get_trick_winner
    from models import Card, Play, Player, RoundState, Team, TrickState
    from store import GameSession


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


def serialize_player(player: Player, session: GameSession) -> dict:
    captured_cards = sort_cards(player.captured_cards)
    bot_id = session.player_bot_ids.get(player.name)
    bot_label = get_ready_bot_spec(bot_id).label if bot_id is not None else None
    return {
        "name": player.name,
        "bot_id": bot_id,
        "bot_label": bot_label,
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


def serialize_round_state(session: GameSession) -> dict:
    round_state = session.game.round_state
    hidden_cards = sort_cards(round_state.hidden_cards)
    return {
        "players": [serialize_player(player, session) for player in round_state.players],
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


def serialize_auction(session: GameSession) -> dict:
    player_order = session.player_names
    auction = session.auction.state

    return {
        "dealer_name": auction.dealer_name,
        "current_bidder_name": auction.current_bidder_name,
        "current_high_bid": auction.highest_bid,
        "highest_bidder_name": auction.highest_bidder_name,
        "passed_player_names": [
            name for name in player_order if name in auction.passed_player_names
        ],
        "active_player_names": auction.active_player_names,
        "bid_history": [
            {
                "bidder_name": event.bidder_name,
                "action": event.action,
                "amount": event.amount,
            }
            for event in auction.bid_history
        ],
        "is_complete": auction.is_complete,
    }


def serialize_match(session: GameSession) -> dict:
    return {
        "round_number": session.round_number,
        "target_score": session.target_score,
        "scores": [
            {"name": unit_name, "points": session.match_scores[unit_name]}
            for unit_name in session.score_unit_names
        ],
        "is_complete": session.is_match_complete,
        "winner_names": session.match_winner_names,
    }


def serialize_game(session: GameSession) -> dict:
    game = session.game
    return {
        "num_players": game.num_players,
        "low": game.low,
        "phase": session.phase,
        "auction": serialize_auction(session),
        "match": serialize_match(session),
        "round": serialize_round_state(session),
    }


def serialize_score_details(score_details: dict) -> dict:
    return {
        "trump": score_details["trump"],
        "high_card": serialize_card(score_details["high_card"]),
        "low_card": serialize_card(score_details["low_card"]),
        "bid_summary": {
            "bidder_name": score_details["bid_summary"].get("bidder_name"),
            "unit_name": score_details["bid_summary"].get("unit_name"),
            "amount": score_details["bid_summary"].get("amount"),
            "points_won": score_details["bid_summary"].get("points_won"),
            "made_bid": score_details["bid_summary"].get("made_bid"),
            "match_delta": score_details["bid_summary"].get("match_delta"),
        },
        "awards": {
            name: {
                "unit_name": award.get("unit_name"),
                "player_name": award.get("player_name"),
                "card": serialize_card(award["card"]) if award.get("card") else None,
                "game_total": award.get("game_total"),
                "tied_unit_names": award.get("tied_unit_names"),
                "reason": award.get("reason"),
            }
            for name, award in score_details["awards"].items()
        },
        "results": [
            {
                "name": result["name"],
                "member_names": result["member_names"],
                "breakdown": result["breakdown"],
                "joker_count": result["joker_count"],
                "game_total": result["game_total"],
                "total_points": result["total_points"],
                "match_delta": result["match_delta"],
                "bid_amount": result.get("bid_amount"),
                "made_bid": result.get("made_bid"),
                "captured_cards": [
                    serialize_card(card) for card in sort_cards(result["captured_cards"])
                ],
            }
            for result in score_details["results"]
        ],
    }
