from __future__ import annotations

try:
    from .constants import RANK_ORDER
    from .models import Card
except ImportError:
    from constants import RANK_ORDER
    from models import Card


SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3}


def sort_cards(
    cards: list[Card] | set[Card],
    descending_rank: bool = False,
    jokers_first: bool = False,
    suit_order: dict[str, int] | None = None,
) -> list[Card]:
    """Sort cards with customizable ordering.

    Args:
        cards: List or set of cards to sort.
        descending_rank: If True, sort ranks in descending order (highest first). Defaults to False.
        jokers_first: If True, jokers appear first; otherwise they appear last. Defaults to False.
        suit_order: Custom suit ordering dict. Defaults to {"C": 0, "D": 1, "H": 2, "S": 3}.

    Returns:
        Sorted list of cards.
    """
    if suit_order is None:
        suit_order = SUIT_ORDER

    def sort_key(card: Card):
        # Joker positioning
        if jokers_first:
            joker_key = 0 if card.is_joker else 1
        else:
            joker_key = 1 if card.is_joker else 0

        # Suit
        suit_key = suit_order.get(card.suit or "", 99)

        # Rank (negate if descending)
        rank_val = RANK_ORDER.get(card.rank or "", 99)
        rank_key = -rank_val if descending_rank else rank_val

        return (joker_key, suit_key, rank_key, card.code)

    return sorted(cards, key=sort_key)
