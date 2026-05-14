import type { Card } from "./types";

const RANK_ORDER: Record<string, number> = {
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
  "6": 6,
  "7": 7,
  "8": 8,
  "9": 9,
  "10": 10,
  J: 11,
  Q: 12,
  K: 13,
  A: 14,
};

const SUIT_ORDER: Record<string, number> = {
  C: 0,
  D: 1,
  H: 2,
  S: 3,
};

export function sortCardsHighToLow(cards: Card[]): Card[] {
  return [...cards].sort((left, right) => {
    if (left.is_joker !== right.is_joker) {
      return left.is_joker ? -1 : 1;
    }

    const leftRank = left.rank ? RANK_ORDER[left.rank] ?? -1 : -1;
    const rightRank = right.rank ? RANK_ORDER[right.rank] ?? -1 : -1;
    if (leftRank !== rightRank) {
      return rightRank - leftRank;
    }

    const leftSuit = left.suit ? SUIT_ORDER[left.suit] ?? 99 : 99;
    const rightSuit = right.suit ? SUIT_ORDER[right.suit] ?? 99 : 99;
    if (leftSuit !== rightSuit) {
      return leftSuit - rightSuit;
    }

    return left.code.localeCompare(right.code);
  });
}
