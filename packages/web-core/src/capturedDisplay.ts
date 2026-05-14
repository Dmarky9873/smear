import type { Card, RoundState } from "./types";

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

export type DisplayCapturedPlayer = {
  cards: Card[];
  count: number;
};

export type DisplayCapturedByPlayer = Record<string, DisplayCapturedPlayer>;

function sortCardsForCapturedDisplay(cards: Card[]): Card[] {
  return [...cards].sort((left, right) => {
    if (left.is_joker !== right.is_joker) {
      return left.is_joker ? 1 : -1;
    }

    const leftSuit = left.suit ? SUIT_ORDER[left.suit] ?? 99 : 99;
    const rightSuit = right.suit ? SUIT_ORDER[right.suit] ?? 99 : 99;
    if (leftSuit !== rightSuit) {
      return leftSuit - rightSuit;
    }

    const leftRank = left.rank ? RANK_ORDER[left.rank] ?? 99 : 99;
    const rightRank = right.rank ? RANK_ORDER[right.rank] ?? 99 : 99;
    if (leftRank !== rightRank) {
      return leftRank - rightRank;
    }

    return left.code.localeCompare(right.code);
  });
}

function getVisibleLowTrumpCard(round: RoundState): Card | null {
  if (!round.trump) {
    return null;
  }

  const visibleTrumpCards = new Map<string, Card>();
  const addCard = (card: Card) => {
    if (!card.is_joker && card.suit === round.trump) {
      visibleTrumpCards.set(card.code, card);
    }
  };

  for (const player of round.players) {
    for (const card of player.cards) {
      addCard(card);
    }
  }

  for (const trick of [...round.trick_history, round.current_trick]) {
    for (const play of trick.plays) {
      addCard(play.card);
    }
  }

  let lowCard: Card | null = null;
  for (const card of visibleTrumpCards.values()) {
    if (
      !lowCard ||
      (card.rank !== null &&
        lowCard.rank !== null &&
        RANK_ORDER[card.rank] < RANK_ORDER[lowCard.rank])
    ) {
      lowCard = card;
    }
  }

  return lowCard;
}

function getPlayerWhoPlayedCard(
  round: RoundState,
  targetCardCode: string,
): string | null {
  for (const trick of [...round.trick_history, round.current_trick]) {
    for (const play of trick.plays) {
      if (play.card.code === targetCardCode) {
        return play.player_name;
      }
    }
  }

  return null;
}

export function buildDisplayCapturedByPlayer(
  round: RoundState,
): DisplayCapturedByPlayer {
  const displayByPlayer = Object.fromEntries(
    round.players.map((player) => [
      player.name,
      {
        cards: sortCardsForCapturedDisplay(player.captured_cards),
        count: player.captured_count,
      },
    ]),
  ) as DisplayCapturedByPlayer;

  const lowCard = getVisibleLowTrumpCard(round);
  if (!lowCard) {
    return displayByPlayer;
  }

  const lowOwnerName = getPlayerWhoPlayedCard(round, lowCard.code);
  if (!lowOwnerName || !displayByPlayer[lowOwnerName]) {
    return displayByPlayer;
  }

  const capturedByName = round.players.find((player) =>
    player.captured_cards.some((card) => card.code === lowCard.code),
  )?.name;

  if (!capturedByName || capturedByName === lowOwnerName) {
    return displayByPlayer;
  }

  displayByPlayer[capturedByName] = {
    cards: displayByPlayer[capturedByName].cards.filter(
      (card) => card.code !== lowCard.code,
    ),
    count: Math.max(0, displayByPlayer[capturedByName].count - 1),
  };

  displayByPlayer[lowOwnerName] = {
    cards: sortCardsForCapturedDisplay([
      ...displayByPlayer[lowOwnerName].cards,
      lowCard,
    ]),
    count: displayByPlayer[lowOwnerName].count + 1,
  };

  return displayByPlayer;
}

export function getDisplayCapturedCountForTeam(
  displayByPlayer: DisplayCapturedByPlayer,
  constituentNames: string[],
): number {
  return constituentNames.reduce(
    (total, name) => total + (displayByPlayer[name]?.count ?? 0),
    0,
  );
}
