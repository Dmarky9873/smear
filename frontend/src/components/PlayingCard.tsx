import type { Card } from "../types";

type PlayingCardProps = {
  card: Card;
  trumpSuit?: string | null;
};

const SUIT_SYMBOLS: Record<string, string> = {
  clubs: "♣",
  diamonds: "♦",
  hearts: "♥",
  spades: "♠",
};

export function PlayingCard({ card, trumpSuit }: PlayingCardProps) {
  const suitSymbol = card.suit ? SUIT_SYMBOLS[card.suit] ?? card.suit : "";
  const isRedSuit = card.suit === "hearts" || card.suit === "diamonds";
  const isTrump = Boolean(trumpSuit && card.suit === trumpSuit);

  return (
    <div
      className={[
        "playing-card",
        isRedSuit ? "playing-card-red" : "",
        isTrump ? "playing-card-trump" : "",
        card.is_joker ? "playing-card-joker" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span className="playing-card-rank">{card.rank}</span>
      <span className="playing-card-suit">{card.is_joker ? "JOKER" : suitSymbol}</span>
    </div>
  );
}
