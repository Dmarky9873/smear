import * as PlayingCardDeck from "@letele/playing-cards";
import type { ComponentType, SVGProps } from "react";

import type { Card } from "../types";

type PlayingCardProps = {
  card: Card;
  isTrump?: boolean;
  compact?: boolean;
  disabled?: boolean;
  className?: string;
  onClick?: () => void;
};

function getCardTitle(card: Card): string {
  if (card.is_joker) {
    return card.code === "J2" ? "Joker 2" : "Joker 1";
  }

  const suitNames: Record<string, string> = {
    C: "Clubs",
    D: "Diamonds",
    H: "Hearts",
    S: "Spades",
  };
  const rankNames: Record<string, string> = {
    A: "Ace",
    J: "Jack",
    Q: "Queen",
    K: "King",
    "10": "Ten",
    "9": "Nine",
    "8": "Eight",
    "7": "Seven",
    "6": "Six",
    "5": "Five",
    "4": "Four",
    "3": "Three",
    "2": "Two",
  };

  const rank = card.rank ? rankNames[card.rank] ?? card.rank : "";
  const suit = card.suit ? suitNames[card.suit] ?? card.suit : "";
  return `${rank} of ${suit}`;
}

type CardSvgComponent = ComponentType<
  SVGProps<SVGSVGElement> & {
    title?: string;
    titleId?: string;
  }
>;

function getCardAssetKey(card: Card): string {
  if (card.is_joker) {
    return card.code;
  }

  const rankMap: Record<string, string> = {
    A: "a",
    J: "j",
    Q: "q",
    K: "k",
  };

  const suit = card.suit ?? "";
  const rank = card.rank ? rankMap[card.rank] ?? card.rank : "";
  return `${suit}${rank}`;
}

function getCardSvg(card: Card): CardSvgComponent | null {
  const assetKey = getCardAssetKey(card);
  return (
    (PlayingCardDeck as Record<string, CardSvgComponent | undefined>)[assetKey] ??
    null
  );
}

export function PlayingCard({
  card,
  isTrump = false,
  compact = false,
  disabled = false,
  className = "",
  onClick,
}: PlayingCardProps) {
  const combinedClassName = [
    "playing-card",
    isTrump ? "playing-card--trump" : "",
    compact ? "playing-card--compact" : "",
    onClick ? "playing-card--clickable" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const accessibleTitle = getCardTitle(card);
  const CardSvg = getCardSvg(card);

  const content = (
    <div className="playing-card__art" aria-hidden="true">
      {CardSvg ? (
        <CardSvg
          className="playing-card__svg"
          title={accessibleTitle}
          style={{ width: "100%", height: "100%" }}
        />
      ) : (
        <div className="playing-card__fallback">
          <strong>{card.code}</strong>
        </div>
      )}
    </div>
  );

  if (onClick) {
    return (
      <button
        type="button"
        className={combinedClassName}
        onClick={onClick}
        disabled={disabled}
        aria-label={accessibleTitle}
        title={accessibleTitle}
      >
        {content}
      </button>
    );
  }

  return (
    <div className={combinedClassName} aria-label={accessibleTitle} title={accessibleTitle}>
      {content}
    </div>
  );
}
