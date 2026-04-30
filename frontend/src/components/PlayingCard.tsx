import type { Card } from "../types";

type PlayingCardProps = {
  card: Card;
  isTrump?: boolean;
  compact?: boolean;
  disabled?: boolean;
  onClick?: () => void;
};

function getCardTitle(card: Card): string {
  if (card.is_joker) {
    return "JOKER";
  }

  return `${card.rank ?? ""}${card.suit ?? ""}`;
}

export function PlayingCard({
  card,
  isTrump = false,
  compact = false,
  disabled = false,
  onClick,
}: PlayingCardProps) {
  const className = [
    "playing-card",
    isTrump ? "playing-card--trump" : "",
    compact ? "playing-card--compact" : "",
    onClick ? "playing-card--clickable" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const content = (
    <>
      <strong>{getCardTitle(card)}</strong>
      <span>{card.is_joker ? "joker" : `${card.rank} ${card.suit}`}</span>
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        onClick={onClick}
        disabled={disabled}
      >
        {content}
      </button>
    );
  }

  return <div className={className}>{content}</div>;
}
