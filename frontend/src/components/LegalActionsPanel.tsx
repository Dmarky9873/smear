import { PlayingCard } from "./PlayingCard";
import type { Card, LegalAction, Player } from "../types";

type LegalActionsPanelProps = {
  actions: LegalAction[];
  currentPlayer: Player | null;
  trump: string | null;
  isTerminal: boolean;
  disabled: boolean;
  onPlay: (cardCode: string) => Promise<void>;
};

function fallbackCard(cardCode: string): Card {
  if (cardCode.startsWith("J")) {
    return { code: cardCode, rank: null, suit: null, is_joker: true };
  }

  return {
    code: cardCode,
    rank: cardCode.slice(0, -1) || null,
    suit: cardCode.slice(-1) || null,
    is_joker: false,
  };
}

export function LegalActionsPanel({
  actions,
  currentPlayer,
  trump,
  isTerminal,
  disabled,
  onPlay,
}: LegalActionsPanelProps) {
  return (
    <section className="panel">
      <h2>Legal Actions</h2>
      {isTerminal ? (
        <p className="muted">The round is terminal. No more actions are available.</p>
      ) : null}
      <div className="card-row">
        {actions.map((action) => {
          const card =
            currentPlayer?.cards.find((candidate) => candidate.code === action.card_code) ??
            fallbackCard(action.card_code);
          return (
            <PlayingCard
              key={action.card_code}
              card={card}
              isTrump={!card.is_joker && trump === card.suit}
              disabled={disabled}
              onClick={() => onPlay(action.card_code)}
            />
          );
        })}
        {!isTerminal && actions.length === 0 ? (
          <span className="muted">No legal actions returned by the backend.</span>
        ) : null}
      </div>
    </section>
  );
}
