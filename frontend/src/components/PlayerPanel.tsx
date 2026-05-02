import { sortCardsHighToLow } from "../cardSort";
import { PlayingCard } from "./PlayingCard";
import type { Player } from "../types";

type PlayerPanelProps = {
  player: Player;
  trump: string | null;
  isCurrentPlayer: boolean;
};

export function PlayerPanel({
  player,
  trump,
  isCurrentPlayer,
}: PlayerPanelProps) {
  const orderedCards = sortCardsHighToLow(player.cards);

  return (
    <section className={`player-panel ${isCurrentPlayer ? "player-panel--current" : ""}`}>
      <div className="player-panel__header">
        <div className="player-panel__title">
          <h3>{player.name}</h3>
          {player.bot_label ? (
            <span className="player-panel__bot-badge">{player.bot_label} bot</span>
          ) : null}
        </div>
        <span>{isCurrentPlayer ? "Current player" : "Waiting"}</span>
      </div>

      <div className="player-panel__group">
        <strong>Hand</strong>
        <div className="card-row">
          {orderedCards.map((card) => (
            <PlayingCard
              key={card.code}
              card={card}
              isTrump={!card.is_joker && trump === card.suit}
            />
          ))}
          {player.cards.length === 0 ? <span className="muted">No cards left.</span> : null}
        </div>
      </div>

      <div className="player-panel__group">
        <strong>Captured ({player.captured_count})</strong>
        <div className="card-row">
          {player.captured_cards.map((card) => (
            <PlayingCard
              key={`${player.name}-${card.code}`}
              card={card}
              compact
              isTrump={!card.is_joker && trump === card.suit}
            />
          ))}
          {player.captured_cards.length === 0 ? (
            <span className="muted">No captured cards.</span>
          ) : null}
        </div>
      </div>
    </section>
  );
}
